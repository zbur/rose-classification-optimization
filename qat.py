"""Quantization-aware training fallback (slide 4: use QAT only if PTQ loses too much).

Starts from the FP32 fine-tuned checkpoint, fuses Conv+BN(+ReLU), inserts fake
quantization (per-channel symmetric INT8 weights, UINT8 activations, matching
ONNX Runtime's QDQ scheme), trains briefly with a small learning rate, and
exports the fake-quantized model to ONNX as QuantizeLinear/DequantizeLinear
pairs that ONNX Runtime executes as INT8.

Note: the exported graph stores weights as float initializers followed by Q/DQ
nodes, so the file is not smaller than FP32 even though execution is INT8.

Example:
    python qat.py --ckpt runs/fp32_224_s0/best.pt --split splits/rose_split.json \
        --out runs/qat_224_s0 --img-size 224
    python run_experiments.py ... --int8-224 runs/qat_224_s0/qat_int8_224.onnx
"""
import argparse
import json
import os

import torch
import torch.nn as nn
from torch.ao.quantization import (FakeQuantize, MovingAverageMinMaxObserver,
                                   MovingAveragePerChannelMinMaxObserver, QConfig, prepare_qat)
from torchvision.models.quantization import mobilenet_v2 as quantizable_mobilenet_v2

from datasets import make_loader
from export_onnx import onnx_export
from model import get_device
from preprocess import load_split
from train import evaluate_loader, set_seed

ORT_QCONFIG = QConfig(
    activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255,
                                      dtype=torch.quint8, qscheme=torch.per_tensor_affine),
    weight=FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, quant_min=-128,
                                  quant_max=127, dtype=torch.qint8,
                                  qscheme=torch.per_channel_symmetric, ch_axis=0),
)


def build_qat_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    num_classes = len(ckpt["classes"])
    model = quantizable_mobilenet_v2(weights=None, quantize=False)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, num_classes))
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch: missing={missing} unexpected={unexpected}")
    model.train()
    model.fuse_model(is_qat=True)
    model.qconfig = ORT_QCONFIG
    prepare_qat(model, inplace=True)
    return model, ckpt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="FP32 checkpoint from train.py.")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--freeze-bn-epoch", type=int, default=2, help="Freeze BN statistics after this epoch.")
    ap.add_argument("--freeze-observer-epoch", type=int, default=4,
                    help="Freeze quantization ranges after this epoch.")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="Fake-quant training is most reliable on CPU/CUDA.")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = get_device(args.device)
    split = load_split(args.split)
    class_names = split["classes"]
    model, ckpt = build_qat_model(args.ckpt)
    img_size = args.img_size or ckpt["img_size"]
    model.to(device)

    train_loader = make_loader(split, "train", img_size, args.batch_size, True, args.workers)
    val_loader = make_loader(split, "val", img_size, args.batch_size, False, args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_f1, history = -1.0, []
    best_path = os.path.join(args.out, "qat_best_state.pt")
    for epoch in range(1, args.epochs + 1):
        model.train()
        if epoch > args.freeze_bn_epoch:
            model.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
        if epoch > args.freeze_observer_epoch:
            model.apply(torch.ao.quantization.disable_observer)
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        val = evaluate_loader(model, val_loader, device, class_names)
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_macro_f1": val["macro_f1"]})
        print(f"[QAT] epoch {epoch}/{args.epochs}  loss {total / max(n, 1):.4f}  val macro-F1 {val['macro_f1']:.4f}")
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to("cpu").eval()
    onnx_path = os.path.join(args.out, f"qat_int8_{img_size}.onnx")
    onnx_export(model, torch.randn(1, 3, img_size, img_size), onnx_path, args.opset)
    with open(os.path.join(args.out, "qat_history.json"), "w") as f:
        json.dump({"best_val_macro_f1": best_f1, "img_size": img_size, "args": vars(args),
                   "history": history}, f, indent=1)
    print(f"Best QAT val macro-F1 {best_f1:.4f}; exported {onnx_path}")


if __name__ == "__main__":
    main()
