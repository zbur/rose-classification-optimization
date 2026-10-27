"""Two-stage fine-tuning of MobileNetV2 (slide 3).

Stage 1: freeze the whole backbone, train the new 4-class head.
Stage 2: unfreeze the final backbone blocks with a small learning rate.
The checkpoint with the best validation Macro-F1 is kept; the test split is
never used here.

Example:
    python train.py --split splits/rose_split.json --img-size 224 --out runs/fp32_224_s0
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from datasets import make_loader
from metrics import classification_metrics
from model import build_model, get_device, param_groups, save_checkpoint, set_bn_mode, set_trainable
from preprocess import load_split


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_loader(model, loader, device, class_names):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        logits = model(x.to(device))
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(y.tolist())
    return classification_metrics(y_true, y_pred, class_names)


def train_one_epoch(model, loader, optimizer, criterion, device, stage, args):
    model.train()
    set_bn_mode(model, stage, args.unfreeze_from, args.freeze_bn)
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def run_stage(stage, model, loaders, criterion, device, args, state, class_names):
    set_trainable(model, stage, args.unfreeze_from)
    if stage == 1:
        epochs = args.epochs_head
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr_head,
                                      weight_decay=args.weight_decay)
    else:
        epochs = args.epochs_ft
        optimizer = torch.optim.AdamW(
            param_groups(model, args.unfreeze_from, args.lr_ft_backbone, args.lr_ft_head),
            weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    stale = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device, stage, args)
        scheduler.step()
        val = evaluate_loader(model, loaders["val"], device, class_names)
        record = {"stage": stage, "epoch": epoch, "train_loss": loss,
                  "val_macro_f1": val["macro_f1"], "val_accuracy": val["accuracy"],
                  "val_recall": {k: v["recall"] for k, v in val["per_class"].items()},
                  "seconds": time.time() - t0}
        state["history"].append(record)
        print(f"[stage {stage}] epoch {epoch:2d}/{epochs}  loss {loss:.4f}  "
              f"val macro-F1 {val['macro_f1']:.4f}  acc {val['accuracy']:.4f}")

        if val["macro_f1"] > state["best_f1"]:
            state["best_f1"] = val["macro_f1"]
            state["best"] = {"stage": stage, "epoch": epoch}
            save_checkpoint(os.path.join(args.out, "best.pt"), model, {
                "classes": class_names, "img_size": args.img_size, "val_macro_f1": val["macro_f1"],
                "stage": stage, "epoch": epoch, "args": vars(args)})
            stale = 0
        else:
            stale += 1
            if stage == 2 and args.patience > 0 and stale >= args.patience:
                print(f"Early stopping stage 2 after {stale} epochs without improvement.")
                break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs-head", type=int, default=5)
    ap.add_argument("--epochs-ft", type=int, default=20)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-ft-head", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--unfreeze-from", type=int, default=14,
                    help="Index into model.features (0-18) from which blocks are unfrozen in stage 2.")
    ap.add_argument("--no-freeze-bn", dest="freeze_bn", action="store_false",
                    help="Let BatchNorm statistics update in the unfrozen blocks.")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = get_device(args.device)
    split = load_split(args.split)
    class_names = split["classes"]

    loaders = {
        "train": make_loader(split, "train", args.img_size, args.batch_size, True, args.workers),
        "val": make_loader(split, "val", args.img_size, args.batch_size, False, args.workers),
    }
    model = build_model(len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    state = {"best_f1": -1.0, "best": None, "history": []}

    print(f"Device: {device}  train={len(loaders['train'].dataset)}  val={len(loaders['val'].dataset)}")
    run_stage(1, model, loaders, criterion, device, args, state, class_names)
    run_stage(2, model, loaders, criterion, device, args, state, class_names)

    # Weights-only file, used to report the PyTorch model size (variant A).
    best = torch.load(os.path.join(args.out, "best.pt"), map_location="cpu", weights_only=False)
    torch.save(best["state_dict"], os.path.join(args.out, "weights_fp32.pt"))
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({"best": state["best"], "best_val_macro_f1": state["best_f1"],
                   "args": vars(args), "history": state["history"]}, f, indent=1)
    print(f"Best val macro-F1 {state['best_f1']:.4f} at {state['best']}; saved to {args.out}/best.pt")


if __name__ == "__main__":
    main()
