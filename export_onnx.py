"""Export a fine-tuned checkpoint to ONNX (FP32) and verify prediction consistency.

Example:
    python export_onnx.py --ckpt runs/fp32_224_s0/best.pt --img-size 224 \
        --split splits/rose_split.json --out models/B_fp32_224.onnx
"""
import argparse
import json
import os

import numpy as np
import torch

from inference import OnnxRunner
from model import load_checkpoint
from preprocess import items_for, load_split, preprocess_image


def onnx_export(model, dummy, out_path, opset):
    kwargs = dict(input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                  opset_version=opset, do_constant_folding=True)
    try:
        torch.onnx.export(model, dummy, out_path, dynamo=False, **kwargs)
    except TypeError:  # older torch without the `dynamo` argument
        torch.onnx.export(model, dummy, out_path, **kwargs)


def verify(model, onnx_path, items, img_size, num_images=64):
    """Compare PyTorch and ONNX Runtime logits / top-1 on validation images."""
    runner = OnnxRunner(onnx_path, threads=0)
    max_diff, agree, n = 0.0, 0, 0
    for item in items[:num_images]:
        x = preprocess_image(item["abs_path"], img_size)[None]
        with torch.inference_mode():
            ref = model(torch.from_numpy(x)).numpy()
        out = runner(x)
        max_diff = max(max_diff, float(np.abs(ref - out).max()))
        agree += int(ref.argmax(1)[0] == out.argmax(1)[0])
        n += 1
    return {"num_images": n, "max_abs_logit_diff": max_diff, "top1_agreement": agree / max(n, 1)}


def export(ckpt_path, img_size, out_path, split_path=None, opset=17, num_verify=64):
    model, meta = load_checkpoint(ckpt_path)
    img_size = img_size or meta["img_size"]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    dummy = torch.randn(1, 3, img_size, img_size)
    onnx_export(model, dummy, out_path, opset)

    report = {"ckpt": ckpt_path, "onnx": out_path, "img_size": img_size,
              "trained_img_size": meta["img_size"], "opset": opset}
    if split_path:
        split = load_split(split_path)
        report["verify"] = verify(model, out_path, items_for(split, "val"), img_size, num_verify)
        v = report["verify"]
        print(f"Verify: max |logit diff| = {v['max_abs_logit_diff']:.2e}, "
              f"top-1 agreement = {v['top1_agreement']:.3f} on {v['num_images']} val images")
        if v["top1_agreement"] < 1.0 or v["max_abs_logit_diff"] > 1e-3:
            print("WARNING: ONNX output differs from PyTorch more than expected.")
    with open(os.path.splitext(out_path)[0] + ".export.json", "w") as f:
        json.dump(report, f, indent=1)
    print(f"Exported {out_path}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=None, help="Default: the size used in training.")
    ap.add_argument("--split", default=None, help="If given, verify consistency on val images.")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--num-verify", type=int, default=64)
    args = ap.parse_args()
    export(args.ckpt, args.img_size, args.out, args.split, args.opset, args.num_verify)


if __name__ == "__main__":
    main()
