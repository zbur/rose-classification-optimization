"""Evaluate a PyTorch checkpoint (.pt) or ONNX model (.onnx): Macro-F1 and per-class recall.

Select settings on --subset val; run --subset test only for the final locked evaluation.

Example:
    python evaluate.py --model models/C_int8_224.onnx --split splits/rose_split.json --subset val
"""
import argparse
import json

import numpy as np

from inference import make_runner
from metrics import classification_metrics
from preprocess import items_for, load_split, preprocess_image


def evaluate(runner, items, img_size, class_names, batch_size=32):
    y_true, y_pred = [], []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        x = np.stack([preprocess_image(it["abs_path"], img_size) for it in batch])
        y_pred.extend(runner(x).argmax(1).tolist())
        y_true.extend(it["label"] for it in batch)
    return classification_metrics(y_true, y_pred, class_names)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--subset", choices=["train", "val", "test"], default="val")
    ap.add_argument("--img-size", type=int, default=None, help="Default: taken from the model.")
    ap.add_argument("--threads", type=int, default=0, help="0 = runtime default.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    split = load_split(args.split)
    runner = make_runner(args.model, threads=args.threads)
    img_size = args.img_size or runner.img_size
    if img_size is None:
        raise SystemExit("Could not infer input size; pass --img-size.")
    m = evaluate(runner, items_for(split, args.subset), img_size, split["classes"])

    print(f"{args.model} [{args.subset}, {img_size}px]  macro-F1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}")
    for name, v in m["per_class"].items():
        print(f"  {name:16s} recall {v['recall']:.4f}  f1 {v['f1']:.4f}  n={v['support']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "subset": args.subset, "img_size": img_size, **m}, f, indent=1)


if __name__ == "__main__":
    main()
