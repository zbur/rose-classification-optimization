"""Post-training static INT8 quantization with ONNX Runtime (slide 4).

Calibration uses training images only (never val/test). The default output is
QDQ format with per-channel symmetric INT8 weights and UINT8 activations.

Example:
    python quantize_int8.py --fp32 models/B_fp32_224.onnx --split splits/rose_split.json \
        --out models/C_int8_224.onnx
"""
import argparse
import json
import os
import random
import tempfile

import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from preprocess import items_for, load_split, preprocess_image


class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, items, input_name, img_size, batch_size=8):
        self.items = items
        self.input_name = input_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.pos = 0

    def get_next(self):
        if self.pos >= len(self.items):
            return None
        batch = self.items[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        x = np.stack([preprocess_image(it["abs_path"], self.img_size) for it in batch])
        return {self.input_name: x.astype(np.float32)}

    def rewind(self):
        self.pos = 0


def model_input(path):
    m = onnx.load(path)
    inp = m.graph.input[0]
    dims = inp.type.tensor_type.shape.dim
    return inp.name, dims[2].dim_value


def sample_calibration_items(split, num, seed):
    """Stratified sample from the training split."""
    train = items_for(split, "train")
    rng = random.Random(seed)
    by_label = {}
    for it in train:
        by_label.setdefault(it["label"], []).append(it)
    per_class = max(1, num // max(len(by_label), 1))
    chosen = []
    for label in sorted(by_label):
        items = by_label[label][:]
        rng.shuffle(items)
        chosen.extend(items[:per_class])
    rng.shuffle(chosen)
    return chosen


def quantize(fp32_path, split_path, out_path, num_calib=320, method="minmax", per_channel=True,
             quant_format="qdq", seed=0):
    split = load_split(split_path)
    input_name, img_size = model_input(fp32_path)
    calib_items = sample_calibration_items(split, num_calib, seed)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        prep_path = os.path.join(tmp, "prep.onnx")
        quant_pre_process(fp32_path, prep_path, skip_symbolic_shape=True)
        reader = ImageCalibrationReader(calib_items, input_name, img_size)
        quantize_static(
            prep_path, out_path, reader,
            quant_format=QuantFormat.QDQ if quant_format == "qdq" else QuantFormat.QOperator,
            per_channel=per_channel,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method={"minmax": CalibrationMethod.MinMax,
                              "entropy": CalibrationMethod.Entropy,
                              "percentile": CalibrationMethod.Percentile}[method],
        )

    report = {"fp32": fp32_path, "int8": out_path, "img_size": img_size,
              "num_calibration_images": len(calib_items), "calibration_split": "train",
              "method": method, "per_channel": per_channel, "format": quant_format,
              "fp32_mb": os.path.getsize(fp32_path) / 2**20, "int8_mb": os.path.getsize(out_path) / 2**20}
    with open(os.path.splitext(out_path)[0] + ".quant.json", "w") as f:
        json.dump(report, f, indent=1)
    print(f"Quantized {fp32_path} -> {out_path} "
          f"({report['fp32_mb']:.2f} MB -> {report['int8_mb']:.2f} MB, {len(calib_items)} calib images)")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-calib", type=int, default=320)
    ap.add_argument("--method", choices=["minmax", "entropy", "percentile"], default="minmax")
    ap.add_argument("--per-tensor", action="store_true", help="Use per-tensor instead of per-channel weights.")
    ap.add_argument("--format", choices=["qdq", "qoperator"], default="qdq")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    quantize(args.fp32, args.split, args.out, args.num_calib, args.method,
             not args.per_tensor, args.format, args.seed)


if __name__ == "__main__":
    main()
