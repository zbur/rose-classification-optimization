"""Deployment preprocessing and split loading (NumPy/PIL only, no torch).

Every variant (PyTorch and ONNX Runtime) uses the same preprocessing so that
differences come from the model/runtime, not from the input pipeline.
"""
import json
import os

import numpy as np
from PIL import Image

from config import IMAGENET_MEAN, IMAGENET_STD, RESIZE_RATIO

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
_STD = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)


def load_rgb(path):
    with Image.open(path) as img:
        return img.convert("RGB")


def preprocess_pil(img, size):
    """Resize shorter side, center crop to size x size, normalize -> (3, H, W) float32."""
    w, h = img.size
    short = int(round(size * RESIZE_RATIO))
    if w <= h:
        new_w, new_h = short, int(round(h * short / w))
    else:
        new_w, new_h = int(round(w * short / h)), short
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (x - _MEAN) / _STD


def preprocess_image(path, size):
    return preprocess_pil(load_rgb(path), size)


def load_split(split_path):
    with open(split_path) as f:
        split = json.load(f)
    # ROSE_DATA_ROOT lets the same split file be used on another machine (e.g. the cluster).
    root = os.environ.get("ROSE_DATA_ROOT", split["root"])
    split["root"] = root
    for item in split["items"]:
        item["abs_path"] = os.path.join(root, item["path"])
    return split


def items_for(split, subset):
    return [it for it in split["items"] if it["split"] == subset]
