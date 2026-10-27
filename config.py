"""Shared constants for the MobileNetV2 edge-inference project."""

# Rose leaf classes (Mendeley dataset p3g46cwz7r, rose subset).
CLASSES = ["healthy", "mosaic_virus", "pest_damage", "botrytis_blight"]

# Keywords used to map dataset folder names to class labels (case-insensitive).
# Adjust with --class-map in prepare_split.py if the folder names differ.
CLASS_KEYWORDS = {
    "healthy": ["healthy", "fresh"],
    "mosaic_virus": ["mosaic"],
    "pest_damage": ["pest", "insect"],
    "botrytis_blight": ["botrytis", "blight"],
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ImageNet normalization (matches the pretrained MobileNetV2 weights).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Eval preprocessing: resize shorter side to size * 256/224, then center crop.
RESIZE_RATIO = 256 / 224

# Experiment variants from slide 5.
VARIANTS = {
    "A": {"runtime": "pytorch", "precision": "fp32", "img_size": 224},
    "B": {"runtime": "onnxruntime", "precision": "fp32", "img_size": 224},
    "C": {"runtime": "onnxruntime", "precision": "int8", "img_size": 224},
    "D": {"runtime": "onnxruntime", "precision": "fp32", "img_size": 160},
    "E": {"runtime": "onnxruntime", "precision": "int8", "img_size": 160},
}

# Proposed quality gate: Macro-F1 loss <= 1 percentage point vs. A.
GATE_MACRO_F1_PP = 1.0
