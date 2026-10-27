# rose-classification-optimization
Optimizing MobileNetV2 to work on low-end mobile devices classifying rose blight using an optimized dataset

## MobileNetV2 for Edge Inference — Implementation

Implements the plan in `MobileNetV2_Improvement_Plan_Visual.pptx` (including speaker notes):
fine-tuning → ONNX Runtime → INT8 quantization → lower input resolution, compared through controlled experiments A–E.

Dataset: the rose subset of Mendeley `p3g46cwz7r`, 4 classes: healthy / mosaic_virus / pest_damage / botrytis_blight.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Files

| File | Slide | Purpose |
|---|---|---|
| `config.py` | — | Classes, keywords, variants A–E, quality gate |
| `prepare_split.py` | 2, 3 | Scan the rose subset, audit labels and duplicates, **group before splitting** (70/15/15, stratified by class) |
| `preprocess.py` | — | Preprocessing shared by all variants (NumPy/PIL only) |
| `datasets.py`, `model.py` | 3 | Data augmentation, MobileNetV2 + new 4-class head, freezing helpers |
| `train.py` | 3 | Two-stage fine-tuning, model selection by validation Macro-F1 |
| `export_onnx.py` | 4 | Export FP32 ONNX and verify predictions match PyTorch |
| `quantize_int8.py` | 4 | ONNX Runtime static PTQ (calibrated on training images only, QDQ, per-channel) |
| `qat.py` | 4 | QAT fallback for when PTQ loses too much accuracy; exports QDQ ONNX |
| `evaluate.py` | 5 | Macro-F1, per-class recall, confusion matrix |
| `benchmark.py` | 5 | batch=1, fixed threads, warmup, median/P95, separate preprocessing latency, file size, peak memory (each measured in a fresh subprocess) |
| `run_experiments.py` | 5 | Build A–E, evaluate and benchmark, apply the gate (Macro-F1 ≤ 1pp below A), select the fastest |
| `aggregate_seeds.py` | 5 | Mean ± std across seeds |
| `run_all.sh` | — | Run the full pipeline in one command |

## Usage

```bash
# 1) Split and audit (check the audit fields in splits/rose_split.json)
python prepare_split.py --data-root /path/to/dataset --subset-keyword rose --out splits/rose_split.json
#    If folder names don't match the classes: --class-map my_map.json  (format {"healthy": ["healthy"], ...})

# 2) Fine-tune (224; also train a 160 model if D/E should use matched training)
python train.py --split splits/rose_split.json --img-size 224 --seed 0 --out runs/fp32_224_s0
python train.py --split splits/rose_split.json --img-size 160 --seed 0 --out runs/fp32_160_s0

# 3) A–E experiments: tune on val, then run test only once at the end
python run_experiments.py --ckpt-224 runs/fp32_224_s0/best.pt --ckpt-160 runs/fp32_160_s0/best.pt \
    --split splits/rose_split.json --subset val --threads 1 --out results/val_s0
python run_experiments.py ... --subset test --out results/test_s0     # final locked run

# 4) (Optional) switch to QAT if PTQ loses too much
python qat.py --ckpt runs/fp32_224_s0/best.pt --split splits/rose_split.json --out runs/qat_224_s0
python run_experiments.py ... --int8-224 runs/qat_224_s0/qat_int8_224.onnx

# Or run everything at once (multiple seeds)
DATA_ROOT=/path/to/dataset SEEDS="0 1 2" bash run_all.sh
```

Output: `results/<run>/results.{json,csv,md}`, containing per-variant metrics, the B−A / C−B / D−B / E−C / E−D comparisons, and the selected variant.

## Default hyperparameters (from the slide notes; tune on val only)

- AdamW, cross-entropy, ImageNet normalization
- Stage 1: freeze the whole backbone, head lr 1e-3, 5 epochs
- Stage 2: unfreeze `features[14:]`, backbone lr 1e-5, head lr 1e-4, cosine schedule, early stopping (patience 5)
- BatchNorm running stats frozen by default (disable with `--no-freeze-bn`)
- INT8: calibrate on 320 training images, MinMax (or `entropy` / `percentile`)

## Notes

- `prepare_split.py` groups images by filename (stripping suffixes such as `_aug3`, `_flip`, `(2)`), exact MD5 duplicates, and dHash near-duplicates. Adjust `--aug-regex` / `--hash-threshold` to the actual filenames and check `label_conflict_groups`.
- Without `--ckpt-160`, D/E run the 224 checkpoint at 160 (shared checkpoint); this choice is recorded in the results.
- Peak memory = subprocess max RSS − max RSS before loading the model; whole-process peak RSS is also recorded.
- QAT-exported ONNX still stores weights as float followed by Q/DQ, so the file is not smaller.
- Run all benchmarks on the same target device and record its details (`platform` in the `benchmark` results).
