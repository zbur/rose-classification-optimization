#!/usr/bin/env bash
# End-to-end pipeline: split -> fine-tune (224 & 160) -> variants A-E -> aggregate seeds.
# Usage: DATA_ROOT=/path/to/dataset SEEDS="0 1 2" SUBSET=val bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the extracted dataset directory}
SEEDS=${SEEDS:-"0"}
SUBSET=${SUBSET:-val}       # switch to "test" only for the final locked run
THREADS=${THREADS:-1}
SPLIT=splits/rose_split.json

[ -f "$SPLIT" ] || python prepare_split.py --data-root "$DATA_ROOT" --subset-keyword rose --out "$SPLIT"

dirs=()
for s in $SEEDS; do
  for size in 224 160; do
    [ -f runs/fp32_${size}_s$s/best.pt ] || \
      python train.py --split "$SPLIT" --img-size $size --seed $s --out runs/fp32_${size}_s$s
  done
  python run_experiments.py --ckpt-224 runs/fp32_224_s$s/best.pt --ckpt-160 runs/fp32_160_s$s/best.pt \
    --split "$SPLIT" --subset "$SUBSET" --threads "$THREADS" --seed $s --out results/${SUBSET}_s$s
  dirs+=(results/${SUBSET}_s$s)
done

python aggregate_seeds.py "${dirs[@]}" --out results/${SUBSET}_summary.md
