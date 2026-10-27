#!/usr/bin/env bash
# Job array: one task per (seed, img_size). Task count = |SEEDS| * |SIZES|.
#SBATCH --job-name=mnv2-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
source "$PROJECT_DIR/hpc/config.sh"
activate_env
cd "$PROJECT_DIR"

read -ra seeds <<< "$SEEDS"
read -ra sizes <<< "$SIZES"
idx=${SLURM_ARRAY_TASK_ID:-0}
seed=${seeds[$((idx / ${#sizes[@]}))]}
size=${sizes[$((idx % ${#sizes[@]}))]}
out="$RUN_DIR/runs/fp32_${size}_s${seed}"

echo "host=$(hostname) seed=$seed size=$size out=$out"
nvidia-smi --query-gpu=name,memory.total --format=csv

if [ -f "$out/best.pt" ] && [ -f "$out/weights_fp32.pt" ]; then
  echo "Already trained, skipping."; exit 0
fi
python train.py --split "$RUN_DIR/splits/rose_split.json" --img-size "$size" --seed "$seed" \
  --workers "${SLURM_CPUS_PER_TASK:-4}" --device cuda --out "$out"
