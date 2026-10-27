#!/usr/bin/env bash
# Export / quantize / evaluate A-E for every seed on a CPU node.
# Latency numbers from a shared cluster node are NOT edge-device numbers; by default
# only accuracy is computed here (BENCH=1 to also benchmark on this node).
#SBATCH --job-name=mnv2-exp
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
set -euo pipefail
source "$PROJECT_DIR/hpc/config.sh"
activate_env
cd "$PROJECT_DIR"

SUBSET=${SUBSET:-val}
extra=()
[ "${BENCH:-0}" = "1" ] || extra+=(--skip-benchmark)

dirs=()
for s in $SEEDS; do
  python run_experiments.py \
    --ckpt-224 "$RUN_DIR/runs/fp32_224_s$s/best.pt" --ckpt-160 "$RUN_DIR/runs/fp32_160_s$s/best.pt" \
    --split "$RUN_DIR/splits/rose_split.json" --subset "$SUBSET" --threads 1 --seed "$s" \
    --out "$RUN_DIR/results/${SUBSET}_s$s" "${extra[@]}"
  dirs+=("$RUN_DIR/results/${SUBSET}_s$s")
done
python aggregate_seeds.py "${dirs[@]}" --out "$RUN_DIR/results/${SUBSET}_summary.md"
