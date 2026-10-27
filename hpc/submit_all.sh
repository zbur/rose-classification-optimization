#!/usr/bin/env bash
# Submit prep -> train array -> experiments with dependencies.
# Usage (on the login node):  bash hpc/submit_all.sh          # SUBSET=test for the final run
set -euo pipefail
source "$(dirname "$0")/config.sh"
mkdir -p "$RUN_DIR/logs"
cd "$RUN_DIR/logs"

read -ra seeds <<< "$SEEDS"
read -ra sizes <<< "$SIZES"
n=$(( ${#seeds[@]} * ${#sizes[@]} ))
common=(--export=ALL,PROJECT_DIR="$PROJECT_DIR")

dep=()
if [ ! -f "$RUN_DIR/splits/rose_split.json" ]; then
  prep=$(sbatch --parsable "${common[@]}" "$PROJECT_DIR/hpc/prep.sbatch")
  echo "prep:        $prep"
  dep=(--dependency=afterok:$prep)
fi
train=$(sbatch --parsable "${common[@]}" "${dep[@]}" --array=0-$((n - 1)) "$PROJECT_DIR/hpc/train.sbatch")
echo "train array: $train  ($n tasks)"
exp=$(sbatch --parsable "${common[@]}" --dependency=afterok:$train "$PROJECT_DIR/hpc/experiments.sbatch")
echo "experiments: $exp"
echo "Logs in $RUN_DIR/logs.  Monitor: squeue -u $USER"
