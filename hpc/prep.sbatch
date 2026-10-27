#!/usr/bin/env bash
#SBATCH --job-name=mnv2-prep
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=%x-%j.out
set -euo pipefail
source "$PROJECT_DIR/hpc/config.sh"
activate_env
cd "$PROJECT_DIR"

python prepare_split.py --data-root "$ROSE_DATA_ROOT" --subset-keyword rose \
  --out "$RUN_DIR/splits/rose_split.json"
