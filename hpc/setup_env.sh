#!/usr/bin/env bash
# One-time environment setup. Run on the login node:  bash hpc/setup_env.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"

module purge
module load "$CONDA_MODULE"
eval "$(conda shell.bash hook)"

# Keep conda envs/packages out of the 75GB home quota if a /projects dir exists.
if [ -n "${CONDA_PREFIX_DIR:-}" ]; then
  conda create -y -p "$CONDA_PREFIX_DIR/$ENV_NAME" python=3.11
  ENV_NAME="$CONDA_PREFIX_DIR/$ENV_NAME"
else
  conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

# CUDA build of PyTorch (wheel bundles its own CUDA runtime).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r "$PROJECT_DIR/requirements.txt"

python -c "import torch, onnxruntime; print('torch', torch.__version__, '| ort', onnxruntime.__version__)"
mkdir -p "$RUN_DIR/logs"
echo "Environment '$ENV_NAME' ready."
