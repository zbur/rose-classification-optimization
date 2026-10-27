# Shared settings for the Explorer cluster scripts. Edit these first.
export PROJECT_DIR=${PROJECT_DIR:-$HOME/mnv2/code}         # where code/ lives on the cluster
export ROSE_DATA_ROOT=${ROSE_DATA_ROOT:-/scratch/$USER/rose_data}  # extracted dataset
export RUN_DIR=${RUN_DIR:-/scratch/$USER/mnv2_runs}        # checkpoints, results, logs
export ENV_NAME=${ENV_NAME:-mnv2}
export CONDA_MODULE=${CONDA_MODULE:-miniconda3}            # check: module avail miniconda anaconda
export SEEDS=${SEEDS:-"0 1 2"}
export SIZES=${SIZES:-"224 160"}

activate_env() {
  module purge
  module load "$CONDA_MODULE"
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
}
