# Training on the Explorer Cluster

| File | Purpose |
|---|---|
| `config.sh` | Paths, conda module, seeds, resolutions (**edit this first**) |
| `setup_env.sh` | One-time conda environment setup (login node) |
| `prep.sbatch` | Data split (`short` CPU partition) |
| `train.sbatch` | GPU job array, one task per (seed, size) |
| `experiments.sbatch` | Export / quantize / evaluate A–E, plus seed aggregation (CPU) |
| `submit_all.sh` | Submit prep → train → experiments with dependencies |

Steps:

```bash
# Local: upload code and data
rsync -av --exclude __pycache__ code/ <user>@login.explorer.northeastern.edu:~/mnv2/code/
rsync -av /path/to/rose_data/ <user>@xfer.discovery.neu.edu:/scratch/<user>/rose_data/

# Cluster
ssh <user>@login.explorer.northeastern.edu
cd ~/mnv2/code
module avail miniconda anaconda        # check the module name; edit hpc/config.sh if needed
bash hpc/setup_env.sh
bash hpc/submit_all.sh                  # SUBSET=test bash hpc/submit_all.sh for the final test run
squeue -u $USER

# Local: fetch results (/scratch is purged after 28 days)
rsync -av <user>@xfer.discovery.neu.edu:/scratch/<user>/mnv2_runs/{runs,results} ./hpc_results/
```
