# 在 Explorer Cluster 上訓練

| 檔案 | 用途 |
|---|---|
| `config.sh` | 路徑、conda module、seeds、解析度（**先改這裡**） |
| `setup_env.sh` | 一次性建立 conda 環境（login node） |
| `prep.sbatch` | 資料切分（`short` CPU partition） |
| `train.sbatch` | GPU job array，每個 task 跑一組 (seed, size) |
| `experiments.sbatch` | 匯出 / 量化 / 評估 A–E，加上 seed 彙整（CPU） |
| `submit_all.sh` | 依相依關係一次送出 prep → train → experiments |

詳細步驟見上層對話或下方指令：

```bash
# 本機：上傳程式與資料
rsync -av --exclude __pycache__ code/ <user>@login.explorer.northeastern.edu:~/mnv2/code/
rsync -av /path/to/rose_data/ <user>@xfer.discovery.neu.edu:/scratch/<user>/rose_data/

# 叢集
ssh <user>@login.explorer.northeastern.edu
cd ~/mnv2/code
module avail miniconda anaconda        # 確認 module 名稱，必要時修改 hpc/config.sh
bash hpc/setup_env.sh
bash hpc/submit_all.sh                  # SUBSET=test bash hpc/submit_all.sh 做最後一次 test
squeue -u $USER

# 本機：取回結果（/scratch 28 天會被清除）
rsync -av <user>@xfer.discovery.neu.edu:/scratch/<user>/mnv2_runs/{runs,results} ./hpc_results/
```
