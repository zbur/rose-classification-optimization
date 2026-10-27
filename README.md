# rose-classification-optimization
Optimizing MobileNetV2 to work on low-end mobile devices classifying rose blight using an optimized dataset

## MobileNetV2 for Edge Inference — 實作

依照 `MobileNetV2_Improvement_Plan_Visual.pptx`（含講者備註）實作：
fine-tuning → ONNX Runtime → INT8 量化 → 降解析度，並以 A–E 控制變因實驗比較。

資料集：Mendeley `p3g46cwz7r` 的 rose 子集，4 類 healthy / mosaic_virus / pest_damage / botrytis_blight。

## 安裝

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 檔案

| 檔案 | 對應投影片 | 功能 |
|---|---|---|
| `config.py` | — | 類別、關鍵字、變體 A–E、品質門檻 |
| `prepare_split.py` | 2, 3 | 掃描 rose 子集、稽核標籤與重複圖、**先分組再切分**（70/15/15，依類別分層） |
| `preprocess.py` | — | 所有變體共用的前處理（只用 NumPy/PIL） |
| `datasets.py`, `model.py` | 3 | 資料增強、MobileNetV2 + 新 4 類 head、凍結工具 |
| `train.py` | 3 | 兩階段 fine-tuning，以 val Macro-F1 選模型 |
| `export_onnx.py` | 4 | 匯出 FP32 ONNX，並驗證與 PyTorch 預測一致 |
| `quantize_int8.py` | 4 | ONNX Runtime 靜態 PTQ（只用 train 影像校正，QDQ、per-channel） |
| `qat.py` | 4 | PTQ 掉太多時才用的 QAT 備案，匯出 QDQ ONNX |
| `evaluate.py` | 5 | Macro-F1、各類 recall、混淆矩陣 |
| `benchmark.py` | 5 | batch=1、固定 thread、warmup、median/P95、前處理延遲分開算、檔案大小、峰值記憶體（每次在獨立子程序量測） |
| `run_experiments.py` | 5 | 產生 A–E，評估加 benchmark，套用門檻（Macro-F1 比 A 少 ≤ 1pp），選最快的 |
| `aggregate_seeds.py` | 5 | 多個 seed 結果取平均 ± 標準差 |
| `run_all.sh` | — | 一鍵跑完整流程 |

## 使用流程

```bash
# 1) 切分與稽核（請檢查 splits/rose_split.json 的 audit 欄位）
python prepare_split.py --data-root /path/to/dataset --subset-keyword rose --out splits/rose_split.json
#    若資料夾名稱對不到類別：--class-map my_map.json  (格式 {"healthy": ["healthy"], ...})

# 2) Fine-tune（224；如果 D/E 要用相同訓練設定，也訓練一個 160）
python train.py --split splits/rose_split.json --img-size 224 --seed 0 --out runs/fp32_224_s0
python train.py --split splits/rose_split.json --img-size 160 --seed 0 --out runs/fp32_160_s0

# 3) A–E 實驗：先在 val 上調整，最後只跑一次 test
python run_experiments.py --ckpt-224 runs/fp32_224_s0/best.pt --ckpt-160 runs/fp32_160_s0/best.pt \
    --split splits/rose_split.json --subset val --threads 1 --out results/val_s0
python run_experiments.py ... --subset test --out results/test_s0     # final locked run

# 4)（選用）PTQ 掉太多時改用 QAT
python qat.py --ckpt runs/fp32_224_s0/best.pt --split splits/rose_split.json --out runs/qat_224_s0
python run_experiments.py ... --int8-224 runs/qat_224_s0/qat_int8_224.onnx

# 或一次跑完全部（多個 seed）
DATA_ROOT=/path/to/dataset SEEDS="0 1 2" bash run_all.sh
```

輸出：`results/<run>/results.{json,csv,md}`，內容包含各變體指標、B−A / C−B / D−B / E−C / E−D 的比較，以及被選中的變體。

## 預設超參數（來自投影片備註，只在 val 上調整）

- AdamW、cross-entropy、ImageNet normalization
- Stage 1：凍結整個 backbone，head lr 1e-3，5 個 epoch
- Stage 2：解凍 `features[14:]`，backbone lr 1e-5、head lr 1e-4，cosine 排程，early stopping（patience 5）
- BatchNorm running stats 預設凍結（`--no-freeze-bn` 可關閉）
- INT8：320 張 train 影像校正，MinMax（可改 `entropy` / `percentile`）

## 注意事項

- `prepare_split.py` 用檔名（去掉 `_aug3`、`_flip`、`(2)` 等後綴）、MD5 完全重複和 dHash 近似重複來分組。請依實際檔名調整 `--aug-regex` / `--hash-threshold`，並檢查 `label_conflict_groups`。
- 如果沒給 `--ckpt-160`，D/E 會直接用 224 的 checkpoint 在 160 上推論（shared checkpoint），結果裡會記錄這個選擇。
- 峰值記憶體 = 子程序 max RSS − 載入模型前的 max RSS；同時也記錄整個程序的 peak RSS。
- QAT 匯出的 ONNX 權重仍以 float 存放再接 Q/DQ，所以檔案大小不會變小。
- 所有 benchmark 都要在同一台目標裝置上跑，並記錄裝置資訊（`benchmark` 結果中的 `platform`）。
