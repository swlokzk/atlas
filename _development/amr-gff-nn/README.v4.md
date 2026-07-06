# README.v4

簡短說明
---------
本檔為 amc-amr-gff-nn 專案的「v4」說明檔，聚焦於最近的結構化與模組化改動。此版本說明如何以「程式庫 + CLI entrypoint」的方式使用本套件（更適合生產化、單元測試與 CI/CD），並補充程式呼叫示例與常用參數。此文件與原 README.md 共存，作為更工程化（modular）使用的操作手冊。

重點變更（v4）
----------------
- 將原 Notebook 與單一腳本流程拆成模組化套件（src 套件化）：核心邏輯（dataset / pipeline / metrics / models / factory）可被程式匯入或透過 CLI 呼叫。
- 新增統一入口：`src/run.py`（或 `python -m src.run`）作為 subcommand 入口（evaluate / deep_analysis / ablation / gating / compare）。
- models 由 factory 管理（`src/models/factory.py`），方便動態載入或替換模型實作。
- 配置集中管理：`src/configs/config.py`（或 YAML/JSON）作為預設參數來源，並支援 CLI 覆寫。
- 結果與視覺化輸出規範化存放至 `outputs/`（圖表、混淆矩陣、TSNE 等）。

快速開始（Prerequisites）
--------------------------
安裝必要套件（建議使用虛擬環境）：
```bash
pip install -r requirements.txt
# OR
pip install torch torchvision timm scipy scikit-learn matplotlib seaborn tqdm pandas
```

資料集
------
使用 RadioML 2016.10a（RML2016.10a）。下載並解壓後，於 CLI 中以 `--data RML2016.10a_dict.pkl` 指定資料檔路徑。

程式結構（範例）
----------------
（下列為主要檔案與目錄 — 請以實際 repo 為準）
```
amc-amr-gff-nn/
├── assets/
├── src/
│   ├── __init__.py
│   ├── run.py                 # CLI / subcommands entrypoint
│   ├── export.py              # 模型匯出 / ONNX / TorchScript (選用)
│   ├── utils.py               # 共用工具（IO、plotting、metrics）
│   ├── configs/
│   │   └── config.py          # 預設設定 / hyperparams
│   ├── core/                  # 分析/實驗模組（evaluate, deep_analysis, ...）
│   │   ├── dataset.py
│   │   ├── evaluate.py
│   │   ├── deep_analysis.py
│   │   ├── ablation.py
│   │   ├── gating_weights.py
│   │   └── cnn_vs_transformer.py
│   └── models/
│       ├── factory.py         # build_model(...)
│       ├── model.py
│       ├── gff_nn.py
│       └── cnn2.py / mod_rec_net.py
├── README.md
└── README.v4.md
```

使用方式（CLI）
---------------
- 通用參數（各子指令共有）：
  - `--weights` : 訓練後的 model weights 檔案（.pth） — 必填
  - `--data`    : RML2016.10a_dict.pkl 的完整路徑 — 必填
  - `--batch-size` : 推理批次大小（預設 256）
  - `--output-dir` : 輸出目錄（預設 outputs）
  - `--device`  : `cpu` 或 `cuda`（預設自動偵測）

- 範例：直接用腳本執行（若 run.py 設為可執行）
```bash
# Evaluate（等於原 01_evaluate.py）
python src/run.py evaluate --weights path/to/model.pth --data path/to/RML2016.10a_dict.pkl --batch-size 256

# 深度分析（等於原 02_deep_analysis.py）
python src/run.py deep_analysis --weights model.pth --data RML2016.10a_dict.pkl

# 消融實驗（等於原 03_ablation.py）
python src/run.py ablation --weights model.pth --data RML2016.10a_dict.pkl

# 門控權重分析
python src/run.py gating --weights model.pth --data RML2016.10a_dict.pkl

# CNN vs Transformer 比較
python src/run.py compare --weights model.pth --data RML2016.10a_dict.pkl
```

（若 prefer module run）
```bash
python -m src.run evaluate --weights ... --data ...
```

輸出說明
--------
- 所有圖表、可視化結果、CSV、與 logs 會儲存在 `--output-dir` 指定的目錄（預設 `outputs/`）。
- 常見輸出檔例：
  - `confusion_matrix_counts.png`
  - `confusion_matrix_normalized.png`
  - `accuracy_vs_snr.png`
  - `per_class_accuracy_vs_snr.png`
  - `tsne_visualization.png`
  - `gating_weights_vs_snr.png`
  - `ablation_*.png`
  - `cnn_vs_transformer_accuracy.png`

程式化呼叫（當作 library 使用）
--------------------------------
如果你想在其他程式中直接使用模型或 pipeline，可以匯入 factory 與工具函式：
```python
from src.models.factory import build_model
from src.core.dataset import RMLDataset
from src.utils import load_weights, predict_batch

# 建構模型
model = build_model(name="gff_nn", num_classes=11)
# 載入權重
load_weights(model, "path/to/model.pth")
# 資料/推理
dataset = RMLDataset("path/to/RML2016.10a_dict.pkl", split="test")
preds = predict_batch(model, dataset, device="cuda")
```
（上述 API 名稱為範例；請以實際 code 中的 function/class 名稱為準）

常見問題（FAQ）
----------------
- 我改了某些檔案路徑，README 需要更新嗎？
  - 若模組或檔名有改動（例如子指令名稱或檔案位置），請把變動的確切檔案路徑告訴我，我會幫你更新 README.v4.md 中的範例指令與範例程式碼。
- 想把 README.v4.md 直接加入 repo？
  - 我可以幫你產生內容與建議的 commit message；若要我直接開 PR 或 push，請明確授權並提供 repository 權限/owner 資訊（或讓我知道要用的分支名）。

維護與開發建議
----------------
- 將 CLI 的 subcommands 套用 unittest / pytest 測試，以確保 refactor 後 CLI 行為���變。
- 將 config（超參數）移到 YAML/JSON，並加入版本記錄，方便在實驗重現時載入 exact config。
- 在模型 factory 與 config 之間標註相容性（哪個 model-name 需要哪些 config keys），減少使用錯誤。

版本紀錄（小節）
----------------
- README.v4.md — 本檔（模組化 / production-friendly）
- 若需更精細的 migration notes（例如���檔名對照表），我可以為你產出一份「遷移對照表」。
