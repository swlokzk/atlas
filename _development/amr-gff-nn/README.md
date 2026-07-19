# AMR-GFFNN

AMC AMR Gated Fusion Former 自動調變識別神經網路。  
A multimodal Gated Fusion Former neural network for automatic modulation recognition.

---

## 概述 | Overview

AMR-GFFNN 是一個基於 PyTorch 的自動調變識別模型，使用三種訊號模態：
- **IQ**：原始同相／正交訊號
- **STFT**：短時傅立葉轉換頻域特徵
- **S-TD**：統計時域特徵

模型透過 gating network 動態融合三種模態，產生輸入相依的模態權重，並使用 Conv1D、Conv2D、Multi-Head Attention 與 Transformer blocks 進行分類。

---

## 類別與狀態 | Category and Status

- **類別 | Category**：Development
- **類型 | Type**：Deep Learning | Neural Network | Signal Processing
- **生命週期 | Lifecycle**：Deployment-ready prototype
- **部署狀態 | Deployment Status**：Validated FP32 ONNX pipeline; INT8 and full production validation pending

---

## 專案結構 | Structure

```text
_development/amr-gff-nn/
├── src/
│   ├── __init__.py
│   ├── run.py                     # Research CLI entrypoint
│   ├── utils.py                   # Research utilities
│   ├── configs/
│   │   └── config.py              # Research model configuration
│   ├── core/
│   │   ├── dataset.py             # RML2016.10a loading and feature preparation
│   │   ├── evaluate_main.py       # Basic evaluation
│   │   ├── deep_analysis.py       # Per-class/SNR and feature analysis
│   │   ├── ablation.py            # Modality ablation studies
│   │   ├── gating_weights.py      # Gating-weight analysis
│   │   └── cnn_vs_transformer.py # Architecture comparison
│   └── models/
│       ├── factory.py             # Model construction utilities
│       ├── model.py               # Canonical GatedFusionFormer
│       ├── gff_nn.py              # Legacy/alternative implementation
│       ├── cnn2.py
│       └── mod_rec_net.py
│
├── deploy/
│   ├── checkpoint.py              # Checkpoint retrieval and compatibility checks
│   ├── preprocessing.py           # Raw IQ → IQ/STFT/S-TD
│   ├── model_wrapper.py           # Production forward wrapper
│   ├── predict.py                 # PyTorch CPU inference baseline
│   ├── export_onnx.py             # FP32 ONNX export
│   ├── validate_onnx.py           # PyTorch/ONNX parity validation
│   ├── quantize_onnx.py           # Dynamic/static INT8 workflow
│   ├── IMPLEMENTATION_NOTES.md    # Deployment implementation notes
│   ├── requirements.txt           # Deployment dependencies
│   ├── constraints.txt            # Pinned dependency constraints
│   ├── config/
│   │   ├── model_config.json
│   │   └── checkpoint_manifest.json
│   └── service/
│       ├── app.py                 # FastAPI inference service
│       ├── schemas.py             # API schemas and validation
│       └── Dockerfile             # CPU deployment image
│
├── artifacts/
│   └── gff-v3/                    # Generated deployment artifacts
│
├── tests/                         # Standard-library unittest suite
├── assets/                        # Research plots and visualizations
├── .python-version                # Python 3.10 deployment target
└── README.md
```

`artifacts/` contains generated model files and reports. Binary deployment artifacts should remain excluded from Git unless repository policy explicitly allows them.

---

## 快速導覽手冊 | Engineering Documentation

本專案區分「研究分析」與「生產部署」兩大管線。具體的環境建置、執行指令與部署契約，請參閱以下獨立文件：

*   🛠️ **[DEVELOPMENT.md](./DEVELOPMENT.md)**：完整工程操作手冊。包含：
    - Python 3.10 環境建置與依賴管理
    - 研究實驗 CLI 指令（消融實驗、Gating 權重分析、架構對比）
    - Deployment Pipeline（ONNX 導出、精度驗證、INT8 量化流程）
    - FastAPI 服務啟動與 Docker 容器化部署
*   📝 **[deploy/IMPLEMENTATION_NOTES.md](./deploy/IMPLEMENTATION_NOTES.md)**：部署實作細節與底層設計筆記。

---


---

## 輸出與展示 | Outputs and Research Artifacts

**資料集 | Dataset**：本專案使用 [RadioML 2016.10a (RML2016.10a)](https://www.deepsig.ai/datasets)。下載並解壓縮後，以 `RML2016.10a_dict.pkl` 作為 `--data` 參數。  
This project uses [RadioML 2016.10a (RML2016.10a)](https://www.deepsig.ai/datasets). After downloading and extracting, use `RML2016.10a_dict.pkl` as the `--data` parameter.

---

## 輸出與展示 | Outputs and Demos

所有圖表、視覺化結果、CSV 與 logs 儲存於 `--output-dir` 指定目錄（預設 `outputs/`）。  
All charts, visualizations, CSVs, and logs are saved in the directory specified by `--output-dir` (default `outputs/`).

### 基本模型評估 | Basic Model Evaluation (`evaluate.py`)

```bash
python src/run.py evaluate --weights path/to/model.pth --data path/to/RML2016.10a_dict.pkl --batch-size 256
```

**輸出 | Outputs:**

- `confusion_matrix_counts.png` — 原始計數混淆矩陣 | Raw count confusion matrix
- `confusion_matrix_normalized.png` — 歸一化混淆矩陣 | Normalized confusion matrix
- `accuracy_vs_snr.png` — 整體 Accuracy vs SNR

### 深度性能分析 | Deep Analysis (`deep_analysis.py`)

```bash
python src/run.py deep_analysis --weights model.pth --data RML2016.10a_dict.pkl
```

**輸出 | Outputs:**

- `per_class_accuracy_vs_snr.png` — 各類別 Accuracy vs SNR
- `confused_categories_high_snr.png` — 高 SNR 易混淆類別柱狀圖
- `tsne_visualization.png` — t-SNE 特徵空間可視化（低 / 高 SNR）

![per_class_acc_vs_snr](assets/per_class_acc_vs_snr.png)
![tsne_visualization](assets/tsne_visualization.png)

### 模態消融實驗 | Ablation Study (`ablation.py`)

```bash
python src/run.py ablation --weights model.pth --data RML2016.10a_dict.pkl
```

**輸出 | Outputs:**

- `ablation_single_modality.png` — 單模態消融 Accuracy vs SNR
- `ablation_pairwise_modality.png` — 成對模態消融 Accuracy vs SNR
- `ablation_confused_categories.png` — 高 SNR 易混淆類別對比

![ablation_single_modality](assets/ablation_single_modality.png)
![ablation_pairwise_modality](assets/ablation_pairwise_modality.png)

### 門控網絡權重分析 | Gating Weights Analysis (`gating_weights.py`)

```bash
python src/run.py gating --weights model.pth --data RML2016.10a_dict.pkl
```

**輸出 | Outputs:**

- `gating_weights_vs_snr.png` — IQ / STFT / S-TD 模態重要性 vs SNR

![gating_weights_vs_snr](assets/gating_weights_vs_snr.png)

### CNN vs Transformer 對比（TODO）| CNN vs Transformer Comparison (TODO)

```bash
python src/run.py compare --weights model.pth --data RML2016.10a_dict.pkl
```

**輸出 | Outputs:**

- `gffnn_compare_acc.png` — 整體 Accuracy vs SNR 對比折線圖
- `gffnn_compare_overall.png` — 整體準確率柱狀圖

---

## 注意事項 | Notes and Limitations

- 可使用模組化方式執行 | Module run alternative:

  ```bash
  python -m src.run evaluate --weights ... --data ...
  ```

- 建議為 CLI subcommands 加入 unittest / pytest 以確保重構後行為一致。Apply unittest / pytest to CLI subcommands to ensure behavior consistency after refactoring.
- 建議將設定（超參數）移至 YAML/JSON 並進行版本追蹤，以確保實驗可重現。Move config (hyperparameters) to YAML/JSON with version tracking for reproducible experiments.

---

## 限制與注意事項 | Limitations

- Full class-level and SNR-level deployment regression requires the RadioML dataset.
- Dynamic INT8 generation has been tested structurally but has not received final accuracy approval.
- Static INT8 calibration is not performed without real calibration data.
- INT8 gating-weight deviation has not been validated on the full dataset.
- Compression comparisons for attention-off, reduced-size, and compact CNN variants remain pending.
- Docker build/runtime smoke testing requires Docker CLI availability.
- Legacy research implementations and model-factory mappings may require further cleanup.
- Research analysis and deployment inference use different dependency scopes.
- The service is a deployment-ready prototype, not a claim of fully validated production performance.
- GGUF/GGML export is not supported or planned for this project.

---

## Related Links

- [Project Catalog](../../catalog/index.md)
- [Repository Root](../../README.md)
- [Deployment Implementation Notes](deploy/IMPLEMENTATION_NOTES.md)

---

## 相關連結 | Related Links

- [專案 Catalog | Project Catalog](../../catalog/index.md)
- [Repository 根目錄 | Repository Root](../../README.md)


