# AMR-GFFNN DEVELOPMENT GUIDE

AMC AMR Gated Fusion Former 自動調變識別神經網路。  
A multimodal Gated Fusion Former neural network for automatic modulation recognition.

---

## 概述 | Overview

AMR-GFFNN 是一個基於 PyTorch 的自動調變識別模型，使用三種訊號模態：

- **IQ**：原始同相／正交訊號
- **STFT**：短時傅立葉轉換頻域特徵
- **S-TD**：統計時域特徵

模型透過 gating network 動態融合三種模態，產生輸入相依的模態權重，並使用 Conv1D、Conv2D、Multi-Head Attention 與 Transformer blocks 進行分類。

本專案同時支援研究分析與部署流程：

- RML2016.10a 資料集推論
- Overall、per-class 與 per-SNR accuracy 分析
- Confusion matrix 與 t-SNE 視覺化
- 單模態與多模態消融實驗
- Gating-weight 分析
- CNN 與 Transformer 模型比較
- PyTorch CPU inference baseline
- FP32 ONNX export
- ONNX Runtime CPU inference
- Dynamic INT8 quantization
- FastAPI inference service
- CPU Docker deployment

目前專案已完成一套可運作的 CPU ONNX deployment prototype。FP32 ONNX 已通過 deterministic parity validation；INT8 精度驗證、static calibration、完整 RadioML class/SNR regression、模型壓縮比較與 Docker runtime smoke test 仍待完成。

---

## 類別與狀態 | Category and Status

- **類別 | Category**：Development
- **類型 | Type**：Deep Learning | Neural Network | Signal Processing
- **生命週期 | Lifecycle**：Deployment-ready prototype
- **部署狀態 | Deployment Status**：Validated FP32 ONNX pipeline; INT8 and full production validation pending
- **標籤 | Tags**：deep-learning, neural-network, gated-fusion-former, automatic-modulation-recognition, onnx, quantization

---

## Canonical Model

目前已驗證可載入部署 checkpoint 的正式模型為：

```text
src.models.model.GatedFusionFormer
```

已驗證配置：

```text
embed_dim       = 96
num_classes     = 11
stft_time_steps = 128
depth           = 4
num_heads       = 4
```

模型支援的 modulation classes：

```text
8PSK, BPSK, CPFSK, GFSK, PAM4,
QAM16, QAM64, QPSK, AM-DSB, AM-SSB, WBFM
```

Checkpoint 使用部署設定與環境變數取得。Private checkpoint token、repository ID 與本機 cache path 不應提交至 repository。

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

## Dataset and Modalities

本專案使用 RadioML 2016.10a：

```text
RML2016.10a_dict.pkl
```

資料集通常包含：

```text
X: IQ samples
Y: modulation labels
Z: SNR values
```

資料集下載與授權資訊請參考 DeepSig RadioML 2016.10a dataset page。

模型使用三種輸入模態：

| 模態 | 說明 | Internal shape |
|---|---|---|
| IQ | 原始 I/Q time-domain signal | `[B, 2, 128]` |
| STFT | Short-Time Fourier Transform magnitude | `[B, 1, 32, 128]` |
| S-TD | Statistical time-domain features | `[B, 2, 128]` |

---

## Research Environment

研究分析與 deployment environment 分開管理。

Deployment target：

```text
Python 3.10
```

安裝 deployment dependencies：

```bash
cd _development/amr-gff-nn

python -m pip install \
  -r deploy/requirements.txt \
  -c deploy/constraints.txt
```

Deployment dependencies include:

```text
torch==2.3.1
numpy==1.24.3
scipy==1.11.2
onnx==1.16.1
onnxruntime==1.18.1
huggingface-hub==0.23.4
python-dotenv==1.0.1
fastapi==0.111.0
pydantic==2.7.4
uvicorn==0.30.1
```

研究腳本可能需要額外的 visualization 與 analysis dependencies；請依研究環境設定安裝，不要將研究套件與最小部署環境混用。

---

## Research Commands

從以下目錄執行：

```bash
cd _development/amr-gff-nn
```

研究 CLI 的共同參數：

| Parameter | Description |
|---|---|
| `--weights` | Trained PyTorch checkpoint path |
| `--data` | Path to `RML2016.10a_dict.pkl` |
| `--batch-size` | Inference batch size |
| `--output-dir` | Output directory |
| `--device` | `cpu`, `cuda`, or automatic selection |

### Basic Evaluation

```bash
python src/run.py evaluate \
  --weights path/to/model.pth \
  --data path/to/RML2016.10a_dict.pkl \
  --batch-size 256
```

Typical outputs:

```text
confusion_matrix_counts.png
confusion_matrix_normalized.png
accuracy_vs_snr.png
```

### Deep Analysis

```bash
python src/run.py deep_analysis \
  --weights path/to/model.pth \
  --data path/to/RML2016.10a_dict.pkl
```

Typical outputs:

```text
per_class_accuracy_vs_snr.png
confused_categories_high_snr.png
tsne_visualization.png
```

### Modality Ablation

```bash
python src/run.py ablation \
  --weights path/to/model.pth \
  --data path/to/RML2016.10a_dict.pkl
```

Typical outputs:

```text
ablation_single_modality.png
ablation_pairwise_modality.png
ablation_confused_categories.png
```

### Gating-Weight Analysis

```bash
python src/run.py gating \
  --weights path/to/model.pth \
  --data path/to/RML2016.10a_dict.pkl
```

Typical output:

```text
gating_weights_vs_snr.png
```

### CNN vs Transformer Comparison

```bash
python src/run.py compare \
  --weights path/to/model.pth \
  --data path/to/RML2016.10a_dict.pkl
```

The comparison workflow exists for research analysis. A complete deployment-oriented benchmark for attention-off and compact CNN variants remains pending.

---

## Deployment Pipeline

```text
Private checkpoint
        ↓
Checkpoint compatibility validation
        ↓
Deterministic raw-IQ preprocessing
        ↓
PyTorch CPU inference baseline
        ↓
FP32 ONNX export
        ↓
PyTorch / ONNX Runtime parity validation
        ↓
Dynamic INT8 experimentation
        ↓
FastAPI service
        ↓
CPU Docker deployment
```

The deployment pipeline does not use GGUF or GGML. ONNX Runtime is the supported deployment runtime.

### Deployment Input Contract

The external service accepts raw IQ data with two equal-length channels:

```text
IQ input: [2, N]
```

For non-128 sample lengths, the deployment preprocessing policy uses deterministic linear interpolation to map the signal to the model target length.

The internal model tensors are:

```text
iq:   [B, 2, 128]
stft: [B, 1, 32, 128]
std:  [B, 2, 128]
```

### STFT Contract

The deployment preprocessing uses:

```text
window   = blackman
nperseg  = 31
noverlap = 30
nfft     = 128
fs       = 1.0
frequency bins = first 32 bins
```

The complex signal is constructed as:

```python
x_complex = I + 1j * Q
```

The STFT magnitude is used as the model input.

### S-TD Contract

The S-TD features are computed as:

```python
std = np.vstack((
    I ** 2 - Q ** 2,
    2 * I * Q,
))
```

The output channel order is:

```text
[I² - Q², 2IQ]
```

### Model Output Contract

The production wrapper exposes:

```text
logits:         [B, 11]
gating_weights: [B, 3]
```

Gating weight order:

```text
[iq, stft, std]
```

The weights are validated outside the exported graph to ensure they are finite, non-negative, and approximately sum to one.

---

## Deployment Commands

All deployment commands target Python 3.10.

### Checkpoint Loading

```bash
python -m deploy.checkpoint
```

The checkpoint loader supports configured private checkpoint retrieval, raw `state_dict` loading, SHA256 verification, and strict model compatibility diagnostics.

### FP32 ONNX Export

```bash
python -m deploy.export_onnx
```

Generated artifact:

```text
artifacts/gff-v3/model.fp32.onnx
```

The export uses ONNX opset 17, dynamic batch size, and fixed modality dimensions.

### PyTorch / ONNX Parity Validation

```bash
python -m deploy.validate_onnx
```

Generated report:

```text
artifacts/gff-v3/parity_report.json
```

Quantization is blocked unless the FP32 parity gate passes.

### Dynamic INT8 Quantization

```bash
python -m deploy.quantize_onnx
```

Generated artifact:

```text
artifacts/gff-v3/model.int8.onnx
```

Dynamic INT8 has been structurally generated and loaded successfully. Accuracy approval on real RadioML data remains pending.

Static INT8 quantization is intentionally blocked when real calibration data is unavailable. The pipeline does not fabricate calibration data.

---

## FastAPI Service

The service provides:

```text
GET  /health
GET  /metadata
POST /v1/classify
```

Start the service from `_development/amr-gff-nn/`:

```bash
uvicorn deploy.service.app:app \
  --host 0.0.0.0 \
  --port 8000
```

The service loads a prebuilt `model.int8.onnx` when available and falls back to `model.fp32.onnx`. Model export and quantization are not executed during service startup.

### Health Check

```bash
curl http://localhost:8000/health
```

### Metadata

```bash
curl http://localhost:8000/metadata
```

### Raw IQ Classification

Example request:

```bash
curl -X POST http://localhost:8000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "iq": [
      [0.10, 0.20, 0.30, 0.40],
      [0.05, 0.15, 0.25, 0.35]
    ]
  }'
```

The service validates:

- exactly two channels;
- equal channel lengths;
- finite numeric values;
- non-empty input;
- configured maximum signal length.

The response contains model prediction, confidence, probability outputs, gating telemetry, latency measurements, and model version. Raw IQ data is not returned in the response.

Representative response:

```json
{
  "prediction": "QPSK",
  "class_id": 7,
  "confidence": 0.945,
  "gating_weights": {
    "iq": 0.26,
    "stft": 0.51,
    "std": 0.23
  },
  "preprocessing_latency_ms": 0.8,
  "inference_latency_ms": 2.1,
  "total_latency_ms": 3.2,
  "model_version": "gff-v3-int8"
}
```

---

## Docker Deployment

The deployment image is CPU-only and uses ONNX Runtime. Deployment artifacts are built before the image is created; the container does not export or quantize the model at startup.

Build from the repository root:

```bash
docker build \
  -f _development/amr-gff-nn/deploy/service/Dockerfile \
  -t amr-gffnn:latest \
  .
```

Run:

```bash
docker run --rm \
  -p 8000:8000 \
  amr-gffnn:latest
```

Verify:

```bash
curl http://localhost:8000/health
```

The Dockerfile runs the service as a non-root user and exposes port `8000`.

Docker build and runtime smoke testing were not performed in the original implementation environment because the Docker CLI was unavailable.

---

## Deployment Validation Status

| Component | Status |
|---|---|
| CPython 3.10 deployment target | Complete |
| Pinned deployment dependencies | Complete |
| Canonical model selection | Complete |
| Strict checkpoint loading | Complete |
| Checkpoint SHA256 manifest | Complete |
| Deterministic raw-IQ preprocessing | Complete |
| PyTorch CPU inference baseline | Complete |
| FP32 ONNX export | Complete |
| ONNX checker validation | Complete |
| ONNX Runtime CPU smoke inference | Complete |
| PyTorch/ONNX parity on deterministic fixtures | Complete |
| Dynamic INT8 graph generation | Complete |
| Dynamic INT8 runtime loading | Complete |
| Dynamic INT8 real-data accuracy validation | Pending RadioML dataset |
| Static INT8 calibration | Pending real calibration data |
| Per-class INT8 evaluation | Pending RadioML dataset |
| Per-SNR INT8 evaluation | Pending RadioML dataset |
| Compression benchmark | Pending |
| Docker build/runtime smoke test | Pending Docker CLI availability |

### FP32 Parity Result

The current deterministic fixtures passed with:

```text
Max logit absolute difference: 9.804964065551758e-06
Mean logit absolute difference: 1.3627789030579152e-06
Max gating absolute difference: 1.7285346984863281e-06
Mean gating absolute difference: 2.862264807390602e-07
Max probability absolute difference: 8.344650268554688e-07
Top-1 agreement: 1.0
```

These results represent deterministic fixture-level validation. They do not replace full class-level and SNR-level evaluation on the RadioML dataset.

### Dynamic INT8 Structural Benchmark

```text
FP32 artifact size: 3,336,046 bytes
INT8 artifact size: 2,926,923 bytes
P50 inference:       approximately 4.14 ms
P95 inference:       approximately 4.83 ms
Throughput:          approximately 240 inferences/second
```

These are structural CPU runtime measurements. They are not final INT8 accuracy approval results.

---

## Tests

The deployment suite uses Python standard-library `unittest`.

Run the available tests from the project directory:

```bash
python -m unittest discover \
  -s tests \
  -p "test_*.py" \
  -v
```

The test suite covers:

- checkpoint loading;
- preprocessing;
- model wrapper;
- PyTorch inference;
- ONNX export;
- ONNX Runtime parity;
- quantization controls;
- API validation.

The latest recorded verification result was:

```text
16 tests passed
```

Full production validation still requires real RadioML data and a Docker-enabled environment.

---

## 輸出與展示 | Outputs and Research Artifacts

Research plots, visualizations, CSV files, and logs are written to the configured output directory.

Typical outputs include:

```text
confusion_matrix_counts.png
confusion_matrix_normalized.png
accuracy_vs_snr.png
per_class_accuracy_vs_snr.png
confused_categories_high_snr.png
tsne_visualization.png
ablation_single_modality.png
ablation_pairwise_modality.png
ablation_confused_categories.png
gating_weights_vs_snr.png
gffnn_compare_acc.png
gffnn_compare_overall.png
```

---

## 限制 | Limitations

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
