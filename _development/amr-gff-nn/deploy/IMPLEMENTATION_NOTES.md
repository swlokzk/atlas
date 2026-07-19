# AMR GFFNN Deployment Implementation Notes

## Canonical Model

- **Candidate:** `src.models.model.GatedFusionFormer`
- **Deployment version:** `gff-v5`
- **Model name:** `gffnn`
- **Reason:** It is the only complete Gated Fusion Former implementation in the
  current repository. It defines the encoder, transformer, normalization,
  gating, and output head used by its forward path.
- **Compatibility status:** Confirmed on 2026-07-12. The private Hugging Face
  checkpoint `gatedfusionformer_v4.0_best_20251025_211831.pth` is a raw
  `state_dict` and passes `strict=True` loading against the candidate model
  using the committed configuration. SHA256:
  `9523ff1097725bd8b15ea1f53e7d0de154eb719ce27412189865abd0325fdc9d`.
  No architecture was modified to fit the checkpoint.

## Legacy Inconsistencies

- `src.models.gff_nn.GatedFusionFormer` is a divergent and incomplete legacy
  implementation; it references symbols that are not defined in that module.
- `src.models.factory.build_model` maps `gffnn` to an undefined `GFFNN` symbol.
- `src.run` is an incomplete interactive fragment rather than a command
  dispatcher.
- Research utilities use inconsistent bare and package-relative imports.
- `src.export` is a legacy generic exporter and is not the deployment export
  contract.

## Observed Preprocessing Contract

- IQ is a two-channel signal with shape `[B, 2, 128]`.
- S-TD is `[I ** 2 - Q ** 2, 2 * I * Q]` with the same signal length.
- STFT uses `scipy.signal.stft` with `fs=1.0`, a Blackman window,
  `nperseg=31`, `noverlap=30`, and `nfft=128`; deployment uses the first 32
  magnitude frequency bins.
- The model adapts STFT time features to 128 steps internally. The exact
  dataset SNR inventory remains unverified until a RadioML dataset is supplied.

## Supported Execution Model

Run commands from `_development/amr-gff-nn` with CPython 3.10:

```bash
conda activate amr-gffnn-py310
python -m pip install -r deploy/requirements.txt
python -m deploy.checkpoint
python -m deploy.export_onnx
python -m deploy.validate_onnx
python -m deploy.quantize_onnx
uvicorn deploy.service.app:app --host 0.0.0.0 --port 8000
```

All code uses package-qualified imports. No module mutates `sys.path` or
requires callers to set `PYTHONPATH`.

The deployment dependency file only references the workspace-approved package
allowlist and the official PyPI index. The supported runtime is CPython 3.10.x;
local Python 3.13 is not a deployment validation environment.

## Multi-Backend Roadmap

- ONNX Runtime CPU is the validated portable reference and fallback backend.
- `AMR_RUNTIME=cpu|cuda|tensorrt|auto` selects the requested provider chain;
  explicit CUDA or TensorRT requests fail when their primary provider is absent.
- CUDA ONNX Runtime, FP16, TensorRT engines, and vendor NPU artifacts require
  separate compatibility, parity, and end-to-end benchmark validation.
- DSP preprocessing remains CPU-side NumPy/SciPy until profiling demonstrates
  that it dominates the end-to-end budget.
- Static Batch-1 ONNX is an upcoming compiler input for a chosen edge/NPU
  target; ONNX alone does not guarantee full accelerator delegation.

## FP32 ONNX Evidence

- **ONNX opset:** 17.
- **Exporter:** `torch.onnx.export` legacy path. The modern `dynamo=True`
  path requires `onnxscript`, which is outside the approved direct-dependency
  allowlist.
- **Graph validation:** `onnx.checker.check_model` and ONNX Runtime
  `CPUExecutionProvider` smoke inference passed.
- **Deterministic parity fixtures:** zero signal, sinusoid, fixed-seed noise,
  and a variable-length interpolation path.
- **Parity metrics:** max logit absolute difference `9.804964e-06`, mean logit
  absolute difference `1.362779e-06`, max gating difference `1.728535e-06`,
  mean gating difference `2.862265e-07`, max probability difference
  `8.344650e-07`, and 100% top-1 agreement over four deterministic fixtures.

## Quantization and Deployment Status

- Dynamic INT8 quantization for `MatMul` and `Gemm` is generated only after
  successful FP32 parity. Softmax and normalization operations remain floating
  point by exclusion.
- Static INT8 QDQ quantization is deliberately blocked until a real,
  deterministic, stratified class/SNR calibration fixture is available.
- `model.int8.onnx` is structurally load-tested by ONNX Runtime. Accuracy,
  per-class/per-SNR degradation, and gating deviation for INT8 are unavailable
  without a RadioML dataset and must not be inferred from synthetic fixtures.
- The FastAPI service reads prebuilt ONNX artifacts only. It never exports or
  quantizes at startup. Set `AMR_VERIFY_CHECKPOINT_ON_STARTUP=true` only when
  an additional private Hugging Face checkpoint checksum verification is
  required at startup.

## Container Commands

The Docker build context is the repository root and requires prebuilt artifacts:

```bash
docker build -f _development/amr-gff-nn/deploy/service/Dockerfile -t amr-gffnn:latest .
docker run --rm -p 8000:8000 amr-gffnn:latest
curl http://localhost:8000/health
```

Docker CLI was unavailable in the implementation environment, so the container
build and runtime smoke test remain an explicit deployment gate.

## Validation Gates

1. Strict checkpoint compatibility confirms the canonical model.
2. The PyTorch CPU baseline validates preprocessing, logits, and gating output.
3. ONNX parity must pass before quantization.
4. Dataset-dependent class/SNR and calibration acceptance results remain
   explicitly unavailable until a real dataset is supplied.
