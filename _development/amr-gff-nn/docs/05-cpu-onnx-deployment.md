# CPU ONNX Deployment

## Reference Flow

```text
Private checkpoint
    -> strict model loading
    -> deterministic preprocessing
    -> PyTorch CPU baseline
    -> FP32 ONNX export
    -> ONNX checker and CPU smoke inference
    -> PyTorch/ONNX parity
    -> optional dynamic INT8 artifact
    -> FastAPI service
```

ONNX Runtime CPU is the portable reference backend and fallback runtime. CUDA and hardware-specific runtimes must be compared against this path rather than replacing it.

## Commands

Run from `_development/amr-gff-nn`:

```bash
python -m deploy.checkpoint
python -m deploy.export_onnx
python -m deploy.validate_onnx
python -m deploy.quantize_onnx
python -m deploy.runtime_check
```

The service reads prebuilt artifacts and does not export or quantize during startup:

```bash
uvicorn deploy.service.app:app --host 0.0.0.0 --port 8000
```

## Existing Artifact Bundle

The repository currently uses:

```text
artifacts/gff-ver/
```

Typical files include `model.fp32.onnx`, `model.int8.onnx`, configuration, preprocessing, checkpoint, and parity manifests. The artifact directory is existing deployment evidence; it is not a claim that a regenerated `gff-v5` artifact bundle has been produced.

## FP32 Evidence

Recorded deterministic fixture results include:

```text
max logit difference:       9.804964e-06
mean logit difference:      1.362779e-06
max gating difference:     1.728535e-06
mean gating difference:    2.862265e-07
max probability difference: 8.344650e-07
top-1 agreement:            100%
```

These are fixture-level parity results. They do not replace real RadioML class and SNR evaluation.

## INT8 Status

Dynamic INT8 has been generated and structurally loaded. It is not accuracy-approved until real RadioML calibration and regression results are available. Static INT8 calibration must not fabricate calibration data.
