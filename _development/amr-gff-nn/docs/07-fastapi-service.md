# FastAPI Service

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Verify that a model session is loaded |
| `GET` | `/metadata` | Return model, artifact, preprocessing, and runtime metadata |
| `POST` | `/v1/classify` | Classify one raw IQ request |

Start the service from `_development/amr-gff-nn`:

```bash
uvicorn deploy.service.app:app --host 0.0.0.0 --port 8000
```

## Startup Behavior

The service selects `model.int8.onnx` when it exists and otherwise uses `model.fp32.onnx`. It requires the artifact, model configuration, and manifest before startup succeeds. Export and quantization are never performed during startup.

The runtime is selected through `AMR_RUNTIME`. Set `AMR_VERIFY_CHECKPOINT_ON_STARTUP=true` only when startup should retrieve and verify the private checkpoint checksum against the manifest.

## Request Contract

```json
{
  "iq": [[0.1, 0.2], [0.05, 0.15]],
  "sample_rate": 1.0
}
```

`iq` must contain exactly two finite, non-empty, equal-length channels. The maximum is 4096 samples per channel. `sample_rate` is optional and must be positive when supplied; it is accepted as metadata and does not trigger resampling.

## Response Behavior

The response contains:

- predicted label and class ID;
- confidence and per-class probabilities;
- ordered gating weights for `iq`, `stft`, and `std`;
- preprocessing, inference, and total latency;
- model version.

Raw IQ values are not returned. A request ID is returned in the `X-Request-ID` response header.

## Metadata

The `/metadata` response includes model version, ONNX Runtime version, requested and selected runtime, active provider, fallback providers, labels, preprocessing version, tensor shapes, and artifact checksum.

## Operational Constraints

The service should not log raw IQ signals by default. Asynchronous dynamic batching is deferred until traffic measurements demonstrate that batch-1 GPU utilization is insufficient and queueing latency is acceptable.
