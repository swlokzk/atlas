# Validation Matrix

## Current Gates

| Area | CPU | CUDA | TensorRT | Real dataset |
|---|---:|---:|---:|---:|
| Checkpoint strict loading | Complete | Shared | N/A | No |
| Preprocessing determinism | Complete | Shared | Shared | No |
| Wrapper output contract | Complete | Required | N/A | No |
| ONNX checker | Complete | Shared | N/A | No |
| FP32 parity | Complete on fixtures | Pending | Pending | Initially no |
| API validation | Complete structurally | Shared | Shared | No |
| Dynamic INT8 loading | Complete structurally | N/A | N/A | No |
| Per-class accuracy | Pending | Pending | Pending | Yes |
| Per-SNR accuracy | Pending | Pending | Pending | Yes |
| Gating deviation | Pending | Pending | Pending | Yes |
| End-to-end benchmark | Structural CPU result | Pending | Pending | No |
| Docker smoke test | Pending CLI | Pending | Pending | No |
| NPU compiler validation | Future | N/A | N/A | Hardware-specific |

## Recorded CPU Results

FP32 ONNX parity on deterministic fixtures:

```text
max logit difference:       9.804964e-06
mean logit difference:      1.362779e-06
max gating difference:      1.728535e-06
mean gating difference:     2.862265e-07
max probability difference: 8.344650e-07
top-1 agreement:             100%
```

Dynamic INT8 structural CPU benchmark:

```text
artifact size: approximately 2.93 MB
P50 inference: approximately 4.14 ms
P95 inference: approximately 4.83 ms
throughput: approximately 240 inferences/second
```

These figures are not real-data accuracy approval.

## Dataset Gate

Before INT8 or production accuracy approval, record:

- dataset checksum and inventory;
- class and SNR split strategy;
- random seed;
- overall accuracy;
- per-class accuracy;
- per-SNR accuracy;
- confusion matrices;
- gating-weight distribution;
- calibration manifest for static quantization.

Suggested initial degradation limits are 1 percentage point overall, 2 percentage points per SNR, and 3 percentage points per class. Review these against observed baseline variance before enforcing them.

## Backend Gate

A CUDA or TensorRT result is acceptable only when the intended provider is active, output parity is measured, fallback behavior is understood, and end-to-end benchmarks include preprocessing and transfer costs.
