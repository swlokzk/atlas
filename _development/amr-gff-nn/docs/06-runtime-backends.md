# Runtime Backends

## Runtime Selection

The service uses the `deploy.runtime` session factory. Select a backend with:

```text
AMR_RUNTIME=auto|cpu|cuda|tensorrt
```

Provider policy:

| Runtime | Provider order | Status |
|---|---|---|
| `cpu` | `CPUExecutionProvider` | Default and validated reference |
| `cuda` | CUDA, CPU | Planned; explicit CUDA availability required |
| `tensorrt` | TensorRT, CUDA, CPU | Planned; hardware validation required |
| `auto` | TensorRT, CUDA, CPU when available | CPU on CPU-only hosts |

An explicit CUDA or TensorRT request fails clearly when its primary provider is unavailable. This prevents silent execution on an unintended backend.

## Shared Boundaries

All backends share:

- the canonical `GatedFusionFormer` wrapper;
- the preprocessing contract;
- ONNX input and output names;
- class and gating order;
- API schema and response behavior;
- parity and accuracy acceptance gates.

DSP preprocessing remains CPU-side NumPy/SciPy initially. Move it to an accelerator only after end-to-end profiling proves it is the dominant cost and a new parity contract is established.

## Measurement

Backend benchmarks must separate:

- preprocessing;
- host-to-device transfer;
- model inference;
- device-to-host transfer;
- postprocessing;
- end-to-end latency;
- P50 and P95 latency;
- throughput;
- memory use.

Batch-1 GPU results must include transfer and preprocessing overhead. Model-only latency is insufficient for adoption.

## Roadmap

1. Validate ONNX Runtime CUDA with the canonical FP32 artifact.
2. Compare CPU, CUDA, and PyTorch outputs on deterministic fixtures.
3. Benchmark batch sizes 1, 8, 32, and 64.
4. Evaluate FP16 only if measured end-to-end performance or memory improves.
5. Build a TensorRT FP16 engine only for a documented target GPU.
6. Evaluate static Batch-1 ONNX for a selected NPU vendor and device.
