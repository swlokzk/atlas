# Project Overview

## Purpose

AMR-GFFNN is a multimodal automatic modulation recognition project. The canonical model combines raw IQ signals, STFT time-frequency features, and statistical time-domain features through a learned gating network.

## Identity

| Field | Value |
|---|---|
| Main model name | `gffnn` |
| Deployment identity | `gff-v5` |
| Canonical class | `src.models.model.GatedFusionFormer` |
| Current validated deployment backend | ONNX Runtime CPU |
| Research framework | PyTorch |
| Dataset target | RadioML 2016.10a |

## Scope

The project has two related but separate workflows:

- Research: training, evaluation, ablation, gating analysis, and architecture comparison.
- Deployment: checkpoint compatibility, deterministic preprocessing, ONNX export, parity validation, quantization experiments, and API serving.

The deployment workflow must not replace the research workflow. Both workflows use the same canonical model contract where applicable, but they may have different dependency and execution requirements.

## Current Status

Complete or structurally validated:

- strict checkpoint loading;
- deterministic deployment preprocessing;
- PyTorch CPU inference baseline;
- FP32 ONNX export and checker validation;
- ONNX Runtime CPU parity validation;
- dynamic INT8 artifact generation and structural loading;
- FastAPI service definition;
- CPU Docker definition.

Pending:

- real RadioML class and SNR regression;
- INT8 accuracy approval and calibration;
- CUDA provider parity and end-to-end benchmarks;
- FP16 and TensorRT evaluation;
- static Batch-1 compiler validation;
- Docker build and runtime smoke testing where Docker is unavailable;
- compression and architecture variant benchmarking.

## Non-Goals

The current deployment plan does not include GGUF, GGML, llama.cpp, custom GGML graphs, immediate TensorRT INT8, generic support for every NPU vendor, or moving DSP preprocessing into the first accelerator graph.
