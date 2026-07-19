# GFFNN Documentation

This directory contains detailed project documentation for the AMR-GFFNN deployment and research workflows.

## Source Of Truth

- `README.md` is the project entry point and status summary.
- `DEVELOPMENT.md` is the operational guide for setup and common commands.
- `deploy/IMPLEMENTATION_NOTES.md` records implementation evidence and decisions.
- This directory contains topic-specific documents that should not duplicate entire manuals.

## Documents

| Document | Purpose |
|---|---|
| [Project Overview](01-project-overview.md) | Scope, naming, lifecycle, and current status |
| [Research Workflow](02-research-workflow.md) | RadioML analysis and research CLI workflows |
| [Canonical Model](03-canonical-model.md) | Model, checkpoint, labels, and output contract |
| [Preprocessing Contract](04-preprocessing-contract.md) | Deterministic IQ, STFT, and S-TD processing |
| [CPU ONNX Deployment](05-cpu-onnx-deployment.md) | Validated checkpoint-to-service deployment path |
| [Runtime Backends](06-runtime-backends.md) | CPU, CUDA, TensorRT, and future NPU runtime policy |
| [FastAPI Service](07-fastapi-service.md) | HTTP API, startup behavior, metadata, and telemetry |
| [Validation Matrix](08-validation-matrix.md) | Test gates, evidence, and pending validation |
| [Artifact Governance](09-artifact-governance.md) | Artifact naming, manifests, checksums, and versioning |

## Version Terminology

`gffnn` is the main model name and `gff-v5` is the current deployment identity in configuration and planning documents.

The repository still contains generated artifacts under `artifacts/gff-v3/`. That directory name identifies the existing artifact bundle and must not be interpreted as proof that a new `gff-v5` artifact bundle has been exported.
