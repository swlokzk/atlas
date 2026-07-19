# Artifact Governance

## Artifact Principles

Generated model files are deployment outputs, not source code. Large binary artifacts should remain excluded from Git unless repository policy explicitly permits them.

Every artifact bundle should be traceable to:

- model name and deployment version;
- checkpoint SHA256;
- source Git commit;
- creation timestamp;
- preprocessing version;
- labels and output order;
- input and output shapes;
- runtime and precision;
- hardware compatibility;
- artifact SHA256.

## Current and Target Paths

The existing repository bundle is:

```text
artifacts/gff-ver/
```

The deployment identity in configuration is `gffnn` / `gff-v5`. A future regenerated bundle should use an explicitly chosen layout and manifest rather than silently overwriting the existing `gff-ver` artifact directory.

A target layout is:

```text
artifacts/gff-v5/
├── shared/
├── cpu/
├── static/
└── gpu/
```

## Manifest Fields

At minimum, manifests should include:

```json
{
  "model_name": "gffnn",
  "model_version": "gff-v5",
  "canonical_class": "src.models.model.GatedFusionFormer",
  "preprocessing_version": "amr-preprocessing-v1",
  "runtime": "onnxruntime",
  "precision": "fp32",
  "tensor_shapes": {},
  "labels": [],
  "checkpoint_sha256": "...",
  "artifact_sha256": "..."
}
```

The exact manifest schema should be kept compatible with the service loader and extended when a backend needs hardware metadata.

## Version Rules

- Do not rename a generated artifact only to make it appear newer.
- Do not call an artifact production-approved without real-data validation.
- Keep model identity, artifact directory, and checkpoint filename as separate fields.
- Record provider and precision independently from model version.
- Preserve checksums whenever an artifact is copied or promoted.
