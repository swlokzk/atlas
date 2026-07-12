# AMR GFFNN Deployment Implementation Notes

## Canonical Model

- **Candidate:** `src.models.model.GatedFusionFormer`
- **Version:** `gff-v3`
- **Reason:** It is the only complete Gated Fusion Former implementation in the
  current repository. It defines the encoder, transformer, normalization,
  gating, and output head used by its forward path.
- **Compatibility status:** Pending a strict load of the private
  Hugging Face checkpoint `gff_v3.pth`. No architecture is modified to fit a
  checkpoint.

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
python -m deploy.checkpoint
python -m deploy.predict --help
python -m deploy.export_onnx --help
python -m deploy.validate_onnx --help
uvicorn deploy.service.app:app --host 0.0.0.0 --port 8000
```

All code uses package-qualified imports. No module mutates `sys.path` or
requires callers to set `PYTHONPATH`.

## Validation Gates

1. Strict checkpoint compatibility confirms the canonical model.
2. The PyTorch CPU baseline validates preprocessing, logits, and gating output.
3. ONNX parity must pass before quantization.
4. Dataset-dependent class/SNR and calibration acceptance results remain
   explicitly unavailable until a real dataset is supplied.
