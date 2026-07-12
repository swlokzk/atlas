# Deployment implementation notes

## Phase 0 audit

The selected production candidate is
`src.models.model.GatedFusionFormer`.  It is the implementation imported by
`src.models.factory`, accepts all three required modalities, supports the
research ablation argument, and its defaults exactly match
`src.configs.config.MODEL_CFG`.

Checkpoint compatibility is **unverified and blocked**: this repository
contains no `.pth`, `.pt`, `.ckpt`, or other checkpoint file.  Existing
`src.utils.load_model` assumes a raw `state_dict`, but that assumption has not
been tested and must not be used to select or alter the architecture.  Supply a
checkpoint before export, parity validation, compression, or quantization.

The authoritative label order is the `CLASSES` list in
`src.configs.config`: `8PSK`, `BPSK`, `CPFSK`, `GFSK`, `PAM4`, `QAM16`,
`QAM64`, `QPSK`, `AM-DSB`, `AM-SSB`, `WBFM`.  Available dataset SNR values are
also unverified because the required RML2016.10a pickle is not present.

## Training preprocessing contract observed

`src.core.dataset.RMLDataset` keeps IQ unchanged, derives S-TD as
`[I**2 - Q**2, 2*I*Q]`, and derives STFT magnitude from `I + 1j*Q` using
SciPy's `signal.stft` with `fs=1.0`, a Blackman window, `nperseg=31`,
`noverlap=30`, and `nfft=128`; it selects the first 32 frequency bins.  It
does not normalize, resample, or explicitly align STFT time steps.  The
production preprocessing module must reproduce this behavior after a real
dataset sample verifies its exact STFT time dimension.

## Known legacy inconsistencies

- `src.models.gff_nn.GatedFusionFormer` is an alternative incomplete model:
  it references undefined `LayerNorm` and `FusionTransformerBlock`.
- `src.models.factory.build_model` imports the selected model but registers an
  undefined `GFFNN`.
- `src.run` is an incomplete interactive stub, while both READMEs describe a
  subcommand dispatcher that does not exist.
- The research modules use imports incompatible with the current package
  layout; Phase 0 adds package markers only and deliberately does not rewrite
  their imports before compatibility tests are available.
- `src.utils` imports nonexistent `src.config` and `src.dataset` modules.
- The existing ONNX exporter can export randomly initialized weights, defaults
  to opset 13, and uses 64 STFT frequency bins, so it is not a deployment
  entrypoint.

## Supported commands at this point

Only package discovery is supported without external model dependencies:

```bash
cd /home/runner/work/atlas/atlas/_development/amr-gff-nn
python -c "import src; import src.configs"
```

Deployment commands, ONNX export, validation, quantization, and the service
are intentionally not advertised until checkpoint and dataset compatibility
gates are completed.
