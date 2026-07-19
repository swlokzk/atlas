# Preprocessing Contract

The preprocessing contract is shared by every inference backend. Accelerator work must not alter it without a new parity gate.

## External Input

The service accepts raw IQ data with shape:

```text
[2, N]
```

The first row is I and the second row is Q. Values must be finite real numbers. Empty channels, unequal channel lengths, and more than 4096 samples per channel are rejected by the API schema.

## Length Policy

The model target length is 128. Any input length other than 128 is deterministically resized by linear interpolation before feature construction. No additional amplitude normalization is applied.

## Model Inputs

```text
iq:  [B, 2, 128]
stft: [B, 1, 32, 128]
std: [B, 2, 128]
```

## STFT

The complex signal is constructed as:

```python
complex_signal = I + 1j * Q
```

The deployment STFT uses:

```text
window          = blackman
fs              = 1.0
nperseg         = 31
noverlap        = 30
nfft             = 128
frequency bins  = first 32 magnitude bins
```

The output is the magnitude of the complex STFT, expanded to one channel.

## S-TD Features

The statistical time-domain feature is:

```python
np.vstack((
    I ** 2 - Q ** 2,
    2 * I * Q,
))
```

The output channel order is `[I^2 - Q^2, 2IQ]`.

## Version

The current preprocessing version is:

```text
amr-preprocessing-v1
```

Preprocessing changes require deterministic fixture comparison against the PyTorch and ONNX reference paths before they can be adopted by a service or accelerator backend.
