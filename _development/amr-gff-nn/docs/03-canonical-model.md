# Canonical Model

## Model Definition

The deployment model is:

```text
src.models.model.GatedFusionFormer
```

The model name exposed by deployment metadata and configuration is `gffnn`.

## Configuration

| Parameter | Value |
|---|---:|
| `embed_dim` | 96 |
| `num_classes` | 11 |
| `signal_length` | 128 |
| `stft_frequency_bins` | 32 |
| `stft_time_steps` | 128 |
| `depth` | 4 |
| `num_heads` | 4 |

## Class Order

The output class order is fixed and must not be changed independently by a backend:

```text
8PSK, BPSK, CPFSK, GFSK, PAM4,
QAM16, QAM64, QPSK, AM-DSB, AM-SSB, WBFM
```

## Checkpoint Contract

The verified checkpoint is a raw PyTorch `state_dict` loaded with `strict=True` against the committed model configuration. Its filename is:

```text
gatedfusionformer_v4.0_best_20251025_211831.pth
```

The recorded SHA256 is:

```text
9523ff1097725bd8b15ea1f53e7d0de154eb719ce27412189865abd0325fdc9d
```

Checkpoint retrieval uses configured private registry credentials. Tokens, local cache paths, and private credentials must not be committed.

## Production Wrapper Contract

The deployment wrapper exposes two outputs:

```text
logits:         [B, 11]
gating_weights: [B, 3]
```

Gating order is:

```text
[iq, stft, std]
```

The service validates that gating values are finite, non-negative, and approximately sum to one. Backends must preserve output names, shapes, class order, and gating order.
