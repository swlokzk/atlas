"""Deterministic raw-IQ preprocessing for production Gated Fusion Former inference."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d


PREPROCESSING_VERSION = "amr-preprocessing-v1"
SIGNAL_LENGTH = 128
STFT_FREQUENCY_BINS = 32
_STFT_NPERSEG = 31
_STFT_NOVERLAP = 30
_STFT_NFFT = 128


class InputValidationError(ValueError):
    """Raised when external raw IQ input violates the deployment contract."""


@dataclass(frozen=True)
class PreprocessedSample:
    """The three modalities for one request, each stored as float32 arrays."""

    iq: np.ndarray
    stft: np.ndarray
    std: np.ndarray


def _coerce_iq(iq: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    try:
        raw = np.asarray(iq)
    except (TypeError, ValueError) as error:
        raise InputValidationError("IQ must be a rectangular numeric array.") from error

    if raw.ndim != 2 or raw.shape[0] != 2:
        raise InputValidationError("IQ must have exactly two channels with shape [2, samples].")
    if raw.shape[1] == 0:
        raise InputValidationError("IQ channels must not be empty.")
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise InputValidationError("IQ values must use a real numeric dtype.")

    result = raw.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise InputValidationError("IQ values must be finite; NaN and Inf are not supported.")
    return result


def _interpolate_to_signal_length(iq: np.ndarray, target_length: int) -> np.ndarray:
    source_length = iq.shape[1]
    if source_length == target_length:
        return iq
    if source_length == 1:
        return np.repeat(iq, target_length, axis=1)

    source_positions = np.linspace(0.0, 1.0, source_length, dtype=np.float64)
    target_positions = np.linspace(0.0, 1.0, target_length, dtype=np.float64)
    interpolate = interp1d(
        source_positions,
        iq,
        axis=1,
        kind="linear",
        bounds_error=True,
        assume_sorted=True,
    )
    return np.asarray(interpolate(target_positions), dtype=np.float32)


def _build_stft(iq: np.ndarray) -> np.ndarray:
    complex_signal = iq[0] + 1j * iq[1]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Input data is complex, switching to return_onesided=False",
        )
        _, _, stft_raw = signal.stft(
            complex_signal,
            fs=1.0,
            window="blackman",
            nperseg=_STFT_NPERSEG,
            noverlap=_STFT_NOVERLAP,
            nfft=_STFT_NFFT,
        )
    stft = np.abs(stft_raw)[:STFT_FREQUENCY_BINS, :]
    if stft.shape[0] != STFT_FREQUENCY_BINS:
        raise InputValidationError(
            f"STFT produced {stft.shape[0]} frequency bins; expected {STFT_FREQUENCY_BINS}."
        )
    return np.expand_dims(stft.astype(np.float32, copy=False), axis=0)


def preprocess_iq(
    iq: Sequence[Sequence[float]] | np.ndarray,
    target_length: int = SIGNAL_LENGTH,
) -> PreprocessedSample:
    """Convert one raw `[2, samples]` IQ signal into all GFF deployment modalities.

    Signals with a length other than 128 use deterministic linear interpolation
    before feature generation. The training code applies no amplitude
    normalization, so deployment performs none before or after interpolation.
    """
    if target_length <= 0:
        raise InputValidationError("target_length must be positive.")

    normalized_iq = _interpolate_to_signal_length(_coerce_iq(iq), target_length)
    in_phase, quadrature = normalized_iq
    std = np.vstack((
        in_phase ** 2 - quadrature ** 2,
        2.0 * in_phase * quadrature,
    )).astype(np.float32, copy=False)
    stft = _build_stft(normalized_iq)

    if not np.isfinite(stft).all() or not np.isfinite(std).all():
        raise InputValidationError("Preprocessing produced non-finite features.")
    return PreprocessedSample(iq=normalized_iq, stft=stft, std=std)


def preprocess_batch(
    iq_batch: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    target_length: int = SIGNAL_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a rectangular raw IQ batch into `[B, ...]` deployment tensors."""
    try:
        raw_batch = np.asarray(iq_batch)
    except (TypeError, ValueError) as error:
        raise InputValidationError("IQ batch must be a rectangular numeric array.") from error

    if raw_batch.ndim != 3 or raw_batch.shape[0] == 0:
        raise InputValidationError("IQ batch must have shape [batch, 2, samples].")

    samples = [preprocess_iq(sample, target_length) for sample in raw_batch]
    return (
        np.stack([sample.iq for sample in samples], axis=0),
        np.stack([sample.stft for sample in samples], axis=0),
        np.stack([sample.std for sample in samples], axis=0),
    )