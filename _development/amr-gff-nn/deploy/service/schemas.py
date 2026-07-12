"""Validated public request and response schemas for AMR inference."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAX_SIGNAL_SAMPLES = 4096


class ClassifyRequest(BaseModel):
    """Raw I/Q request. Sample-rate metadata is accepted but not resampled."""

    model_config = ConfigDict(extra="forbid")
    iq: list[list[float]]
    sample_rate: float | None = Field(default=None, gt=0)

    @field_validator("iq")
    @classmethod
    def validate_iq(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 2:
            raise ValueError("iq must contain exactly two channels.")
        if not value[0] or not value[1]:
            raise ValueError("iq channels must not be empty.")
        if len(value[0]) != len(value[1]):
            raise ValueError("iq channels must have equal lengths.")
        if len(value[0]) > MAX_SIGNAL_SAMPLES:
            raise ValueError(f"iq must not exceed {MAX_SIGNAL_SAMPLES} samples per channel.")
        if not all(math.isfinite(sample) for channel in value for sample in channel):
            raise ValueError("iq values must be finite.")
        return value


class ClassifyResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prediction: str
    class_id: int
    confidence: float
    probabilities: dict[str, float]
    gating_weights: dict[str, float]
    preprocessing_latency_ms: float
    inference_latency_ms: float
    total_latency_ms: float
    model_version: str
