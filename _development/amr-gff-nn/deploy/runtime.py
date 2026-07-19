"""ONNX Runtime provider selection for portable GFFNN deployment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import onnxruntime as ort


RUNTIME_PROVIDERS: dict[str, tuple[str, ...]] = {
    "cpu": ("CPUExecutionProvider",),
    "cuda": ("CUDAExecutionProvider", "CPUExecutionProvider"),
    "tensorrt": ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"),
}


@dataclass(frozen=True)
class RuntimeSelection:
    """Requested and resolved ONNX Runtime provider configuration."""

    requested: str
    selected: str
    providers: tuple[str, ...]


def resolve_runtime(
    requested: str | None = None,
    available_providers: Sequence[str] | None = None,
) -> RuntimeSelection:
    """Resolve an ONNX Runtime provider chain without creating a session."""
    requested_runtime = (requested or os.getenv("AMR_RUNTIME", "auto")).strip().lower()
    if requested_runtime not in {"auto", *RUNTIME_PROVIDERS}:
        supported = ", ".join(("auto", *RUNTIME_PROVIDERS))
        raise RuntimeError(f"Unsupported AMR_RUNTIME={requested_runtime!r}; expected one of: {supported}.")

    available = set(available_providers or ort.get_available_providers())
    if requested_runtime == "auto":
        for candidate in ("tensorrt", "cuda", "cpu"):
            if RUNTIME_PROVIDERS[candidate][0] in available:
                return RuntimeSelection("auto", candidate, RUNTIME_PROVIDERS[candidate])
        raise RuntimeError("No supported ONNX Runtime execution provider is available.")

    primary_provider = RUNTIME_PROVIDERS[requested_runtime][0]
    if primary_provider not in available:
        raise RuntimeError(
            f"AMR_RUNTIME={requested_runtime} requires {primary_provider}, but available providers are: "
            f"{', '.join(sorted(available)) or 'none'}."
        )
    return RuntimeSelection(requested_runtime, requested_runtime, RUNTIME_PROVIDERS[requested_runtime])


def create_session(
    artifact_path: Path | str,
    requested: str | None = None,
    session_options: ort.SessionOptions | None = None,
) -> tuple[ort.InferenceSession, RuntimeSelection]:
    """Create a session with the selected provider chain."""
    selection = resolve_runtime(requested)
    session = ort.InferenceSession(str(artifact_path), sess_options=session_options, providers=list(selection.providers))
    return session, selection