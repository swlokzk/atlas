"""Strict deterministic PyTorch-to-ONNX Runtime parity validation for GFF v3."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from deploy.export_onnx import ARTIFACT_DIR, export_fp32_onnx
from deploy.model_wrapper import GFFInferenceWrapper
from deploy.preprocessing import preprocess_batch


DEFAULT_THRESHOLDS = {
    "max_logit_abs": 1e-4,
    "mean_logit_abs": 1e-5,
    "max_gating_abs": 1e-4,
    "mean_gating_abs": 1e-5,
    "max_probability_abs": 1e-5,
}


class OnnxParityError(RuntimeError):
    """Raised when ONNX output diverges from the PyTorch reference baseline."""


def _fixtures() -> np.ndarray:
    generator = np.random.default_rng(29)
    phase = np.linspace(0.0, 2.0 * np.pi, 128, dtype=np.float32)
    sinusoid = np.vstack((np.cos(phase), np.sin(phase)))
    noise = generator.normal(0.0, 0.25, size=(2, 128)).astype(np.float32)
    variable_length = np.vstack((np.linspace(-1.0, 1.0, 63), np.linspace(1.0, -1.0, 63)))
    iq, _, _ = preprocess_batch(np.stack((np.zeros((2, 128)), sinusoid, noise)))
    variable_iq, _, _ = preprocess_batch(np.expand_dims(variable_length, axis=0))
    return np.concatenate((iq, variable_iq), axis=0)


def _metrics(reference: np.ndarray, candidate: np.ndarray, prefix: str) -> dict[str, float]:
    difference = np.abs(reference - candidate)
    return {
        f"max_{prefix}_abs": float(difference.max()),
        f"mean_{prefix}_abs": float(difference.mean()),
    }


def validate_onnx_parity(
    artifact_path: Path | None = None,
    output_dir: Path = ARTIFACT_DIR,
    thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    """Run deterministic raw-IQ parity checks and write a report or failure case."""
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    artifact_path = artifact_path or output_dir / "model.fp32.onnx"
    if not artifact_path.is_file():
        artifact_path = export_fp32_onnx(output_dir)

    wrapper, _, _ = GFFInferenceWrapper.from_checkpoint()
    raw_iq = _fixtures()
    iq, stft, std = preprocess_batch(raw_iq)
    with torch.inference_mode():
        pytorch_logits, pytorch_gating = wrapper(
            torch.from_numpy(iq), torch.from_numpy(stft), torch.from_numpy(std)
        )
        wrapper.validate_outputs(pytorch_logits, pytorch_gating)
        pytorch_probabilities = torch.softmax(pytorch_logits, dim=1).numpy()

    session = ort.InferenceSession(str(artifact_path), providers=["CPUExecutionProvider"])
    onnx_logits, onnx_gating = session.run(
        ["logits", "gating_weights"], {"iq": iq, "stft": stft, "std": std}
    )
    onnx_probabilities = np.exp(onnx_logits - onnx_logits.max(axis=1, keepdims=True))
    onnx_probabilities /= onnx_probabilities.sum(axis=1, keepdims=True)

    pytorch_logits_array = pytorch_logits.numpy()
    pytorch_gating_array = pytorch_gating.numpy()
    metrics = {
        **_metrics(pytorch_logits_array, onnx_logits, "logit"),
        **_metrics(pytorch_gating_array, onnx_gating, "gating"),
        "max_probability_abs": float(np.abs(pytorch_probabilities - onnx_probabilities).max()),
        "top1_agreement": float(
            np.mean(np.argmax(pytorch_logits_array, axis=1) == np.argmax(onnx_logits, axis=1))
        ),
        "sample_count": int(iq.shape[0]),
    }
    failed = (
        metrics["max_logit_abs"] >= thresholds["max_logit_abs"]
        or metrics["mean_logit_abs"] >= thresholds["mean_logit_abs"]
        or metrics["max_gating_abs"] >= thresholds["max_gating_abs"]
        or metrics["mean_gating_abs"] >= thresholds["mean_gating_abs"]
        or metrics["max_probability_abs"] >= thresholds["max_probability_abs"]
        or metrics["top1_agreement"] != 1.0
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validation_scope": "deterministic synthetic and preprocessing fixtures; dataset coverage unavailable",
        "thresholds": thresholds,
        "metrics": metrics,
        "passed": not failed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parity_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failed:
        np.savez_compressed(
            output_dir / "parity_failure.npz",
            iq=iq,
            stft=stft,
            std=std,
            pytorch_logits=pytorch_logits_array,
            onnx_logits=onnx_logits,
            pytorch_gating=pytorch_gating_array,
            onnx_gating=onnx_gating,
        )
        raise OnnxParityError(f"ONNX parity failed: {json.dumps(metrics, sort_keys=True)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FP32 ONNX parity against PyTorch.")
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()
    report = validate_onnx_parity(args.artifact, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()