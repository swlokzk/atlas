"""PyTorch CPU reference predictor for AMR Gated Fusion Former deployment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from deploy.model_wrapper import GFFInferenceWrapper
from deploy.preprocessing import preprocess_iq


def predict_iq(
    iq: Sequence[Sequence[float]] | np.ndarray,
    wrapper: GFFInferenceWrapper,
    model_config: dict[str, object],
) -> dict[str, object]:
    """Run deterministic CPU inference and return JSON-safe prediction telemetry."""
    started = time.perf_counter()
    preprocessed = preprocess_iq(iq, target_length=int(model_config["signal_length"]))
    preprocessing_latency_ms = (time.perf_counter() - started) * 1000.0

    iq_tensor = torch.from_numpy(preprocessed.iq).unsqueeze(0)
    stft_tensor = torch.from_numpy(preprocessed.stft).unsqueeze(0)
    std_tensor = torch.from_numpy(preprocessed.std).unsqueeze(0)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        logits, gating_weights = wrapper(iq_tensor, stft_tensor, std_tensor)
        wrapper.validate_outputs(logits, gating_weights)
        probabilities = torch.softmax(logits, dim=1)[0]
    inference_latency_ms = (time.perf_counter() - inference_started) * 1000.0

    class_id = int(torch.argmax(probabilities).item())
    labels = list(model_config["labels"])
    values = probabilities.cpu().tolist()
    gating_values = gating_weights[0].cpu().tolist()
    total_latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "prediction": labels[class_id],
        "class_id": class_id,
        "confidence": float(values[class_id]),
        "probabilities": {label: float(value) for label, value in zip(labels, values)},
        "gating_weights": {
            "iq": float(gating_values[0]),
            "stft": float(gating_values[1]),
            "std": float(gating_values[2]),
        },
        "preprocessing_latency_ms": preprocessing_latency_ms,
        "inference_latency_ms": inference_latency_ms,
        "total_latency_ms": total_latency_ms,
        "model_version": str(model_config["model_version"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GFF PyTorch CPU inference from raw IQ JSON.")
    parser.add_argument("--input", required=True, type=Path, help="JSON file containing an `iq` field")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "iq" not in payload:
        raise ValueError("Input JSON must contain an `iq` field.")

    wrapper, model_config, _ = GFFInferenceWrapper.from_checkpoint()
    print(json.dumps(predict_iq(payload["iq"], wrapper, model_config), indent=2))


if __name__ == "__main__":
    main()