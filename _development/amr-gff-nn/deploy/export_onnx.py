"""Export and structurally validate the canonical GFF checkpoint as FP32 ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from deploy.model_wrapper import GFFInferenceWrapper
from deploy.preprocessing import PREPROCESSING_VERSION


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "gff-v3"
OPSET_VERSION = 17


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_fp32_onnx(output_dir: Path = ARTIFACT_DIR) -> Path:
    """Export the strict checkpoint-backed production wrapper and validate it locally."""
    wrapper, config, checkpoint_info = GFFInferenceWrapper.from_checkpoint()
    wrapper.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.fp32.onnx"
    batch_size = 1
    iq = torch.zeros(batch_size, 2, int(config["signal_length"]), dtype=torch.float32)
    stft = torch.zeros(
        batch_size,
        1,
        int(config["stft_frequency_bins"]),
        int(config["stft_time_steps"]),
        dtype=torch.float32,
    )
    std = torch.zeros(batch_size, 2, int(config["signal_length"]), dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (iq, stft, std),
        output_path,
        input_names=["iq", "stft", "std"],
        output_names=["logits", "gating_weights"],
        dynamic_axes={
            "iq": {0: "batch"},
            "stft": {0: "batch"},
            "std": {0: "batch"},
            "logits": {0: "batch"},
            "gating_weights": {0: "batch"},
        },
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    outputs = session.run(
        ["logits", "gating_weights"],
        {"iq": iq.numpy(), "stft": stft.numpy(), "std": std.numpy()},
    )
    if outputs[0].shape != (batch_size, int(config["num_classes"])):
        raise RuntimeError(f"Unexpected ONNX logits shape: {outputs[0].shape}")
    if outputs[1].shape != (batch_size, 3):
        raise RuntimeError(f"Unexpected ONNX gating-weight shape: {outputs[1].shape}")
    if not np.isfinite(outputs[0]).all() or not np.isfinite(outputs[1]).all():
        raise RuntimeError("ONNX Runtime smoke inference produced non-finite outputs.")

    (output_dir / "model_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output_dir / "labels.json").write_text(json.dumps(config["labels"], indent=2) + "\n", encoding="utf-8")
    (output_dir / "preprocessing.json").write_text(
        json.dumps(
            {
                "preprocessing_version": PREPROCESSING_VERSION,
                "signal_length": config["signal_length"],
                "stft": {
                    "window": "blackman",
                    "nperseg": 31,
                    "noverlap": 30,
                    "nfft": 128,
                    "frequency_bins": config["stft_frequency_bins"],
                },
                "std_formula": "[I ** 2 - Q ** 2, 2 * I * Q]",
                "variable_length_policy": "linear_interpolation_to_128_before_features",
                "normalization": "none",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_manifest_source = PROJECT_ROOT / "deploy" / "config" / "checkpoint_manifest.json"
    shutil.copyfile(checkpoint_manifest_source, output_dir / "checkpoint_manifest.json")
    manifest = {
        "artifact_name": output_path.name,
        "model_version": config["model_version"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sha256": _sha256(output_path),
        "opset_version": OPSET_VERSION,
        "exporter": "torch.onnx.export legacy exporter; dynamo exporter unavailable without non-whitelisted onnxscript",
        "runtime": "onnxruntime==1.18.1 CPUExecutionProvider",
        "checkpoint_sha256": checkpoint_info.sha256,
        "preprocessing_version": PREPROCESSING_VERSION,
        "tensor_shapes": {
            "iq": ["batch", 2, int(config["signal_length"])],
            "stft": ["batch", 1, int(config["stft_frequency_bins"]), int(config["stft_time_steps"])],
            "std": ["batch", 2, int(config["signal_length"])],
            "logits": ["batch", int(config["num_classes"])],
            "gating_weights": ["batch", 3],
        },
        "labels": config["labels"],
        "quantization_state": "fp32",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GFF v3 checkpoint to FP32 ONNX.")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()
    artifact_path = export_fp32_onnx(args.output_dir)
    print(f"exported_onnx={artifact_path}")


if __name__ == "__main__":
    main()