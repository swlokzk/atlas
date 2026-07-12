"""ONNX Runtime quantization flow gated by validated FP32 parity."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

from deploy.export_onnx import ARTIFACT_DIR
from deploy.preprocessing import preprocess_batch


class CalibrationDataRequiredError(RuntimeError):
    """Raised when static INT8 quantization lacks a verified real-data fixture."""


def inspect_graph(model_path: Path) -> dict[str, int]:
    """Return operator counts to document attention, normalization, and gating graph surfaces."""
    graph = onnx.load(model_path).graph
    counts: dict[str, int] = {}
    for node in graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def _benchmark(session: ort.InferenceSession, iterations: int = 25) -> dict[str, float]:
    raw = np.zeros((1, 2, 128), dtype=np.float32)
    iq, stft, std = preprocess_batch(raw)
    inputs = {"iq": iq, "stft": stft, "std": std}
    for _ in range(3):
        session.run(None, inputs)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        session.run(None, inputs)
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "p50_inference_ms": float(np.percentile(samples, 50)),
        "p95_inference_ms": float(np.percentile(samples, 95)),
        "throughput_per_second": float(1000.0 / np.mean(samples)),
    }


def quantize_dynamic_int8(output_dir: Path = ARTIFACT_DIR) -> Path:
    """Quantize MatMul/Gemm weights to INT8 while keeping softmax/norm in FP32."""
    parity_report_path = output_dir / "parity_report.json"
    source_path = output_dir / "model.fp32.onnx"
    destination_path = output_dir / "model.int8.onnx"
    if not source_path.is_file():
        raise FileNotFoundError("FP32 ONNX artifact is missing; export and validate it first.")
    if not parity_report_path.is_file() or not json.loads(parity_report_path.read_text())["passed"]:
        raise RuntimeError("FP32 ONNX parity report is missing or failed; quantization is blocked.")

    quantize_dynamic(
        model_input=source_path,
        model_output=destination_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        extra_options={"EnableSubgraph": True},
    )
    session = ort.InferenceSession(str(destination_path), providers=["CPUExecutionProvider"])
    benchmark = _benchmark(session)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quantization_configuration": {
            "strategy": "dynamic_int8",
            "weight_type": "QInt8",
            "included_operators": ["MatMul", "Gemm"],
            "excluded_sensitive_operators": ["Softmax", "LayerNormalization"],
        },
        "source_graph_operators": inspect_graph(source_path),
        "quantized_graph_operators": inspect_graph(destination_path),
        "fp32_size_bytes": source_path.stat().st_size,
        "int8_size_bytes": destination_path.stat().st_size,
        "latency_benchmark": benchmark,
        "accuracy_degradation": "unavailable_without_real_dataset",
        "per_class_degradation": "unavailable_without_real_dataset",
        "per_snr_degradation": "unavailable_without_real_dataset",
        "gating_weight_deviation": "pending_int8_parity_validation",
    }
    (output_dir / "quantization_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return destination_path


def quantize_static_int8(calibration_npz: Path) -> None:
    """Reserve static QDQ quantization for a supplied, dataset-derived calibration fixture."""
    if not calibration_npz.is_file():
        raise CalibrationDataRequiredError(
            "Static INT8 requires a deterministic calibration fixture generated from the real dataset. "
            "No fixture was supplied, so calibration will not be fabricated."
        )
    raise NotImplementedError(
        "Static quantization is intentionally gated pending a real, stratified class/SNR calibration fixture."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize validated GFF FP32 ONNX artifacts.")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--static-calibration", type=Path, default=None)
    args = parser.parse_args()
    if args.static_calibration is not None:
        quantize_static_int8(args.static_calibration)
    else:
        print(f"quantized_onnx={quantize_dynamic_int8(args.output_dir)}")


if __name__ == "__main__":
    main()