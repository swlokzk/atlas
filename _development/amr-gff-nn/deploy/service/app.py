"""CPU-only prebuilt ONNX Runtime AMR inference service."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request

from deploy.checkpoint import download_checkpoint, inspect_checkpoint
from deploy.preprocessing import preprocess_iq
from deploy.runtime import RuntimeSelection, create_session
from deploy.service.schemas import ClassifyRequest, ClassifyResponse


LOGGER = logging.getLogger("amr_gffnn.service")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "gff-v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ServiceRuntime:
    def __init__(self) -> None:
        self.session: ort.InferenceSession | None = None
        self.artifact_path: Path | None = None
        self.config: dict[str, Any] = {}
        self.manifest: dict[str, Any] = {}
        self.runtime_selection: RuntimeSelection | None = None

    def load(self) -> None:
        artifact_dir = Path(os.getenv("AMR_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR)))
        preferred = artifact_dir / "model.int8.onnx"
        artifact_path = preferred if preferred.is_file() else artifact_dir / "model.fp32.onnx"
        config_path = artifact_dir / "model_config.json"
        manifest_path = artifact_dir / "manifest.json"
        if not artifact_path.is_file() or not config_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("Prebuilt ONNX artifact, config, or manifest is missing; export before startup.")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if artifact_path.name == "model.fp32.onnx" and self.manifest.get("sha256") != _sha256(artifact_path):
            raise RuntimeError("FP32 ONNX artifact checksum does not match manifest.")
        if os.getenv("AMR_VERIFY_CHECKPOINT_ON_STARTUP", "false").lower() == "true":
            checkpoint_info = inspect_checkpoint(download_checkpoint())
            if checkpoint_info.sha256 != self.manifest.get("checkpoint_sha256"):
                raise RuntimeError("Downloaded checkpoint checksum does not match ONNX artifact manifest.")
        self.session, self.runtime_selection = create_session(artifact_path)
        self.artifact_path = artifact_path


runtime = ServiceRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.load()
    LOGGER.info("model_session_loaded artifact=%s", runtime.artifact_path.name if runtime.artifact_path else "unknown")
    yield
    runtime.session = None


app = FastAPI(title="AMR Gated Fusion Former", version="gff-v3", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
def health() -> dict[str, object]:
    if runtime.session is None or runtime.artifact_path is None:
        raise HTTPException(status_code=503, detail="ONNX Runtime session is not loaded.")
    return {"status": "ok", "artifact": runtime.artifact_path.name, "runtime": "onnxruntime"}


@app.get("/metadata")
def metadata() -> dict[str, object]:
    if runtime.session is None or runtime.artifact_path is None:
        raise HTTPException(status_code=503, detail="ONNX Runtime session is not loaded.")
    return {
        "model_version": runtime.config["model_version"],
        "runtime": f"onnxruntime=={ort.__version__}",
        "runtime_requested": runtime.runtime_selection.requested if runtime.runtime_selection else "unknown",
        "runtime_selected": runtime.runtime_selection.selected if runtime.runtime_selection else "unknown",
        "provider": runtime.session.get_providers()[0],
        "fallback_providers": runtime.session.get_providers()[1:],
        "labels": runtime.config["labels"],
        "preprocessing_version": runtime.manifest["preprocessing_version"],
        "input_shapes": runtime.manifest["tensor_shapes"],
        "output_shapes": {
            "logits": runtime.manifest["tensor_shapes"]["logits"],
            "gating_weights": runtime.manifest["tensor_shapes"]["gating_weights"],
        },
        "artifact_checksum": _sha256(runtime.artifact_path),
    }


@app.post("/v1/classify", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest) -> ClassifyResponse:
    if runtime.session is None:
        raise HTTPException(status_code=503, detail="ONNX Runtime session is not loaded.")
    started = time.perf_counter()
    features = preprocess_iq(payload.iq, target_length=int(runtime.config["signal_length"]))
    preprocessing_latency_ms = (time.perf_counter() - started) * 1000.0
    inference_started = time.perf_counter()
    logits, gating_weights = runtime.session.run(
        ["logits", "gating_weights"],
        {
            "iq": features.iq[None, ...],
            "stft": features.stft[None, ...],
            "std": features.std[None, ...],
        },
    )
    inference_latency_ms = (time.perf_counter() - inference_started) * 1000.0
    probabilities = np.exp(logits[0] - logits[0].max())
    probabilities /= probabilities.sum()
    weights = gating_weights[0]
    if not np.isfinite(probabilities).all() or not np.isfinite(weights).all() or np.any(weights < 0):
        raise HTTPException(status_code=500, detail="Model returned invalid numerical output.")
    if not np.isclose(weights.sum(), 1.0, atol=1e-5):
        raise HTTPException(status_code=500, detail="Model returned invalid gating weights.")
    labels = runtime.config["labels"]
    class_id = int(np.argmax(probabilities))
    total_latency_ms = (time.perf_counter() - started) * 1000.0
    LOGGER.info(
        "classification_complete class_id=%d confidence=%.6f preprocessing_ms=%.3f inference_ms=%.3f total_ms=%.3f model_version=%s",
        class_id, probabilities[class_id], preprocessing_latency_ms, inference_latency_ms, total_latency_ms, runtime.config["model_version"],
    )
    return ClassifyResponse(
        prediction=labels[class_id],
        class_id=class_id,
        confidence=float(probabilities[class_id]),
        probabilities={label: float(value) for label, value in zip(labels, probabilities)},
        gating_weights={"iq": float(weights[0]), "stft": float(weights[1]), "std": float(weights[2])},
        preprocessing_latency_ms=preprocessing_latency_ms,
        inference_latency_ms=inference_latency_ms,
        total_latency_ms=total_latency_ms,
        model_version=runtime.config["model_version"],
    )