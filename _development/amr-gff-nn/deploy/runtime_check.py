"""Print the active deployment runtime and available acceleration backends."""

from __future__ import annotations

import platform
from pathlib import Path

import onnxruntime as ort

from deploy.hardware import inspect_hardware
from deploy.runtime import create_session, resolve_runtime


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "gff-v3"


def main() -> None:
    artifact_dir = Path(DEFAULT_ARTIFACT_DIR)
    artifact_path = artifact_dir / "model.int8.onnx"
    if not artifact_path.is_file():
        artifact_path = artifact_dir / "model.fp32.onnx"
    selection = resolve_runtime()
    hardware = inspect_hardware()
    print(f"python_version={platform.python_version()}")
    print(f"pytorch_version={hardware['pytorch_version']}")
    print(f"pytorch_cuda_version={hardware['pytorch_cuda_version']}")
    print(f"torch_cuda_available={hardware['cuda_available']}")
    print(f"gpu_count={hardware['gpu_count']}")
    print(f"gpu_names={hardware['gpu_names']}")
    print(f"onnxruntime_version={ort.__version__}")
    print(f"available_ort_providers={ort.get_available_providers()}")
    print(f"runtime_requested={selection.requested}")
    print(f"runtime_selected={selection.selected}")
    print(f"selected_model_artifact={artifact_path}")
    if artifact_path.is_file():
        session, _ = create_session(artifact_path)
        print(f"actual_session_providers={session.get_providers()}")
    else:
        print("actual_session_providers=artifact_missing")


if __name__ == "__main__":
    main()