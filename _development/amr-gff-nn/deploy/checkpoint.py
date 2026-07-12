"""Secure checkpoint retrieval and strict compatibility checks for GFF v3."""

from __future__ import annotations

import argparse
import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_FILENAME = "gatedfusionformer_v4.0_best_20251025_211831.pth"
_STATE_DICT_KEYS = (
    "state_dict",
    "model_state_dict",
    "model",
    "net",
    "network",
)


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be securely retrieved or strictly loaded."""


@dataclass(frozen=True)
class CheckpointInfo:
    """Non-secret metadata about a downloaded and inspected checkpoint."""

    repository_id: str
    filename: str
    local_path: Path
    sha256: str
    checkpoint_format: str
    state_dict_key: str | None


def load_deployment_environment() -> None:
    """Load local deployment variables without overriding process-level secrets."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _required_environment(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise CheckpointError(
            f"Required environment variable {name} is not configured. "
            f"Set it in {PROJECT_ROOT / '.env'} or the process environment."
        )
    return value


def checkpoint_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 checksum without loading the checkpoint into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(
    filename: str | None = None,
    repository_id: str | None = None,
    revision: str | None = None,
) -> Path:
    """Download the configured private checkpoint to the Hugging Face cache."""
    load_deployment_environment()
    resolved_filename = filename or os.getenv(
        "HF_CHECKPOINT_FILENAME", DEFAULT_CHECKPOINT_FILENAME
    )
    resolved_repository_id = repository_id or _required_environment("HF_REPO_ID")
    token = _required_environment("HF_TOKEN")
    try:
        return Path(
            hf_hub_download(
                repo_id=resolved_repository_id,
                filename=resolved_filename,
                token=token,
                revision=revision or os.getenv("HF_REVISION"),
            )
        )
    except Exception as error:
        raise CheckpointError(
            f"Unable to download {resolved_filename!r} from "
            f"Hugging Face repository {resolved_repository_id!r}."
        ) from error


def _is_raw_state_dict(payload: Mapping[str, Any]) -> bool:
    return bool(payload) and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    )


def extract_state_dict(payload: Any) -> tuple[dict[str, torch.Tensor], str, str | None]:
    """Extract a raw or common wrapped state dict while rejecting unknown payloads."""
    if not isinstance(payload, Mapping):
        raise CheckpointError(
            "Checkpoint payload is not a mapping and cannot provide a state_dict."
        )
    if _is_raw_state_dict(payload):
        return dict(payload), "raw_state_dict", None

    for key in _STATE_DICT_KEYS:
        candidate = payload.get(key)
        if isinstance(candidate, Mapping) and _is_raw_state_dict(candidate):
            return dict(candidate), f"wrapped:{key}", key

    raise CheckpointError(
        "Checkpoint does not contain a supported raw state_dict or wrapped key "
        f"({', '.join(_STATE_DICT_KEYS)})."
    )


def _normalize_data_parallel_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def inspect_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> CheckpointInfo:
    """Inspect the checkpoint with PyTorch's restricted weights-only loader."""
    if not path.is_file():
        raise CheckpointError(f"Checkpoint file does not exist: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as error:
        raise CheckpointError(
            "Unable to load checkpoint using PyTorch weights_only=True. "
            "Only trusted tensor state dictionaries are supported."
        ) from error

    _, checkpoint_format, state_dict_key = extract_state_dict(payload)
    load_deployment_environment()
    return CheckpointInfo(
        repository_id=_required_environment("HF_REPO_ID"),
        filename=path.name,
        local_path=path.resolve(),
        sha256=checkpoint_sha256(path),
        checkpoint_format=checkpoint_format,
        state_dict_key=state_dict_key,
    )


def load_checkpoint_strict(
    model: torch.nn.Module,
    path: Path,
    map_location: str | torch.device = "cpu",
) -> CheckpointInfo:
    """Strictly load a trusted checkpoint and report architecture incompatibility."""
    if not path.is_file():
        raise CheckpointError(f"Checkpoint file does not exist: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
        state_dict, _, _ = extract_state_dict(payload)
        model.load_state_dict(_normalize_data_parallel_keys(state_dict), strict=True)
    except CheckpointError:
        raise
    except RuntimeError as error:
        raise CheckpointError(
            "Checkpoint is incompatible with the selected canonical model. "
            "The model architecture was not modified."
        ) from error
    except Exception as error:
        raise CheckpointError("Unable to load checkpoint state_dict strictly.") from error
    return inspect_checkpoint(path, map_location=map_location)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and inspect GFF v3 checkpoint")
    parser.add_argument("--filename", default=None, help="Override HF checkpoint filename")
    parser.add_argument("--revision", default=None, help="Optional HF repository revision")
    args = parser.parse_args()

    path = download_checkpoint(filename=args.filename, revision=args.revision)
    info = inspect_checkpoint(path)
    print(f"repository_id={info.repository_id}")
    print(f"filename={info.filename}")
    print(f"local_path={info.local_path}")
    print(f"sha256={info.sha256}")
    print(f"checkpoint_format={info.checkpoint_format}")


if __name__ == "__main__":
    main()