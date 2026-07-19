"""Hardware information used by deployment runtime diagnostics."""

from __future__ import annotations

from typing import Any

import torch


def inspect_hardware() -> dict[str, Any]:
    """Return serializable PyTorch and CUDA availability details."""
    cuda_available = torch.cuda.is_available()
    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if cuda_available else []
    return {
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "gpu_names": gpu_names,
    }