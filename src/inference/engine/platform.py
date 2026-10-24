"""抽象推论 Worker 使用的加速器与装置观测能力。"""

from __future__ import annotations

from typing import Any, Protocol


class Platform(Protocol):
    """Model Runner 使用的装置、记忆体与资源释放介面。"""

    name: str

    def resolve_device(self, requested: str) -> Any: ...

    def gpu_status(self, device: Any) -> dict[str, Any]: ...

    def peak_memory(self, device: Any) -> dict[str, int]: ...

    def reset_peak_memory(self, device: Any) -> None: ...

    def release_cached_memory(self) -> None: ...


class TorchCudaPlatform:
    """本机 CUDA 平台；其他装置后端可实作相同契约。"""

    name = "torch_cuda"

    def resolve_device(self, requested: str) -> Any:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; this model deployment requires a CUDA device.")
        device = torch.device(requested)
        if device.type != "cuda":
            raise RuntimeError(f"Configured device must use CUDA, received: {device}.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"Configured CUDA device is unavailable: {device}.")
        return device

    def gpu_status(self, device: Any) -> dict[str, Any]:
        import torch

        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return {
            "gpu": torch.cuda.get_device_name(device),
            "vram_free_bytes": int(free_bytes),
            "vram_total_bytes": int(total_bytes),
        }

    def peak_memory(self, device: Any) -> dict[str, int]:
        import torch

        return {
            "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
        }

    def reset_peak_memory(self, device: Any) -> None:
        import torch

        torch.cuda.reset_peak_memory_stats(device)

    def release_cached_memory(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_platform(name: str) -> Platform:
    """依模型 YAML 的平台识别码建立已登记平台。"""
    if name == TorchCudaPlatform.name:
        return TorchCudaPlatform()
    raise ValueError(f"Unsupported inference platform: {name}.")
