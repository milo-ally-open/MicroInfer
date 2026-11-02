"""视觉 Transformer 使用的 LayerNorm。"""

import time

import torch
import torch.nn.functional as F
from torch import nn


class Qwen3VLVisionLayerNorm(nn.Module):
    """沿 patch hidden 维执行仿射 LayerNorm，权重键与 nn.LayerNorm 一致。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.normalized_shape = (hidden_size,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(hidden_states, self.normalized_shape, self.weight, self.bias, self.eps)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    reference = nn.LayerNorm(64, eps=1e-6).to(device=device, dtype=dtype)
    local = Qwen3VLVisionLayerNorm(64, eps=1e-6).to(device=device, dtype=dtype)
    local.load_state_dict(reference.state_dict(), strict=True)

    hidden_states = torch.randn(2, 7, 64, device=device, dtype=dtype)
    with torch.inference_mode():
        expected = reference(hidden_states)
        actual = local(hidden_states)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    print(f"Qwen3VLVisionLayerNorm parity passed on {device}.")

    # Bench：测量 BF16（CUDA）或 FP32（CPU）LayerNorm 前向。
    for _ in range(5):
        local(hidden_states)
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_event.record()
        for _ in range(50):
            local(hidden_states)
        end_event.record()
        end_event.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event) / 50
    else:
        started_at = time.perf_counter()
        for _ in range(50):
            local(hidden_states)
        elapsed_ms = (time.perf_counter() - started_at) * 1000 / 50
    print(f"Bench: vision LayerNorm average={elapsed_ms:.4f} ms/call on {device}.")

    # Profile：统计 LayerNorm 内部 reduction 与逐元素算子。
    profile_activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=profile_activities,
        record_shapes=True,
        profile_memory=True,
    ) as profile:
        local(hidden_states)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        profile_sort = "self_cuda_time_total"
    else:
        profile_sort = "self_cpu_time_total"
    print("Profile: vision LayerNorm operators")
    print(profile.key_averages().table(sort_by=profile_sort, row_limit=10))
