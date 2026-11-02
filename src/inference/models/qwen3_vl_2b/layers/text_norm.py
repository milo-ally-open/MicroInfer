"""Qwen3-VL 文字层 RMSNorm。"""

import time

import torch
from torch import nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextRMSNorm as _ReferenceRMSNorm,
)


class Qwen3VLTextRMSNorm(nn.Module):
    """在 hidden 维计算均方根，使用 fp32 累加后乘可训练缩放权重。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.to(torch.float32)
        variance = values.pow(2).mean(-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    reference = _ReferenceRMSNorm(64)
    local = Qwen3VLTextRMSNorm(64)
    local.load_state_dict(reference.state_dict(), strict=True)
    local.to(device).eval()
    reference.to(device).eval()

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    hidden_states = torch.randn(2, 7, 64, device=device, dtype=dtype)
    with torch.inference_mode():
        expected = reference(hidden_states)
        actual = local(hidden_states)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    print(f"Qwen3VLTextRMSNorm parity passed on {device}.")

    # Bench：保持输入 shape 与 dtype 固定，测量 RMSNorm 平均延迟。
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
    print(f"Bench: text RMSNorm average={elapsed_ms:.4f} ms/call on {device}.")

    # Profile：采集归一化的算子时间、输入 shape 和显存使用。
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
    print("Profile: text RMSNorm operators")
    print(profile.key_averages().table(sort_by=profile_sort, row_limit=10))
