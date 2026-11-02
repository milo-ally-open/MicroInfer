"""Qwen3-VL 文字骨干的门控 SwiGLU MLP。"""

import time

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextMLP as _ReferenceMLP,
)


class Qwen3VLTextMLP(nn.Module):
    """gate 和 up 两路投影逐元素相乘，再由 down 投影回 hidden 维。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(hidden_states))
        value = self.up_proj(hidden_states)
        return self.down_proj(gate * value)


if __name__ == "__main__":
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    config = Qwen3VLTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_parameters={
            "rope_theta": 500000.0,
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
        },
    )
    reference = _ReferenceMLP(config)
    local = Qwen3VLTextMLP(config)
    local.load_state_dict(reference.state_dict(), strict=True)
    local.to(device).eval()
    reference.to(device).eval()

    hidden_states = torch.randn(2, 5, config.hidden_size, device=device)
    with torch.inference_mode():
        expected = reference(hidden_states)
        actual = local(hidden_states)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    print(f"Qwen3VLTextMLP parity passed on {device}.")

    # Bench：用同一 hidden state 多次执行 SwiGLU MLP 前向。
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
    print(f"Bench: text MLP average={elapsed_ms:.4f} ms/call on {device}.")

    # Profile：查看两次线性投影、激活及其临时张量的成本。
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
    print("Profile: text MLP operators")
    print(profile.key_averages().table(sort_by=profile_sort, row_limit=12))
