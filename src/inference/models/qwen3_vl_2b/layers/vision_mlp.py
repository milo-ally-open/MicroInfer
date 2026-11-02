"""视觉 Transformer block 中的两层前馈 MLP。"""

import time

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionMLP as _ReferenceMLP,
)


class Qwen3VLVisionMLP(nn.Module):
    """Linear → 官方激活函数 → Linear。"""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return self.linear_fc2(hidden_states)

if __name__ == "__main__":
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    config = Qwen3VLVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=64,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "axial"},
    )
    reference = _ReferenceMLP(config)
    local = Qwen3VLVisionMLP(config)
    local.load_state_dict(reference.state_dict(), strict=True)
    local.to(device).eval()
    reference.to(device).eval()

    hidden_states = torch.randn(8, config.hidden_size, device=device)
    with torch.inference_mode():
        expected = reference(hidden_states)
        actual = local(hidden_states)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    print(f"Qwen3VLVisionMLP parity passed on {device}.")

    # Bench：固定 patch hidden shape，重复运行视觉前馈层。
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
    print(f"Bench: vision MLP average={elapsed_ms:.4f} ms/call on {device}.")

    # Profile：观察视觉 MLP 两个 Linear 和激活的算子分布。
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
    print("Profile: vision MLP operators")
    print(profile.key_averages().table(sort_by=profile_sort, row_limit=12))
