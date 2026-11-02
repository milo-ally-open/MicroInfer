"""视觉时空 Patch 的 3D 卷积嵌入。"""

import time

import torch
from torch import nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionPatchEmbed as _ReferencePatchEmbed,
)


class Qwen3VLVisionPatchEmbed(nn.Module):
    """将 temporal×height×width patch 映射为视觉 hidden 向量。"""

    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ]
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 恢复展平 patch 的 Conv3D 输入布局，并匹配卷积权重 dtype。
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = hidden_states.to(dtype=self.proj.weight.dtype)
        return self.proj(hidden_states).view(-1, self.embed_dim)

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
    reference = _ReferencePatchEmbed(config)
    local = Qwen3VLVisionPatchEmbed(config)
    local.load_state_dict(reference.state_dict(), strict=True)
    local.to(device).eval()
    reference.to(device).eval()

    patches = torch.randn(6, 24, device=device)
    with torch.inference_mode():
        expected = reference(patches)
        actual = local(patches)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    print(f"Qwen3VLVisionPatchEmbed parity passed on {device}.")

    # Bench：固定 patch 展平布局，测量 Conv3D Patch Embedding。
    for _ in range(5):
        local(patches)
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_event.record()
        for _ in range(50):
            local(patches)
        end_event.record()
        end_event.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event) / 50
    else:
        started_at = time.perf_counter()
        for _ in range(50):
            local(patches)
        elapsed_ms = (time.perf_counter() - started_at) * 1000 / 50
    print(f"Bench: vision patch embedding average={elapsed_ms:.4f} ms/call on {device}.")

    # Profile：记录 Conv3D 和 reshape 的 shape、耗时及显存活动。
    profile_activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=profile_activities,
        record_shapes=True,
        profile_memory=True,
    ) as profile:
        local(patches)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        profile_sort = "self_cuda_time_total"
    else:
        profile_sort = "self_cpu_time_total"
    print("Profile: vision patch embedding operators")
    print(profile.key_averages().table(sort_by=profile_sort, row_limit=12))
