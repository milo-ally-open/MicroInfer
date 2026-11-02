"""视觉塔根据图像高度、宽度坐标生成二维 Axial RoPE。"""

import time

import torch
from torch import nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionRotaryEmbedding as _ReferenceRoPE,
)


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """分别计算 H/W 两轴频率，拼接后复制为旋转向量的两半。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rope_type = config.rope_parameters["rope_type"]
        if self.rope_type != "axial":
            raise ValueError(f'Qwen3VL 视觉 RoPE 只支持 axial，收到 {self.rope_type}')

        inv_freq, self.attention_scaling = self.compute_axial_rope_parameters(config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_axial_rope_parameters(config, device=None, **kwargs):
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None)
        if dim is None:
            dim = config.hidden_size // config.num_heads
        spatial_dim = dim // 2

        exponents = (
            torch.arange(0, spatial_dim, 2, dtype=torch.float, device=device)
            / spatial_dim
        )
        inv_freq = 1.0 / (base**exponents)
        return inv_freq.to(device), 1.0

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids 第一行是 H 坐标，第二行是 W 坐标。
        coordinates = position_ids[..., None].float()
        frequencies = coordinates * self.inv_freq.to(device=hidden_states.device, dtype=torch.float)
        cos = self.recomposition_frequencies(frequencies.cos() * self.attention_scaling)
        sin = self.recomposition_frequencies(frequencies.sin() * self.attention_scaling)
        return cos, sin

    def recomposition_frequencies(self, frequencies: torch.Tensor) -> torch.Tensor:
        height_freq, width_freq = frequencies[:, 0], frequencies[:, 1]
        half = torch.cat((height_freq, width_freq), dim=-1)
        return torch.cat((half, half), dim=-1)


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
    reference = _ReferenceRoPE(config).to(device)
    local = Qwen3VLVisionRotaryEmbedding(config).to(device)

    hidden_states = torch.zeros(6, config.hidden_size, device=device)
    positions = torch.arange(6, device=device)
    position_ids = torch.stack((positions % 3, positions // 3))
    with torch.inference_mode():
        expected = reference(hidden_states, position_ids)
        actual = local(hidden_states, position_ids)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    print(f"Qwen3VLVisionRotaryEmbedding parity passed on {device}.")

    # Bench：对固定数量的视觉 patch 坐标计算 Axial RoPE。
    for _ in range(5):
        local(hidden_states, position_ids)
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_event.record()
        for _ in range(50):
            local(hidden_states, position_ids)
        end_event.record()
        end_event.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event) / 50
    else:
        started_at = time.perf_counter()
        for _ in range(50):
            local(hidden_states, position_ids)
        elapsed_ms = (time.perf_counter() - started_at) * 1000 / 50
    print(f"Bench: vision RoPE average={elapsed_ms:.4f} ms/call on {device}.")

    # Profile：查看 Axial 频率计算、sin/cos 与坐标重排的执行成本。
    profile_activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=profile_activities,
        record_shapes=True,
        profile_memory=True,
    ) as profile:
        local(hidden_states, position_ids)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        profile_sort = "self_cuda_time_total"
    else:
        profile_sort = "self_cpu_time_total"
    print("Profile: vision RoPE operators")
    print(profile.key_averages().table(sort_by=profile_sort, row_limit=12))
