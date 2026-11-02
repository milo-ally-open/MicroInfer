"""视觉 Patch Merger 与 DeepStack 视觉特征投影。"""

import time

import torch
from torch import nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionPatchMerger as _ReferenceMerger,
)


class Qwen3VLVisionPatchMerger(nn.Module):
    """归一化并合并空间 patch，再用 GELU MLP 投影到语言 hidden 维。"""

    def __init__(self, config, use_postshuffle_norm: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size * config.spatial_merge_size**2
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_size = self.hidden_size if use_postshuffle_norm else config.hidden_size
        self.norm = nn.LayerNorm(norm_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # DeepStack 在 shuffle 后归一化；主 Merger 在 shuffle 前归一化。
        if self.use_postshuffle_norm:
            hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = self.norm(hidden_states).view(-1, self.hidden_size)

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

    # 普通 Merger 和 DeepStack Merger 的 LayerNorm 输入位置不同，分别对齐。
    for use_postshuffle_norm in (False, True):
        reference = _ReferenceMerger(
            config,
            use_postshuffle_norm=use_postshuffle_norm,
        )
        local = Qwen3VLVisionPatchMerger(
            config,
            use_postshuffle_norm=use_postshuffle_norm,
        )
        local.load_state_dict(reference.state_dict(), strict=True)
        local.to(device).eval()
        reference.to(device).eval()

        input_width = local.hidden_size if use_postshuffle_norm else config.hidden_size
        hidden_states = torch.randn(8, input_width, device=device)
        with torch.inference_mode():
            expected = reference(hidden_states)
            actual = local(hidden_states)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

        # Bench：分别保留普通 Patch Merger 和 DeepStack Merger 的测量结果。
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
        merger_name = "DeepStack Merger" if use_postshuffle_norm else "Patch Merger"
        print(f"Bench: {merger_name} average={elapsed_ms:.4f} ms/call on {device}.")

        # Profile：单独采集当前 Merger 结构的线性层、GELU 与归一化算子。
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
        print(f"Profile: {merger_name} operators")
        print(profile.key_averages().table(sort_by=profile_sort, row_limit=12))

    print(f"Qwen3VLVisionPatchMerger parity passed on {device}.")
