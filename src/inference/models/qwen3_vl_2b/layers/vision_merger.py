"""视觉 Patch Merger 与 DeepStack 视觉特征投影。"""

import torch
import torch.utils.benchmark
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
        reference = _ReferenceMerger(config, use_postshuffle_norm=use_postshuffle_norm)
        local = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=use_postshuffle_norm)
        local.load_state_dict(reference.state_dict(), strict=True)
        local.to(device).eval()
        reference.to(device).eval()

        input_width = local.hidden_size if use_postshuffle_norm else config.hidden_size
        hidden_states = torch.randn(8, input_width, device=device)
        merger_name = "DeepStack Merger" if use_postshuffle_norm else "Patch Merger"

        components = [
            (merger_name, reference, local, (hidden_states,), (hidden_states,), {}, {}, 1e-5, 1e-5),
        ]

        profile_activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
        profile_sort = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"

        for component_name, baseline, enhanced, baseline_args, enhanced_args, baseline_kwargs, enhanced_kwargs, atol, rtol in components:
            baseline_output = None
            baseline_median = None
            for stage, module, args, kwargs in (
                ("baseline", baseline, baseline_args, baseline_kwargs),
                ("enhanced", enhanced, enhanced_args, enhanced_kwargs),
            ):
                print(f"================= {stage} =================")
                print("   test:")
                print(f"       component={component_name}")
                with torch.inference_mode():
                    output = module(*args, **kwargs)
                if stage == "baseline":
                    torch.testing.assert_close(output, output, atol=atol, rtol=rtol)
                    baseline_output = output
                    output_tensor = output if isinstance(output, torch.Tensor) else output[0]
                    print(f"       PASS, shape={tuple(output_tensor.shape)}, dtype={output_tensor.dtype}, device={output_tensor.device}")
                else:
                    torch.testing.assert_close(output, baseline_output, atol=atol, rtol=rtol)
                    if isinstance(output, torch.Tensor):
                        max_abs_error = (output - baseline_output).abs().max().item()
                        output_tensor = output
                    else:
                        max_abs_error = max(
                            (
                                (actual - expected).abs().max().item()
                                for actual, expected in zip(output, baseline_output)
                                if isinstance(actual, torch.Tensor)
                            ),
                            default=0.0,
                        )
                        output_tensor = output[0]
                    print(f"       parity=PASS, max_abs_error={max_abs_error:.3e}, shape={tuple(output_tensor.shape)}")
                print("   bench:")
                with torch.inference_mode():
                    bench = torch.utils.benchmark.Timer(
                        stmt="module(*args, **kwargs)",
                        globals={"module": module, "args": args, "kwargs": kwargs},
                    ).blocked_autorange(min_run_time=0.2)
                if stage == "baseline":
                    baseline_median = bench.median
                    print(f"       median_ms={bench.median * 1000:.4f}, iqr_ms={bench.iqr * 1000:.4f}")
                else:
                    print(
                        f"       median_ms={bench.median * 1000:.4f}, iqr_ms={bench.iqr * 1000:.4f}, speedup={baseline_median / bench.median:.3f}x"
                    )
                print("   profile:")
                with torch.inference_mode():
                    with torch.profiler.profile(
                        activities=profile_activities,
                        record_shapes=True,
                        profile_memory=True,
                    ) as profile:
                        module(*args, **kwargs)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                print(profile.key_averages().table(sort_by=profile_sort, row_limit=12))
