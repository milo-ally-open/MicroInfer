"""视觉塔根据图像高度、宽度坐标生成二维 Axial RoPE。"""

import torch
import torch.utils.benchmark
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

    components = [
        ("vision RoPE", reference, local, (hidden_states, position_ids), (hidden_states, position_ids), {}, {}, 1e-5, 1e-5),
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
