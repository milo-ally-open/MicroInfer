"""Qwen3-VL 文字三轴 MRoPE 频率计算。"""

import torch
import torch.utils.benchmark
from torch import nn
from transformers.modeling_rope_utils import (
    ROPE_INIT_FUNCTIONS,
    dynamic_rope_update,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextRotaryEmbedding as _ReferenceRoPE,
)

class Qwen3VLTextRotaryEmbedding(nn.Module):
    """按时间、高度、宽度位置计算频率，并按官方 MRoPE 规则重排。"""

    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_type = config.rope_parameters["rope_type"]

        rope_init = ROPE_INIT_FUNCTIONS.get(self.rope_type, self.compute_default_rope_parameters)
        inv_freq, self.attention_scaling = rope_init(config, device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.mrope_section = config.rope_parameters.get("mrope_section", [24, 20, 20])

    @staticmethod
    def compute_default_rope_parameters(config, device=None, **kwargs):
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None)
        if dim is None:
            dim = config.hidden_size // config.num_attention_heads

        exponents = torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim
        inv_freq = 1.0 / (base**exponents)
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 三组 position ids 分别对应时间、高度和宽度三个位置轴。
        position_ids = position_ids.expand(3, -1, -1)
        frequencies = position_ids[..., None].float() * self.inv_freq.to(
            device=hidden_states.device,
            dtype=torch.float,
        )
        cos = self.recomposition_frequencies(frequencies.cos() * self.attention_scaling)
        sin = self.recomposition_frequencies(frequencies.sin() * self.attention_scaling)
        return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)

    def recomposition_frequencies(self, frequencies: torch.Tensor) -> torch.Tensor:
        """将 T/H/W 三轴频率按 mrope_section 交织后复制为两半。"""
        combined = frequencies[0].clone()
        for axis, offset in enumerate((1, 2), start=1):
            indices = slice(offset, self.mrope_section[axis] * 3, 3)
            combined[..., indices] = frequencies[axis, ..., indices]
        return torch.cat((combined, combined), dim=-1)

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
    reference = _ReferenceRoPE(config).to(device)
    local = Qwen3VLTextRotaryEmbedding(config).to(device)
    hidden_states = torch.randn(2, 5, config.hidden_size, device=device)
    position_ids = torch.arange(5, device=device).view(1, 1, 5).expand(3, 2, 5)

    components = [
        ("text MRoPE", reference, local, (hidden_states, position_ids), (hidden_states, position_ids), {}, {}, 1e-5, 1e-5),
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
