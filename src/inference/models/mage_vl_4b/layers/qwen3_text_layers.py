"""Mage-VL 文字骨干使用的 Qwen3 正规化与 MLP 层。"""

from __future__ import annotations

import torch
import torch.utils.benchmark
from torch import nn
from torch.nn import functional as F

class Qwen3RMSNorm(nn.Module):
    """使用 FP32 累加的 Qwen3 RMSNorm。"""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return self.weight * values.to(input_dtype)

class Qwen3MLP(nn.Module):
    """以独立投影实现 Qwen3 的 gated MLP。"""

    def __init__(self, config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3MLP as TransformersQwen3MLP,
        Qwen3RMSNorm as TransformersQwen3RMSNorm,
    )

    torch.manual_seed(37)
    config = Qwen3Config(hidden_size=32, intermediate_size=64, num_attention_heads=4)
    baseline_norm = TransformersQwen3RMSNorm(config.hidden_size, config.rms_norm_eps).eval()
    engine_norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps).eval()
    engine_norm.load_state_dict(baseline_norm.state_dict(), strict=True)
    baseline_mlp = TransformersQwen3MLP(config).eval()
    engine_mlp = Qwen3MLP(config).eval()
    engine_mlp.load_state_dict(baseline_mlp.state_dict(), strict=True)
    hidden_states = torch.randn(2, 5, config.hidden_size)

    components = [
        ("Qwen3 RMSNorm", baseline_norm, engine_norm, (hidden_states,), (hidden_states,), {}, {}, None, None, 1e-6, 1e-6, False),
        ("Qwen3 MLP", baseline_mlp, engine_mlp, (hidden_states,), (hidden_states,), {}, {}, None, None, 1e-6, 1e-6, False),
    ]
    profile_activities = [torch.profiler.ProfilerActivity.CPU]
    if test_device.type == "cuda":
        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
    profile_sort = "self_cuda_time_total" if test_device.type == "cuda" else "self_cpu_time_total"

    for (
        component_name,
        baseline,
        enhanced,
        baseline_args,
        enhanced_args,
        baseline_kwargs,
        enhanced_kwargs,
        baseline_selector,
        enhanced_selector,
        atol,
        rtol,
        compare_hidden_states,
    ) in components:
        baseline_output = None
        baseline_raw_output = None
        baseline_median = None
        for stage, module, args, kwargs, selector in (
            ("baseline", baseline, baseline_args, baseline_kwargs, baseline_selector),
            ("enhanced", enhanced, enhanced_args, enhanced_kwargs, enhanced_selector),
        ):
            print(f"================= {stage} =================")
            print("   test:")
            print(f"       component={component_name}")
            with torch.inference_mode():
                raw_output = module(*args, **kwargs)
            if isinstance(selector, str):
                output = raw_output[selector] if isinstance(raw_output, dict) else getattr(raw_output, selector)
            elif selector is not None:
                output = raw_output[selector]
            else:
                output = raw_output
            if stage == "baseline":
                torch.testing.assert_close(output, output, atol=atol, rtol=rtol)
                baseline_output, baseline_raw_output = output, raw_output
                output_tensor = output if isinstance(output, torch.Tensor) else output[0]
                print(f"       PASS, shape={tuple(output_tensor.shape)}, dtype={output_tensor.dtype}, device={output_tensor.device}")
            else:
                torch.testing.assert_close(output, baseline_output, atol=atol, rtol=rtol)
                if compare_hidden_states:
                    torch.testing.assert_close(
                        tuple(raw_output.hidden_states), tuple(baseline_raw_output.hidden_states), atol=atol, rtol=rtol
                    )
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
            if test_device.type == "cuda":
                torch.cuda.synchronize(test_device)
            print(profile.key_averages().table(sort_by=profile_sort, row_limit=12))
