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
    with torch.inference_mode():
        engine_output = (engine_norm(hidden_states), engine_mlp(hidden_states))
        baseline_output = (baseline_norm(hidden_states), baseline_mlp(hidden_states))
    print("==========engine===========")
    print(f"RMSNorm={engine_output[0][0, 0, :4].tolist()}")
    print(f"MLP={engine_output[1][0, 0, :4].tolist()}")
    print("==========baseline==========")
    print(f"RMSNorm={baseline_output[0][0, 0, :4].tolist()}")
    print(f"MLP={baseline_output[1][0, 0, :4].tolist()}")
    torch.testing.assert_close(engine_output[0], baseline_output[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(engine_output[1], baseline_output[1], atol=1e-6, rtol=1e-6)
    print(f"Mage-VL Qwen3 RMSNorm and MLP match the Transformers baseline on {test_device}.")

    # Bench：独立测量 RMSNorm 和 Qwen3 MLP，便于判断逐层瓶颈。
    with torch.inference_mode():
        norm_bench = torch.utils.benchmark.Timer(
            stmt="engine_norm(hidden_states)", globals=locals()
        ).blocked_autorange(min_run_time=0.3)
        mlp_bench = torch.utils.benchmark.Timer(
            stmt="engine_mlp(hidden_states)", globals=locals()
        ).blocked_autorange(min_run_time=0.3)
    print(f"Bench: Qwen3RMSNorm {norm_bench}")
    print(f"Bench: Qwen3MLP {mlp_bench}")

    # Profile：分别记录 Norm 与 MLP 算子，不把两种成本混在一起。
    profile_activities = [torch.profiler.ProfilerActivity.CPU]
    if test_device.type == "cuda":
        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.inference_mode():
        with torch.profiler.profile(
            activities=profile_activities,
            record_shapes=True,
            profile_memory=True,
        ) as norm_profile:
            engine_norm(hidden_states)
        with torch.profiler.profile(
            activities=profile_activities,
            record_shapes=True,
            profile_memory=True,
        ) as mlp_profile:
            engine_mlp(hidden_states)
    if test_device.type == "cuda":
        torch.cuda.synchronize(test_device)
        profile_sort = "self_cuda_time_total"
    else:
        profile_sort = "self_cpu_time_total"
    print("Profile: Qwen3RMSNorm operators")
    print(norm_profile.key_averages().table(sort_by=profile_sort, row_limit=10))
    print("Profile: Qwen3MLP operators")
    print(mlp_profile.key_averages().table(sort_by=profile_sort, row_limit=12))
