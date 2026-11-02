"""Mage-VL 可查看的标准多头注意力投影与矩阵计算。"""

from __future__ import annotations

import math

import torch
import torch.utils.benchmark
from torch import nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    """以 `nn.MultiheadAttention` 兼容的权重名称计算多头注意力。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        batch_first: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim < 1 or num_heads < 1 or embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError("dropout must be finite and between zero and one.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.batch_first:
            query, key, value = (item.transpose(0, 1) for item in (query, key, value))
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("Attention inputs must be rank-3 [batch, sequence, embedding] tensors.")
        if query.shape[0] != key.shape[0] or key.shape[:2] != value.shape[:2]:
            raise ValueError("Attention inputs must have matching batch and key/value sequence dimensions.")
        if query.shape[1] < 1 or key.shape[1] < 1:
            raise ValueError("Attention query and key sequences must not be empty.")
        if any(item.shape[-1] != self.embed_dim for item in (query, key, value)):
            raise ValueError(f"Attention input embedding dimensions must equal {self.embed_dim}.")
        batch_size, query_length, _ = query.shape
        key_length = key.shape[1]
        if key_padding_mask is not None and key_padding_mask.shape != (batch_size, key_length):
            raise ValueError("key_padding_mask must have shape [batch, key sequence].")
        if key_padding_mask is not None and key_padding_mask.dtype != torch.bool and not key_padding_mask.is_floating_point():
            raise ValueError("key_padding_mask must use boolean or floating-point values.")
        q_weight, k_weight, v_weight = self.in_proj_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = self.in_proj_bias.chunk(3, dim=0)
        q = F.linear(query, q_weight, q_bias)
        k = F.linear(key, k_weight, k_bias)
        v = F.linear(value, v_weight, v_bias)
        q = q.view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, key_length, self.num_heads, self.head_dim).transpose(1, 2)

        # QKᵀ 先按 head_dim 缩放；Softmax 以 FP32 累积降低半精度溢位风险。
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            mask = key_padding_mask.to(device=scores.device)
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask[:, None, None, :], torch.finfo(scores.dtype).min)
            else:
                scores = scores + mask[:, None, None, :].to(dtype=scores.dtype)
        probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        probabilities = F.dropout(probabilities, p=self.dropout, training=self.training)
        attended = torch.matmul(probabilities, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, query_length, self.embed_dim)
        attended = self.out_proj(attended)
        if not self.batch_first:
            attended = attended.transpose(0, 1)
        weights = probabilities.mean(dim=1) if need_weights else None
        return attended, weights

if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
    torch.manual_seed(7)
    reference = nn.MultiheadAttention(24, 4, batch_first=True).eval()
    implementation = MultiHeadAttention(24, 4).eval()
    implementation.load_state_dict(reference.state_dict(), strict=True)
    query = torch.randn(2, 3, 24)
    key = torch.randn(2, 7, 24)
    value = torch.randn(2, 7, 24)
    padding_mask = torch.tensor([[False, False, False, False, False, True, True], [False, False, False, False, False, False, True]])

    additive_mask = torch.zeros_like(padding_mask, dtype=query.dtype).masked_fill(padding_mask, -torch.inf)

    components = [
        (
            "attention with padding mask",
            reference,
            implementation,
            (query, key, value),
            (query, key, value),
            {"key_padding_mask": padding_mask, "need_weights": True},
            {"key_padding_mask": padding_mask, "need_weights": True},
            None,
            None,
            1e-6,
            1e-5,
            False,
        ),
        (
            "attention with additive mask",
            reference,
            implementation,
            (query, key, value),
            (query, key, value),
            {"key_padding_mask": additive_mask, "need_weights": False},
            {"key_padding_mask": additive_mask, "need_weights": False},
            None,
            None,
            1e-6,
            1e-5,
            False,
        ),
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
