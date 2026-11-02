"""Qwen3-VL 文字 GQA Attention：投影、QK-Norm、MRoPE、Cache 与注意力核。"""

from __future__ import annotations

import torch
import torch.utils.benchmark
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextAttention as _ReferenceAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from .text_norm import Qwen3VLTextRMSNorm

class Qwen3VLTextAttention(nn.Module):
    """实现官方文字 Attention 的前向计算，并保持其权重布局。"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_type = (
            config.layer_types[layer_idx]
            if hasattr(config, "layer_types")
            else None
        )
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Q 有 query heads，K/V 只有 key-value heads，支持 GQA 参数布局。
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # 将投影结果拆成 heads，并在每个 head 的通道维上归一化 Q/K。
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # 使用上层按文本/图像位置生成的 MRoPE，并将新 K/V 写入请求 Cache。
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
            )

        # 选择配置指定的 Transformers backend，随后合并 heads 并做输出投影。
        attention = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        dropout = 0.0 if not self.training else self.attention_dropout
        output, weights = attention(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=dropout,
            scaling=self.scaling,
            **kwargs,
        )
        output = output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(output), weights

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
    config._attn_implementation = "sdpa"

    reference = _ReferenceAttention(config, layer_idx=0)
    local = Qwen3VLTextAttention(config, layer_idx=0)
    local.load_state_dict(reference.state_dict(), strict=True)
    local.to(device).eval()
    reference.to(device).eval()

    batch_size, sequence_length = 2, 5
    hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size, device=device)
    cos = torch.randn(batch_size, sequence_length, config.head_dim, device=device)
    sin = torch.randn_like(cos)
    attention_mask = torch.triu(
        torch.full(
            (batch_size, 1, sequence_length, sequence_length),
            torch.finfo(hidden_states.dtype).min,
            device=device,
        ),
        diagonal=1,
    )

    components = [
        (
            "text attention",
            reference,
            local,
            (hidden_states, (cos, sin), attention_mask),
            (hidden_states, (cos, sin), attention_mask),
            {},
            {},
            1e-5,
            1e-5,
        ),
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
