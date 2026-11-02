"""由 Norm、GQA Attention 和 SwiGLU MLP 组合成文字 Decoder Layer。"""

import torch
import torch.utils.benchmark
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextDecoderLayer as _ReferenceDecoderLayer,
)

from .text_attention import Qwen3VLTextAttention
from .text_mlp import Qwen3VLTextMLP
from .text_norm import Qwen3VLTextRMSNorm

class Qwen3VLTextDecoderLayer(nn.Module):
    """Pre-Norm Transformer block，子模块名与官方 checkpoint 键保持一致。"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3VLTextAttention(config, layer_idx)
        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        **kwargs,
    ) -> torch.Tensor:
        # 第一子层：归一化后计算自注意力，再与输入 hidden state 相加。
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        attention_output, _ = self.self_attn(
            hidden_states=normalized,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + attention_output

        # 第二子层：归一化后计算 SwiGLU MLP，再加上第二条残差。
        residual = hidden_states
        normalized = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(normalized)

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

    reference = _ReferenceDecoderLayer(config, layer_idx=0)
    local = Qwen3VLTextDecoderLayer(config, layer_idx=0)
    local.load_state_dict(reference.state_dict(), strict=True)
    local.to(device).eval()
    reference.to(device).eval()

    batch_size, sequence_length = 2, 5
    hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size, device=device)
    cos = torch.randn(batch_size, sequence_length, config.head_dim, device=device)
    sin = torch.randn_like(cos)

    components = [
        ("text decoder layer", reference, local, (hidden_states, (cos, sin)), (hidden_states, (cos, sin)), {}, {}, 1e-5, 1e-5),
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
