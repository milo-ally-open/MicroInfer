"""视觉 Transformer Block：LayerNorm、分段 Attention、MLP 和残差。"""

import torch
import torch.utils.benchmark
from torch import nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionBlock as _ReferenceBlock,
)

from .vision_attention import Qwen3VLVisionAttention
from .vision_mlp import Qwen3VLVisionMLP
from .vision_norm import Qwen3VLVisionLayerNorm

class Qwen3VLVisionBlock(nn.Module):
    """视觉残差块，保留官方子模块命名与前向计算顺序。"""

    def __init__(self, config, attn_implementation: str = "sdpa"):
        super().__init__()
        self.norm1 = Qwen3VLVisionLayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config)
        self.norm2 = Qwen3VLVisionLayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = Qwen3VLVisionMLP(config)
        self.attn_implementation = attn_implementation

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # cu_seqlens 标记每张图片的 patch 边界，Attention 不跨图像混合信息。
        attention_input = self.norm1(hidden_states)
        attention_output = self.attn(
            attention_input,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + attention_output

        # 视觉前馈层使用独立的归一化和残差路径。
        mlp_input = self.norm2(hidden_states)
        return hidden_states + self.mlp(mlp_input)

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
    config._attn_implementation = "sdpa"

    reference = _ReferenceBlock(config)
    local = Qwen3VLVisionBlock(config)
    local.load_state_dict(reference.state_dict(), strict=True)
    local.to(device).eval()
    reference.to(device).eval()

    hidden_states = torch.randn(8, config.hidden_size, device=device)
    cos = torch.ones(8, config.hidden_size // config.num_heads, device=device)
    sin = torch.zeros_like(cos)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)

    components = [
        (
            "vision block",
            reference,
            local,
            (hidden_states, cu_seqlens, (cos, sin)),
            (hidden_states, cu_seqlens, (cos, sin)),
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
