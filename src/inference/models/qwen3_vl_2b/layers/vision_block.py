"""视觉 Transformer Block：LayerNorm、分段 Attention、MLP 和残差。"""

import time

import torch
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

    with torch.inference_mode():
        expected = reference(hidden_states, cu_seqlens, (cos, sin))
        actual = local(hidden_states, cu_seqlens, (cos, sin))

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    print(f"Qwen3VLVisionBlock parity passed on {device}.")

    # Bench：对完整视觉残差块做预热和重复计时。
    for _ in range(5):
        local(hidden_states, cu_seqlens, (cos, sin))
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_event.record()
        for _ in range(50):
            local(hidden_states, cu_seqlens, (cos, sin))
        end_event.record()
        end_event.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event) / 50
    else:
        started_at = time.perf_counter()
        for _ in range(50):
            local(hidden_states, cu_seqlens, (cos, sin))
        elapsed_ms = (time.perf_counter() - started_at) * 1000 / 50
    print(f"Bench: vision block average={elapsed_ms:.4f} ms/call on {device}.")

    # Profile：分解视觉块内 Norm、Attention、MLP 与残差算子。
    profile_activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=profile_activities,
        record_shapes=True,
        profile_memory=True,
    ) as profile:
        local(hidden_states, cu_seqlens, (cos, sin))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        profile_sort = "self_cuda_time_total"
    else:
        profile_sort = "self_cpu_time_total"
    print("Profile: vision block operators")
    print(profile.key_averages().table(sort_by=profile_sort, row_limit=15))
