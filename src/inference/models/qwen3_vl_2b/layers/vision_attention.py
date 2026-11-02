"""视觉塔变长 patch 序列的非因果 Attention。"""

import time

import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionAttention as _ReferenceAttention,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
    get_max_seqlen,
    is_flash_attention_requested,
)


class Qwen3VLVisionAttention(nn.Module):
    """合并 QKV 投影，并按 cu_seqlens 隔离不同图像的 patch 序列。"""

    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        max_seqlen: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        sequence_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(1, 0, 2, 3).unbind(0)

        # 先对 Q/K 施加二维视觉 RoPE，再整理成 backend 需要的布局。
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb_vision(query, key, cos, sin)
        query, key, value = (
            tensor.transpose(0, 1).unsqueeze(0)
            for tensor in (query, key, value)
        )

        attention = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        dropout = 0.0 if not self.training else self.attention_dropout

        if is_flash_attention_requested(self.config):
            # FlashAttention 直接消费边界数组，支持变长图像序列。
            max_seqlen = get_max_seqlen(
                cu_seqlens,
                self.config,
                kwargs={"max_seqlen": max_seqlen},
            )
            output, _ = attention(
                self,
                query,
                key,
                value,
                attention_mask=None,
                scaling=self.scaling,
                dropout=dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # 其他 backend 逐图运行，防止图像之间发生注意力信息混合。
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            query_splits, key_splits, value_splits = [
                torch.split(tensor, lengths, dim=2)
                for tensor in (query, key, value)
            ]
            outputs = [
                attention(
                    self,
                    query_part,
                    key_part,
                    value_part,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for query_part, key_part, value_part in zip(
                    query_splits,
                    key_splits,
                    value_splits,
                )
            ]
            output = torch.cat(outputs, dim=1)

        output = output.reshape(sequence_length, -1).contiguous()
        return self.proj(output)


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

    reference = _ReferenceAttention(config)
    local = Qwen3VLVisionAttention(config)
    local.load_state_dict(reference.state_dict(), strict=True)
    local.to(device).eval()
    reference.to(device).eval()

    sequence_length = 8
    hidden_states = torch.randn(sequence_length, config.hidden_size, device=device)
    cos = torch.ones(sequence_length, config.hidden_size // config.num_heads, device=device)
    sin = torch.zeros_like(cos)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)

    with torch.inference_mode():
        expected = reference(hidden_states, cu_seqlens, (cos, sin))
        actual = local(hidden_states, cu_seqlens, (cos, sin))

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    print(f"Qwen3VLVisionAttention parity passed on {device}.")

    # Bench：输入 patch 数和图像边界固定，统计视觉 Attention 延迟。
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
    print(f"Bench: vision attention average={elapsed_ms:.4f} ms/call on {device}.")

    # Profile：查看视觉 QKV 投影、分段 Attention 和输出投影的成本。
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
    print("Profile: vision attention operators")
    print(profile.key_averages().table(sort_by=profile_sort, row_limit=15))
