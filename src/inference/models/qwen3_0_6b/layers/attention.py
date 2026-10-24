"""Qwen3 GQA、Q/K RMSNorm、RoPE 与 paged KV 写读。"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from ..components.paged_kv_cache import PagedKVCache
from .layers import Qwen3RMSNorm
from .rotary_embedding import apply_rotary, rotary_embeddings


class Qwen3Attention(nn.Module):
    """依 request block table 执行 Qwen3 grouped-query self-attention。"""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        rope_theta: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if min(hidden_size, num_attention_heads, num_key_value_heads, head_dim) < 1:
            raise ValueError("Qwen3 attention dimensions must be positive.")
        if num_attention_heads % num_key_value_heads:
            raise ValueError("Qwen3 query heads must be divisible by key/value heads.")
        if head_dim % 2 or not math.isfinite(rope_theta) or rope_theta <= 0:
            raise ValueError("Qwen3 RoPE requires a positive theta and an even head dimension.")
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rope_theta = float(rope_theta)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False, device=device, dtype=dtype)
        self.q_norm = Qwen3RMSNorm(head_dim, rms_norm_eps, device=device, dtype=dtype)
        self.k_norm = Qwen3RMSNorm(head_dim, rms_norm_eps, device=device, dtype=dtype)

    def forward_batch(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_index: int,
        request_ids: Sequence[str],
        token_lengths: Sequence[int],
        position_offsets: Sequence[int],
        block_tables: Sequence[Sequence[int]],
        cache: PagedKVCache,
    ) -> torch.Tensor:
        batch_size, query_width, _ = hidden_states.shape
        if query_width < 1:
            raise ValueError("Qwen3 attention requires at least one query token.")
        if hidden_states.shape[-1] != self.q_proj.in_features:
            raise ValueError("Qwen3 attention hidden size does not match its projection weights.")
        query = self.q_proj(hidden_states).view(
            batch_size, query_width, self.num_attention_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch_size, query_width, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch_size, query_width, self.num_key_value_heads, self.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)

        position_ids = torch.zeros((batch_size, query_width), device=hidden_states.device, dtype=torch.long)
        for index, (offset, length) in enumerate(zip(position_offsets, token_lengths, strict=True)):
            position_ids[index, :length] = torch.arange(
                offset, offset + length, device=hidden_states.device, dtype=torch.long
            )
        cosine, sine = rotary_embeddings(
            position_ids,
            head_dim=self.head_dim,
            theta=self.rope_theta,
            dtype=hidden_states.dtype,
        )
        query, key = apply_rotary(query, key, cosine, sine)

        key_lengths: list[int] = []
        for index, (request_id, offset, length, block_table) in enumerate(
            zip(request_ids, position_offsets, token_lengths, block_tables, strict=True)
        ):
            key_lengths.append(
                cache.write(
                    request_id=request_id,
                    layer_index=layer_index,
                    block_table=block_table,
                    position_offset=int(offset),
                    key=key[index, :, :length, :],
                    value=value[index, :, :length, :],
                )
            )

        key_states, value_states = cache.read_batch(
            layer_index=layer_index,
            request_ids=request_ids,
            block_tables=block_tables,
            lengths=key_lengths,
        )
        key_positions = torch.arange(key_states.shape[2], device=hidden_states.device)
        query_positions = position_ids
        query_lengths_tensor = torch.tensor(token_lengths, device=hidden_states.device).unsqueeze(1)
        key_lengths_tensor = torch.tensor(key_lengths, device=hidden_states.device).unsqueeze(1)
        query_valid = torch.arange(query_width, device=hidden_states.device).unsqueeze(0) < query_lengths_tensor
        key_valid = key_positions.unsqueeze(0) < key_lengths_tensor
        causal = key_positions.view(1, 1, -1) <= query_positions.unsqueeze(-1)
        attention_mask = (
            query_valid.unsqueeze(-1) & key_valid.unsqueeze(1) & causal
        ).unsqueeze(1)
        # 以分组 einsum 直接计算 GQA，避免将 K/V 实体复制成 query head 数量。
        groups = self.num_attention_heads // self.num_key_value_heads
        grouped_query = query.reshape(
            batch_size, self.num_key_value_heads, groups, query_width, self.head_dim
        )
        scores = torch.einsum("bhgqd,bhkd->bhgqk", grouped_query, key_states)
        scores = scores * (self.head_dim ** -0.5)
        scores = scores.masked_fill(~attention_mask.unsqueeze(2), torch.finfo(scores.dtype).min)

        # softmax 使用 FP32 累加；padding query 的全遮罩列转为零，避免产生 NaN。
        probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        probabilities = torch.nan_to_num(probabilities, nan=0.0)
        grouped_output = torch.einsum("bhgqk,bhkd->bhgqd", probabilities, value_states)
        attended = grouped_output.reshape(
            batch_size, self.num_attention_heads, query_width, self.head_dim
        )
        attended = attended * query_valid[:, None, :, None].to(dtype=attended.dtype)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, query_width, self.num_attention_heads * self.head_dim
        )
        return self.o_proj(attended)


if __name__ == "__main__":
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention as TransformersQwen3Attention,
        Qwen3RotaryEmbedding,
    )

    torch.manual_seed(89)
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
    )
    config.rope_theta = float(config.rope_parameters["rope_theta"])
    config._attn_implementation = "eager"
    baseline = TransformersQwen3Attention(config, layer_idx=0).eval()
    engine = Qwen3Attention(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).eval()
    engine.load_state_dict(baseline.state_dict(), strict=True)
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=2,
        block_size_tokens=4,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    request_id = "attention-forward-check"
    block_table = (1, 0)
    cache.bind(request_id, block_table)
    hidden_states = torch.randn(1, 5, config.hidden_size)
    position_ids = torch.arange(hidden_states.shape[1]).unsqueeze(0)
    causal_mask = torch.zeros(1, 1, 5, 5).masked_fill(
        torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1), -torch.inf
    )
    position_embeddings = Qwen3RotaryEmbedding(config)(hidden_states, position_ids)
    try:
        with torch.inference_mode():
            engine_output = engine.forward_batch(
                hidden_states,
                layer_index=0,
                request_ids=(request_id,),
                token_lengths=(5,),
                position_offsets=(0,),
                block_tables=(block_table,),
                cache=cache,
            )
            baseline_output = baseline(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
            )[0]
        print("==========engine===========")
        print(f"shape={tuple(engine_output.shape)}, sample={engine_output[0, 0, :4].tolist()}")
        print("==========baseline==========")
        print(f"shape={tuple(baseline_output.shape)}, sample={baseline_output[0, 0, :4].tolist()}")
        torch.testing.assert_close(engine_output, baseline_output, atol=1e-5, rtol=1e-5)
    finally:
        cache.release(request_id)
    print("Qwen3 paged attention matches the Transformers eager baseline on CPU.")
