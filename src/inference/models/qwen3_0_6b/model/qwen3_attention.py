"""Qwen3 GQA、Q/K RMSNorm、RoPE 與 paged KV 寫讀。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .paged_kv_cache import PagedKVCache
from .qwen3_layers import Qwen3RMSNorm
from .rotary_embedding import apply_rotary, rotary_embeddings


class Qwen3Attention(nn.Module):
    """依 request block table 執行 Qwen3 grouped-query self-attention。"""

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
        try:
            attended = F.scaled_dot_product_attention(
                query,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=True,
            )
        except TypeError:
            # 舊版 PyTorch 沒有 enable_gqa 時，以 repeat_interleave 保持 GQA 語義。
            repeats = self.num_attention_heads // self.num_key_value_heads
            attended = F.scaled_dot_product_attention(
                query,
                key_states.repeat_interleave(repeats, dim=1),
                value_states.repeat_interleave(repeats, dim=1),
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, query_width, self.num_attention_heads * self.head_dim
        )
        return self.o_proj(attended)

