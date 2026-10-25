"""Mage-VL Qwen3 文字骨干的 GQA 与 Cache 更新。"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from transformers.cache_utils import Cache

from .qwen3_text_layers import Qwen3RMSNorm


class Qwen3Attention(nn.Module):
    """明确执行 Q/K/V、RoPE、GQA、因果遮罩与 Cache 读写。"""

    def __init__(self, config, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        if min(self.num_heads, self.num_key_value_heads, int(config.hidden_size)) < 1:
            raise ValueError("Qwen3 attention dimensions must be positive.")
        configured_head_dim = getattr(config, "head_dim", None)
        self.head_dim = int(configured_head_dim or (int(config.hidden_size) // self.num_heads))
        if self.head_dim < 1:
            raise ValueError("Qwen3 attention head dimension must be positive.")
        if self.num_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError("Qwen3 attention requires divisible GQA heads and an even rotary head dimension.")
        self.groups = self.num_heads // self.num_key_value_heads
        rope_parameters = getattr(config, "rope_parameters", None)
        if not isinstance(rope_parameters, dict) or "rope_theta" not in rope_parameters:
            raise ValueError("Qwen3 config must define rope_parameters.rope_theta.")
        self.rope_theta = float(rope_parameters["rope_theta"])
        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0))
        self.query_chunk_size = int(getattr(config, "attention_query_chunk_size", 128))
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("Qwen3 RoPE theta must be finite and positive.")
        if not math.isfinite(self.attention_dropout) or not 0 <= self.attention_dropout <= 1:
            raise ValueError("Qwen3 attention dropout must be finite and between zero and one.")
        if self.query_chunk_size < 1:
            raise ValueError("Qwen3 attention query chunk size must be positive.")
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        cache_position: torch.Tensor,
        use_cache: bool,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if hidden_states.ndim != 3 or hidden_states.shape[1] < 1:
            raise ValueError("Qwen3 attention hidden states must have shape [batch, non-empty sequence, hidden].")
        if hidden_states.shape[-1] != self.q_proj.in_features:
            raise ValueError("Qwen3 attention hidden size does not match its projection weights.")
        batch_size, query_length, _ = hidden_states.shape
        if position_ids.shape != (batch_size, query_length):
            raise ValueError("Qwen3 position_ids must match the hidden-state batch and query dimensions.")
        if cache_position.numel() != query_length:
            raise ValueError("Qwen3 cache_position must contain one position per query token.")
        query = self.q_proj(hidden_states).view(batch_size, query_length, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)

        position_ids = position_ids.to(device=hidden_states.device, dtype=torch.long)
        frequencies = position_ids.float().unsqueeze(-1) * self._inverse_frequency(hidden_states.device)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        cosine = angles.cos().to(dtype=query.dtype).unsqueeze(1)
        sine = angles.sin().to(dtype=query.dtype).unsqueeze(1)
        query = query * cosine + self._rotate_half(query) * sine
        key = key * cosine + self._rotate_half(key) * sine

        if use_cache and past_key_values is not None:
            key, value = past_key_values.update(
                key,
                value,
                self.layer_index,
                {"cos": cosine, "sin": sine, "cache_position": cache_position},
            )

        key_length = key.shape[-2]
        query_positions = cache_position.to(device=hidden_states.device, dtype=torch.long).reshape(-1)
        key_positions = torch.arange(key_length, device=hidden_states.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions[-query_length:].unsqueeze(1)
        causal_mask = causal_mask.view(1, 1, query_length, key_length)

        groups = self.groups
        grouped_query = query.reshape(batch_size, self.num_key_value_heads, groups, query_length, self.head_dim)
        chunk_outputs: list[torch.Tensor] = []
        chunk_weights: list[torch.Tensor] = []
        for start in range(0, query_length, self.query_chunk_size):
            end = min(start + self.query_chunk_size, query_length)
            scores = torch.einsum(
                "bhgqd,bhkd->bhgqk", grouped_query[..., start:end, :], key
            ) * (self.head_dim**-0.5)
            chunk_causal_mask = causal_mask[..., start:end, :]
            scores = scores.masked_fill(~chunk_causal_mask.unsqueeze(2), torch.finfo(scores.dtype).min)
            if attention_mask is not None:
                if attention_mask.ndim == 2:
                    key_mask = attention_mask[:, None, None, :key_length].to(dtype=torch.bool)
                    scores = scores.masked_fill(~key_mask.unsqueeze(2), torch.finfo(scores.dtype).min)
                elif attention_mask.ndim == 4:
                    mask = attention_mask[..., -query_length:, :key_length][..., start:end, :]
                    if mask.dtype == torch.bool:
                        scores = scores.masked_fill(~mask.unsqueeze(2), torch.finfo(scores.dtype).min)
                    else:
                        scores = scores + mask.unsqueeze(2)

            probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
            probabilities = F.dropout(probabilities, p=self.attention_dropout, training=self.training)
            chunk_outputs.append(torch.einsum("bhgqk,bhkd->bhgqd", probabilities, value))
            if output_attentions:
                chunk_weights.append(probabilities)

        attended = torch.cat(chunk_outputs, dim=-2).reshape(
            batch_size, self.num_heads, query_length, self.head_dim
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, query_length, -1)
        weights = torch.cat(chunk_weights, dim=-2) if chunk_weights else None
        return self.o_proj(attended), weights

    def _inverse_frequency(self, device: torch.device) -> torch.Tensor:
        indices = torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
        return 1.0 / (self.rope_theta ** (indices / self.head_dim))

    @staticmethod
    def _rotate_half(values: torch.Tensor) -> torch.Tensor:
        first, second = values.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention as TransformersQwen3Attention,
        Qwen3RotaryEmbedding,
    )

    torch.manual_seed(31)
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
    config._attn_implementation = "eager"
    config.attention_query_chunk_size = 2
    baseline = TransformersQwen3Attention(config, layer_idx=0).eval()
    engine = Qwen3Attention(config, layer_index=0).eval()
    engine.load_state_dict(baseline.state_dict(), strict=True)
    hidden_states = torch.randn(2, 5, config.hidden_size)
    position_ids = torch.arange(hidden_states.shape[1]).expand(hidden_states.shape[0], -1)
    cache_position = torch.arange(hidden_states.shape[1])
    causal_mask = torch.zeros(1, 1, 5, 5).masked_fill(
        torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1), -torch.inf
    )
    baseline_rotary = Qwen3RotaryEmbedding(config)
    position_embeddings = baseline_rotary(hidden_states, position_ids)
    with torch.inference_mode():
        engine_output, _ = engine(
            hidden_states,
            position_ids=position_ids,
            attention_mask=causal_mask,
            past_key_values=None,
            cache_position=cache_position,
            use_cache=False,
        )
        baseline_output, _ = baseline(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            past_key_values=None,
            cache_position=cache_position,
        )
    print("==========engine===========")
    print(f"shape={tuple(engine_output.shape)}, sample={engine_output[0, 0, :4].tolist()}")
    print("==========baseline==========")
    print(f"shape={tuple(baseline_output.shape)}, sample={baseline_output[0, 0, :4].tolist()}")
    torch.testing.assert_close(engine_output, baseline_output, atol=1e-5, rtol=1e-5)
    print(f"Qwen3 attention matches the Transformers eager baseline on {test_device}.")
