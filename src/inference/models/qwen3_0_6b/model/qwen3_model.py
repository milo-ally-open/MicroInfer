"""固定 Qwen3-0.6B decoder-only 計算圖與批次 logits 入口。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from .paged_kv_cache import PagedKVCache
from .qwen3_attention import Qwen3Attention
from .qwen3_layers import Qwen3MLP, Qwen3RMSNorm


class Qwen3DecoderLayer(nn.Module):
    """一層 Qwen3 pre-norm decoder block。"""

    def __init__(self, config: Any, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        head_dim = int(getattr(config, "head_dim", hidden_size // int(config.num_attention_heads)))
        self.input_layernorm = Qwen3RMSNorm(
            hidden_size, float(config.rms_norm_eps), device=device, dtype=dtype
        )
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_attention_heads=int(config.num_attention_heads),
            num_key_value_heads=int(config.num_key_value_heads),
            head_dim=head_dim,
            rms_norm_eps=float(config.rms_norm_eps),
            rope_theta=float(config.rope_theta),
            device=device,
            dtype=dtype,
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            hidden_size, float(config.rms_norm_eps), device=device, dtype=dtype
        )
        self.mlp = Qwen3MLP(
            hidden_size, intermediate_size, device=device, dtype=dtype
        )

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
        hidden_states = hidden_states + self.self_attn.forward_batch(
            self.input_layernorm(hidden_states),
            layer_index=layer_index,
            request_ids=request_ids,
            token_lengths=token_lengths,
            position_offsets=position_offsets,
            block_tables=block_tables,
            cache=cache,
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Qwen3Model(nn.Module):
    """Embedding、decoder stack 與 final norm。"""

    def __init__(self, config: Any, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            int(config.vocab_size), int(config.hidden_size), device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config, device=device, dtype=dtype)
            for _ in range(int(config.num_hidden_layers))
        )
        self.norm = Qwen3RMSNorm(
            int(config.hidden_size), float(config.rms_norm_eps), device=device, dtype=dtype
        )


class Qwen3ForCausalLM(nn.Module):
    """與固定 Qwen3 checkpoint key 對齊的本地語言模型。"""

    def __init__(self, config: Any, *, dtype: torch.dtype, device: torch.device | str = "meta") -> None:
        super().__init__()
        target_device = torch.device(device)
        self.config = config
        self.model = Qwen3Model(config, device=target_device, dtype=dtype)
        self.lm_head = nn.Linear(
            int(config.hidden_size), int(config.vocab_size), bias=False, device=target_device, dtype=dtype
        )

    def forward_batch(
        self,
        input_ids: torch.Tensor,
        *,
        token_lengths: Sequence[int],
        position_offsets: Sequence[int],
        request_ids: Sequence[str],
        block_tables: Sequence[Sequence[int]],
        cache: PagedKVCache,
    ) -> torch.Tensor:
        batch_size, query_width = input_ids.shape
        if not (
            batch_size == len(token_lengths) == len(position_offsets) == len(request_ids) == len(block_tables)
        ):
            raise ValueError("Qwen3 batch tensors and sequence metadata must have matching sizes.")
        if any(length < 1 or length > query_width for length in token_lengths):
            raise ValueError("Qwen3 token lengths must fit the padded input batch.")
        hidden_states = self.model.embed_tokens(input_ids)
        for layer_index, layer in enumerate(self.model.layers):
            hidden_states = layer.forward_batch(
                hidden_states,
                layer_index=layer_index,
                request_ids=request_ids,
                token_lengths=token_lengths,
                position_offsets=position_offsets,
                block_tables=block_tables,
                cache=cache,
            )
        hidden_states = self.model.norm(hidden_states)
        row_indices = torch.arange(batch_size, device=input_ids.device)
        last_indices = torch.tensor(token_lengths, device=input_ids.device) - 1
        last_hidden_states = hidden_states[row_indices, last_indices]
        return self.lm_head(last_hidden_states)

    def tie_weights(self) -> None:
        """套用 config 的 tied word embeddings；不接受不一致的重複權重。"""
        if not bool(getattr(self.config, "tie_word_embeddings", False)):
            return
        input_weight = self.model.embed_tokens.weight
        output_weight = self.lm_head.weight
        if input_weight.device.type != "meta" and not torch.equal(input_weight, output_weight):
            raise ValueError("Qwen3 tied embedding and lm_head checkpoint tensors differ.")
        self.lm_head.weight = input_weight

    def to_empty_cpu(self) -> Qwen3ForCausalLM:
        """為 meta 計算圖配置 CPU 權重儲存，避免先建立隨機 FP32 權重。"""
        return self.to_empty(device=torch.device("cpu"))

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

