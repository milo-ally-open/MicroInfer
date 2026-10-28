"""Mage-VL 专用 Qwen3 文字骨干，保留上游权重名称与 Cache 契约。"""

from __future__ import annotations

import torch
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..layers.qwen3_text_attention import Qwen3Attention
from ..layers.qwen3_text_layers import Qwen3MLP, Qwen3RMSNorm


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 pre-norm Transformer block。"""

    def __init__(self, config, layer_index: int) -> None:
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Qwen3Attention(config, layer_index)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        attention_output, attention_weights = self.self_attn(self.input_layernorm(hidden_states), **kwargs)
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, attention_weights


class Qwen3Model(nn.Module):
    """以本地可替换 Decoder layers 执行 Qwen3 文字计算图。"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config, index) for index in range(config.num_hidden_layers)
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)

    def get_input_embeddings(self) -> nn.Embedding:
        """提供 Transformers 量化器与生成接口所需的输入嵌入访问器。"""
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, query_length, _ = inputs_embeds.shape
        use_cache = self.config.use_cache if use_cache is None else use_cache
        past_length = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + query_length, device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = inputs_embeds
        layer_kwargs = {
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
        }
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, attention_weights = layer(hidden_states, **layer_kwargs)
            if output_attentions:
                all_attentions += (attention_weights,)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
        return output if return_dict else output.to_tuple()


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model as TransformersQwen3Model

    torch.manual_seed(11)
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    config._attn_implementation = "eager"
    config.attention_query_chunk_size = 2
    reference = TransformersQwen3Model(config).eval()
    implementation = Qwen3Model(config).eval()
    implementation.load_state_dict(reference.state_dict(), strict=True)

    input_ids = torch.tensor([[4, 8, 15, 16, 23]])
    attention_mask = torch.ones_like(input_ids)
    expected_prefill = reference(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True
    )
    actual_prefill = implementation(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True
    )
    torch.testing.assert_close(
        actual_prefill.last_hidden_state, expected_prefill.last_hidden_state, atol=1e-5, rtol=1e-5
    )

    next_ids = torch.tensor([[42]])
    next_mask = torch.ones((1, 6), dtype=torch.long)
    position_ids = torch.tensor([[5]])
    expected_decode = reference(
        input_ids=next_ids,
        attention_mask=next_mask,
        position_ids=position_ids,
        past_key_values=expected_prefill.past_key_values,
        use_cache=True,
        return_dict=True,
    )
    actual_decode = implementation(
        input_ids=next_ids,
        attention_mask=next_mask,
        position_ids=position_ids,
        past_key_values=actual_prefill.past_key_values,
        use_cache=True,
        return_dict=True,
    )
    torch.testing.assert_close(
        actual_decode.last_hidden_state, expected_decode.last_hidden_state, atol=1e-5, rtol=1e-5
    )
    print("==========engine===========")
    print(f"prefill={actual_prefill.last_hidden_state[0, -1, :4].tolist()}")
    print(f"decode={actual_decode.last_hidden_state[0, -1, :4].tolist()}")
    print("==========baseline==========")
    print(f"prefill={expected_prefill.last_hidden_state[0, -1, :4].tolist()}")
    print(f"decode={expected_decode.last_hidden_state[0, -1, :4].tolist()}")
    print(f"Mage-VL Qwen3 text backbone matches Transformers eager Prefill and Cache Decode on {test_device}.")
