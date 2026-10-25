"""固定 Qwen3-0.6B decoder-only 计算图与批次 logits 入口。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from transformers import Qwen3Config

from ..components.paged_kv_cache import PagedKVCache
from ..layers.attention import Qwen3Attention
from ..layers.layers import Qwen3MLP, Qwen3RMSNorm


class Qwen3DecoderLayer(nn.Module):
    """一层 Qwen3 pre-norm decoder block。"""

    def __init__(self, config: Qwen3Config, *, device: torch.device, dtype: torch.dtype) -> None:
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
        self.post_attention_layernorm = Qwen3RMSNorm(hidden_size, float(config.rms_norm_eps), device=device, dtype=dtype)
        self.mlp = Qwen3MLP(hidden_size, intermediate_size, device=device, dtype=dtype)

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
    """Embedding、decoder stack 与 final norm。"""

    def __init__(self, config: Qwen3Config, *, device: torch.device, dtype: torch.dtype) -> None:
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
    """与固定 Qwen3 checkpoint key 对齐的本地语言模型。"""

    def __init__(self, config: Qwen3Config, *, dtype: torch.dtype, device: torch.device | str = "meta") -> None:
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
        if input_ids.ndim != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("Qwen3 input_ids must have shape [non-empty batch, non-empty token sequence].")
        batch_size, query_width = input_ids.shape
        if not (
            batch_size == len(token_lengths) == len(position_offsets) == len(request_ids) == len(block_tables)
        ):
            raise ValueError("Qwen3 batch tensors and sequence metadata must have matching sizes.")
        if len(set(request_ids)) != batch_size:
            raise ValueError("Qwen3 batch request IDs must be unique.")
        if any(length < 1 or length > query_width for length in token_lengths):
            raise ValueError("Qwen3 token lengths must fit the padded input batch.")
        if any(offset < 0 for offset in position_offsets):
            raise ValueError("Qwen3 position offsets must be non-negative.")
        embedding = self.model.embed_tokens.weight
        expected_head_dim = int(
            getattr(self.config, "head_dim", None)
            or (int(self.config.hidden_size) // int(self.config.num_attention_heads))
        )
        if (
            cache.num_layers != len(self.model.layers)
            or cache.num_key_value_heads != int(self.config.num_key_value_heads)
            or cache.head_dim != expected_head_dim
            or cache.device != embedding.device
            or cache.dtype != embedding.dtype
            or input_ids.device != embedding.device
        ):
            raise ValueError("Qwen3 model, input tensor and paged KV cache layouts must match.")
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
        """套用 config 的 tied word embeddings；不接受不一致的重复权重。"""
        if not bool(getattr(self.config, "tie_word_embeddings", False)):
            return
        input_weight = self.model.embed_tokens.weight
        output_weight = self.lm_head.weight
        if input_weight.device.type != "meta" and not torch.equal(input_weight, output_weight):
            raise ValueError("Qwen3 tied embedding and lm_head checkpoint tensors differ.")
        self.lm_head.weight = input_weight

    def to_empty_cpu(self) -> Qwen3ForCausalLM:
        """为 meta 计算图配置 CPU 权重储存，避免先建立随机 FP32 权重。"""
        return self.to_empty(device=torch.device("cpu"))

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM as TransformersQwen3ForCausalLM

    torch.manual_seed(17)
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
    # 正式 checkpoint 的 RoPE 参数来自模型设定；合成设定需同步清除 Transformers 的旧式缩放预设。
    config.rope_theta = 1_000_000.0
    config.rope_parameters = {"rope_type": "default", "rope_theta": config.rope_theta}
    config._attn_implementation = "eager"
    reference = TransformersQwen3ForCausalLM(config).eval()
    implementation = Qwen3ForCausalLM(config, dtype=torch.float32, device=test_device).eval()
    implementation.load_state_dict(reference.state_dict(), strict=True)

    request_id = "qwen3-parity"
    block_table = (0, 1)
    cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=2,
        block_size_tokens=4,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        device=test_device,
        dtype=torch.float32,
    )
    cache.bind(request_id, block_table)
    prompt = torch.tensor([[4, 8, 15, 16, 23]])
    decode_token = torch.tensor([[42]])
    try:
        with torch.inference_mode():
            expected_prefill = reference(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                use_cache=True,
                return_dict=True,
            )
            actual_prefill = implementation.forward_batch(
                prompt,
                token_lengths=(prompt.shape[-1],),
                position_offsets=(0,),
                request_ids=(request_id,),
                block_tables=(block_table,),
                cache=cache,
            )
            torch.testing.assert_close(
                actual_prefill, expected_prefill.logits[:, -1, :], atol=1e-5, rtol=1e-5
            )

            expected_decode = reference(
                input_ids=decode_token,
                attention_mask=torch.ones((1, prompt.shape[-1] + 1), dtype=torch.long),
                past_key_values=expected_prefill.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            actual_decode = implementation.forward_batch(
                decode_token,
                token_lengths=(1,),
                position_offsets=(prompt.shape[-1],),
                request_ids=(request_id,),
                block_tables=(block_table,),
                cache=cache,
            )
            torch.testing.assert_close(
                actual_decode, expected_decode.logits[:, -1, :], atol=1e-5, rtol=1e-5
            )
    finally:
        cache.release(request_id)
    print("==========engine===========")
    print(f"prefill={actual_prefill[0, :4].tolist()}")
    print(f"decode={actual_decode[0, :4].tolist()}")
    print("==========baseline==========")
    print(f"prefill={expected_prefill.logits[0, -1, :4].tolist()}")
    print(f"decode={expected_decode.logits[0, -1, :4].tolist()}")
    print(f"Qwen3 local model with paged KV matches Transformers eager Prefill and Decode on {test_device}.")
