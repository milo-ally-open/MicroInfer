"""Mage-VL 视觉归一化、RoPE 注意力及 eager attention 计算。"""

# 本模块沿用 microsoft/Mage-VL revision d88b153285f1633a61b2f693c59c8576693af185（Apache-2.0）实现。

from typing import Optional

import torch
import torch.nn as nn

from ..model.configuration_mage_vl import MageVLVisionConfig
from .siglip_mlp import SiglipMLP
from .vision_rope import apply_rotary_pos_emb


def get_norm_layer(config):
    if config.layer_norm_type == "rms_norm":
        return nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    else:
        return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Eager attention; query/key/value are expected as ``(B, H, L, D)``."""
    # 使用明确的 QKᵀ、softmax、权重乘 V 路径，让注意力数值流程可查看与替换。
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(
                ~attention_mask, torch.finfo(attn_weights.dtype).min
            )
        else:
            attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()  # (B, L, H, D)
    return attn_output, attn_weights


class MageVLVisionAttention(nn.Module):
    """以明确 eager 计算执行 Mage-VL 视觉注意力及 RoPE。"""

    def __init__(self, config: MageVLVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.num_key_value_groups = 1  # 视觉 Q、K、V 使用相同的 head 数。
        self.scale = self.head_dim**-0.5
        self.scaling = self.scale  # alias expected by some attention interfaces
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, q_len, _ = hidden_states.size()
        # (B, L, 3*H*D) -> (B, L, 3, H, D) -> 3 x (B, L, H, D) -> 3 x (B, H, L, D)
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(batch_size, q_len, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 1, 3, 4)
            .unbind(0)
        )
        query_states = q.transpose(1, 2)
        key_states = k.transpose(1, 2)
        value_states = v.transpose(1, 2)

        if rotary_pos_emb is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, rotary_pos_emb)

        dropout = 0.0 if not self.training else self.attention_dropout

        if cu_seqlens is not None:
            # 打包帧彼此独立，依累积长度切开后逐段计算，避免跨帧互相注意。
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            splits = [torch.split(t, lengths, dim=2) for t in (query_states, key_states, value_states)]
            attn_outputs = [
                eager_attention_forward(
                    self,
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    attention_mask=None,
                    scaling=self.scale,
                    dropout=dropout,
                )[0]
                for q_chunk, k_chunk, v_chunk in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)
        else:
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                if attn_mask.shape[0] == 1 and batch_size > 1:
                    attn_mask = attn_mask.expand(batch_size, -1, -1)
                attn_mask = attn_mask.unsqueeze(1)  # (B, 1, L, L)
            attn_output, _ = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=attn_mask,
                scaling=self.scale,
                dropout=dropout,
            )

        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)
        attn_output = self.proj(attn_output)

        return attn_output, None


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
    import importlib.util
    import sys
    import types
    from pathlib import Path

    official_snapshot = Path(__file__).resolve().parents[5] / "checkpoints" / "microsoft-Mage-VL-d88b153"
    official_package_name = "_mage_vl_official_d88b153"
    official_package = types.ModuleType(official_package_name)
    official_package.__path__ = [str(official_snapshot)]
    sys.modules[official_package_name] = official_package
    official_name = f"{official_package_name}.modeling_mage_vl"
    official_spec = importlib.util.spec_from_file_location(official_name, official_snapshot / "modeling_mage_vl.py")
    if official_spec is None or official_spec.loader is None:
        raise RuntimeError(f"Unable to load the official Mage-VL source at {official_snapshot}.")
    official = importlib.util.module_from_spec(official_spec)
    sys.modules[official_name] = official
    official_spec.loader.exec_module(official)

    torch.manual_seed(65)
    config = MageVLVisionConfig(hidden_size=64, num_attention_heads=2, attention_dropout=0.0)
    reference_config = official.MageVLVisionConfig(hidden_size=64, num_attention_heads=2, attention_dropout=0.0)
    config._attn_implementation = "eager"
    reference_config._attn_implementation = "eager"
    engine = MageVLVisionAttention(config).eval()
    baseline = official.MageVLVisionAttention(reference_config).eval()
    engine.load_state_dict(baseline.state_dict(), strict=True)

    hidden_states = torch.randn(1, 8, config.hidden_size)
    grid_thw = torch.tensor([[2, 2, 2]])
    frequencies = official.VisionRotaryEmbedding(reference_config)(grid_thw)
    rotary_pos_emb = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(0)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    with torch.inference_mode():
        engine_output = engine(
            hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            cu_seqlens=cu_seqlens,
            max_seqlen=4,
        )[0]
        baseline_output = baseline(
            hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            cu_seqlens=cu_seqlens,
            max_seqlen=4,
        )[0]
    torch.testing.assert_close(engine_output, baseline_output, atol=1e-6, rtol=1e-6)
    print("==========engine===========")
    print(f"shape={tuple(engine_output.shape)}, sample={engine_output[0, 0, :4].tolist()}")
    print("==========baseline==========")
    print(f"official Mage-VL sample={baseline_output[0, 0, :4].tolist()}")
    print(f"Mage-VL vision attention with per-frame sequence boundaries matches the official source on {test_device}.")
