"""Mage-VL 视觉 RoPE、Patch、Attention 与 Encoder 元件。"""

# 本档由固定 revision 的上游实作拆分调整；来源：microsoft/Mage-VL，revision d88b153285f1633a61b2f693c59c8576693af185（Apache-2.0）。
import math
from dataclasses import dataclass
from typing import Any, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import LayerNorm

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    replace_return_docstrings,
)

from ..model.configuration_mage_vl import MageVLConfig, MageVLVisionConfig
from .siglip_mlp import SiglipMLP


class VisionRotaryEmbedding(nn.Module):
    """
    3D (T,H,W) Rotary frequency constructor with 4:6:6 split.
    Supports both grid_thw-based and explicit position-based RoPE computation.
    """

    def __init__(self, config: MageVLVisionConfig):
        super().__init__()
        if config.num_attention_heads < 1 or config.hidden_size < 1:
            raise ValueError("Vision hidden size and attention head count must be positive.")
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("Vision hidden size must be divisible by the attention head count.")
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta

        if head_dim % 16:
            raise ValueError("Vision rotary head_dim must be divisible by 16.")
        half = head_dim // 2
        if half % 16:
            raise ValueError("Half of vision rotary head_dim must be divisible by 16 for the 4:6:6 split.")
        if not math.isfinite(base) or base <= 0:
            raise ValueError("Vision rotary theta must be finite and positive.")

        self.head_dim = head_dim
        self.half = half
        self.base = base

        # 4:6:6 split for T:H:W
        unit = half // 16
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit

        self.register_buffer(
            "inv_freq_t",
            1.0 / (base ** (torch.arange(self.t_size, dtype=torch.float32) / self.t_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_h",
            1.0 / (base ** (torch.arange(self.h_size, dtype=torch.float32) / self.h_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_w",
            1.0 / (base ** (torch.arange(self.w_size, dtype=torch.float32) / self.w_size)),
            persistent=False,
        )

    def forward(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Compute rotary position embeddings from grid_thw (Qwen2VL style).

        Args:
            grid_thw: [num_samples, 3] tensor with [t, h, w] for each sample

        Returns:
            freqs: [total_seq_len, half] tensor of position frequencies
        """
        device = grid_thw.device
        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        all_freqs = []
        for sample_thw in grid_thw:
            t, h, w = sample_thw[0].item(), sample_thw[1].item(), sample_thw[2].item()

            # Compute frequency tables
            ft = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv_t)
            fh = torch.outer(torch.arange(h, device=device, dtype=torch.float32), inv_h)
            fw = torch.outer(torch.arange(w, device=device, dtype=torch.float32), inv_w)

            # Build position indices for this sample
            t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
            h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
            w_ids = torch.arange(w, device=device).repeat(h).repeat(t)

            # Concatenate frequencies: [seq_len, half]
            sample_freqs = torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)
            all_freqs.append(sample_freqs)

        return torch.cat(all_freqs, dim=0)

    def forward_from_positions(self, patch_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute rotary position embeddings from explicit patch positions.

        Args:
            patch_positions: [seq_len, 3] tensor with [t, h, w] positions for each patch

        Returns:
            freqs: [seq_len, half] tensor of position frequencies
        """
        device = patch_positions.device
        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        t_pos = patch_positions[:, 0].float()
        h_pos = patch_positions[:, 1].float()
        w_pos = patch_positions[:, 2].float()

        ft = torch.outer(t_pos, inv_t)
        fh = torch.outer(h_pos, inv_h)
        fw = torch.outer(w_pos, inv_w)

        return torch.cat([ft, fh, fw], dim=-1)

    def forward_with_thw(self, t: int, h: int, w: int, device=None) -> torch.Tensor:
        """
        Compute rotary position embeddings from explicit t, h, w dimensions.

        Args:
            t: Number of temporal frames
            h: Number of height patches
            w: Number of width patches
            device: Target device

        Returns:
            freqs: [t*h*w, half] tensor of position frequencies
        """
        if device is None:
            device = self.inv_freq_t.device

        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        ft = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv_t)
        fh = torch.outer(torch.arange(h, device=device, dtype=torch.float32), inv_h)
        fw = torch.outer(torch.arange(w, device=device, dtype=torch.float32), inv_w)

        t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
        h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
        w_ids = torch.arange(w, device=device).repeat(h).repeat(t)

        freqs = torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)
        return freqs


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------


class MageVLVisionEmbeddings(nn.Module):
    """
    Patch embedding layer that converts pre-processed patches to embeddings.

    This module is designed to receive patches that have already been extracted
    and arranged by the Qwen2VL image processor in 2x2 block spatial order.

    Input format: [total_patches, num_channels, patch_size, patch_size]
    Output format: [total_patches, embed_dim]
    """

    def __init__(self, config: MageVLVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.in_channels = config.num_channels

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        hidden_states = hidden_states.view(-1, self.in_channels, self.patch_size, self.patch_size)
        hidden_states = self.patch_embedding(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)

        return hidden_states


# ---------------------------------------------------------------------------
# Patch Merger
# ---------------------------------------------------------------------------


class MageVLVisionPatchMerger(nn.Module):
    """
    Patch merger that merges spatial_merge_size x spatial_merge_size patches into one.

    This module is designed to work with Qwen2VL-style patch processing where patches
    are already arranged in 2x2 block order by the image processor.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        layer_norm_eps: float = 1e-05,
        use_patch_position_encoding: bool = False,
        patch_position_encoding_type: str = "absolute",
        max_position_embeddings: int = 8192,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm(context_dim, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )
        self.spatial_merge_size = spatial_merge_size
        self.use_patch_position_encoding = use_patch_position_encoding
        self.patch_position_encoding_type = patch_position_encoding_type

        if self.use_patch_position_encoding:
            if self.patch_position_encoding_type != "absolute":
                raise ValueError(
                    f"Unknown patch_position_encoding_type: {self.patch_position_encoding_type}. "
                    "Only 'absolute' is supported."
                )
            self.pos_emb_h = nn.Embedding(max_position_embeddings, dim)
            self.pos_emb_w = nn.Embedding(max_position_embeddings, dim)

    def forward(self, x: torch.Tensor, patch_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Merge patches from Qwen2VL-style input.

        The input patches are already arranged in 2x2 block order by the image processor,
        so we simply need to apply LayerNorm, reshape, and project through MLP.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size] or [seq_len, hidden_size]
               where seq_len = t * h * w (patches in 2x2 block order)

        Returns:
            Merged tensor of shape [batch_size, seq_len // spatial_merge_size^2, dim]
            or [seq_len // spatial_merge_size^2, dim]
        """
        if patch_positions is not None and patch_positions.dim() == 3:
            patch_positions = patch_positions.squeeze(0)

        x = self.ln_q(x).view(-1, self.hidden_size)
        x = self.mlp(x)

        if self.use_patch_position_encoding and patch_positions is not None:
            pp = patch_positions.view(-1, self.spatial_merge_size**2, 3)
            pp = pp[:, 0, :]
            pp = (pp // self.spatial_merge_size).long()

            x = x + self.pos_emb_h(pp[:, 1]) + self.pos_emb_w(pp[:, 2])

        return x


def rotate_half(x):
    """
    Interleaved rotation to match Source model's implementation.
    (x1, x2, x3, x4) -> (-x2, x1, -x4, x3)
    """
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def get_norm_layer(config):
    if config.layer_norm_type == "rms_norm":
        return nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    else:
        return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


def apply_rotary_pos_emb(q, k, freqs):
    # q, k: (B, H, L, D)
    # freqs: (B, L, D)
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    # We need to broadcast freqs to match heads
    # (B, L, D) -> (B, 1, L, D)
    # Keep the same dtype as q, k to avoid memory doubling from float32 promotion
    cos = freqs.cos().unsqueeze(1).float()
    sin = freqs.sin().unsqueeze(1).float()

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


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
    # 使用明确的 QKᵀ、softmax、权重乘 V 路径，让注意力数值流程可检视与替换。
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
            # 打包影格彼此独立，依累积长度切开后逐段计算，避免跨影格互相注意。
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


class MageVLVisionEncoderLayer(nn.Module):
    """Mage-VL 视觉 Encoder 层，采前置正规化与明确 eager 注意力。"""

    def __init__(self, config: MageVLVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = MageVLVisionAttention(config)
        self.layer_norm1 = get_norm_layer(config)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = get_norm_layer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            output_attentions=output_attentions,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attn_weights) if output_attentions else (hidden_states,)
        return outputs


class MageVLVisionEncoder(nn.Module):
    def __init__(self, config: MageVLVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MageVLVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # Gradient checkpointing support
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    output_attentions,
                    cu_seqlens,
                    max_seqlen,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    output_attentions=output_attentions,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


if __name__ == "__main__":
    import torch.nn.functional as functional
    from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig
    from transformers.models.siglip2.modeling_siglip2 import Siglip2EncoderLayer

    torch.manual_seed(61)
    config = MageVLVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_channels=3,
        image_size=8,
        patch_size=2,
        out_hidden_size=32,
        spatial_merge_size=2,
        layer_norm_eps=1e-6,
        layer_norm_type="layer_norm",
        attention_dropout=0.0,
    )

    # Check the three-axis RoPE against its explicit frequency-table definition.
    rope = VisionRotaryEmbedding(config).eval()
    grid_thw = torch.tensor([[2, 2, 2]])
    positions = torch.tensor(
        [[t, h, w] for t in range(2) for h in range(2) for w in range(2)], dtype=torch.long
    )
    unit = rope.half // 16
    inv_t = 1.0 / (config.rope_theta ** (torch.arange(4 * unit).float() / (4 * unit)))
    inv_h = 1.0 / (config.rope_theta ** (torch.arange(6 * unit).float() / (6 * unit)))
    inv_w = 1.0 / (config.rope_theta ** (torch.arange(6 * unit).float() / (6 * unit)))
    baseline_rope = torch.cat(
        (
            torch.outer(positions[:, 0].float(), inv_t),
            torch.outer(positions[:, 1].float(), inv_h),
            torch.outer(positions[:, 2].float(), inv_w),
        ),
        dim=-1,
    )
    engine_rope = rope(grid_thw)
    torch.testing.assert_close(engine_rope, baseline_rope, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(rope.forward_from_positions(positions), baseline_rope, atol=1e-6, rtol=1e-6)
    print("==========engine===========")
    print(f"VisionRotaryEmbedding={engine_rope[0, :4].tolist()}")
    print("==========baseline==========")
    print(f"three-axis frequency table={baseline_rope[0, :4].tolist()}")

    # Verify patch projection order with the equivalent convolution operation.
    embeddings = MageVLVisionEmbeddings(config).eval()
    pixel_patches = torch.randn(4, config.num_channels, config.patch_size, config.patch_size)
    with torch.inference_mode():
        engine_embeddings = embeddings(pixel_patches)
        baseline_embeddings = functional.conv2d(
            pixel_patches,
            embeddings.patch_embedding.weight,
            bias=None,
            stride=config.patch_size,
        ).view(-1, config.hidden_size)
    torch.testing.assert_close(engine_embeddings, baseline_embeddings, atol=1e-6, rtol=1e-6)
    print("==========engine===========")
    print(f"MageVLVisionEmbeddings={engine_embeddings[0, :4].tolist()}")
    print("==========baseline==========")
    print(f"conv2d patch projection={baseline_embeddings[0, :4].tolist()}")

    # Compare the packed eager attention and full encoder block with Transformers SigLIP2.
    siglip_config = Siglip2VisionConfig(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=config.num_attention_heads,
        hidden_act=config.hidden_act,
        layer_norm_eps=config.layer_norm_eps,
        attention_dropout=0.0,
    )
    siglip_config._attn_implementation = "eager"
    baseline_layer = Siglip2EncoderLayer(siglip_config).eval()
    engine_layer = MageVLVisionEncoderLayer(config).eval()
    with torch.no_grad():
        engine_layer.layer_norm1.load_state_dict(baseline_layer.layer_norm1.state_dict(), strict=True)
        engine_layer.layer_norm2.load_state_dict(baseline_layer.layer_norm2.state_dict(), strict=True)
        engine_layer.mlp.load_state_dict(baseline_layer.mlp.state_dict(), strict=True)
        engine_layer.self_attn.qkv.weight.copy_(
            torch.cat(
                (
                    baseline_layer.self_attn.q_proj.weight,
                    baseline_layer.self_attn.k_proj.weight,
                    baseline_layer.self_attn.v_proj.weight,
                ),
                dim=0,
            )
        )
        engine_layer.self_attn.qkv.bias.copy_(
            torch.cat(
                (
                    baseline_layer.self_attn.q_proj.bias,
                    baseline_layer.self_attn.k_proj.bias,
                    baseline_layer.self_attn.v_proj.bias,
                ),
                dim=0,
            )
        )
        engine_layer.self_attn.proj.load_state_dict(baseline_layer.self_attn.out_proj.state_dict(), strict=True)
    hidden_states = torch.randn(2, 5, config.hidden_size)
    normalized = engine_layer.layer_norm1(hidden_states)
    with torch.inference_mode():
        engine_attention, _ = engine_layer.self_attn(normalized)
        baseline_attention = baseline_layer.self_attn(normalized)[0]
        engine_output = engine_layer(hidden_states)[0]
        baseline_output = baseline_layer(hidden_states, attention_mask=None)
    torch.testing.assert_close(engine_attention, baseline_attention, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(engine_output, baseline_output, atol=1e-5, rtol=1e-5)
    print("==========engine===========")
    print(f"MageVLVisionEncoderLayer={engine_output[0, 0, :4].tolist()}")
    print("==========baseline==========")
    print(f"Siglip2EncoderLayer={baseline_output[0, 0, :4].tolist()}")

    # Check the patch merger against explicit normalization, flattening, and projection.
    merger = MageVLVisionPatchMerger(
        dim=config.out_hidden_size,
        context_dim=16,
        spatial_merge_size=2,
        layer_norm_eps=config.layer_norm_eps,
    ).eval()
    patch_group = torch.randn(4, 16)
    with torch.inference_mode():
        engine_merged = merger(patch_group)
        normalized_group = functional.layer_norm(
            patch_group,
            (16,),
            merger.ln_q.weight,
            merger.ln_q.bias,
            merger.ln_q.eps,
        ).reshape(1, 64)
        baseline_merged = functional.linear(normalized_group, merger.mlp[0].weight, merger.mlp[0].bias)
        baseline_merged = functional.gelu(baseline_merged)
        baseline_merged = functional.linear(baseline_merged, merger.mlp[2].weight, merger.mlp[2].bias)
    torch.testing.assert_close(engine_merged, baseline_merged, atol=1e-6, rtol=1e-6)
    print("==========engine===========")
    print(f"MageVLVisionPatchMerger={engine_merged[0, :4].tolist()}")
    print("==========baseline==========")
    print(f"explicit patch merge={baseline_merged[0, :4].tolist()}")
    print("Mage-VL vision encoder modules match their Transformers/PyTorch baselines on CPU.")
