"""Mage-VL 視覺塔與 Transformers 預訓練模型基底。"""

# 本檔由固定 revision 的上游實作拆分調整；來源：microsoft/Mage-VL，revision d88b153285f1633a61b2f693c59c8576693af185（Apache-2.0）。
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import LayerNorm

from transformers import AutoModel
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.siglip.modeling_siglip import SiglipMLP
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    replace_return_docstrings,
)
from transformers.utils.generic import is_flash_attention_requested

from .configuration_mage_vl import MageVLConfig, MageVLVisionConfig
from .vision_encoder import (
    MageVLVisionEmbeddings,
    MageVLVisionEncoder,
    MageVLVisionPatchMerger,
    VisionRotaryEmbedding,
    get_norm_layer,
)


class MageVLPreTrainedModel(PreTrainedModel):
    config_class = MageVLConfig
    base_model_prefix = "model"
    input_modalities = ("image", "video", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["MageVLVisionEncoderLayer", "Qwen3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    def _init_weights(self, module):
        super()._init_weights(module)
        # Re-initialize VisionRotaryEmbedding inv_freq buffers.
        # These are registered with persistent=False, so they are not in the checkpoint
        # state_dict. When ``from_pretrained`` materializes the model from meta tensors,
        # the values in these buffers end up uninitialized. Mirror Qwen3-VL by explicitly
        # filling them here so RoPE produces the correct frequencies post-load.
        if isinstance(module, VisionRotaryEmbedding):
            base = module.base
            with torch.no_grad():
                inv_t = 1.0 / (base ** (torch.arange(module.t_size, dtype=torch.float32) / module.t_size))
                inv_h = 1.0 / (base ** (torch.arange(module.h_size, dtype=torch.float32) / module.h_size))
                inv_w = 1.0 / (base ** (torch.arange(module.w_size, dtype=torch.float32) / module.w_size))
                module.inv_freq_t.copy_(inv_t.to(module.inv_freq_t.device))
                module.inv_freq_h.copy_(inv_h.to(module.inv_freq_h.device))
                module.inv_freq_w.copy_(inv_w.to(module.inv_freq_w.device))


class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """
    Multi-Head Attention Pooling with a learned probe (PMA-style).
    """

    def __init__(self, config: MageVLVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        attn_output, _ = self.attention(probe, hidden_states, hidden_states)

        residual = attn_output
        attn_output = self.norm(attn_output)
        attn_output = residual + self.mlp(attn_output)

        return attn_output[:, 0]


# ---------------------------------------------------------------------------
# Vision Model
# ---------------------------------------------------------------------------


class MageVLVisionPretrainedModel(MageVLPreTrainedModel):
    """
    Mage-VL Vision Model.

    This vision model is designed to work with Qwen2VL-style image processing:
        - Receives pre-processed patches in 2x2 block spatial order
        - Applies RoPE with matching 2x2 block layout conversion
        - Accepts explicit patch_positions for RoPE computation

    Input format:
        hidden_state: [total_patches, num_channels, patch_size, patch_size]
        grid_thw: [num_samples, 3] with [t, h, w] for each sample
    """

    def __init__(self, config: MageVLVisionConfig):
        super().__init__(config)
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        # Vision components
        self.embeddings = MageVLVisionEmbeddings(config)
        self.layernorm_pre = get_norm_layer(config)
        self.encoder = MageVLVisionEncoder(config)
        self.video_rope = VisionRotaryEmbedding(config)

        if config.use_head:
            self.layernorm_post = get_norm_layer(config)
            self.head = Siglip2MultiheadAttentionPoolingHead(config)
        else:
            self.layernorm_post = None
            self.head = None

        self.merger = MageVLVisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            layer_norm_eps=config.layer_norm_eps,
            use_patch_position_encoding=getattr(config, "use_patch_position_encoding", False),
            patch_position_encoding_type=getattr(config, "patch_position_encoding_type", "absolute"),
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
        )

        self.post_init()

    def _build_cu_seqlens(
        self,
        grid_thw: torch.Tensor,
        total_patches: int,
        fixed_t: Optional[int] = 4,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, int]:
        if grid_thw is None or grid_thw.numel() == 0:
            # Fallback for no grid_thw: treat as single sequence
            return torch.tensor([0, total_patches], dtype=torch.int32, device=device), total_patches

        if device is None:
            device = grid_thw.device

        cu_seqlens = [0]
        max_seqlen = 0
        total_entries = grid_thw.shape[0]
        current_len = 0

        # Calculate cumulative lengths: split sequences based on fixed_t if provided
        for idx in range(total_entries):
            t_val = grid_thw[idx, 0].item()
            h_val = grid_thw[idx, 1].item()
            w_val = grid_thw[idx, 2].item()

            if fixed_t is not None and fixed_t > 0 and t_val > fixed_t:
                # Split large t into chunks of fixed_t
                num_full_windows = t_val // fixed_t
                remainder = t_val % fixed_t

                # Add full windows
                for _ in range(num_full_windows):
                    chunk_patches = fixed_t * int(h_val) * int(w_val)
                    current_len += chunk_patches
                    max_seqlen = max(max_seqlen, chunk_patches)
                    cu_seqlens.append(current_len)

                # Add remainder if any
                if remainder > 0:
                    chunk_patches = remainder * int(h_val) * int(w_val)
                    current_len += chunk_patches
                    max_seqlen = max(max_seqlen, chunk_patches)
                    cu_seqlens.append(current_len)
            else:
                # Standard case: add as one chunk
                chunk_patches = t_val * int(h_val) * int(w_val)
                current_len += chunk_patches
                max_seqlen = max(max_seqlen, chunk_patches)
                cu_seqlens.append(current_len)

        last_len = cu_seqlens[-1]
        if last_len != total_patches:
            raise ValueError(
                "cu_seqlens calculation mismatch:\n"
                f"- total_patches: {total_patches}\n"
                f"- calculated total: {last_len}\n"
                f"- grid_thw: {grid_thw}"
            )

        return torch.tensor(cu_seqlens, dtype=torch.int32, device=device), max_seqlen

    def _build_block_attention_mask(
        self,
        grid_thw: torch.Tensor,
        total_patches: int,
        fixed_t: Optional[int] = 4,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        if grid_thw is None or grid_thw.numel() == 0:
            return None

        if device is None:
            device = grid_thw.device

        lengths = []
        total_entries = grid_thw.shape[0]

        for idx in range(total_entries):
            t_val = grid_thw[idx, 0].item()
            h_val = grid_thw[idx, 1].item()
            w_val = grid_thw[idx, 2].item()

            if fixed_t is not None and fixed_t > 0 and t_val > fixed_t:
                # Split large t into chunks of fixed_t
                num_full_windows = t_val // fixed_t
                remainder = t_val % fixed_t

                # Add full windows
                for _ in range(num_full_windows):
                    lengths.append(fixed_t * int(h_val) * int(w_val))

                # Add remainder if any
                if remainder > 0:
                    lengths.append(remainder * int(h_val) * int(w_val))
            else:
                lengths.append(t_val * int(h_val) * int(w_val))

        total_len = sum(lengths)
        if total_len != total_patches:
            raise ValueError(
                "Block attention mask length mismatch:\n"
                f"- total_patches: {total_patches}\n"
                f"- total_len: {total_len}\n"
                f"- grid_thw: {grid_thw}"
            )

        attn_mask = torch.ones((total_len, total_len), dtype=torch.bool, device=device)
        start = 0
        for size in lengths:
            end = start + size
            attn_mask[start:end, start:end] = False
            start = end

        return attn_mask

    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=MageVLVisionConfig)
    def forward(
        self,
        hidden_state: torch.Tensor,
        grid_thw: Optional[torch.Tensor] = None,
        patch_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        skip_merger: Optional[bool] = False,
    ) -> Union[tuple, BaseModelOutputWithPooling]:
        r"""
        Forward pass for vision model.

        This method accepts pre-processed patches from Qwen2VL image processor and applies
        RoPE (Rotary Position Embedding) in 2x2 block layout to match the spatial arrangement
        of patches.

        Args:
            hidden_state: Pre-processed patches from Qwen2VL processor.
                Shape: [total_patches, num_channels, patch_size, patch_size]
            grid_thw: Grid sizes tensor of shape [num_samples, 3] with [t, h, w] for each sample.
                Required for computing RoPE and handling visible indices.
            patch_positions: Optional explicit patch positions for RoPE computation.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            return_dict: Whether to return a ModelOutput instead of tuple.
            skip_merger: If True, skip patch merger (useful for consistency checking).

        Returns:
            BaseModelOutputWithPooling with last_hidden_state containing merged features.
        """
        output_attentions = (
            output_attentions if output_attentions is not None else getattr(self.config, "output_attentions", False)
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )
        return_dict = True if return_dict is None else return_dict

        # 1. Embeddings
        # Note: embeddings returns [total_patches, embed_dim], we need to add batch dimension
        hidden_states = self.embeddings(hidden_state)
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)  # [1, total_patches, embed_dim]
        batch_size, total_patches, _ = hidden_states.shape

        # 2. RoPE Construction
        if patch_positions is not None and patch_positions.dim() == 3:
            patch_positions = patch_positions.squeeze(0)
        freqs_visible = self.video_rope.forward_from_positions(patch_positions)

        # Concatenate D/2 + D/2 -> D for applying rope
        freqs_visible = torch.cat([freqs_visible, freqs_visible], dim=-1)
        if freqs_visible.dim() == 2:
            freqs_visible = freqs_visible.unsqueeze(0)

        # 3. Pre-Norm & Encoder
        hidden_states = self.layernorm_pre(hidden_states)

        cu_seqlens, max_seqlen = self._build_cu_seqlens(
            grid_thw=grid_thw,
            total_patches=total_patches,
            fixed_t=getattr(self.config, "frame_windows_size", 4),
            device=hidden_states.device,
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=None,
            rotary_pos_emb=freqs_visible,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always get hidden states to use -2 layer
            return_dict=True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # Use second-to-last layer output for better feature representation
        if encoder_outputs.hidden_states is not None and len(encoder_outputs.hidden_states) >= 2 and not skip_merger:
            sequence_output = encoder_outputs.hidden_states[-1]
        else:
            sequence_output = encoder_outputs[0]

        # Post-Norm
        if self.layernorm_post is not None:
            sequence_output = self.layernorm_post(sequence_output)

        # Skip merger for consistency check with original ViT
        if skip_merger:
            pooled_output = None
            if self.head is not None:
                pooled_output = self.head(sequence_output)

            if not return_dict:
                return (sequence_output, pooled_output) + (
                    encoder_outputs.hidden_states if output_hidden_states else None,
                )
            return BaseModelOutputWithPooling(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
                attentions=encoder_outputs.attentions if output_attentions else None,
            )

        # Patch merger: input patches are already in 2x2 block order from Qwen2VL processor
        merged_output = self.merger(sequence_output, patch_positions=patch_positions)

        if not return_dict:
            return (merged_output,) + (encoder_outputs.hidden_states if output_hidden_states else None,)

        return BaseModelOutputWithPooling(
            last_hidden_state=merged_output,
            pooler_output=None,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None,
        )
