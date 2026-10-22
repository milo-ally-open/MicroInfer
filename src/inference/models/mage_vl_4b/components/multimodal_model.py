"""Mage-VL 多模態視覺與語言模型組裝。"""

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


from .model_outputs import MageVLModelOutputWithPast
from .vision_tower import MageVLPreTrainedModel, MageVLVisionPretrainedModel

@auto_docstring
class MageVLModel(MageVLPreTrainedModel):
    base_model_prefix = ""
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: MageVLConfig
    _no_split_modules = ["MageVLVisionEncoderLayer", "Qwen3DecoderLayer"]

    def __init__(self, config: MageVLConfig):
        super().__init__(config)
        self.visual = MageVLVisionPretrainedModel._from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.streammind_gate = None
        self._streammind_model_path = None

        # Initialize weights and apply final processing
        self.post_init()

    def _load_streammind_gate(self):
        if self.streammind_gate is not None:
            return self.streammind_gate
        from safetensors.torch import load_file
        from .streammind_gate import StreamMindGate

        if not self._streammind_model_path:
            raise RuntimeError(
                "StreamMind gate path is unavailable. Load the model with "
                "MageVLStreamMindForConditionalGeneration.from_pretrained()."
            )
        gate = StreamMindGate(self.config.text_config.hidden_size)
        gate_path = Path(self._streammind_model_path) / "streammind_gate.safetensors"
        if not gate_path.is_file():
            from huggingface_hub import hf_hub_download

            gate_path = Path(
                hf_hub_download(
                    repo_id=self._streammind_model_path,
                    filename="streammind_gate.safetensors",
                )
            )
        state = load_file(str(gate_path))
        gate.load_state_dict(state, strict=True)
        gate.to(
            device=next(self.visual.parameters()).device,
            dtype=next(self.visual.parameters()).dtype,
        ).eval()
        self.streammind_gate = gate
        return gate

    def _streammind_vision_tokens(self, pixel_values, image_grid_thw, patch_positions=None):
        pixel_values = pixel_values.type(self.visual.embeddings.patch_embedding.weight.dtype)
        rope = self.visual.video_rope
        saved_rope = (rope.inv_freq_t, rope.inv_freq_h, rope.inv_freq_w)
        try:
            # The StreamMind checkpoint was trained after the whole OV model,
            # including non-persistent RoPE buffers, was cast to BF16.
            rope.inv_freq_t = rope.inv_freq_t.to(pixel_values.dtype)
            rope.inv_freq_h = rope.inv_freq_h.to(pixel_values.dtype)
            rope.inv_freq_w = rope.inv_freq_w.to(pixel_values.dtype)
            vision_output = self.visual(
                pixel_values,
                grid_thw=image_grid_thw,
                patch_positions=patch_positions,
            )
        finally:
            rope.inv_freq_t, rope.inv_freq_h, rope.inv_freq_w = saved_rope
        merged = vision_output.last_hidden_state.reshape(-1, vision_output.last_hidden_state.shape[-1])
        merge = self.visual.spatial_merge_size
        time = int(image_grid_thw[:, 0].sum().item())
        patches_per_time = int(
            (image_grid_thw[0, 1].item() // merge)
            * (image_grid_thw[0, 2].item() // merge)
        )
        vision_tokens = merged.reshape(1, time, patches_per_time, -1)
        return vision_tokens

    def streammind_gate_forward(self, pixel_values, image_grid_thw, patch_positions=None):
        """Run the gate on one segment without changing the base LLM visual path."""
        vision_tokens = self._streammind_vision_tokens(
            pixel_values, image_grid_thw, patch_positions=patch_positions
        )
        return self._load_streammind_gate()(vision_tokens)

    def streammind_gate_forward_segments(self, segments):
        """Run EPFE continuously over a list of codec segments from one stream."""
        tokens = [
            self._streammind_vision_tokens(
                segment["pixel_values"],
                segment["image_grid_thw"],
                patch_positions=segment.get("patch_positions"),
            )
            for segment in segments
        ]
        lengths = [token.shape[1] for token in tokens]
        boundaries = torch.tensor(lengths).cumsum(0).tolist()
        return self._load_streammind_gate()(
            torch.cat(tokens, dim=1), response_positions=boundaries
        )

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
        patch_positions=None,
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos: Pre-processed patches from Qwen2VL processor.
                `torch.FloatTensor` of shape `(total_patches, num_channels, patch_size, patch_size)`
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        # Convert to correct dtype
        pixel_values_videos = pixel_values_videos.type(self.visual.embeddings.patch_embedding.weight.dtype)

        # Forward through vision model with grid_thw
        vision_output = self.visual(pixel_values_videos, grid_thw=video_grid_thw, patch_positions=patch_positions)

        # Extract the actual tensor from BaseModelOutputWithPooling
        if hasattr(vision_output, "last_hidden_state"):
            video_embeds = vision_output.last_hidden_state
        else:
            video_embeds = vision_output[0]  # Fallback for tuple output

        # Compute split sizes from video_grid_thw or from input shape
        if video_grid_thw is not None:
            split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        else:
            # Compute from input shape
            batch_size = pixel_values_videos.shape[0]
            split_sizes = [video_embeds.shape[1]] * batch_size

        # Split embeddings per video
        if len(split_sizes) > 1:
            video_embeds = torch.split(video_embeds.view(-1, video_embeds.shape[-1]), split_sizes)
        else:
            video_embeds = [video_embeds.view(-1, video_embeds.shape[-1])]

        return video_embeds

    def get_image_features(
        self, pixel_values, image_grid_thw: Optional[torch.LongTensor] = None, patch_positions=None
    ):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values: Pre-processed patches from Qwen2VL processor.
                - `torch.FloatTensor` of shape `(total_patches, num_channels, patch_size, patch_size)`
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        # Standard format from Qwen2VL processor
        if pixel_values.dim() == 2:
            # Convert to correct dtype
            pixel_values = pixel_values.type(self.visual.embeddings.patch_embedding.weight.dtype)

            # Forward through vision model with grid_thw
            vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, patch_positions=patch_positions)

            # Extract the actual tensor from BaseModelOutputWithPooling
            if hasattr(vision_output, "last_hidden_state"):
                image_embeds = vision_output.last_hidden_state
            else:
                image_embeds = vision_output[0]

            # Compute split sizes from grid_thw
            if image_grid_thw is not None:
                split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
            else:
                # Fallback: assume single image
                split_sizes = [image_embeds.shape[0] if image_embeds.dim() == 2 else image_embeds.shape[1]]

            # Split embeddings per image
            image_embeds_flat = image_embeds.view(-1, image_embeds.shape[-1])
            if len(split_sizes) > 1:
                image_embeds = list(torch.split(image_embeds_flat, split_sizes))
            else:
                image_embeds = [image_embeds_flat]

            return image_embeds
        else:
            raise ValueError(
                f"Unsupported pixel_values shape: expected 4D tensor [total_patches, C, H, W], "
                f"got {pixel_values.shape if hasattr(pixel_values, 'shape') else type(pixel_values)}"
            )

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        patch_positions: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, MageVLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        patch_positions (`torch.LongTensor` of shape `(total_patches, 3)` or `(1, total_patches, 3)`, *optional*):
            Explicit per-patch `(t, h, w)` position indices used by the vision tower to compute 3D rotary
            position embeddings (and the optional absolute position embedding inside the patch merger).
            `total_patches` is the sum of `t * h * w` across all images and videos in the batch, matching
            the layout produced by the Qwen2VL-style image processor.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
              Indices depicting the position of the input sequence tokens in the sequence. Contrarily to
              `position_ids`, this tensor is not affected by padding.

        Note: see the top-level ``MageVLForConditionalGeneration.forward``
        docstring; currently video flows in via the ``image_grid_thw`` / ``pixel_values``
        alias, so ``pixel_values_videos`` / ``video_grid_thw`` /
        ``second_per_grid_ts`` are unused at this layer.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = True if return_dict is None else return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_embeds = None

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, patch_positions=patch_positions)

        if image_embeds is not None:
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(
                pixel_values_videos, video_grid_thw, patch_positions=patch_positions
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # Use simple 1D position_ids
        if position_ids is None:
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                position_ids = (
                    torch.arange(seq_length, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
                )

            # Handle cache_position for generation
            if cache_position is not None and cache_position[0] != 0:
                position_ids = position_ids + cache_position[0]

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = MageVLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return output if return_dict else output.to_tuple()
