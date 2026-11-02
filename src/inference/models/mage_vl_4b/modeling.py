"""Mage-VL 模型装配入口：组合视觉塔、Qwen3 文本塔与条件生成头。"""

# 本模块沿用 microsoft/Mage-VL 固定 revision d88b153285f1633a61b2f693c59c8576693af185（Apache-2.0）。
from typing import Any, Optional, Union

import torch
import torch.utils.benchmark
from torch import nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple

from .architecture.configuration_mage_vl import MageVLConfig
from .architecture.model_outputs import MageVLCausalLMOutputWithPast, MageVLModelOutputWithPast
from .architecture.multimodal_inputs import prepare_multimodal_inputs
from .architecture.streammind import MageVLStreamMindMixin
from .architecture.text_backbone import Qwen3Model
from .architecture.vision_features import MageVLVisionFeatureMixin
from .architecture.vision_tower import MageVLPreTrainedModel, MageVLVisionPretrainedModel

__all__ = ["MageVLForConditionalGeneration", "MageVLModel", "MageVLPreTrainedModel", "build_position_ids"]


def build_position_ids(inputs_embeds: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """依完整 attention mask 产生位置，并只返回目前 query 对应的位置。"""
    batch_size, query_length, _ = inputs_embeds.shape
    if attention_mask is None:
        return torch.arange(query_length, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids[:, -query_length:]


@auto_docstring
class MageVLModel(MageVLVisionFeatureMixin, MageVLStreamMindMixin, MageVLPreTrainedModel):
    base_model_prefix = ""
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: MageVLConfig
    _no_split_modules = ["MageVLVisionEncoderLayer", "Qwen3DecoderLayer"]

    def __init__(self, config: MageVLConfig):
        super().__init__(config)
        self.visual = MageVLVisionPretrainedModel._from_config(config.vision_config)
        self.language_model = Qwen3Model(config.text_config)
        self.streammind_gate = None
        self._streammind_model_path = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

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

        inputs_embeds = prepare_multimodal_inputs(
            self,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            patch_positions=patch_positions,
        )

        # Use simple 1D position_ids
        if position_ids is None:
            position_ids = build_position_ids(inputs_embeds, attention_mask)

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


@auto_docstring
class MageVLForConditionalGeneration(MageVLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.model = MageVLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        model.model._streammind_model_path = str(pretrained_model_name_or_path)
        return model

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
        patch_positions=None,
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, patch_positions=patch_positions)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def streammind_gate_forward(self, pixel_values, image_grid_thw, patch_positions=None):
        return self.model.streammind_gate_forward(
            pixel_values,
            image_grid_thw,
            patch_positions=patch_positions,
        )

    def streammind_gate_forward_segments(self, segments):
        return self.model.streammind_gate_forward_segments(segments)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        patch_positions: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, MageVLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
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

        Note (native-video alias):
            The companion ``MageVLProcessor.__call__(videos=...)`` does NOT
            pass ``pixel_values_videos`` / ``video_grid_thw`` / ``second_per_grid_ts``
            to this forward. Instead it aliases the video patch tensor as
            ``pixel_values=`` and ``image_grid_thw=``, so video inputs share the
            same code path as multi-image inputs (Mage-ViT is purely
            spatial; temporal information is carried by per-frame ``<X.X seconds>``
            text tags emitted by the processor). The ``*_videos`` and
            ``second_per_grid_ts`` kwargs are kept declared here only for API
            completeness and future use (e.g. 3D mRoPE / ``get_rope_index``); they
            are NOT consumed by the current Mage-ViT encoder.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor
        >>> from inference.models.mage_vl_4b.architecture.configuration_mage_vl import MageVLConfig
        >>> from inference.models.mage_vl_4b.modeling import MageVLForConditionalGeneration

        >>> model_path = "/path/to/local/mage-vl-snapshot"
        >>> model_config = MageVLConfig.from_pretrained(model_path, local_files_only=True)
        >>> model = MageVLForConditionalGeneration.from_pretrained(
        ...     model_path, config=model_config, local_files_only=True
        ... )
        >>> processor = AutoProcessor.from_pretrained(
        ...     model_path, trust_remote_code=True, local_files_only=True
        ... )

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            patch_positions=patch_positions,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
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

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return MageVLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        patch_positions=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            patch_positions=patch_positions,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # After the prefill iteration, drop image inputs so the vision tower
        # isn't re-run on decode steps. Gating on `is_first_iteration` (the
        # Qwen3-VL convention) is the only reliable signal in transformers
        # 5.x: `past_key_values` is non-None even on the first call (an empty
        # DynamicCache is created up-front by `generate`), and `cache_position`
        # may be `None` for remote-code models.
        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = [
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
            "patch_positions",
        ]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
                elif key == "patch_positions":
                    if image_grid_thw is not None and image_grid_thw.numel() > 0 and image_nums.sum() > 0:
                        samples = torch.split(image_grid_thw, list(image_nums))
                        lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    elif video_grid_thw is not None and video_grid_thw.numel() > 0 and video_nums.sum() > 0:
                        samples = torch.split(video_grid_thw, list(video_nums))
                        lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    else:
                        continue
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model as TransformersQwen3Model

    from .architecture.configuration_mage_vl import MageVLConfig, MageVLVisionConfig

    torch.manual_seed(53)
    text_config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    text_config._attn_implementation = "eager"
    vision_config = MageVLVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_channels=3,
        patch_size=2,
        out_hidden_size=32,
        spatial_merge_size=2,
    )
    vision_config._attn_implementation = "eager"
    config = MageVLConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=95,
        video_token_id=96,
    )
    engine = MageVLModel(config).eval()
    baseline = TransformersQwen3Model(text_config).eval()
    baseline.load_state_dict(engine.language_model.state_dict(), strict=True)
    input_ids = torch.tensor([[3, 7, 11, 19, 23]])
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        engine_output = engine(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        baseline_output = baseline(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    print("==========engine===========")
    print(f"shape={tuple(engine_output.last_hidden_state.shape)}, sample={engine_output.last_hidden_state[0, -1, :4].tolist()}")
    print("==========baseline==========")
    print(f"shape={tuple(baseline_output.last_hidden_state.shape)}, sample={baseline_output.last_hidden_state[0, -1, :4].tolist()}")
    torch.testing.assert_close(
        engine_output.last_hidden_state, baseline_output.last_hidden_state, atol=1e-5, rtol=1e-5
    )
    print(f"Mage-VL text-only multimodal model matches the Transformers Qwen3 baseline on {test_device}.")

    import importlib.util
    import sys
    import types
    from pathlib import Path

    official_snapshot = Path(__file__).resolve().parents[4] / "checkpoints" / "microsoft-Mage-VL-d88b153"
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

    reference_text_config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    reference_text_config._attn_implementation = "eager"
    reference_vision_config = official.MageVLVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_channels=3,
        patch_size=2,
        out_hidden_size=32,
        spatial_merge_size=2,
    )
    reference_vision_config._attn_implementation = "eager"
    reference_config = official.MageVLConfig(
        text_config=reference_text_config,
        vision_config=reference_vision_config,
        image_token_id=95,
        video_token_id=96,
    )
    official_model = official.MageVLModel(reference_config).eval()
    official_model.load_state_dict(engine.state_dict(), strict=True)
    multimodal_input_ids = torch.tensor([[3, 95, 11]])
    image_grid_thw = torch.tensor([[1, 2, 2]])
    patch_positions = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]])
    pixel_values = torch.randn(4, 12)
    with torch.inference_mode():
        engine_vision_output = engine(
            input_ids=multimodal_input_ids,
            attention_mask=torch.ones_like(multimodal_input_ids),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            patch_positions=patch_positions,
            use_cache=False,
        )
        official_vision_output = official_model(
            input_ids=multimodal_input_ids,
            attention_mask=torch.ones_like(multimodal_input_ids),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            patch_positions=patch_positions,
            use_cache=False,
        )
    torch.testing.assert_close(
        engine_vision_output.last_hidden_state,
        official_vision_output.last_hidden_state,
        atol=1e-5,
        rtol=1e-5,
    )
    print("==========engine===========")
    print(f"MageVLModel image output={engine_vision_output.last_hidden_state[0, -1, :4].tolist()}")
    print("==========baseline==========")
    print(f"official MageVLModel image output={official_vision_output.last_hidden_state[0, -1, :4].tolist()}")
    print(f"Mage-VL multimodal model forward matches the official fixed-revision source on {test_device}.")

    # Bench：分别测量文本路径与完整图像融合路径，不纳入模型构造和权重复制。
    with torch.inference_mode():
        text_model_bench = torch.utils.benchmark.Timer(
            stmt="engine(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)",
            globals=locals(),
        ).blocked_autorange(min_run_time=0.3)
        image_model_bench = torch.utils.benchmark.Timer(
            stmt="engine(input_ids=multimodal_input_ids, attention_mask=torch.ones_like(multimodal_input_ids), pixel_values=pixel_values, image_grid_thw=image_grid_thw, patch_positions=patch_positions, use_cache=False)",
            globals=locals(),
        ).blocked_autorange(min_run_time=0.3)
    print(f"Bench: MageVLModel text path {text_model_bench}")
    print(f"Bench: MageVLModel image path {image_model_bench}")

    # Profile：分别采集文本骨干和多模态图像注入的算子及峰值临时显存。
    profile_activities = [torch.profiler.ProfilerActivity.CPU]
    if test_device.type == "cuda":
        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.inference_mode():
        with torch.profiler.profile(
            activities=profile_activities,
            record_shapes=True,
            profile_memory=True,
        ) as text_model_profile:
            engine(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        with torch.profiler.profile(
            activities=profile_activities,
            record_shapes=True,
            profile_memory=True,
        ) as image_model_profile:
            engine(
                input_ids=multimodal_input_ids,
                attention_mask=torch.ones_like(multimodal_input_ids),
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                patch_positions=patch_positions,
                use_cache=False,
            )
    if test_device.type == "cuda":
        torch.cuda.synchronize(test_device)
        profile_sort = "self_cuda_time_total"
    else:
        profile_sort = "self_cpu_time_total"
    print("Profile: MageVLModel text-path operators")
    print(text_model_profile.key_averages().table(sort_by=profile_sort, row_limit=20))
    print("Profile: MageVLModel multimodal image-path operators")
    print(image_model_profile.key_averages().table(sort_by=profile_sort, row_limit=20))
