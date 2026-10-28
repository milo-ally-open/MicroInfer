"""Mage-VL 基础多模态模型：装配视觉塔、Qwen3 文本塔与特征接口。"""

# 本模块沿用 microsoft/Mage-VL 固定 revision d88b153285f1633a61b2f693c59c8576693af185（Apache-2.0）。
from typing import Optional, Union

import torch
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring

from .configuration_mage_vl import MageVLConfig
from .model_outputs import MageVLModelOutputWithPast
from .multimodal_inputs import prepare_multimodal_inputs
from .streammind import MageVLStreamMindMixin
from .vision_features import MageVLVisionFeatureMixin
from .vision_tower import MageVLPreTrainedModel, MageVLVisionPretrainedModel
from .text_backbone import Qwen3Model


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


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model as TransformersQwen3Model

    from .configuration_mage_vl import MageVLConfig, MageVLVisionConfig

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
