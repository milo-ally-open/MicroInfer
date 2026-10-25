"""Mage-VL 图片／视频特征编码、切分与占位符校验。"""

# 本模块拆分自 microsoft/Mage-VL revision d88b153285f1633a61b2f693c59c8576693af185（Apache-2.0）。

from typing import Optional
import torch


class MageVLVisionFeatureMixin:
    """封装视觉塔到多模态 token 特征之间的模型接口。"""

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


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
    import importlib.util
    import sys
    import types
    from pathlib import Path
    from types import SimpleNamespace

    import torch.nn as nn

    from .configuration_mage_vl import MageVLVisionConfig
    from .vision_tower import MageVLVisionPretrainedModel

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

    torch.manual_seed(69)
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
    )
    reference_config = official.MageVLVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_channels=3,
        image_size=8,
        patch_size=2,
        out_hidden_size=32,
        spatial_merge_size=2,
    )
    config._attn_implementation = "eager"
    reference_config._attn_implementation = "eager"
    engine_visual = MageVLVisionPretrainedModel(config).eval()
    baseline_visual = official.MageVLVisionPretrainedModel(reference_config).eval()
    engine_visual.load_state_dict(baseline_visual.state_dict(), strict=True)
    embedding = nn.Embedding(100, 32)
    model_config = SimpleNamespace(image_token_id=95, video_token_id=96)
    engine_owner = SimpleNamespace(
        visual=engine_visual,
        config=model_config,
        get_input_embeddings=lambda: embedding,
    )
    baseline_owner = SimpleNamespace(
        visual=baseline_visual,
        config=model_config,
        get_input_embeddings=lambda: embedding,
    )
    image_grid_thw = torch.tensor([[1, 2, 2]])
    patch_positions = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]])
    image_pixels = torch.randn(4, 12)
    video_pixels = torch.randn(4, 3, 2, 2)
    input_ids = torch.tensor([[3, 95, 4, 96]])
    inputs_embeds = embedding(input_ids)
    with torch.inference_mode():
        engine_images = MageVLVisionFeatureMixin.get_image_features(
            engine_owner, image_pixels, image_grid_thw, patch_positions=patch_positions
        )
        baseline_images = official.MageVLModel.get_image_features(
            baseline_owner, image_pixels, image_grid_thw, patch_positions=patch_positions
        )
        engine_videos = MageVLVisionFeatureMixin.get_video_features(
            engine_owner, video_pixels, image_grid_thw, patch_positions=patch_positions
        )
        baseline_videos = official.MageVLModel.get_video_features(
            baseline_owner, video_pixels, image_grid_thw, patch_positions=patch_positions
        )
        engine_masks = MageVLVisionFeatureMixin.get_placeholder_mask(
            engine_owner,
            input_ids,
            inputs_embeds,
            image_features=torch.cat(engine_images, dim=0),
            video_features=torch.cat(engine_videos, dim=0),
        )
        baseline_masks = official.MageVLModel.get_placeholder_mask(
            baseline_owner,
            input_ids,
            inputs_embeds,
            image_features=torch.cat(baseline_images, dim=0),
            video_features=torch.cat(baseline_videos, dim=0),
        )
    for engine_values, baseline_values in ((engine_images, baseline_images), (engine_videos, baseline_videos)):
        assert len(engine_values) == len(baseline_values)
        for engine_value, baseline_value in zip(engine_values, baseline_values):
            torch.testing.assert_close(engine_value, baseline_value, atol=1e-6, rtol=1e-6)
    for engine_mask, baseline_mask in zip(engine_masks, baseline_masks):
        torch.testing.assert_close(engine_mask, baseline_mask, atol=0, rtol=0)
    print("==========engine===========")
    print(f"image={tuple(engine_images[0].shape)}, video={tuple(engine_videos[0].shape)}")
    print("==========baseline==========")
    print(f"official image={tuple(baseline_images[0].shape)}, video={tuple(baseline_videos[0].shape)}")
    print(f"Mage-VL image/video feature splitting and placeholder masks match the official source on {test_device}.")
