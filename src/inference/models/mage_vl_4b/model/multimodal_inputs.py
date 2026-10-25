"""把视觉塔输出写入文字序列中的 image／video 占位 token。"""

# 按模型既有的视觉特征／placeholder 语义整理输入注入流程。

import torch


def prepare_multimodal_inputs(
    model,
    *,
    input_ids,
    inputs_embeds,
    pixel_values,
    pixel_values_videos,
    image_grid_thw,
    video_grid_thw,
    patch_positions,
):
    """将图片及视频 patch 编码后注入语言模型输入 embedding。"""
    if inputs_embeds is None:
        inputs_embeds = model.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        image_embeds = model.get_image_features(pixel_values, image_grid_thw, patch_positions=patch_positions)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds = model.get_video_features(
            pixel_values_videos, video_grid_thw, patch_positions=patch_positions
        )
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    return inputs_embeds


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
    import importlib.util
    import sys
    import types
    from pathlib import Path

    from transformers import Qwen3Config

    from .configuration_mage_vl import MageVLConfig, MageVLVisionConfig
    from .mage_vl import MageVLModel

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

    torch.manual_seed(73)
    text_config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        eos_token_id=None,
    )
    text_config._attn_implementation = "eager"
    reference_text_config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        eos_token_id=None,
    )
    reference_text_config._attn_implementation = "eager"
    vision_config = MageVLVisionConfig(
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
    reference_vision_config = official.MageVLVisionConfig(
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
    vision_config._attn_implementation = "eager"
    reference_vision_config._attn_implementation = "eager"
    config = MageVLConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=95,
        video_token_id=96,
    )
    reference_config = official.MageVLConfig(
        text_config=reference_text_config,
        vision_config=reference_vision_config,
        image_token_id=95,
        video_token_id=96,
    )
    engine = MageVLModel(config).eval()
    baseline = official.MageVLModel(reference_config).eval()
    baseline.load_state_dict(engine.state_dict(), strict=True)

    input_ids = torch.tensor([[1, 95, 2, 96, 3]])
    attention_mask = torch.ones_like(input_ids)
    image_grid_thw = torch.tensor([[1, 2, 2]])
    video_grid_thw = torch.tensor([[1, 2, 2]])
    patch_positions = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]])
    pixel_values = torch.randn(4, 12)
    pixel_values_videos = torch.randn(4, 3, 2, 2)
    with torch.inference_mode():
        engine_inputs_embeds = prepare_multimodal_inputs(
            engine,
            input_ids=input_ids,
            inputs_embeds=None,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            patch_positions=patch_positions,
        )
        baseline_inputs_embeds = baseline.get_input_embeddings()(input_ids)
        baseline_images = torch.cat(
            baseline.get_image_features(pixel_values, image_grid_thw, patch_positions=patch_positions), dim=0
        ).to(baseline_inputs_embeds.device, baseline_inputs_embeds.dtype)
        baseline_image_mask, _ = baseline.get_placeholder_mask(
            input_ids, baseline_inputs_embeds, image_features=baseline_images
        )
        baseline_inputs_embeds = baseline_inputs_embeds.masked_scatter(baseline_image_mask, baseline_images)
        baseline_videos = torch.cat(
            baseline.get_video_features(
                pixel_values_videos, video_grid_thw, patch_positions=patch_positions
            ),
            dim=0,
        ).to(baseline_inputs_embeds.device, baseline_inputs_embeds.dtype)
        _, baseline_video_mask = baseline.get_placeholder_mask(
            input_ids, baseline_inputs_embeds, video_features=baseline_videos
        )
        baseline_inputs_embeds = baseline_inputs_embeds.masked_scatter(baseline_video_mask, baseline_videos)
        engine_output = engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            patch_positions=patch_positions,
            use_cache=False,
        )
        baseline_output = baseline(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            patch_positions=patch_positions,
            use_cache=False,
        )
    torch.testing.assert_close(engine_inputs_embeds, baseline_inputs_embeds, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        engine_output.last_hidden_state, baseline_output.last_hidden_state, atol=1e-5, rtol=1e-5
    )
    print("==========engine===========")
    print(f"embeddings={tuple(engine_inputs_embeds.shape)}, output={tuple(engine_output.last_hidden_state.shape)}")
    print("==========baseline==========")
    print(f"official embeddings={tuple(baseline_inputs_embeds.shape)}, output={tuple(baseline_output.last_hidden_state.shape)}")
    print(f"Mage-VL image/video embedding injection and model forward match the official source on {test_device}.")
