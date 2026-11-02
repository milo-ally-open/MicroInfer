"""用本地层组件显式组装 Qwen3-VL，并复用 Transformers 官方多模态前向。"""

from __future__ import annotations

import torch
import torch.utils.benchmark
from torch import nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration as _TransformersForConditionalGeneration,
    Qwen3VLModel as _TransformersModel,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextModel as _TransformersTextModel,
    Qwen3VLVisionModel as _TransformersVisionModel,
)

from .layers import (
    Qwen3VLLMHead,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextEmbedding,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionBlock,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionRotaryEmbedding,
)


class Qwen3VLVisionModel(_TransformersVisionModel):
    """视觉塔：Patch Embed → 位置编码 → Vision Blocks → Merger。"""

    def __init__(self, config, *inputs, **kwargs) -> None:
        # 直接初始化 HF 公共基类，避免先创建另一套官方视觉层再替换。
        Qwen3VLPreTrainedModel.__init__(self, config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size**2

        self.patch_embed = Qwen3VLVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.interpolation_align_corners = True
        self.interpolation_mode = "bilinear"
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(config)
        self.blocks = nn.ModuleList((Qwen3VLVisionBlock(config) for _ in range(config.depth)))
        self.merger = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True)
            for _ in self.deepstack_visual_indexes
        )
        self.gradient_checkpointing = False
        self.post_init()


class Qwen3VLTextModel(_TransformersTextModel):
    """文本塔：Token Embedding → Decoder Layers → Final RMSNorm。"""

    def __init__(self, config) -> None:
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = Qwen3VLTextEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )
        self.layers = nn.ModuleList(
            Qwen3VLTextDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class Qwen3VLModel(_TransformersModel):
    """多模态主干：视觉塔与文本塔在官方模型前向中协同工作。"""

    def __init__(self, config) -> None:
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.visual = Qwen3VLVisionModel(config.vision_config)
        self.language_model = Qwen3VLTextModel(config.text_config)
        self.rope_deltas = None
        self.post_init()


class Qwen3VLForConditionalGeneration(_TransformersForConditionalGeneration):
    """完整生成模型：多模态主干加词表投影头。"""

    def __init__(self, config) -> None:
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.model = Qwen3VLModel(config)
        self.lm_head = Qwen3VLLMHead(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.post_init()


__all__ = [
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLModel",
    "Qwen3VLTextModel",
    "Qwen3VLVisionModel",
]


if __name__ == "__main__":
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )

    # 用 CUDA（可用时）构造小型模型，对照 HF 完整模型的标准模块树。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    text_config = Qwen3VLTextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rope_parameters={
            "rope_theta": 500000.0,
            "rope_type": "default",
            "mrope_section": [1, 1, 2],
        },
    )
    vision_config = Qwen3VLVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        depth=1,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=32,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "axial"},
    )
    vision_config._attn_implementation = "sdpa"
    text_config._attn_implementation = "sdpa"
    config = Qwen3VLConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=126,
        video_token_id=127,
    )
    baseline = _TransformersForConditionalGeneration(config).eval().to(device)
    local = Qwen3VLForConditionalGeneration(config).eval().to(device)
    local.load_state_dict(baseline.state_dict(), strict=True)
    input_ids = torch.tensor([[3, 7, 11, 19]], device=device)
    attention_mask = torch.ones_like(input_ids)

    # Test：同权重、同文本输入，验证显式组装后的完整模型 logits。
    with torch.inference_mode():
        expected = baseline(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        actual = local(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    print(f"Qwen3VLForConditionalGeneration parity passed on {device}.")

    # 用一个合并后图像 token 和 4 个视觉 patch 验证两座塔的完整连接。
    image_input_ids = torch.tensor([[3, config.image_token_id, 11]], device=device)
    image_attention_mask = torch.ones_like(image_input_ids)
    mm_token_type_ids = torch.tensor([[0, 1, 0]], device=device)
    image_grid_thw = torch.tensor([[1, 2, 2]], device=device)
    pixel_values = torch.randn(4, 24, device=device)
    with torch.inference_mode():
        expected_image_logits = baseline(
            input_ids=image_input_ids,
            attention_mask=image_attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
        ).logits
        actual_image_logits = local(
            input_ids=image_input_ids,
            attention_mask=image_attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
        ).logits
    torch.testing.assert_close(actual_image_logits, expected_image_logits, atol=1e-05, rtol=1e-05)
    print(f"Qwen3VL image-to-text full-model parity passed on {device}.")

    # Bench：分别测量纯文本和图文前向，排除权重加载时间。
    with torch.inference_mode():
        text_model_bench = torch.utils.benchmark.Timer(
            stmt="local(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)",
            globals=locals(),
        ).blocked_autorange(min_run_time=0.3)
        image_model_bench = torch.utils.benchmark.Timer(
            stmt="local(input_ids=image_input_ids, attention_mask=image_attention_mask, mm_token_type_ids=mm_token_type_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw, use_cache=False)",
            globals=locals(),
        ).blocked_autorange(min_run_time=0.3)
    print(f"Bench: Qwen3VL text path {text_model_bench}")
    print(f"Bench: Qwen3VL image path {image_model_bench}")

    # Profile：分开观察文本塔和图像编码/注入/文本生成路径的算子。
    profile_activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.inference_mode():
        with torch.profiler.profile(
            activities=profile_activities,
            record_shapes=True,
            profile_memory=True,
        ) as text_model_profile:
            local(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        with torch.profiler.profile(
            activities=profile_activities,
            record_shapes=True,
            profile_memory=True,
        ) as image_model_profile:
            local(
                input_ids=image_input_ids,
                attention_mask=image_attention_mask,
                mm_token_type_ids=mm_token_type_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_cache=False,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        profile_sort = "self_cuda_time_total"
    else:
        profile_sort = "self_cpu_time_total"
    print("Profile: Qwen3VL text-path operators")
    print(text_model_profile.key_averages().table(sort_by=profile_sort, row_limit=20))
    print("Profile: Qwen3VL image-path operators")
    print(image_model_profile.key_averages().table(sort_by=profile_sort, row_limit=20))
