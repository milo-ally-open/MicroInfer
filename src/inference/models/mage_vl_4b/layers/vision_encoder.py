"""Mage-VL 视觉组件兼容导出与 CUDA 优先、CPU 回退的前向对照入口。"""

# 组件实现拆分自 microsoft/Mage-VL revision d88b153285f1633a61b2f693c59c8576693af185（Apache-2.0）。

import torch
import torch.utils.benchmark

from .vision_rope import VisionRotaryEmbedding, apply_rotary_pos_emb, rotate_half
from .vision_embeddings import MageVLVisionEmbeddings, MageVLVisionPatchMerger
from .vision_attention import MageVLVisionAttention, eager_attention_forward, get_norm_layer
from .vision_stack import MageVLVisionEncoder, MageVLVisionEncoderLayer


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
    import torch.nn.functional as functional
    from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig
    from transformers.models.siglip2.modeling_siglip2 import Siglip2EncoderLayer

    from ..architecture.configuration_mage_vl import MageVLConfig, MageVLVisionConfig

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
    print(f"Mage-VL vision encoder modules match their Transformers/PyTorch baselines on {test_device}.")

    # Bench：分别量测 RoPE、patch embedding、Attention、Encoder Layer 和 Merger。
    with torch.inference_mode():
        rope_bench = torch.utils.benchmark.Timer(
            stmt="rope(grid_thw)", globals=locals()
        ).blocked_autorange(min_run_time=0.2)
        patch_bench = torch.utils.benchmark.Timer(
            stmt="embeddings(pixel_patches)", globals=locals()
        ).blocked_autorange(min_run_time=0.2)
        attention_bench = torch.utils.benchmark.Timer(
            stmt="engine_layer.self_attn(normalized)", globals=locals()
        ).blocked_autorange(min_run_time=0.2)
        encoder_layer_bench = torch.utils.benchmark.Timer(
            stmt="engine_layer(hidden_states)", globals=locals()
        ).blocked_autorange(min_run_time=0.2)
        merger_bench = torch.utils.benchmark.Timer(
            stmt="merger(patch_group)", globals=locals()
        ).blocked_autorange(min_run_time=0.2)
    print(f"Bench: VisionRotaryEmbedding {rope_bench}")
    print(f"Bench: MageVLVisionEmbeddings {patch_bench}")
    print(f"Bench: MageVLVisionAttention {attention_bench}")
    print(f"Bench: MageVLVisionEncoderLayer {encoder_layer_bench}")
    print(f"Bench: MageVLVisionPatchMerger {merger_bench}")

    # Profile：每个视觉子组件单独开一个 profiler，便于直接比较热点。
    profile_activities = [torch.profiler.ProfilerActivity.CPU]
    if test_device.type == "cuda":
        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.inference_mode():
        with torch.profiler.profile(
            activities=profile_activities,
            record_shapes=True,
            profile_memory=True,
        ) as rope_profile:
            rope(grid_thw)
        with torch.profiler.profile(
            activities=profile_activities,
            record_shapes=True,
            profile_memory=True,
        ) as patch_profile:
            embeddings(pixel_patches)
        with torch.profiler.profile(
            activities=profile_activities,
            record_shapes=True,
            profile_memory=True,
        ) as attention_profile:
            engine_layer.self_attn(normalized)
        with torch.profiler.profile(
            activities=profile_activities,
            record_shapes=True,
            profile_memory=True,
        ) as encoder_layer_profile:
            engine_layer(hidden_states)
        with torch.profiler.profile(
            activities=profile_activities,
            record_shapes=True,
            profile_memory=True,
        ) as merger_profile:
            merger(patch_group)
    if test_device.type == "cuda":
        torch.cuda.synchronize(test_device)
        profile_sort = "self_cuda_time_total"
    else:
        profile_sort = "self_cpu_time_total"
    print("Profile: VisionRotaryEmbedding operators")
    print(rope_profile.key_averages().table(sort_by=profile_sort, row_limit=10))
    print("Profile: MageVLVisionEmbeddings operators")
    print(patch_profile.key_averages().table(sort_by=profile_sort, row_limit=10))
    print("Profile: MageVLVisionAttention operators")
    print(attention_profile.key_averages().table(sort_by=profile_sort, row_limit=12))
    print("Profile: MageVLVisionEncoderLayer operators")
    print(encoder_layer_profile.key_averages().table(sort_by=profile_sort, row_limit=15))
    print("Profile: MageVLVisionPatchMerger operators")
    print(merger_profile.key_averages().table(sort_by=profile_sort, row_limit=12))
