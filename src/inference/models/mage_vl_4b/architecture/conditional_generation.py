"""Mage-VL 生成模型兼容导出；完整架构在包根目录的 ``modeling.py`` 组装。"""

import torch

from ..modeling import MageVLForConditionalGeneration, MageVLModel, MageVLPreTrainedModel

__all__ = ["MageVLForConditionalGeneration", "MageVLModel", "MageVLPreTrainedModel"]


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3ForCausalLM as TransformersQwen3ForCausalLM,
    )

    from .configuration_mage_vl import MageVLConfig, MageVLVisionConfig

    torch.manual_seed(59)
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
    config = MageVLConfig(text_config=text_config, vision_config=vision_config)
    engine = MageVLForConditionalGeneration(config).eval()
    baseline = TransformersQwen3ForCausalLM(text_config).eval()
    baseline.model.load_state_dict(engine.model.language_model.state_dict(), strict=True)
    baseline.lm_head.load_state_dict(engine.lm_head.state_dict(), strict=True)
    input_ids = torch.tensor([[2, 5, 13, 29]])
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        engine_output = engine(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        baseline_output = baseline(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    print("==========engine===========")
    print(f"shape={tuple(engine_output.logits.shape)}, sample={engine_output.logits[0, -1, :4].tolist()}")
    print("==========baseline==========")
    print(f"shape={tuple(baseline_output.logits.shape)}, sample={baseline_output.logits[0, -1, :4].tolist()}")
    torch.testing.assert_close(engine_output.logits, baseline_output.logits, atol=1e-5, rtol=1e-5)
    print(f"Mage-VL conditional generation text path matches the Transformers Qwen3 baseline on {test_device}.")

    # Bench：测量从 input IDs 到完整 logits 的生成模型前向。
    with torch.inference_mode():
        generation_bench = torch.utils.benchmark.Timer(
            stmt="engine(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)",
            globals=locals(),
        ).blocked_autorange(min_run_time=0.3)
    print(f"Bench: MageVLForConditionalGeneration {generation_bench}")

    # Profile：包含文本主干、LM Head 和 logits 输出，记录各算子及 shape。
    profile_activities = [torch.profiler.ProfilerActivity.CPU]
    if test_device.type == "cuda":
        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.inference_mode():
        with torch.profiler.profile(
            activities=profile_activities,
            record_shapes=True,
            profile_memory=True,
        ) as generation_profile:
            engine(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    if test_device.type == "cuda":
        torch.cuda.synchronize(test_device)
        profile_sort = "self_cuda_time_total"
    else:
        profile_sort = "self_cpu_time_total"
    print("Profile: MageVLForConditionalGeneration operators")
    print(generation_profile.key_averages().table(sort_by=profile_sort, row_limit=20))
