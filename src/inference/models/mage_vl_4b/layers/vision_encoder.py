"""Mage-VL 视觉组件兼容导出与 CUDA 优先、CPU 回退的前向对照入口。"""

# 组件实现拆分自 microsoft/Mage-VL revision d88b153285f1633a61b2f693c59c8576693af185（Apache-2.0）。

import torch
import torch.utils.benchmark

from .vision_rope import VisionRotaryEmbedding, apply_rotary_pos_emb, rotate_half
from .vision_embeddings import MageVLVisionEmbeddings, MageVLVisionPatchMerger
from .vision_attention import MageVLVisionAttention, eager_attention_forward, get_norm_layer
from .vision_stack import MageVLVisionEncoder, MageVLVisionEncoderLayer

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    import importlib.util
    import sys
    import types
    from pathlib import Path

    from ..architecture.configuration_mage_vl import MageVLVisionConfig

    official_snapshot = Path(__file__).resolve().parents[5] / "checkpoints" / "microsoft-Mage-VL-d88b153"
    official_package_name = "_mage_vl_official_d88b153"
    official_package = types.ModuleType(official_package_name)
    official_package.__path__ = [str(official_snapshot)]
    sys.modules[official_package_name] = official_package
    official_name = f"{official_package_name}.modeling_mage_vl"
    official_spec = importlib.util.spec_from_file_location(official_name, official_snapshot / "modeling_mage_vl.py")
    if official_spec is None or official_spec.loader is None:
        raise RuntimeError(f"Unable to load official Mage-VL source: {official_snapshot}")
    official = importlib.util.module_from_spec(official_spec)
    sys.modules[official_name] = official
    official_spec.loader.exec_module(official)

    # 聚合入口对照完整视觉编码器；单个算子的基线在各自模块中测试。
    config = MageVLVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        attention_dropout=0.0,
    )
    baseline_config = official.MageVLVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    baseline_config._attn_implementation = "eager"
    baseline = official.MageVLVisionEncoder(baseline_config).to(device).eval()
    enhanced = MageVLVisionEncoder(config).to(device).eval()
    enhanced.load_state_dict(baseline.state_dict(), strict=True)
    hidden_states = torch.randn(1, 8, config.hidden_size, device=device)
    grid_thw = torch.tensor([[2, 2, 2]], device=device)
    frequencies = official.VisionRotaryEmbedding(baseline_config)(grid_thw)
    rotary_pos_emb = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(0)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
    call_kwargs = {
        "rotary_pos_emb": rotary_pos_emb,
        "output_hidden_states": True,
        "return_dict": True,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": 4,
    }
    components = [
        (
            "Mage-VL vision encoder aggregate",
            baseline,
            enhanced,
            (hidden_states,),
            (hidden_states,),
            call_kwargs,
            call_kwargs,
            "last_hidden_state",
            "last_hidden_state",
            1e-6,
            1e-6,
            True,
        ),
    ]
    profile_activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
    profile_sort = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"

    for (
        component_name,
        baseline,
        enhanced,
        baseline_args,
        enhanced_args,
        baseline_kwargs,
        enhanced_kwargs,
        baseline_selector,
        enhanced_selector,
        atol,
        rtol,
        compare_hidden_states,
    ) in components:
        baseline_output = None
        baseline_raw_output = None
        baseline_median = None
        for stage, module, args, kwargs, selector in (
            ("baseline", baseline, baseline_args, baseline_kwargs, baseline_selector),
            ("enhanced", enhanced, enhanced_args, enhanced_kwargs, enhanced_selector),
        ):
            print(f"================= {stage} =================")
            print("   test:")
            print(f"       component={component_name}")
            with torch.inference_mode():
                raw_output = module(*args, **kwargs)
            output = getattr(raw_output, selector) if isinstance(selector, str) else raw_output
            if stage == "baseline":
                torch.testing.assert_close(output, output, atol=atol, rtol=rtol)
                baseline_output, baseline_raw_output = output, raw_output
                print(f"       PASS, shape={tuple(output.shape)}, dtype={output.dtype}, device={output.device}")
            else:
                torch.testing.assert_close(output, baseline_output, atol=atol, rtol=rtol)
                if compare_hidden_states:
                    torch.testing.assert_close(
                        tuple(raw_output.hidden_states),
                        tuple(baseline_raw_output.hidden_states),
                        atol=atol,
                        rtol=rtol,
                    )
                max_abs_error = (output - baseline_output).abs().max().item()
                print(f"       parity=PASS, max_abs_error={max_abs_error:.3e}, shape={tuple(output.shape)}")

            print("   bench:")
            with torch.inference_mode():
                bench = torch.utils.benchmark.Timer(
                    stmt="module(*args, **kwargs)",
                    globals={"module": module, "args": args, "kwargs": kwargs},
                ).blocked_autorange(min_run_time=0.2)
            if stage == "baseline":
                baseline_median = bench.median
                print(f"       median_ms={bench.median * 1000:.4f}, iqr_ms={bench.iqr * 1000:.4f}")
            else:
                print(
                    f"       median_ms={bench.median * 1000:.4f}, iqr_ms={bench.iqr * 1000:.4f}, speedup={baseline_median / bench.median:.3f}x"
                )

            print("   profile:")
            with torch.inference_mode():
                with torch.profiler.profile(
                    activities=profile_activities,
                    record_shapes=True,
                    profile_memory=True,
                ) as profile:
                    module(*args, **kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            print(profile.key_averages().table(sort_by=profile_sort, row_limit=15))
