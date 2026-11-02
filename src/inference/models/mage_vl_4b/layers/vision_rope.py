"""Mage-VL 视觉三维旋转位置编码。"""

# 本模块沿用 microsoft/Mage-VL revision d88b153285f1633a61b2f693c59c8576693af185（Apache-2.0）实现。

import math
import torch
import torch.utils.benchmark
import torch.nn as nn
from ..architecture.configuration_mage_vl import MageVLVisionConfig


class VisionRotaryEmbedding(nn.Module):
    """
    3D (T,H,W) Rotary frequency constructor with 4:6:6 split.
    Supports both grid_thw-based and explicit position-based RoPE computation.
    """

    def __init__(self, config: MageVLVisionConfig):
        super().__init__()
        if config.num_attention_heads < 1 or config.hidden_size < 1:
            raise ValueError("Vision hidden size and attention head count must be positive.")
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("Vision hidden size must be divisible by the attention head count.")
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta

        if head_dim % 16:
            raise ValueError("Vision rotary head_dim must be divisible by 16.")
        half = head_dim // 2
        if half % 16:
            raise ValueError("Half of vision rotary head_dim must be divisible by 16 for the 4:6:6 split.")
        if not math.isfinite(base) or base <= 0:
            raise ValueError("Vision rotary theta must be finite and positive.")

        self.head_dim = head_dim
        self.half = half
        self.base = base

        # 4:6:6 split for T:H:W
        unit = half // 16
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit

        self.register_buffer(
            "inv_freq_t",
            1.0 / (base ** (torch.arange(self.t_size, dtype=torch.float32) / self.t_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_h",
            1.0 / (base ** (torch.arange(self.h_size, dtype=torch.float32) / self.h_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_w",
            1.0 / (base ** (torch.arange(self.w_size, dtype=torch.float32) / self.w_size)),
            persistent=False,
        )

    def forward(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Compute rotary position embeddings from grid_thw (Qwen2VL style).

        Args:
            grid_thw: [num_samples, 3] tensor with [t, h, w] for each sample

        Returns:
            freqs: [total_seq_len, half] tensor of position frequencies
        """
        device = grid_thw.device
        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        all_freqs = []
        for sample_thw in grid_thw:
            t, h, w = sample_thw[0].item(), sample_thw[1].item(), sample_thw[2].item()

            # Compute frequency tables
            ft = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv_t)
            fh = torch.outer(torch.arange(h, device=device, dtype=torch.float32), inv_h)
            fw = torch.outer(torch.arange(w, device=device, dtype=torch.float32), inv_w)

            # Build position indices for this sample
            t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
            h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
            w_ids = torch.arange(w, device=device).repeat(h).repeat(t)

            # Concatenate frequencies: [seq_len, half]
            sample_freqs = torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)
            all_freqs.append(sample_freqs)

        return torch.cat(all_freqs, dim=0)

    def forward_from_positions(self, patch_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute rotary position embeddings from explicit patch positions.

        Args:
            patch_positions: [seq_len, 3] tensor with [t, h, w] positions for each patch

        Returns:
            freqs: [seq_len, half] tensor of position frequencies
        """
        device = patch_positions.device
        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        t_pos = patch_positions[:, 0].float()
        h_pos = patch_positions[:, 1].float()
        w_pos = patch_positions[:, 2].float()

        ft = torch.outer(t_pos, inv_t)
        fh = torch.outer(h_pos, inv_h)
        fw = torch.outer(w_pos, inv_w)

        return torch.cat([ft, fh, fw], dim=-1)

    def forward_with_thw(self, t: int, h: int, w: int, device=None) -> torch.Tensor:
        """
        Compute rotary position embeddings from explicit t, h, w dimensions.

        Args:
            t: Number of temporal frames
            h: Number of height patches
            w: Number of width patches
            device: Target device

        Returns:
            freqs: [t*h*w, half] tensor of position frequencies
        """
        if device is None:
            device = self.inv_freq_t.device

        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        ft = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv_t)
        fh = torch.outer(torch.arange(h, device=device, dtype=torch.float32), inv_h)
        fw = torch.outer(torch.arange(w, device=device, dtype=torch.float32), inv_w)

        t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
        h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
        w_ids = torch.arange(w, device=device).repeat(h).repeat(t)

        freqs = torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)
        return freqs


def rotate_half(x):
    """
    Interleaved rotation to match Source model's implementation.
    (x1, x2, x3, x4) -> (-x2, x1, -x4, x3)
    """
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rotary_pos_emb(q, k, freqs):
    # q, k: (B, H, L, D)
    # freqs: (B, L, D)
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    # We need to broadcast freqs to match heads
    # (B, L, D) -> (B, 1, L, D)
    # Keep the same dtype as q, k to avoid memory doubling from float32 promotion
    cos = freqs.cos().unsqueeze(1).float()
    sin = freqs.sin().unsqueeze(1).float()

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
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

    torch.manual_seed(61)
    config = MageVLVisionConfig(hidden_size=64, num_attention_heads=2, rope_theta=10000.0)
    reference_config = official.MageVLVisionConfig(hidden_size=64, num_attention_heads=2, rope_theta=10000.0)
    engine = VisionRotaryEmbedding(config).eval()
    baseline = official.VisionRotaryEmbedding(reference_config).eval()
    grid_thw = torch.tensor([[2, 2, 2], [1, 2, 2]])
    positions = torch.tensor(
        [[t, h, w] for t, h, w in ((t, h, w) for t in range(2) for h in range(2) for w in range(2))]
        + [[0, h, w] for h in range(2) for w in range(2)],
        dtype=torch.long,
    )
    rotate_input = torch.randn(1, 2, 8, 32)
    query = torch.randn(1, 2, 8, 32)
    key = torch.randn_like(query)
    rotary_frequencies = torch.randn(1, 8, 32)

    components = [
        ("Mage-VL vision RoPE", baseline, engine, (grid_thw,), (grid_thw,), {}, {}, None, None, 0, 0, False),
        (
            "Mage-VL rotate_half",
            official.rotate_half,
            rotate_half,
            (rotate_input,),
            (rotate_input,),
            {},
            {},
            None,
            None,
            0,
            0,
            False,
        ),
        (
            "Mage-VL apply_rotary_pos_emb",
            official.apply_rotary_pos_emb,
            apply_rotary_pos_emb,
            (query, key, rotary_frequencies),
            (query, key, rotary_frequencies),
            {},
            {},
            None,
            None,
            1e-6,
            1e-6,
            False,
        ),
        (
            "Mage-VL RoPE from positions",
            baseline.forward_from_positions,
            engine.forward_from_positions,
            (positions,),
            (positions,),
            {},
            {},
            None,
            None,
            0,
            0,
            False,
        ),
        (
            "Mage-VL RoPE from THW",
            baseline.forward_with_thw,
            engine.forward_with_thw,
            (2, 2, 2),
            (2, 2, 2),
            {},
            {},
            None,
            None,
            0,
            0,
            False,
        ),
    ]
    profile_activities = [torch.profiler.ProfilerActivity.CPU]
    if test_device.type == "cuda":
        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
    profile_sort = "self_cuda_time_total" if test_device.type == "cuda" else "self_cpu_time_total"

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
            if isinstance(selector, str):
                output = raw_output[selector] if isinstance(raw_output, dict) else getattr(raw_output, selector)
            elif selector is not None:
                output = raw_output[selector]
            else:
                output = raw_output
            if stage == "baseline":
                torch.testing.assert_close(output, output, atol=atol, rtol=rtol)
                baseline_output, baseline_raw_output = output, raw_output
                output_tensor = output if isinstance(output, torch.Tensor) else output[0]
                print(f"       PASS, shape={tuple(output_tensor.shape)}, dtype={output_tensor.dtype}, device={output_tensor.device}")
            else:
                torch.testing.assert_close(output, baseline_output, atol=atol, rtol=rtol)
                if compare_hidden_states:
                    torch.testing.assert_close(
                        tuple(raw_output.hidden_states), tuple(baseline_raw_output.hidden_states), atol=atol, rtol=rtol
                    )
                if isinstance(output, torch.Tensor):
                    max_abs_error = (output - baseline_output).abs().max().item()
                    output_tensor = output
                else:
                    max_abs_error = max(
                        (
                            (actual - expected).abs().max().item()
                            for actual, expected in zip(output, baseline_output)
                            if isinstance(actual, torch.Tensor)
                        ),
                        default=0.0,
                    )
                    output_tensor = output[0]
                print(f"       parity=PASS, max_abs_error={max_abs_error:.3e}, shape={tuple(output_tensor.shape)}")
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
            if test_device.type == "cuda":
                torch.cuda.synchronize(test_device)
            print(profile.key_averages().table(sort_by=profile_sort, row_limit=12))
