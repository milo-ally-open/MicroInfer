"""视觉塔变长 patch 序列的非因果 Attention。"""

import torch
import torch.utils.benchmark
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionAttention as _ReferenceAttention,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
    get_max_seqlen,
    is_flash_attention_requested,
)

class Qwen3VLVisionAttention(nn.Module):
    """合并 QKV 投影，并按 cu_seqlens 隔离不同图像的 patch 序列。"""

    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        max_seqlen: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        sequence_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(sequence_length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(1, 0, 2, 3).unbind(0)

        # 先对 Q/K 施加二维视觉 RoPE，再整理成 backend 需要的布局。
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb_vision(query, key, cos, sin)
        query, key, value = (
            tensor.transpose(0, 1).unsqueeze(0)
            for tensor in (query, key, value)
        )

        attention = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        dropout = 0.0 if not self.training else self.attention_dropout

        if is_flash_attention_requested(self.config):
            # FlashAttention 直接消费边界数组，支持变长图像序列。
            max_seqlen = get_max_seqlen(cu_seqlens, self.config, kwargs={"max_seqlen": max_seqlen})
            output, _ = attention(
                self,
                query,
                key,
                value,
                attention_mask=None,
                scaling=self.scaling,
                dropout=dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # 其他 backend 逐图运行，防止图像之间发生注意力信息混合。
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            query_splits, key_splits, value_splits = [
                torch.split(tensor, lengths, dim=2)
                for tensor in (query, key, value)
            ]
            outputs = [
                attention(
                    self,
                    query_part,
                    key_part,
                    value_part,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for query_part, key_part, value_part in zip(query_splits, key_splits, value_splits)
            ]
            output = torch.cat(outputs, dim=1)

        output = output.reshape(sequence_length, -1).contiguous()
        return self.proj(output)

if __name__ == "__main__":
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    config = Qwen3VLVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=64,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "axial"},
    )
    config._attn_implementation = "sdpa"

    reference = _ReferenceAttention(config)
    local = Qwen3VLVisionAttention(config)
    local.load_state_dict(reference.state_dict(), strict=True)
    local.to(device).eval()
    reference.to(device).eval()

    sequence_length = 8
    hidden_states = torch.randn(sequence_length, config.hidden_size, device=device)
    cos = torch.ones(sequence_length, config.hidden_size // config.num_heads, device=device)
    sin = torch.zeros_like(cos)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)

    components = [
        (
            "vision attention",
            reference,
            local,
            (hidden_states, cu_seqlens, (cos, sin)),
            (hidden_states, cu_seqlens, (cos, sin)),
            {},
            {},
            1e-5,
            1e-5,
        ),
    ]

    profile_activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
    profile_sort = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"

    for component_name, baseline, enhanced, baseline_args, enhanced_args, baseline_kwargs, enhanced_kwargs, atol, rtol in components:
        baseline_output = None
        baseline_median = None
        for stage, module, args, kwargs in (
            ("baseline", baseline, baseline_args, baseline_kwargs),
            ("enhanced", enhanced, enhanced_args, enhanced_kwargs),
        ):
            print(f"================= {stage} =================")
            print("   test:")
            print(f"       component={component_name}")
            with torch.inference_mode():
                output = module(*args, **kwargs)
            if stage == "baseline":
                torch.testing.assert_close(output, output, atol=atol, rtol=rtol)
                baseline_output = output
                output_tensor = output if isinstance(output, torch.Tensor) else output[0]
                print(f"       PASS, shape={tuple(output_tensor.shape)}, dtype={output_tensor.dtype}, device={output_tensor.device}")
            else:
                torch.testing.assert_close(output, baseline_output, atol=atol, rtol=rtol)
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
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            print(profile.key_averages().table(sort_by=profile_sort, row_limit=12))
