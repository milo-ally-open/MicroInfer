"""文字 token Embedding 与词表输出 Linear。"""

import torch
import torch.utils.benchmark
from torch import nn

class Qwen3VLTextEmbedding(nn.Embedding):
    """将 token id 查表映射为 hidden state，保持官方参数布局。"""

class Qwen3VLLMHead(nn.Linear):
    """将最终 hidden state 投影为词表 logits。"""

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    # Embedding 和 LM Head 分别与模型使用的 PyTorch 基础模块对照。
    embedding_reference = nn.Embedding(128, 64).to(device)
    embedding = Qwen3VLTextEmbedding(128, 64).to(device)
    embedding.load_state_dict(embedding_reference.state_dict(), strict=True)
    token_ids = torch.randint(0, 128, (2, 7), device=device)

    head_reference = nn.Linear(64, 128, bias=False).to(device)
    lm_head = Qwen3VLLMHead(64, 128, bias=False).to(device)
    lm_head.load_state_dict(head_reference.state_dict(), strict=True)
    hidden_states = torch.randn(2, 7, 64, device=device)

    components = [
        ("token embedding", embedding_reference, embedding, (token_ids,), (token_ids,), {}, {}, 1e-5, 1e-5),
        ("LM head", head_reference, lm_head, (hidden_states,), (hidden_states,), {}, {}, 1e-5, 1e-5),
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
