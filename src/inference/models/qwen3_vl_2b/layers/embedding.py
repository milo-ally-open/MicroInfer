"""文字 token Embedding 与词表输出 Linear。"""

import time

import torch
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

    with torch.inference_mode():
        expected_embedding = embedding_reference(token_ids)
        actual_embedding = embedding(token_ids)
        expected_logits = head_reference(hidden_states)
        actual_logits = lm_head(hidden_states)

    torch.testing.assert_close(actual_embedding, expected_embedding, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_logits, expected_logits, atol=1e-5, rtol=1e-5)
    print(f"Qwen3VL embedding/head parity passed on {device}.")

    # Bench：分别测量 token Embedding 和 LM Head 的平均前向延迟。
    for _ in range(5):
        embedding(token_ids)
        lm_head(hidden_states)
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_event.record()
        for _ in range(50):
            embedding(token_ids)
            lm_head(hidden_states)
        end_event.record()
        end_event.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event) / 100
    else:
        started_at = time.perf_counter()
        for _ in range(50):
            embedding(token_ids)
            lm_head(hidden_states)
        elapsed_ms = (time.perf_counter() - started_at) * 1000 / 100
    print(f"Bench: embedding+lm_head average={elapsed_ms:.4f} ms/call on {device}.")

    # Profile：采集两个组件的算子、输入 shape 和显存活动。
    profile_activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=profile_activities,
        record_shapes=True,
        profile_memory=True,
    ) as profile:
        embedding(token_ids)
        lm_head(hidden_states)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        profile_sort = "self_cuda_time_total"
    else:
        profile_sort = "self_cpu_time_total"
    print("Profile: embedding + lm_head operators")
    print(profile.key_averages().table(sort_by=profile_sort, row_limit=12))
