"""把连续生成的 Qwen3-VL token IDs 稳定地转换为文本 delta。"""

from inference.network.output_processor import IncrementalDetokenizer


class Qwen3VLDetokenizer(IncrementalDetokenizer):
    """缓冲跨 token 的不完整 UTF-8/Unicode 片段，避免 SSE 输出乱码。"""


if __name__ == "__main__":
    import time

    import torch

    class FakeTokenizer:
        def decode(self, token_ids, **options):
            return "hello"[:len(token_ids)]

    detokenizer = Qwen3VLDetokenizer(FakeTokenizer())
    output = detokenizer.push(1) + detokenizer.push(2) + detokenizer.finish()
    assert output == "he"
    print("Test: Qwen3VLDetokenizer stable incremental output passed.")

    started_at = time.perf_counter()
    for _ in range(1000):
        current = Qwen3VLDetokenizer(FakeTokenizer())
        current.push(1)
        current.push(2)
        current.finish()
    print(f"Bench: detokenizer two-token stream={(time.perf_counter() - started_at) * 1e3 / 1000:.4f} ms/request.")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as profile:
        current = Qwen3VLDetokenizer(FakeTokenizer())
        current.push(1)
        current.push(2)
        current.finish()
    print("Profile: Qwen3VLDetokenizer")
    print(profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
