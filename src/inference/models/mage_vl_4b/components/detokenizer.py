"""按生成顺序把 Mage-VL token IDs 增量还原为稳定文本片段。"""

from __future__ import annotations

from inference.network.output_processor import IncrementalDetokenizer


class MageVLDetokenizer(IncrementalDetokenizer):
    """Mage-VL 的 SSE 文本边界，缓冲跨 token 的不完整字元。"""


if __name__ == "__main__":
    import time

    import torch

    class FakeTokenizer:
        def decode(self, token_ids, **options):
            return "hello"[:len(token_ids)]

    detokenizer = MageVLDetokenizer(FakeTokenizer())
    text = detokenizer.push(1) + detokenizer.push(2) + detokenizer.finish()
    assert text == "he"
    print("Test: MageVLDetokenizer stable incremental output passed.")

    started_at = time.perf_counter()
    for _ in range(1000):
        current = MageVLDetokenizer(FakeTokenizer())
        current.push(1)
        current.push(2)
        current.finish()
    print(f"Bench: detokenizer two-token stream={(time.perf_counter() - started_at) * 1e3 / 1000:.4f} ms/request.")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as profile:
        current = MageVLDetokenizer(FakeTokenizer())
        current.push(1)
        current.push(2)
        current.finish()
    print("Profile: MageVLDetokenizer")
    print(profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
