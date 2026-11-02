"""提供完整序列解码和逐 token Detokenizer 工厂。"""

from __future__ import annotations

from typing import Any

from .detokenizer import Qwen3VLDetokenizer
from .tokenizer import Qwen3VLTokenizer


class Qwen3VLDecoder:
    def __init__(self, tokenizer: Qwen3VLTokenizer) -> None:
        self.tokenizer = tokenizer

    def decode(self, token_ids: Any) -> str:
        return self.tokenizer.decode(token_ids)

    def create_detokenizer(self) -> Qwen3VLDetokenizer:
        return Qwen3VLDetokenizer(self.tokenizer.tokenizer)


if __name__ == "__main__":
    import time
    from types import SimpleNamespace

    import torch

    class FakeBackend:
        def decode(self, token_ids, **options):
            return "hello"

    tokenizer = Qwen3VLTokenizer(SimpleNamespace(tokenizer=FakeBackend()))
    decoder = Qwen3VLDecoder(tokenizer)
    assert decoder.decode([1, 2]) == "hello"
    assert decoder.create_detokenizer() is not None
    print("Test: Qwen3VLDecoder full decode and detokenizer construction passed.")

    started_at = time.perf_counter()
    for _ in range(1000):
        decoder.decode([1, 2])
    print(f"Bench: decoder={(time.perf_counter() - started_at) * 1e3 / 1000:.4f} ms/call.")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as profile:
        decoder.decode([1, 2])
    print("Profile: Qwen3VLDecoder")
    print(profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
