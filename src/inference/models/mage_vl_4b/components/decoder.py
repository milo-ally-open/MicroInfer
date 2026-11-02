"""将完整 Mage-VL 输出 token 序列解码为响应文字。"""

from __future__ import annotations

from typing import Any

from .detokenizer import MageVLDetokenizer
from .tokenizer import MageVLTokenizer


class MageVLDecoder:
    """提供整段输出解码；流式输出由独立 MageVLDetokenizer 负责。"""

    def __init__(self, tokenizer: MageVLTokenizer) -> None:
        self.tokenizer = tokenizer

    def decode(self, token_ids: Any) -> str:
        if hasattr(token_ids, "detach"):
            token_ids = token_ids.detach().cpu().tolist()
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise ValueError("Mage-VL Decoder expects one generated sequence.")
            token_ids = token_ids[0]
        detokenizer = MageVLDetokenizer(self.tokenizer)
        chunks = [detokenizer.push(int(token_id)) for token_id in token_ids]
        chunks.append(detokenizer.finish())
        return "".join(chunks).strip()


if __name__ == "__main__":
    import time

    import torch

    class FakeBackend:
        def decode(self, token_ids, **options):
            return "hello"

    decoder = MageVLDecoder(MageVLTokenizer(FakeBackend()))
    assert decoder.decode([1, 2]) == "hello"
    print("Test: MageVLDecoder full sequence decode passed.")

    started_at = time.perf_counter()
    for _ in range(1000):
        decoder.decode([1, 2])
    print(f"Bench: decoder={(time.perf_counter() - started_at) * 1e3 / 1000:.4f} ms/call.")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as profile:
        decoder.decode([1, 2])
    print("Profile: MageVLDecoder")
    print(profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
