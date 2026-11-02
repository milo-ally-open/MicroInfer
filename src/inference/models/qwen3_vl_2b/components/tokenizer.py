"""通过 Hugging Face AutoProcessor 执行 Qwen3-VL 官方 Chat Template 与 tokenize。"""

from __future__ import annotations

from typing import Any


class Qwen3VLTokenizer:
    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def tokenize_chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    def decode(self, token_ids: Any) -> str:
        if hasattr(token_ids, "detach"):
            token_ids = token_ids.detach().cpu().tolist()
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise ValueError("Qwen3-VL decoder expects one generated sequence.")
            token_ids = token_ids[0]
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()


if __name__ == "__main__":
    import time
    from types import SimpleNamespace

    import torch

    class FakeBackend:
        def decode(self, token_ids, **options):
            return "hello"

    class FakeProcessor:
        tokenizer = FakeBackend()

        def apply_chat_template(self, messages, **options):
            assert options["add_generation_prompt"] is True
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    tokenizer = Qwen3VLTokenizer(FakeProcessor())
    messages = [{"role": "user", "content": "hello"}]
    encoded = tokenizer.tokenize_chat(messages)
    decoded = tokenizer.decode([1, 2])
    assert encoded["input_ids"].tolist() == [[1, 2, 3]]
    assert decoded == "hello"
    print("Test: Qwen3VLTokenizer chat template and decode passed.")

    started_at = time.perf_counter()
    for _ in range(1000):
        tokenizer.tokenize_chat(messages)
        tokenizer.decode([1, 2])
    print(f"Bench: tokenizer encode+decode={(time.perf_counter() - started_at) * 1e3 / 1000:.4f} ms/call.")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as profile:
        tokenizer.tokenize_chat(messages)
        tokenizer.decode([1, 2])
    print("Profile: Qwen3VLTokenizer")
    print(profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
