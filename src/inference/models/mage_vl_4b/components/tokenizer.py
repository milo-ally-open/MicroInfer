"""从 Transformers Processor 中抽出 Mage-VL 的文字 Tokenizer 边界。"""

from __future__ import annotations

from typing import Any


class MageVLTokenizer:
    """只封装词表相关操作；图片与视频预处理仍由模型 Processor 完成。"""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    @property
    def eos_token_id(self) -> Any:
        return getattr(self.backend, "eos_token_id", None)

    def apply_chat_template(self, messages: list[dict[str, Any]], **options: Any) -> Any:
        return self.backend.apply_chat_template(messages, **options)

    def encode(self, text: str, **options: Any) -> Any:
        return self.backend.encode(text, **options)

    def decode(self, token_ids: Any, **options: Any) -> str:
        if hasattr(token_ids, "detach"):
            token_ids = token_ids.detach().cpu().tolist()
        try:
            return self.backend.decode(
                token_ids,
                skip_special_tokens=options.get("skip_special_tokens", True),
                clean_up_tokenization_spaces=options.get("clean_up_tokenization_spaces", False),
            )
        except TypeError:
            return self.backend.decode(
                token_ids, skip_special_tokens=options.get("skip_special_tokens", True)
            )

    def convert_tokens_to_ids(self, token: str) -> int:
        return int(self.backend.convert_tokens_to_ids(token))


if __name__ == "__main__":
    import time

    import torch

    class FakeBackend:
        eos_token_id = 2

        def apply_chat_template(self, messages, **options):
            return "<user>hello</user>"

        def encode(self, text, **options):
            return [1, 2]

        def decode(self, token_ids, **options):
            return "hello"

        def convert_tokens_to_ids(self, token):
            return 2

    tokenizer = MageVLTokenizer(FakeBackend())
    assert tokenizer.eos_token_id == 2
    assert tokenizer.apply_chat_template([{"role": "user", "content": "hello"}]) == "<user>hello</user>"
    assert tokenizer.encode("hello") == [1, 2]
    assert tokenizer.decode([1, 2]) == "hello"
    assert tokenizer.convert_tokens_to_ids("</s>") == 2
    print("Test: MageVLTokenizer template, encode, decode and token lookup passed.")

    started_at = time.perf_counter()
    for _ in range(1000):
        tokenizer.encode("hello")
        tokenizer.decode([1, 2])
    print(f"Bench: tokenizer encode+decode={(time.perf_counter() - started_at) * 1e3 / 1000:.4f} ms/call.")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as profile:
        tokenizer.encode("hello")
        tokenizer.decode([1, 2])
    print("Profile: MageVLTokenizer")
    print(profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
