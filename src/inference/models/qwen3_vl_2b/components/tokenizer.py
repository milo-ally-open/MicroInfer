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
