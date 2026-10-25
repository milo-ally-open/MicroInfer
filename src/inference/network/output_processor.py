"""提供跨模型共用的增量 Token Detokenizer。"""

from __future__ import annotations

from typing import Any


class IncrementalDetokenizer:
    """以累积 token IDs 增量解码，缓冲不完整的 Unicode replacement。"""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self._token_ids: list[int] = []
        self._emitted_chars = 0
        self._emitted_text = ""

    def push(self, token_id: int) -> str:
        self._token_ids.append(int(token_id))
        text = self._decode().strip()
        self._validate_emitted_prefix(text)
        replacement_index = text.find("\ufffd")
        stable_text = text if replacement_index < 0 else text[:replacement_index]
        if len(stable_text) <= self._emitted_chars:
            return ""
        delta = stable_text[self._emitted_chars :]
        self._emitted_chars = len(stable_text)
        self._emitted_text += delta
        return delta

    def finish(self) -> str:
        text = self._decode().strip()
        self._validate_emitted_prefix(text)
        delta = text[self._emitted_chars :]
        self._emitted_chars = len(text)
        self._emitted_text += delta
        return delta

    def _decode(self) -> str:
        try:
            return self.tokenizer.decode(
                self._token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self.tokenizer.decode(self._token_ids, skip_special_tokens=True)

    def _validate_emitted_prefix(self, text: str) -> None:
        if len(text) < self._emitted_chars or not text.startswith(self._emitted_text):
            raise RuntimeError("Tokenizer revised text that had already been streamed to the client.")
