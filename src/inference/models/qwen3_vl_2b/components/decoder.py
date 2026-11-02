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
