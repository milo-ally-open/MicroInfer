"""将完整 Qwen3 输出 token 序列解码为响应文字。"""

from __future__ import annotations

from typing import Any

from .detokenizer import Qwen3Detokenizer
from .tokenizer import Qwen3Tokenizer


class Qwen3Decoder:
    """提供完整输出解码与逐 token 流式解码两个入口。"""

    def __init__(self, tokenizer: Qwen3Tokenizer | Any) -> None:
        self.tokenizer = tokenizer if isinstance(tokenizer, Qwen3Tokenizer) else Qwen3Tokenizer(tokenizer)

    def create_detokenizer(self) -> Qwen3Detokenizer:
        return Qwen3Detokenizer(self.tokenizer.require_backend())

    def decode(self, token_ids: tuple[int, ...]) -> str:
        detokenizer = self.create_detokenizer()
        chunks = [detokenizer.push(token_id) for token_id in token_ids]
        chunks.append(detokenizer.finish())
        return "".join(chunks)
