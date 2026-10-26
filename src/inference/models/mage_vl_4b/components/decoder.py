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
