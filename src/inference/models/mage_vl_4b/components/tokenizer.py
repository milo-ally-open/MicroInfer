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
