"""管理 Qwen3 词表，并提供模型专属的对话 Tokenize 边界。"""

from __future__ import annotations

from typing import Any


class Qwen3Tokenizer:
    """封装 Hugging Face Tokenizer；不负责请求校验或生成参数。"""

    def __init__(self, backend: Any = None) -> None:
        self.backend = backend

    def start(self) -> None:
        if self.backend is None:
            from ..utils.loader import MODEL_REVISION, LOCAL_MODEL_PATH, load_tokenizer

            self.backend = load_tokenizer(LOCAL_MODEL_PATH, MODEL_REVISION)

    def close(self) -> None:
        self.backend = None

    def tokenize_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
    ) -> tuple[int, ...]:
        tokenizer = self.require_backend()
        options: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "enable_thinking": enable_thinking,
        }
        if tools:
            options["tools"] = tools
        try:
            encoded = tokenizer.apply_chat_template(messages, **options)
        except Exception as exc:
            raise ValueError(f"Could not format Qwen3 chat messages: {exc}") from exc
        raw_ids = encoded["input_ids"]
        if hasattr(raw_ids, "tolist"):
            raw_ids = raw_ids.tolist()
        if raw_ids and isinstance(raw_ids[0], list):
            if len(raw_ids) != 1:
                raise ValueError("Qwen3 Tokenizer expects one unpadded chat prompt.")
            raw_ids = raw_ids[0]
        token_ids = tuple(int(token_id) for token_id in raw_ids)
        if not token_ids:
            raise ValueError("Qwen3 chat template produced an empty token sequence.")
        return token_ids

    def decode(self, token_ids: tuple[int, ...] | list[int]) -> str:
        tokenizer = self.require_backend()
        return tokenizer.decode(token_ids, skip_special_tokens=True)

    def require_backend(self) -> Any:
        if self.backend is None:
            raise RuntimeError("Qwen3 Tokenizer has not been started.")
        return self.backend
