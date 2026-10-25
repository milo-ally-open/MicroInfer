"""定义模型专属聊天输入处理器的 API Server 契约。"""

from __future__ import annotations

from typing import Any, Protocol

from inference.engine.sequence import PreparedChatInput


class StreamOutputProcessor(Protocol):
    """将 Worker token 事件转成可安全输出的增量文字。"""

    def push(self, token_id: int) -> str: ...

    def finish(self) -> str: ...


class ChatInputProcessor(Protocol):
    """在 API 边界完成模型专属的聊天格式化与 Tokenize。"""

    def start(self) -> None: ...

    def prepare_chat(self, request_body: dict[str, Any]) -> PreparedChatInput: ...

    def create_stream_output_processor(self) -> StreamOutputProcessor: ...

    def decode_tokens(self, token_ids: tuple[int, ...]) -> str: ...

    def close(self) -> None: ...
