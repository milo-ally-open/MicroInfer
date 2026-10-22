"""定義模型專屬聊天輸入處理器的 API Server 契約。"""

from __future__ import annotations

from typing import Any, Protocol

from engine.contracts import PreparedChatInput


class StreamOutputProcessor(Protocol):
    """將 Worker token 事件轉成可安全輸出的增量文字。"""

    def push(self, token_id: int) -> str: ...

    def finish(self) -> str: ...


class ChatInputProcessor(Protocol):
    """在 API 邊界完成模型專屬的聊天格式化與 Tokenize。"""

    def start(self) -> None: ...

    def prepare_chat(self, request_body: dict[str, Any]) -> PreparedChatInput: ...

    def create_stream_output_processor(self) -> StreamOutputProcessor: ...

    def decode_tokens(self, token_ids: tuple[int, ...]) -> str: ...

    def close(self) -> None: ...
