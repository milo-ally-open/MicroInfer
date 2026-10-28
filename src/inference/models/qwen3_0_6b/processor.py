"""组合 Qwen3 Tokenizer、Encoder、Decoder，维持 API Server 处理器接口。"""

from __future__ import annotations

from typing import Any

from inference.network.input_processor import ChatInputProcessor, StreamOutputProcessor
from inference.network.tool_calling import normalize_chat_messages
from inference.engine.sequence import InferenceError, PreparedChatInput

from .components.decoder import Qwen3Decoder
from .components.encoder import Qwen3Encoder, build_sampling_params, prepare_chat_input
from .components.tokenizer import Qwen3Tokenizer


class Qwen3InputProcessor(ChatInputProcessor):
    """API 兼容门面；具体编码、分词、解码和增量解码由独立组件负责。"""

    def __init__(self) -> None:
        self.tokenizer = Qwen3Tokenizer()
        self.encoder: Qwen3Encoder | None = None
        self.decoder: Qwen3Decoder | None = None

    def start(self) -> None:
        if self.encoder is None or self.decoder is None:
            self.tokenizer.start()
            self.encoder = Qwen3Encoder()
            self.decoder = Qwen3Decoder(self.tokenizer)

    def close(self) -> None:
        self.tokenizer.close()
        self.encoder = None
        self.decoder = None

    def prepare_chat(self, request_body: dict[str, Any]) -> PreparedChatInput:
        if self.encoder is None:
            raise RuntimeError("Qwen3 API input processor has not been started.")
        try:
            plan = self.encoder.prepare(request_body)
            token_ids = self.tokenizer.tokenize_chat(
                plan.messages,
                tools=plan.template_tools,
                enable_thinking=plan.enable_thinking,
            )
            return self.encoder.build(plan, token_ids)
        except ValueError as exc:
            raise InferenceError(400, str(exc)) from exc

    def create_stream_output_processor(self) -> StreamOutputProcessor:
        if self.decoder is None:
            raise RuntimeError("Qwen3 API input processor has not been started.")
        return self.decoder.create_detokenizer()

    def decode_tokens(self, token_ids: tuple[int, ...]) -> str:
        if self.decoder is None:
            raise RuntimeError("Qwen3 API input processor has not been started.")
        return self.decoder.decode(token_ids)


def normalize_text_messages(raw_messages: Any) -> list[dict[str, Any]]:
    """兼容旧名称；实际消息标准化由共用聊天协议组件完成。"""
    return normalize_chat_messages(raw_messages)


__all__ = [
    "Qwen3InputProcessor",
    "Qwen3Tokenizer",
    "Qwen3Encoder",
    "Qwen3Decoder",
    "build_sampling_params",
    "prepare_chat_input",
    "normalize_text_messages",
]
