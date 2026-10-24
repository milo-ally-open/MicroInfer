"""在 API Server 将 OpenAI 聊天请求转为 Qwen3 模型输入。"""

from __future__ import annotations

from typing import Any
from transformers import PreTrainedTokenizerBase

from api.input_processor import ChatInputProcessor, StreamOutputProcessor
from api.output_processor import IncrementalDetokenizer
from engine.contracts import PreparedChatInput
from engine.errors import InferenceError

from .utils import loader as config
from .utils.loader import load_tokenizer


class Qwen3InputProcessor(ChatInputProcessor):
    """载入 CPU Tokenizer，准备 prompt 与 sampling 参数，不载入模型权重。"""

    def __init__(self) -> None:
        self.tokenizer: PreTrainedTokenizerBase | None = None

    def start(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = load_tokenizer(config.LOCAL_MODEL_PATH, config.MODEL_REVISION)

    def close(self) -> None:
        self.tokenizer = None

    def prepare_chat(self, request_body: dict[str, Any]) -> PreparedChatInput:
        if self.tokenizer is None:
            raise RuntimeError("Qwen3 API input processor has not been started.")
        try:
            return prepare_chat_input(self.tokenizer, request_body)
        except ValueError as exc:
            raise InferenceError(400, str(exc)) from exc

    def create_stream_output_processor(self) -> StreamOutputProcessor:
        if self.tokenizer is None:
            raise RuntimeError("Qwen3 API input processor has not been started.")
        return IncrementalDetokenizer(self.tokenizer)

    def decode_tokens(self, token_ids: tuple[int, ...]) -> str:
        output_processor = self.create_stream_output_processor()
        chunks = [output_processor.push(token_id) for token_id in token_ids]
        chunks.append(output_processor.finish())
        return "".join(chunks)


def build_sampling_params(request_body: dict[str, Any]) -> dict[str, Any]:
    defaults = config.GENERATION_CONFIG
    raw_max_tokens = request_body.get("max_completion_tokens", request_body.get("max_tokens"))
    max_new_tokens = int(raw_max_tokens if raw_max_tokens is not None else defaults["max_new_tokens"])
    if max_new_tokens < 1:
        raise ValueError("max_tokens must be positive.")
    num_beams = int(request_body.get("num_beams", defaults.get("num_beams", 1)))
    if num_beams != 1:
        raise ValueError("Qwen3 stepwise decoding supports num_beams=1.")
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": float(request_body.get("temperature", defaults.get("temperature", 0.0))),
        "top_p": float(request_body.get("top_p", defaults.get("top_p", 1.0))),
        "top_k": int(request_body.get("top_k", defaults.get("top_k", 0))),
        "min_p": request_body.get("min_p", defaults.get("min_p")),
        "repetition_penalty": float(
            request_body.get("repetition_penalty", defaults.get("repetition_penalty", 1.0))
        ),
        "no_repeat_ngram_size": int(
            request_body.get("no_repeat_ngram_size", defaults.get("no_repeat_ngram_size", 0))
        ),
    }


def prepare_chat_input(tokenizer: PreTrainedTokenizerBase, request_body: dict[str, Any]) -> PreparedChatInput:
    messages = normalize_text_messages(request_body.get("messages"))
    options = build_sampling_params(request_body)
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            enable_thinking=bool(config.GENERATION_CONFIG.get("enable_thinking", False)),
        )
    except Exception as exc:
        raise ValueError(f"Could not format Qwen3 chat messages: {exc}") from exc

    raw_ids = inputs["input_ids"]
    if hasattr(raw_ids, "tolist"):
        raw_ids = raw_ids.tolist()
    if raw_ids and isinstance(raw_ids[0], list):
        if len(raw_ids) != 1:
            raise ValueError("Qwen3 chat input processor expected exactly one prompt.")
        raw_ids = raw_ids[0]
    token_ids = tuple(int(token_id) for token_id in raw_ids)
    if not token_ids:
        raise ValueError("Qwen3 chat template produced an empty prompt.")
    if len(token_ids) + options["max_new_tokens"] > config.MAX_CONTEXT_TOKENS:
        raise ValueError(
            f"The prompt and requested output exceed the configured {config.MAX_CONTEXT_TOKENS}-token context limit."
        )
    return PreparedChatInput(
        model_alias=config.MODEL_ALIAS,
        input_ids=token_ids,
        sampling_params=options,
    )


def normalize_text_messages(raw_messages: Any) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty list.")
    messages: list[dict[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object.")
        role = str(message.get("role", ""))
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    raise ValueError("Qwen3-0.6B accepts text-only message content.")
                parts.append(str(part.get("text", "")))
            content = "".join(parts)
        if not isinstance(content, str):
            raise ValueError("Qwen3 message content must be text.")
        messages.append({"role": role, "content": content})
    return messages
