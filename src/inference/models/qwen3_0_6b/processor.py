"""在 API Server 将 OpenAI 聊天请求转为 Qwen3 模型输入。"""

from __future__ import annotations

from typing import Any
from transformers import PreTrainedTokenizerBase

from inference.network.input_processor import ChatInputProcessor, StreamOutputProcessor
from inference.network.output_processor import IncrementalDetokenizer
from inference.network.tool_calling import normalize_chat_messages
from inference.engine.sequence import PreparedChatInput
from inference.engine.sequence import InferenceError

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
    messages = normalize_chat_messages(request_body.get("messages"))
    tools = request_body.get("tools") or []
    tool_choice = request_body.get("tool_choice", "auto" if tools else "none")
    template_tools = tools if tool_choice == "auto" else None
    if template_tools:
        messages = _with_single_tool_call_policy(messages)
    options = build_sampling_params(request_body)
    try:
        template_options: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "enable_thinking": bool(config.GENERATION_CONFIG.get("enable_thinking", False)),
        }
        if template_tools:
            template_options["tools"] = template_tools
        inputs = tokenizer.apply_chat_template(messages, **template_options)
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


def normalize_text_messages(raw_messages: Any) -> list[dict[str, Any]]:
    """旧呼叫点的兼容名称；工具历史由共用标准化器完整保留。"""
    return normalize_chat_messages(raw_messages)


def _with_single_tool_call_policy(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Qwen3 上游模板允许同轮多调用；首版在提示中限制为单个调用。"""
    policy = (
        "For this turn, call at most one function. If another function is needed, "
        "wait for the first tool result before requesting the next function."
    )
    result = [dict(message) for message in messages]
    if result and result[0].get("role") == "system":
        result[0]["content"] = f"{result[0].get('content', '')}\n\n{policy}".strip()
    else:
        result.insert(0, {"role": "system", "content": policy})
    return result
