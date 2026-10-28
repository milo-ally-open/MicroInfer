"""把 OpenAI chat 请求编码为 Qwen3 prompt token IDs 与采样参数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inference.engine.sequence import PreparedChatInput
from inference.network.tool_calling import normalize_chat_messages


def _model_settings() -> Any:
    """仅在编码请求时加载已由 Launcher/EngineClient 选定的模型 YAML。"""
    from ..utils import loader

    return loader


@dataclass(frozen=True)
class Qwen3EncodingPlan:
    """由 Encoder 产出的规范消息与生成参数，等待 Tokenizer 转成 IDs。"""

    messages: list[dict[str, Any]]
    template_tools: list[dict[str, Any]] | None
    enable_thinking: bool
    sampling_params: dict[str, Any]


class Qwen3Encoder:
    """负责请求规范化、工具调用约束、生成参数及 prompt 上下文校验。"""

    def prepare(self, request_body: dict[str, Any]) -> Qwen3EncodingPlan:
        settings = _model_settings()
        messages = normalize_chat_messages(request_body.get("messages"))
        tools = request_body.get("tools") or []
        tool_choice = request_body.get("tool_choice", "auto" if tools else "none")
        template_tools = tools if tool_choice == "auto" else None
        if template_tools:
            messages = _with_single_tool_call_policy(messages)
        return Qwen3EncodingPlan(
            messages=messages,
            template_tools=template_tools,
            enable_thinking=bool(settings.GENERATION_CONFIG.get("enable_thinking", False)),
            sampling_params=build_sampling_params(request_body),
        )

    def build(self, plan: Qwen3EncodingPlan, token_ids: tuple[int, ...]) -> PreparedChatInput:
        settings = _model_settings()
        if not token_ids:
            raise ValueError("Qwen3 Tokenizer produced an empty prompt.")
        if len(token_ids) + plan.sampling_params["max_new_tokens"] > settings.MAX_CONTEXT_TOKENS:
            raise ValueError(
                f"The prompt and requested output exceed the configured "
                f"{settings.MAX_CONTEXT_TOKENS}-token context limit."
            )
        return PreparedChatInput(
            model_alias=settings.MODEL_ALIAS,
            input_ids=token_ids,
            sampling_params=plan.sampling_params,
        )

    def encode(self, request_body: dict[str, Any], tokenizer: Any) -> PreparedChatInput:
        """组合 Encoder 与 Tokenizer，供 API facade 和 Worker 旧入口使用。"""
        from .tokenizer import Qwen3Tokenizer

        model_tokenizer = tokenizer if isinstance(tokenizer, Qwen3Tokenizer) else Qwen3Tokenizer(tokenizer)
        plan = self.prepare(request_body)
        token_ids = model_tokenizer.tokenize_chat(
            plan.messages,
            tools=plan.template_tools,
            enable_thinking=plan.enable_thinking,
        )
        return self.build(plan, token_ids)


def build_sampling_params(request_body: dict[str, Any]) -> dict[str, Any]:
    defaults = _model_settings().GENERATION_CONFIG
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


def prepare_chat_input(tokenizer: Any, request_body: dict[str, Any]) -> PreparedChatInput:
    """保留旧函数入口，供 Worker 与逐算子脚本共用新的 Encoder。"""
    return Qwen3Encoder().encode(request_body, tokenizer)


def _with_single_tool_call_policy(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
