"""為標準文字模型提供共用的 Tokenizer／Chat Template 輸入處理。"""

from __future__ import annotations

import math
from typing import Any

from api.input_processor import ChatInputProcessor, StreamOutputProcessor
from api.output_processor import IncrementalDetokenizer
from engine.contracts import PreparedChatInput
from engine.errors import InferenceError
from engine.model_config import model_source, selected_config, selected_model


class DefaultChatInputProcessor(ChatInputProcessor):
    """使用上游 Tokenizer 的 chat_template，不要求模型包撰寫薄型包裝器。"""

    def __init__(self) -> None:
        self.tokenizer: Any = None
        self.model_alias = ""
        self.config: dict[str, Any] = {}

    def start(self) -> None:
        if self.tokenizer is not None:
            return
        from transformers import AutoTokenizer

        self.model_alias, model_spec = selected_model()
        self.config = selected_config("INFERENCE_CONFIG")
        model_settings = self.config.get("model", {})
        runtime = self.config.get("runtime", {})
        if not isinstance(model_settings, dict) or not isinstance(runtime, dict):
            raise ValueError("model and runtime settings must be mappings.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source(self.config),
            revision=str(model_settings["revision"]),
            local_files_only=bool(runtime.get("local_files_only", True)),
            trust_remote_code=bool(model_settings.get("trust_remote_code", False)),
        )
        if not callable(getattr(self.tokenizer, "apply_chat_template", None)):
            self.tokenizer = None
            raise ValueError("The standard text profile requires a Tokenizer with apply_chat_template().")
        if not getattr(self.tokenizer, "chat_template", None):
            self.tokenizer = None
            raise ValueError("The standard text profile requires a model chat_template.")

    def close(self) -> None:
        self.tokenizer = None

    def prepare_chat(self, request_body: dict[str, Any]) -> PreparedChatInput:
        if self.tokenizer is None:
            raise RuntimeError("The standard chat input processor has not been started.")
        try:
            messages = _normalize_messages(request_body.get("messages"))
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            if isinstance(encoded, dict):
                if "input_ids" not in encoded:
                    raise ValueError("The model chat_template result does not contain input_ids.")
                encoded = encoded["input_ids"]
            elif hasattr(encoded, "input_ids"):
                encoded = encoded.input_ids
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            if encoded and isinstance(encoded[0], list):
                if len(encoded) != 1:
                    raise ValueError("The standard text profile expects one unpadded prompt per request.")
                encoded = encoded[0]
            input_ids = tuple(int(value) for value in encoded)
            if not input_ids:
                raise ValueError("The model chat_template returned an empty prompt.")
            options = _sampling_options(
                request_body,
                self.config.get("generation", {}),
                self.tokenizer.eos_token_id,
            )
            api_config = self.config.get("api", {})
            context_limit = int(api_config.get("max_context_tokens", 0))
            if context_limit > 0 and len(input_ids) + options["max_new_tokens"] > context_limit:
                raise ValueError(f"The prompt and requested output exceed the configured {context_limit}-token context limit.")
            return PreparedChatInput(
                model_alias=self.model_alias,
                input_ids=input_ids,
                sampling_params=options,
            )
        except (TypeError, KeyError, ValueError) as exc:
            raise InferenceError(400, str(exc)) from exc

    def create_stream_output_processor(self) -> StreamOutputProcessor:
        if self.tokenizer is None:
            raise RuntimeError("The standard chat input processor has not been started.")
        return IncrementalDetokenizer(self.tokenizer)

    def decode_tokens(self, token_ids: tuple[int, ...]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("The standard chat input processor has not been started.")
        processor = self.create_stream_output_processor()
        text = "".join(processor.push(token_id) for token_id in token_ids)
        return text + processor.finish()


def _normalize_messages(raw_messages: Any) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty list.")
    messages: list[dict[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object.")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("Each message requires a role.")
        content = message.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    raise ValueError("The standard text profile accepts text-only message content.")
                parts.append(str(part.get("text", "")))
            content = "".join(parts)
        if not isinstance(content, str):
            raise ValueError("Message content must be text.")
        messages.append({"role": role, "content": content})
    return messages


def _sampling_options(body: dict[str, Any], defaults: Any, eos_token_id: Any) -> dict[str, Any]:
    if not isinstance(defaults, dict):
        raise ValueError("generation settings must be a mapping.")
    num_beams = int(body.get("num_beams", defaults.get("num_beams", 1)))
    if num_beams != 1:
        raise ValueError("The standard text profile supports num_beams=1 only.")
    raw_max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    max_new_tokens = int(
        raw_max_tokens
        if raw_max_tokens is not None
        else defaults.get("max_new_tokens", defaults.get("max_tokens_default", 128))
    )
    temperature = float(body.get("temperature", defaults.get("temperature", 0.0)))
    top_p = float(body.get("top_p", defaults.get("top_p", 1.0)))
    top_k = int(body.get("top_k", defaults.get("top_k", 0)))
    repetition_penalty = float(body.get("repetition_penalty", defaults.get("repetition_penalty", 1.0)))
    no_repeat_ngram_size = int(body.get("no_repeat_ngram_size", defaults.get("no_repeat_ngram_size", 0)))
    min_p = body.get("min_p", defaults.get("min_p"))
    numeric_options = (temperature, top_p, repetition_penalty)
    if min_p is not None:
        numeric_options += (float(min_p),)
    if (
        max_new_tokens < 1
        or any(not math.isfinite(value) for value in numeric_options)
        or temperature < 0
        or not 0 < top_p <= 1
        or top_k < 0
        or repetition_penalty <= 0
        or no_repeat_ngram_size < 0
        or (min_p is not None and not 0 <= float(min_p) <= 1)
    ):
        raise ValueError("Invalid generation parameter for the standard text profile.")
    if not isinstance(eos_token_id, (list, tuple, set)):
        eos_token_ids = [] if eos_token_id is None else [int(eos_token_id)]
    else:
        eos_token_ids = [int(value) for value in eos_token_id]
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "eos_token_ids": eos_token_ids,
    }
