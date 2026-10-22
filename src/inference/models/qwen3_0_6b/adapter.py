"""將純文字聊天請求轉為 Qwen3 逐步生成工作。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from engine.contracts import ExecutionPhase, ScheduledWorkItem, WorkerResult
from engine.engine import InferenceResult
from engine.errors import InferenceError
from . import config


class Qwen3ModelAdapter:
    """處理 Qwen3 chat template、生成參數及逐請求結果。"""

    def __init__(self, model_runner: Any) -> None:
        self.model_runner = model_runner
        self._tasks: dict[str, dict[str, Any]] = {}

    def health_info(self) -> dict[str, Any]:
        return {
            "backend": "qwen3_native_torch",
            "model": config.MODEL_ALIAS,
            "revision": config.MODEL_REVISION,
            "quantization": "none",
            "live_video_enabled": False,
            "kv_cache_profile": self.model_runner.kv_cache_profile(),
            "worker_kv_cache": self.model_runner.cache_snapshot(),
            "gpu_peak_memory": self.model_runner.peak_memory(),
        }

    def generation_options(self, request_body: dict[str, Any]) -> dict[str, Any]:
        options = config.GENERATION_CONFIG
        max_new_tokens = request_body.get("max_completion_tokens", request_body.get("max_tokens"))
        max_new_tokens = int(max_new_tokens if max_new_tokens is not None else options["max_new_tokens"])
        if max_new_tokens < 1:
            raise InferenceError(400, "max_tokens must be positive.")
        num_beams = int(request_body.get("num_beams", options.get("num_beams", 1)))
        if num_beams != 1:
            raise InferenceError(400, "Qwen3 stepwise decoding supports num_beams=1.")
        return {
            "max_new_tokens": max_new_tokens,
            "temperature": float(request_body.get("temperature", options.get("temperature", 0.0))),
            "top_p": float(request_body.get("top_p", options.get("top_p", 1.0))),
            "top_k": int(request_body.get("top_k", options.get("top_k", 0))),
            "min_p": request_body.get("min_p", options.get("min_p")),
            "repetition_penalty": float(request_body.get("repetition_penalty", options.get("repetition_penalty", 1.0))),
            "no_repeat_ngram_size": int(request_body.get("no_repeat_ngram_size", options.get("no_repeat_ngram_size", 0))),
        }

    def prepare_chat_generation(self, request_id: str, request_body: dict[str, Any]) -> dict[str, int]:
        messages = self._text_messages(request_body.get("messages"))
        options = self.generation_options(request_body)
        tokenizer = self.model_runner.tokenizer
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=bool(config.GENERATION_CONFIG.get("enable_thinking", False)),
            )
        except Exception as exc:
            raise InferenceError(400, f"Could not format Qwen3 chat messages: {exc}") from exc
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        if prompt_tokens + options["max_new_tokens"] > config.MAX_CONTEXT_TOKENS:
            raise InferenceError(
                400,
                f"The prompt and requested output exceed the configured {config.MAX_CONTEXT_TOKENS}-token context limit.",
            )
        self.model_runner.begin_generation(request_id, dict(inputs), options)
        self._tasks[request_id] = {"prompt_tokens": prompt_tokens}
        return {
            "prompt_tokens": prompt_tokens,
            "max_new_tokens": options["max_new_tokens"],
            "prefill_min_chunk_tokens": 1,
        }

    def bind_generation_cache(self, request_id: str, block_ids: tuple[int, ...]) -> None:
        self.model_runner.bind_generation_cache(request_id, block_ids)

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> tuple[WorkerResult, ...]:
        return self.model_runner.execute_batch(requests)

    def finish_chat_generation(self, request_id: str) -> InferenceResult:
        task = self._task(request_id)
        generated_ids, finish_reason = self.model_runner.finish_generation(request_id)
        result = InferenceResult(
            content=self.model_runner.decode(generated_ids),
            prompt_tokens=int(task["prompt_tokens"]),
            completion_tokens=int(generated_ids.numel()),
            finish_reason=finish_reason,
        )
        self._tasks.pop(request_id, None)
        return result

    def cancel_generation(self, request_id: str) -> None:
        self.model_runner.cancel_generation(request_id)
        self._tasks.pop(request_id, None)

    def _task(self, request_id: str) -> dict[str, Any]:
        try:
            return self._tasks[request_id]
        except KeyError as exc:
            raise RuntimeError(f"Unknown Qwen3 chat request: {request_id}") from exc

    @staticmethod
    def _text_messages(raw_messages: Any) -> list[dict[str, str]]:
        if not isinstance(raw_messages, list) or not raw_messages:
            raise InferenceError(400, "messages must be a non-empty list.")
        messages: list[dict[str, str]] = []
        for message in raw_messages:
            if not isinstance(message, dict):
                raise InferenceError(400, "Each message must be an object.")
            role = str(message.get("role", ""))
            content = message.get("content", "")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, dict) or part.get("type") != "text":
                        raise InferenceError(400, "Qwen3-0.6B accepts text-only message content.")
                    parts.append(str(part.get("text", "")))
                content = "".join(parts)
            if not isinstance(content, str):
                raise InferenceError(400, "Qwen3 message content must be text.")
            messages.append({"role": role, "content": content})
        return messages
