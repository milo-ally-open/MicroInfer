"""将 Engine 请求适配到 Qwen3-VL Encoder、Runner 与 InferenceResult。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from inference.engine.sequence import InferenceError, InferenceResult, ScheduledWorkItem, WorkerResult
from .encoder import Qwen3VLEncoder
from ..utils import loader as config


class Qwen3VLModelAdapter:
    def __init__(self, model_runner: Any) -> None:
        self.model_runner = model_runner
        self.encoder = Qwen3VLEncoder(model_runner)
        self._tasks: dict[str, dict[str, int]] = {}

    def health_info(self) -> dict[str, Any]:
        return {
            "backend": "transformers",
            "model": config.MODEL_ID,
            "revision": config.MODEL_REVISION,
            "device": str(self.model_runner.input_device) if self.model_runner.model is not None else "not_loaded",
            "kv_cache_backend": "transformers_dynamic_cache",
            "kv_cache_profile": self.model_runner.kv_cache_profile(),
            "worker_kv_cache": self.model_runner.cache_snapshot(),
            "gpu_peak_memory": self.model_runner.peak_memory(),
        }

    def generation_options(self, request_body: dict[str, Any]) -> dict[str, Any]:
        return self.encoder.generation_options(request_body)

    def prepare_chat_generation(self, request_id: str, request_body: dict[str, Any]) -> dict[str, int]:
        if request_id in self._tasks:
            raise ValueError(f"Generation request already exists: {request_id}")
        model_inputs, prompt_tokens, options = self.encoder.encode_chat(request_body)
        try:
            self.model_runner.begin_generation(request_id, model_inputs, options)
        except (TypeError, ValueError) as exc:
            raise InferenceError(400, str(exc)) from exc
        self._tasks[request_id] = {"prompt_tokens": prompt_tokens}
        return {
            "prompt_tokens": prompt_tokens,
            "max_new_tokens": int(options["max_new_tokens"]),
            "prefill_min_chunk_tokens": self.model_runner.prefill_min_chunk_tokens(request_id),
        }

    def bind_generation_cache(self, request_id: str, block_ids: tuple[int, ...]) -> None:
        self.model_runner.bind_generation_cache(request_id, block_ids)

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> tuple[WorkerResult, ...]:
        if len(requests) != 1:
            raise RuntimeError("Qwen3-VL is configured for one active sequence to isolate MRoPE state.")
        return self.model_runner.execute_batch(requests)

    def finish_chat_generation(self, request_id: str) -> InferenceResult:
        task = self._task(request_id)
        generated_ids, finish_reason = self.model_runner.finish_generation(request_id)
        return InferenceResult(
            content=self.model_runner.decode(generated_ids),
            prompt_tokens=task["prompt_tokens"],
            completion_tokens=int(generated_ids.numel()),
            finish_reason=finish_reason,
            token_ids=tuple(int(token_id) for token_id in generated_ids.tolist()),
        )

    def cancel_generation(self, request_id: str) -> None:
        self.model_runner.cancel_generation(request_id)
        self._tasks.pop(request_id, None)

    def _task(self, request_id: str) -> dict[str, int]:
        try:
            task = self._tasks.pop(request_id)
        except KeyError as exc:
            raise RuntimeError(f"Unknown Qwen3-VL request: {request_id}") from exc
        return task
