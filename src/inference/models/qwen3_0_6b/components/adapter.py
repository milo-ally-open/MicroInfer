"""将纯文字聊天请求转为 Qwen3 逐步生成工作。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from inference.engine.contracts import (
    ExecutionPhase,
    InferenceResult,
    PreparedChatInput,
    ScheduledWorkItem,
    WorkerResult,
)
from inference.engine.contracts import InferenceError
from ..utils import loader as config
from ..processor import build_sampling_params, prepare_chat_input


class Qwen3ModelAdapter:
    """处理 Qwen3 chat template、生成参数及逐请求结果。"""

    def __init__(self, model_runner: Any) -> None:
        self.model_runner = model_runner
        self._tasks: dict[str, dict[str, Any]] = {}

    def health_info(self) -> dict[str, Any]:
        return {
            "backend": "qwen3_native_torch",
            "model": config.MODEL_ALIAS,
            "revision": config.MODEL_REVISION,
            "platform": self.model_runner.platform.name,
            "quantization": "none",
            "live_video_enabled": False,
            "kv_cache_profile": self.model_runner.kv_cache_profile(),
            "worker_kv_cache": self.model_runner.cache_snapshot(),
            "gpu_peak_memory": self.model_runner.peak_memory(),
        }

    def generation_options(self, request_body: dict[str, Any]) -> dict[str, Any]:
        try:
            return build_sampling_params(request_body)
        except ValueError as exc:
            raise InferenceError(400, str(exc)) from exc

    def prepare_chat_generation(
        self,
        request_id: str,
        request_body: dict[str, Any],
        prepared_input: PreparedChatInput | None = None,
    ) -> dict[str, int]:
        if prepared_input is None:
            if self.model_runner.tokenizer is None:
                raise RuntimeError("Qwen3 tokenizer is not loaded in the Model Worker.")
            try:
                prepared_input = prepare_chat_input(self.model_runner.tokenizer, request_body)
            except ValueError as exc:
                raise InferenceError(400, str(exc)) from exc
        if prepared_input.model_alias != config.MODEL_ALIAS:
            raise InferenceError(400, "Prepared input model does not match the Qwen3 service.")
        prompt_tokens = len(prepared_input.input_ids)
        options = dict(prepared_input.sampling_params)
        self.model_runner.begin_generation(
            request_id,
            {"input_ids": prepared_input.input_ids},
            options,
        )
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
            content="",
            prompt_tokens=int(task["prompt_tokens"]),
            completion_tokens=int(generated_ids.numel()),
            finish_reason=finish_reason,
            token_ids=tuple(int(token_id) for token_id in generated_ids.detach().cpu().tolist()),
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
