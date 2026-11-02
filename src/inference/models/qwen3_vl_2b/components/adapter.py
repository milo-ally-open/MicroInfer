"""将 Engine 请求适配到 Qwen3-VL Encoder、Runner 与 InferenceResult。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

if __name__ == "__main__":
    import os
    from pathlib import Path

    _model_dir = Path(__file__).resolve().parents[1]
    _repo_root = _model_dir.parents[3]
    os.environ.setdefault("INFERENCE_PROJECT_ROOT", str(_repo_root))
    os.environ.setdefault("INFERENCE_MODEL_ALIAS", "qwen3-vl-2b")
    os.environ.setdefault("INFERENCE_MODEL_MANIFEST", str(_model_dir / "profiles/model.yml"))
    os.environ.setdefault("INFERENCE_CONFIG", str(_model_dir / "profiles/inference.yml"))
    os.environ.setdefault("INFERENCE_NETWORK_CONFIG", str(_model_dir / "profiles/network.yml"))

from inference.engine.sequence import InferenceError, InferenceResult, ScheduledWorkItem, WorkerResult
from .encoder import Qwen3VLEncoder
from .. import loader as config


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


if __name__ == "__main__":
    import time
    from types import SimpleNamespace

    import torch

    runner = SimpleNamespace(
        model=None,
        kv_cache_profile=lambda: {"bytes_per_token": 0},
        cache_snapshot=lambda: {"active_requests": 0, "cached_tokens": 0},
        peak_memory=lambda: {"peak_allocated": 0, "peak_reserved": 0},
    )
    adapter = Qwen3VLModelAdapter(runner)
    options = adapter.generation_options({"max_tokens": 2, "temperature": 0})
    assert options["max_new_tokens"] == 2 and options["do_sample"] is False
    assert adapter.health_info()["device"] == "not_loaded"
    print("Test: Qwen3VLModelAdapter generation options and health contract passed.")

    started_at = time.perf_counter()
    for _ in range(1000):
        adapter.generation_options({"max_tokens": 2, "temperature": 0})
    print(f"Bench: adapter option validation={(time.perf_counter() - started_at) * 1e3 / 1000:.4f} ms/request.")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as profile:
        adapter.generation_options({"max_tokens": 2, "temperature": 0})
        adapter.health_info()
    print("Profile: Qwen3VLModelAdapter")
    print(profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))
