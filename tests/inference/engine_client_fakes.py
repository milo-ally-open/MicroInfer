"""提供不載入 Torch 或模型權重的 EngineClient 多程序 fixture。"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Sequence

INFERENCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "inference"
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from engine.contracts import ExecutionPhase, ScheduledWorkItem, WorkerResult
from engine.engine import InferenceEngine, InferenceResult
from engine.worker import InProcessModelWorker, MultiprocessModelWorker


class FakeRunner:
    def load(self) -> None:
        return None

    def close(self) -> None:
        return None

    def gpu_status(self) -> dict[str, Any]:
        return {"gpu": "fake"}


class FakeAdapter:
    def __init__(self, runner: FakeRunner) -> None:
        self.runner = runner
        self.tasks: dict[str, dict[str, int]] = {}

    def health_info(self) -> dict[str, Any]:
        return {
            "backend": "transformers",
            "model": "microsoft/Mage-VL",
            "revision": "fake-revision",
            "quantization": "none",
            "kv_cache_profile": {"bytes_per_token": 8},
        }

    def prepare_chat_generation(self, request_id: str, body: dict[str, Any]) -> dict[str, int]:
        if body.get("fail"):
            raise ValueError("fake preparation failure")
        self.tasks[request_id] = {"decode_steps": 0}
        return {"prompt_tokens": 3, "max_new_tokens": 2}

    def bind_generation_cache(self, request_id: str, block_ids: tuple[int, ...]) -> None:
        if request_id not in self.tasks:
            raise RuntimeError("fake task was not prepared")

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> tuple[WorkerResult, ...]:
        results: list[WorkerResult] = []
        for request in requests:
            if request.phase is ExecutionPhase.PREFILL:
                results.append(WorkerResult(
                    request.request_id,
                    prompt_tokens_processed=request.prompt_offset + request.token_budget,
                ))
                continue
            time.sleep(0.15)
            task = self.tasks[request.request_id]
            task["decode_steps"] += 1
            step = task["decode_steps"]
            results.append(
                WorkerResult(
                    request_id=request.request_id,
                    text_delta=("h" if step == 1 else "i"),
                    generated_tokens=1,
                    finished=step == 2,
                    finish_reason="stop" if step == 2 else None,
                )
            )
        return tuple(results)

    def finish_chat_generation(self, request_id: str) -> InferenceResult:
        self.tasks.pop(request_id)
        return InferenceResult("hi", 3, 2, "stop")

    def cancel_generation(self, request_id: str) -> None:
        self.tasks.pop(request_id, None)

    def generation_options(self, request_body: dict[str, Any]) -> dict[str, Any]:
        return {"max_new_tokens": 2}

    def validate_live_video_backend(self, backend: str, codec_engine: str | None) -> None:
        return None


def create_model_worker() -> InProcessModelWorker:
    runner = FakeRunner()
    return InProcessModelWorker(runner, FakeAdapter(runner))


def create_engine_core() -> InferenceEngine:
    worker = MultiprocessModelWorker(
        "engine_client_fakes:create_model_worker",
        startup_timeout_seconds=10,
        request_timeout_seconds=10,
    )
    return InferenceEngine(
        worker,
        model_id="microsoft/Mage-VL",
        max_concurrent_requests=2,
        max_waiting_requests=2,
        max_num_scheduled_tokens=16,
        batch_wait_seconds=0.01,
        kv_cache_block_size_tokens=2,
        kv_cache_capacity_tokens=64,
    )
