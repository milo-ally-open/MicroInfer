"""以假 Worker 與 Runner 驗證 Engine、token 輪轉及 KV block 生命週期。"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from collections.abc import Sequence
from pathlib import Path

INFERENCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "inference"
sys.path.insert(0, str(INFERENCE_ROOT))

from engine.engine import InferenceEngine, InferenceResult
from engine.contracts import ExecutionPhase, ScheduledWorkItem, WorkerResult
from engine.errors import InferenceError
from engine.scheduler import TokenScheduler
from engine.worker import InProcessModelWorker


class FakeRunner:
    def __init__(self) -> None:
        self.loaded = 0
        self.closed = 0
        self.first_prefill = threading.Event()
        self.release_prefill = threading.Event()
        self.second_prepared = threading.Event()
        self.block_first_prefill = False
        self.block = False
        self.entered = threading.Event()

    def load(self) -> None:
        self.loaded += 1

    def close(self) -> None:
        self.closed += 1

    def gpu_status(self) -> dict[str, str]:
        return {"gpu": "fake"}


class FakeAdapter:
    def __init__(self, runner: FakeRunner) -> None:
        self.runner = runner
        self._lock = threading.Lock()
        self._tasks: dict[str, dict[str, object]] = {}
        self.turns: list[tuple[str, str]] = []

    def health_info(self) -> dict[str, object]:
        return {"backend": "fake", "kv_cache_profile": {"bytes_per_token": 8}}

    def prepare_chat_generation(self, request_id: str, body: dict[str, object]) -> dict[str, int]:
        with self._lock:
            self._tasks[request_id] = {"id": body.get("id"), "steps": 0}
        if body.get("id") == "B":
            self.runner.second_prepared.set()
        return {"prompt_tokens": int(body.get("prompt_tokens", 2)), "max_new_tokens": int(body.get("max_tokens", 2))}

    def prepare_video_generation(
        self, request_id: str, path: str, filename: str | None, body: dict[str, object], upload_bytes: int, staging_seconds: float
    ) -> dict[str, int]:
        self.assert_video_exists(path, upload_bytes)
        self._tasks[request_id] = {"id": "video", "steps": 0}
        return {"prompt_tokens": 2, "max_new_tokens": 2}

    @staticmethod
    def assert_video_exists(path: str, upload_bytes: int) -> None:
        if not Path(path).is_file() or Path(path).stat().st_size != upload_bytes:
            raise AssertionError("Engine did not pass a staged video file to the Worker.")

    def bind_generation_cache(self, request_id: str, block_ids: tuple[int, ...]) -> None:
        self._tasks[request_id]["block_ids"] = block_ids

    def prefill_generation(self, request_id: str, token_budget: int) -> dict[str, object]:
        with self._lock:
            task = self._tasks[request_id]
            task["steps"] = 1
            self.turns.append((str(task["id"]), "prefill"))
        self.runner.entered.set()
        if self.runner.block_first_prefill and not self.runner.first_prefill.is_set():
            self.runner.first_prefill.set()
            self.runner.release_prefill.wait(timeout=3)
        if self.runner.block:
            self.runner.release_prefill.wait(timeout=3)
        return {"text": "a", "generated_tokens": 1, "done": False}

    def decode_generation_step(self, request_id: str, token_budget: int) -> dict[str, object]:
        with self._lock:
            task = self._tasks[request_id]
            task["steps"] = int(task["steps"]) + 1
            self.turns.append((str(task["id"]), "decode"))
        return {"text": "b", "generated_tokens": 1, "done": True}

    def finish_chat_generation(self, request_id: str) -> InferenceResult:
        self._tasks.pop(request_id)
        return InferenceResult("ab", 2, 2, "stop")

    def finish_video_generation(self, request_id: str) -> InferenceResult:
        self._tasks.pop(request_id)
        return InferenceResult("video", 2, 2, "stop", {"video": {"upload_bytes": 4}})

    def cancel_generation(self, request_id: str) -> None:
        self._tasks.pop(request_id, None)

    def generation_options(self, body: dict[str, object]) -> dict[str, object]:
        return {"max_new_tokens": int(body.get("max_tokens", 2))}

    def validate_live_video_backend(self, backend: str, codec_engine: str | None) -> None:
        return None

    def prepare_live_frames_generation(
        self, request_id: str, question: str, frames: list[object], options: dict[str, object]
    ) -> dict[str, int]:
        self._tasks[request_id] = {"id": "live", "steps": 0}
        return {"prompt_tokens": 2, "max_new_tokens": 2}

    def finish_live_generation(self, request_id: str) -> InferenceResult:
        self._tasks.pop(request_id)
        return InferenceResult("live", 2, 2, "stop", {"generation_seconds": 0.01})

    def analyze_live_frames(self, question: str, frames: list[object], options: dict[str, object]) -> dict[str, object]:
        return {"answer": "legacy"}


class ChunkedFakeAdapter(FakeAdapter):
    def prepare_chat_generation(self, request_id: str, body: dict[str, object]) -> dict[str, int]:
        self._tasks[request_id] = {"id": body.get("id"), "steps": 0}
        return {
            "prompt_tokens": 7,
            "max_new_tokens": 2,
            "prefill_min_chunk_tokens": 2,
        }

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> tuple[WorkerResult, ...]:
        results = []
        for request in requests:
            if request.phase is ExecutionPhase.PREFILL:
                processed = request.prompt_offset + request.token_budget
                final_chunk = processed == 7
                results.append(WorkerResult(
                    request.request_id,
                    text_delta="A" if final_chunk else "",
                    generated_tokens=1 if final_chunk else 0,
                    prompt_tokens_processed=processed,
                ))
            else:
                results.append(WorkerResult(
                    request.request_id,
                    text_delta="B",
                    generated_tokens=1,
                    finished=True,
                    finish_reason="stop",
                ))
        return tuple(results)

    def finish_chat_generation(self, request_id: str) -> InferenceResult:
        self._tasks.pop(request_id)
        return InferenceResult("AB", 7, 2, "stop")


class FakeLiveSessions:
    def __init__(self) -> None:
        self.stopped = 0

    def stop_all(self) -> None:
        self.stopped += 1

    def active_count(self) -> int:
        return 0


class EngineContractTests(unittest.TestCase):
    def make_engine(
        self, *, capacity: int = 2, waiting: int = 2, cache_tokens: int = 128
    ) -> tuple[InferenceEngine, FakeRunner, FakeAdapter]:
        runner = FakeRunner()
        adapter = FakeAdapter(runner)
        worker = InProcessModelWorker(runner, adapter)
        engine = InferenceEngine(
            worker,
            model_id="fake-model",
            max_concurrent_requests=capacity,
            max_waiting_requests=waiting,
            max_num_scheduled_tokens=32,
            kv_cache_block_size_tokens=4,
            kv_cache_capacity_tokens=cache_tokens,
        )
        return engine, runner, adapter

    def test_engine_owns_worker_lifecycle_and_health(self) -> None:
        engine, runner, _ = self.make_engine()
        self.assertFalse(engine.ready)
        engine.start()
        engine.start()
        self.assertTrue(engine.ready)
        self.assertEqual(engine.health()["gpu"], "fake")
        self.assertEqual(engine.health()["kv_cache"]["capacity_tokens"], 128)
        self.assertEqual(engine.health()["kv_cache"]["estimated_bytes"], 0)
        engine.close()
        engine.close()
        self.assertEqual(engine.state, "closed")
        self.assertEqual(runner.loaded, 1)
        self.assertEqual(runner.closed, 1)

    def test_scheduler_builds_prefill_and_decode_batches_with_token_budget(self) -> None:
        scheduler = TokenScheduler(
            max_num_seqs=4,
            max_waiting_requests=4,
            max_num_scheduled_tokens=8,
            batch_wait_seconds=0,
        )
        scheduler.add_request("A", prompt_tokens=3, max_output_tokens=2)
        scheduler.add_request("B", prompt_tokens=4, max_output_tokens=2)
        prefill = scheduler.schedule_batch()
        self.assertIsNotNone(prefill)
        assert prefill is not None
        self.assertEqual([item.phase for item in prefill.work_items], [ExecutionPhase.PREFILL, ExecutionPhase.PREFILL])
        self.assertEqual(prefill.total_scheduled_tokens, 7)
        scheduler.complete_batch([
            WorkerResult("A", text_delta="a", generated_tokens=1, prompt_tokens_processed=3),
            WorkerResult("B", text_delta="b", generated_tokens=1, prompt_tokens_processed=4),
        ])
        decode = scheduler.schedule_batch()
        self.assertIsNotNone(decode)
        assert decode is not None
        self.assertEqual([item.phase for item in decode.work_items], [ExecutionPhase.DECODE, ExecutionPhase.DECODE])
        self.assertEqual(decode.total_scheduled_tokens, 2)
        scheduler.complete_batch([
            WorkerResult("A", text_delta="x", generated_tokens=1, finished=True),
            WorkerResult("B", text_delta="y", generated_tokens=1, finished=True),
        ])
        self.assertEqual(scheduler.snapshot()["running"], 0)

    def test_scheduler_requeues_prefill_chunks_and_mixes_decode(self) -> None:
        scheduler = TokenScheduler(
            max_num_seqs=2,
            max_waiting_requests=0,
            max_num_scheduled_tokens=5,
            prefill_chunk_size_tokens=3,
            batch_wait_seconds=0,
        )
        scheduler.add_request("long", prompt_tokens=8, max_output_tokens=4)
        scheduler.add_request("short", prompt_tokens=2, max_output_tokens=4)

        first = scheduler.schedule_batch()
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(
            [(item.request_id, item.phase, item.prompt_offset, item.token_budget)
             for item in first.work_items],
            [
                ("long", ExecutionPhase.PREFILL, 0, 3),
                ("short", ExecutionPhase.PREFILL, 0, 2),
            ],
        )
        scheduler.complete_batch([
            WorkerResult("long", prompt_tokens_processed=3),
            WorkerResult("short", text_delta="s", generated_tokens=1, prompt_tokens_processed=2),
        ])

        mixed = scheduler.schedule_batch()
        self.assertIsNotNone(mixed)
        assert mixed is not None
        self.assertEqual(
            [(item.request_id, item.phase, item.prompt_offset, item.token_budget)
             for item in mixed.work_items],
            [
                ("short", ExecutionPhase.DECODE, 0, 1),
                ("long", ExecutionPhase.PREFILL, 3, 3),
            ],
        )
        scheduler.complete_batch([
            WorkerResult("short", text_delta="t", generated_tokens=1),
            WorkerResult("long", prompt_tokens_processed=6),
        ])
        self.assertEqual(scheduler.snapshot()["running"], 2)
        scheduler.cancel_request("long")
        scheduler.cancel_request("short")
        self.assertEqual(scheduler.snapshot()["running"], 0)

    def test_chunked_stream_publishes_only_generated_tokens_and_releases_lease(self) -> None:
        runner = FakeRunner()
        adapter = ChunkedFakeAdapter(runner)
        engine = InferenceEngine(
            InProcessModelWorker(runner, adapter),
            model_id="fake-model",
            max_concurrent_requests=2,
            max_waiting_requests=2,
            max_num_scheduled_tokens=5,
            prefill_chunk_size_tokens=3,
            kv_cache_block_size_tokens=4,
            kv_cache_capacity_tokens=64,
            batch_wait_seconds=0,
        )
        engine.start()
        try:
            events = list(engine.stream_chat({"id": "chunked"}))
            self.assertEqual(
                [event.text for event in events if event.kind == "text"],
                ["A", "B"],
            )
            self.assertEqual(engine.scheduler.snapshot()["scheduled_rounds"], 4)
            self.assertEqual(engine.scheduler.snapshot()["running"], 0)
            self.assertEqual(engine.kv_cache.snapshot()["used_blocks"], 0)
        finally:
            engine.close()

    def test_chat_uses_prefill_decode_and_releases_cache_blocks(self) -> None:
        engine, _, adapter = self.make_engine()
        engine.start()
        result = engine.generate_chat({"id": "A"})
        self.assertEqual(result.content, "ab")
        self.assertEqual(adapter.turns, [("A", "prefill"), ("A", "decode")])
        self.assertEqual(engine.kv_cache.snapshot()["used_blocks"], 0)
        self.assertEqual(engine.scheduler.snapshot()["running"], 0)
        engine.close()

    def test_stream_chat_publishes_incremental_text_events(self) -> None:
        engine, _, _ = self.make_engine()
        engine.start()
        events = list(engine.stream_chat({"id": "A"}))
        self.assertEqual([event.kind for event in events], ["start", "text", "text", "finish"])
        self.assertEqual([event.text for event in events if event.kind == "text"], ["a", "b"])
        engine.close()

    def test_partial_prepare_failure_uses_common_cleanup(self) -> None:
        engine, _, adapter = self.make_engine()
        engine.start()

        def fail_after_partial_setup(request_id: str, body: dict[str, object]) -> dict[str, int]:
            adapter._tasks[request_id] = {"id": body.get("id"), "steps": 0}
            raise RuntimeError("preparation failed after allocating request state")

        adapter.prepare_chat_generation = fail_after_partial_setup
        with self.assertRaisesRegex(RuntimeError, "preparation failed"):
            engine.generate_chat({"id": "A"})
        self.assertEqual(adapter._tasks, {})
        self.assertEqual(engine.scheduler.snapshot()["running"], 0)
        self.assertEqual(engine.kv_cache.snapshot()["used_blocks"], 0)
        engine.close()

    def test_scheduler_rotates_active_requests_between_decode_turns(self) -> None:
        engine, runner, adapter = self.make_engine(capacity=2)
        engine.start()
        runner.block_first_prefill = True
        first = threading.Thread(target=lambda: list(engine.stream_chat({"id": "A"})))
        second = threading.Thread(target=lambda: list(engine.stream_chat({"id": "B"})))
        first.start()
        self.assertTrue(runner.first_prefill.wait(timeout=1))
        second.start()
        self.assertTrue(runner.second_prepared.wait(timeout=1))
        runner.release_prefill.set()
        first.join(timeout=2)
        second.join(timeout=2)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(
            adapter.turns[:3],
            [("A", "prefill"), ("A", "decode"), ("B", "prefill")],
        )
        self.assertEqual(engine.scheduler.snapshot()["scheduled_rounds"], 3)
        engine.close()

    def test_scheduler_rejects_when_sequence_and_waiting_limits_are_full(self) -> None:
        engine, runner, _ = self.make_engine(capacity=1, waiting=0)
        engine.start()
        runner.block = True
        first = threading.Thread(target=engine.generate_chat, args=({"id": "A"},))
        first.start()
        self.assertTrue(runner.entered.wait(timeout=1))
        with self.assertRaises(InferenceError) as raised:
            engine.generate_chat({"id": "B"})
        self.assertEqual(raised.exception.status_code, 429)
        runner.release_prefill.set()
        first.join(timeout=2)
        self.assertFalse(first.is_alive())
        engine.close()

    def test_cancelled_stream_releases_scheduler_and_cache(self) -> None:
        engine, _, _ = self.make_engine(capacity=1, waiting=0)
        engine.start()
        stream = engine.stream_chat({"id": "A"})
        self.assertEqual(next(stream).kind, "start")
        self.assertGreater(engine.kv_cache.snapshot()["used_blocks"], 0)
        stream.close()
        self.assertEqual(engine.kv_cache.snapshot()["used_blocks"], 0)
        self.assertEqual(engine.scheduler.snapshot()["running"], 0)
        self.assertEqual(engine.generate_chat({"id": "B"}).content, "ab")
        engine.close()

    def test_ipc_cancel_before_generation_context_is_registered(self) -> None:
        engine, _, adapter = self.make_engine(capacity=1, waiting=0)
        engine.start()
        request_id = "cancel-before-context"
        engine.register_request(request_id)
        engine.cancel_request(request_id)
        stream = engine.stream_chat({"id": "A"}, request_id=request_id)
        with self.assertRaises(GeneratorExit):
            next(stream)
        engine.unregister_request(request_id)
        self.assertEqual(adapter._tasks, {})
        self.assertEqual(engine.scheduler.snapshot()["running"], 0)
        self.assertEqual(engine.kv_cache.snapshot()["used_blocks"], 0)
        engine.close()

    def test_video_upload_is_spooled_and_forwarded_by_path(self) -> None:
        import io

        engine, _, _ = self.make_engine()
        engine.start()
        result = engine.understand_video(io.BytesIO(b"clip"), "clip.mp4", {"question": "describe"})
        self.assertEqual(result.content, "video")
        self.assertEqual(result.metadata["video"]["upload_bytes"], 4)
        engine.close()

    def test_live_generation_uses_same_step_scheduler(self) -> None:
        engine, _, adapter = self.make_engine()
        engine.start()
        admitted, result = engine.try_live_inference("analyze_live_frames", "question", [], {"max_new_tokens": 2})
        self.assertTrue(admitted)
        self.assertEqual(result["answer"], "live")
        self.assertEqual(adapter.turns[-2:], [("live", "prefill"), ("live", "decode")])
        self.assertEqual(engine.kv_cache.snapshot()["used_blocks"], 0)
        engine.close()

    def test_engine_rejects_invalid_sequence_capacity(self) -> None:
        runner = FakeRunner()
        with self.assertRaisesRegex(ValueError, "positive"):
            InferenceEngine(InProcessModelWorker(runner, FakeAdapter(runner)), model_id="fake", max_concurrent_requests=0)


if __name__ == "__main__":
    unittest.main()
