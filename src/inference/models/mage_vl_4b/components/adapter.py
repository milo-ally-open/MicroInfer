"""将共用 Engine 工作转成 Mage-VL 输入、逐步生成与标准化结果。"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from inference.engine.sequence import ExecutionPhase, ScheduledWorkItem, WorkerResult
from inference.engine.sequence import InferenceResult
from inference.engine.sequence import InferenceError
from inference.processors.video import probe_video_metadata, sample_video_frames

from ..utils import loader as config
from .encoder import MageVLEncoder
from ..utils.loader import (
    CODEC_ENGINE,
    CODEC_MAX_PIXELS,
    CODEC_TARGET_CANVAS,
    MAX_VIDEO_DURATION_SECONDS,
    MODEL_ID,
    MODEL_REVISION,
    VIDEO_BACKEND,
    VIDEO_MAX_FRAME_SIDE,
    VIDEO_MAX_FRAMES,
)


class MageVLModelAdapter:
    """负责 Mage-VL 专属前处理、生成状态与 Engine 结果。"""

    def __init__(self, model_runner: Any) -> None:
        self.model_runner = model_runner
        self.encoder = MageVLEncoder(model_runner)
        self._tasks: dict[str, dict[str, Any]] = {}

    def generation_options(self, request_body: dict[str, Any]) -> dict[str, Any]:
        return self.encoder.generation_args(request_body)

    def validate_live_video_backend(self, backend: str, codec_engine: str | None) -> None:
        if backend != "codec":
            return
        self.encoder.require_codec_runtime(codec_engine)
        self.encoder.require_processor_codec_support()
        if codec_engine == "dcvc-rt":
            self.encoder.codec_processor_config(
                codec_engine, config.LIVE_SETTINGS.live_codec_target_canvas, None
            )

    def health_info(self) -> dict[str, Any]:
        return {
            "backend": "transformers",
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "quantization": config.QUANTIZATION_CONFIG["method"],
            "live_video_enabled": True,
            "kv_cache_profile": self.model_runner.kv_cache_profile(),
            "worker_kv_cache": self.model_runner.kv_cache_snapshot(),
        }

    def prepare_chat_generation(self, request_id: str, request_body: dict[str, Any]) -> dict[str, int]:
        inputs, prompt_tokens, options = self.encoder.encode_chat(request_body)
        self.model_runner.begin_generation(request_id, inputs, options)
        self._tasks[request_id] = {
            "kind": "chat",
            "prompt_tokens": prompt_tokens,
            "generation_options": options,
            "started_at": time.perf_counter(),
        }
        return self._generation_descriptor(request_id, prompt_tokens, options)

    def prepare_video_generation(
        self,
        request_id: str,
        video_path: str,
        filename: str | None,
        request_body: dict[str, Any],
        upload_bytes: int,
        staging_seconds: float,
    ) -> dict[str, int]:
        question = str(request_body.get("question", "")).strip()
        if not question:
            raise InferenceError(400, "question must not be empty.")
        video_backend = request_body.get("video_backend") or VIDEO_BACKEND
        codec_engine = request_body.get("codec_engine") or CODEC_ENGINE
        generation_body = {
            key: value
            for key, value in request_body.items()
            if key not in {"question", "video_backend", "codec_engine"} and value is not None
        }
        options = self.encoder.generation_args(generation_body)
        request_started = time.perf_counter()
        self.model_runner.reset_peak_memory()
        destination = Path(video_path)
        preprocess_started = time.perf_counter()
        if video_backend == "frames":
            video_input, video_metadata = sample_video_frames(
                destination,
                max_frames=VIDEO_MAX_FRAMES,
                max_frame_side=VIDEO_MAX_FRAME_SIDE,
                max_frame_pixels=config.MAX_IMAGE_PIXELS,
                max_duration_seconds=MAX_VIDEO_DURATION_SECONDS,
                resize=self.encoder.resize_image,
            )
            video_metadata.update(backend="frames", codec_engine=None)
        elif video_backend == "codec":
            video_metadata = probe_video_metadata(
                destination,
                ffprobe_bin=config.CODEC_FFPROBE_BIN,
                max_duration_seconds=MAX_VIDEO_DURATION_SECONDS,
            )
            video_metadata.update(backend="codec", codec_engine=codec_engine or CODEC_ENGINE)
            video_input = destination
        else:
            raise InferenceError(400, f"Unsupported video_backend: {video_backend}")
        decode_seconds = time.perf_counter() - preprocess_started
        processor_started = time.perf_counter()
        inputs, prompt_tokens, canvas_count = self.encoder.encode_video(
            question,
            video_input,
            backend=video_backend,
            codec_engine=codec_engine,
            target_canvas=CODEC_TARGET_CANVAS,
            max_pixels=CODEC_MAX_PIXELS,
            codec_cache_root=destination.parent / "codec-cache" if video_backend == "codec" else None,
        )
        processor_seconds = time.perf_counter() - processor_started
        if video_backend == "codec":
            video_metadata["codec_canvas_count"] = canvas_count
        self.encoder.check_context(prompt_tokens, options)
        self.model_runner.begin_generation(request_id, inputs, options)
        self._tasks[request_id] = {
            "kind": "video",
            "prompt_tokens": prompt_tokens,
            "generation_options": options,
            "started_at": request_started,
            "generation_started": time.perf_counter(),
            "video": {**video_metadata, "upload_bytes": upload_bytes, "filename": filename},
            "staging_seconds": staging_seconds,
            "decode_seconds": decode_seconds,
            "processor_seconds": processor_seconds,
        }
        return self._generation_descriptor(request_id, prompt_tokens, options)

    def prepare_live_frames_generation(
        self, request_id: str, question: str, frames: list[Any], generation_options: dict[str, Any]
    ) -> dict[str, int]:
        started = time.perf_counter()
        inputs, prompt_tokens, _ = self.encoder.encode_video(
            question, frames, backend="frames"
        )
        self.encoder.check_context(prompt_tokens, generation_options)
        self.model_runner.begin_generation(request_id, inputs, generation_options)
        self._tasks[request_id] = {
            "kind": "live",
            "prompt_tokens": prompt_tokens,
            "generation_options": generation_options,
            "started_at": started,
        }
        return self._generation_descriptor(request_id, prompt_tokens, generation_options)

    def prepare_live_video_generation(
        self,
        request_id: str,
        question: str,
        video_path: Path,
        generation_options: dict[str, Any],
        codec_engine: str | None,
        target_canvas: int,
        max_pixels: int,
        codec_cache_root: Path | None,
    ) -> dict[str, int]:
        started = time.perf_counter()
        inputs, prompt_tokens, canvas_count = self.encoder.encode_video(
            question,
            video_path,
            backend="codec",
            codec_engine=codec_engine,
            target_canvas=target_canvas,
            max_pixels=max_pixels,
            codec_cache_root=codec_cache_root,
        )
        self.encoder.check_context(prompt_tokens, generation_options)
        self.model_runner.begin_generation(request_id, inputs, generation_options)
        self._tasks[request_id] = {
            "kind": "live",
            "prompt_tokens": prompt_tokens,
            "generation_options": generation_options,
            "started_at": started,
            "codec_canvas_count": canvas_count,
        }
        return self._generation_descriptor(request_id, prompt_tokens, generation_options)

    def bind_generation_cache(self, request_id: str, block_ids: tuple[int, ...]) -> None:
        self.model_runner.bind_generation_cache(request_id, block_ids)

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> tuple[WorkerResult, ...]:
        """将同轮混合 work items 交给模型 Runner，逐请求返回生成结果。"""
        for request in requests:
            task = self._task(request.request_id)
            if request.phase is ExecutionPhase.PREFILL and (
                request.prompt_offset + request.token_budget > task["prompt_tokens"]
            ):
                raise RuntimeError("Scheduler prefill chunk exceeds the prepared prompt.")
            if request.phase is ExecutionPhase.DECODE and request.token_budget != 1:
                raise RuntimeError("Decode scheduling must assign exactly one token per turn.")
        return self.model_runner.execute_batch(requests)

    def _generation_descriptor(
        self, request_id: str, prompt_tokens: int, options: dict[str, Any]
    ) -> dict[str, int]:
        return {
            "prompt_tokens": prompt_tokens,
            "max_new_tokens": int(options["max_new_tokens"]),
            "prefill_min_chunk_tokens": self.model_runner.prefill_min_chunk_tokens(request_id),
        }

    def finish_chat_generation(self, request_id: str) -> InferenceResult:
        task = self._task(request_id)
        generated_ids, finish_reason = self.model_runner.finish_generation(request_id)
        answer = self.model_runner.decode(generated_ids)
        result = InferenceResult(
            answer,
            task["prompt_tokens"],
            int(generated_ids.numel()),
            finish_reason,
        )
        self._tasks.pop(request_id, None)
        return result

    def finish_video_generation(self, request_id: str) -> InferenceResult:
        task = self._task(request_id)
        generated_ids, finish_reason = self.model_runner.finish_generation(request_id)
        answer = self.model_runner.decode(generated_ids)
        generated_at = time.perf_counter()
        result = InferenceResult(
            answer,
            task["prompt_tokens"],
            int(generated_ids.numel()),
            finish_reason,
            metadata={
                "video": task["video"],
                "timing_seconds": {
                    "staging_and_decode": round(task["staging_seconds"] + task["decode_seconds"], 4),
                    "processor": round(task["processor_seconds"], 4),
                    "generation": round(generated_at - task["generation_started"], 4),
                    "server_total": round(generated_at - task["started_at"] + task["staging_seconds"], 4),
                },
                "gpu_memory_bytes": self.model_runner.peak_memory(),
            },
        )
        self._tasks.pop(request_id, None)
        return result

    def finish_live_generation(self, request_id: str) -> InferenceResult:
        task = self._task(request_id)
        generated_ids, finish_reason = self.model_runner.finish_generation(request_id)
        answer = self.model_runner.decode(generated_ids)
        result = InferenceResult(
            answer,
            task["prompt_tokens"],
            int(generated_ids.numel()),
            finish_reason,
            metadata={"generation_seconds": round(time.perf_counter() - task["started_at"], 3)},
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
            raise RuntimeError(f"Unknown Mage-VL generation request: {request_id}") from exc
