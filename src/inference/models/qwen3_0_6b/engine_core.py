"""建立 Qwen3-0.6B 的 Engine Core 與獨立 GPU Worker。"""

from __future__ import annotations

from engine.engine import InferenceEngine
from engine.worker import InProcessModelWorker, MultiprocessModelWorker
from . import config


def create_model_worker() -> InProcessModelWorker:
    """由 GPU Worker 子程序載入 Qwen3 Runner。"""
    from .adapter import Qwen3ModelAdapter
    from .model_runner import Qwen3ModelRunner

    runner = Qwen3ModelRunner()
    return InProcessModelWorker(runner, Qwen3ModelAdapter(runner))


def create_engine_core() -> InferenceEngine:
    """由 Engine Core 子程序建立共用 Scheduler、邏輯 KV lease 與 GPU Worker。"""
    worker = MultiprocessModelWorker(
        "models.qwen3_0_6b.engine_core:create_model_worker",
        startup_timeout_seconds=config.WORKER_STARTUP_TIMEOUT,
        request_timeout_seconds=config.WORKER_REQUEST_TIMEOUT,
    )
    return InferenceEngine(
        worker,
        model_id=config.MODEL_ALIAS,
        max_concurrent_requests=config.MAX_NUM_SEQS,
        max_waiting_requests=config.MAX_WAITING_REQUESTS,
        max_num_scheduled_tokens=config.MAX_NUM_SCHEDULED_TOKENS,
        prefill_chunk_size_tokens=config.PREFILL_CHUNK_SIZE_TOKENS,
        batch_wait_seconds=config.BATCH_WAIT_SECONDS,
        kv_cache_block_size_tokens=config.KV_CACHE_BLOCK_SIZE_TOKENS,
        kv_cache_capacity_tokens=config.KV_CACHE_CAPACITY_TOKENS,
        max_video_bytes=1,
    )
