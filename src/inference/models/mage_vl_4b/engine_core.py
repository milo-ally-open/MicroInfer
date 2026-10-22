"""建立 Mage-VL 模型的 Engine Core 與 GPU Worker。"""

from __future__ import annotations

from engine.engine import InferenceEngine
from engine.worker import InProcessModelWorker, MultiprocessModelWorker

from . import config


def create_model_worker() -> InProcessModelWorker:
    """由 GPU Worker 子程序呼叫，延後載入模型 Runner 與權重。"""
    from .adapter import MageVLModelAdapter
    from .model_runner import MageVLModelRunner

    model_runner = MageVLModelRunner()
    return InProcessModelWorker(model_runner, MageVLModelAdapter(model_runner))


def create_engine_core() -> InferenceEngine:
    """由獨立 Engine Core 子程序呼叫，建立 Scheduler、KV 管理及 GPU Worker。"""
    worker = MultiprocessModelWorker(
        "models.mage_vl_4b.engine_core:create_model_worker",
        startup_timeout_seconds=config.WORKER_STARTUP_TIMEOUT,
        request_timeout_seconds=config.WORKER_REQUEST_TIMEOUT,
    )
    return InferenceEngine(
        worker,
        model_id=config.MODEL_ID,
        max_concurrent_requests=config.MAX_NUM_SEQS,
        max_waiting_requests=config.MAX_WAITING_REQUESTS,
        max_num_scheduled_tokens=config.MAX_NUM_SCHEDULED_TOKENS,
        prefill_chunk_size_tokens=config.PREFILL_CHUNK_SIZE_TOKENS,
        batch_wait_seconds=config.BATCH_WAIT_SECONDS,
        kv_cache_block_size_tokens=config.KV_CACHE_BLOCK_SIZE_TOKENS,
        kv_cache_capacity_tokens=config.KV_CACHE_CAPACITY_TOKENS,
        max_video_bytes=config.MAX_VIDEO_BYTES,
    )
