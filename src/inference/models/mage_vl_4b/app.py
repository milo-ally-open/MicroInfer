"""組裝 API Server 與獨立 Mage-VL Model Worker。"""

from __future__ import annotations

from api.app import create_app
from engine.engine import InferenceEngine
from engine.worker import InProcessModelWorker, MultiprocessModelWorker
from media.live_sessions import LiveSessionManager

from . import config


def create_model_worker() -> InProcessModelWorker:
    """由 Worker 子程序呼叫，延後載入 GPU 模型程式。"""
    from .adapter import MageVLModelAdapter
    from .model_runner import MageVLModelRunner

    model_runner = MageVLModelRunner()
    return InProcessModelWorker(model_runner, MageVLModelAdapter(model_runner))


def _resize_live_frame(image: object, max_side: int) -> object:
    """在 API Server 以 Pillow 調整 Live frame，不匯入模型 Processor 或 GPU Runtime。"""
    from PIL import Image

    resampling = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }.get(config.IMAGE_RESAMPLING)
    if resampling is None:
        raise RuntimeError(f"Unsupported image resampling mode in config: {config.IMAGE_RESAMPLING}")
    image.thumbnail((max_side, max_side), resampling)
    return image


worker = MultiprocessModelWorker(
    "models.mage_vl_4b.app:create_model_worker",
    startup_timeout_seconds=config.WORKER_STARTUP_TIMEOUT,
    request_timeout_seconds=config.WORKER_REQUEST_TIMEOUT,
)
engine = InferenceEngine(
    worker,
    model_id=config.MODEL_ID,
    max_concurrent_requests=config.MAX_NUM_SEQS,
    max_waiting_requests=config.MAX_WAITING_REQUESTS,
    max_num_scheduled_tokens=config.MAX_NUM_SCHEDULED_TOKENS,
    kv_cache_block_size_tokens=config.KV_CACHE_BLOCK_SIZE_TOKENS,
    kv_cache_capacity_tokens=config.KV_CACHE_CAPACITY_TOKENS,
    max_video_bytes=config.MAX_VIDEO_BYTES,
)
engine.attach_live_sessions(LiveSessionManager(
    settings=config.LIVE_SETTINGS,
    try_inference=engine.try_live_inference,
    transform_frame=_resize_live_frame,
))

app = create_app(
    engine=engine,
    model_id=config.MODEL_ID,
    title="Mage-VL-4B API",
    request_limits={
        "/v1/chat/completions": ("Chat request", config.MAX_CHAT_REQUEST_BYTES),
        "/v1/videos/understand": ("Video upload request", config.MAX_VIDEO_REQUEST_BYTES),
        "/v1/live/sessions": ("Live session request", config.MAX_LIVE_REQUEST_BYTES),
        "/v1/live/mediamtx/auth": ("MediaMTX authorization request", config.MAX_LIVE_REQUEST_BYTES),
    },
)
