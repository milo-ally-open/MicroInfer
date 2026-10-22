"""組裝 API Server，並透過 EngineClient 啟動獨立 Engine Core。"""

from __future__ import annotations

from api.app import create_app
from engine.client import EngineClient
from media.live_sessions import LiveSessionManager

from . import config


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


engine = EngineClient(
    "models.mage_vl_4b.engine_core:create_engine_core",
    startup_timeout_seconds=config.ENGINE_CORE_STARTUP_TIMEOUT,
    request_timeout_seconds=config.ENGINE_CORE_REQUEST_TIMEOUT,
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
