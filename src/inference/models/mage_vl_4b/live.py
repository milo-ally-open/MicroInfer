"""建立 Mage-VL 专属的 Live session 管理器。"""

from __future__ import annotations

from typing import Any

from media.live_sessions import LiveSessionManager

from .utils import loader as config


def _resize_live_frame(image: Any, max_side: int) -> Any:
    """依模型设定调整 Live frame，并保留长宽比例。"""
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


def create_live_session_manager(engine: Any) -> LiveSessionManager:
    """由共用 API 装配器呼叫，将 Mage-VL 即时推流接到共用 Engine。"""
    return LiveSessionManager(
        settings=config.LIVE_SETTINGS,
        try_inference=engine.try_live_inference,
        transform_frame=_resize_live_frame,
    )
