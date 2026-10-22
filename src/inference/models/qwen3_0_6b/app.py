"""Qwen3-0.6B 純文字 OpenAI 風格 API。"""

from __future__ import annotations

from api.app import create_app
from engine.client import EngineClient
from . import config


engine = EngineClient(
    "models.qwen3_0_6b.engine_core:create_engine_core",
    startup_timeout_seconds=config.ENGINE_CORE_STARTUP_TIMEOUT,
    request_timeout_seconds=config.ENGINE_CORE_REQUEST_TIMEOUT,
    max_video_bytes=1,
)

app = create_app(
    engine=engine,
    model_id=config.MODEL_ALIAS,
    title="Qwen3-0.6B API",
    request_limits={
        "/v1/chat/completions": ("Chat request", config.MAX_CHAT_REQUEST_BYTES),
    },
    enable_video=False,
    enable_live_sessions=False,
)
