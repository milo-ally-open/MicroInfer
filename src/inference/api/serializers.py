"""將共用推論結果序列化為目前相容的 HTTP API 欄位。"""

from __future__ import annotations

import time
import uuid
from typing import Any

from engine.engine import InferenceResult


def chat_completion(result: InferenceResult, model_id: str, *, video: bool = False) -> dict[str, Any]:
    """建立聊天或視訊理解 endpoint 使用的 JSON response。"""
    response: dict[str, Any] = {
        "id": f"{'videocmpl' if video else 'chatcmpl'}-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.content},
            "finish_reason": result.finish_reason,
        }],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    }
    if video:
        response["object"] = "magevl4b.video.understanding"
        response.update(result.metadata)
    return response


def chat_completion_chunk(
    request_id: str,
    created: int,
    model_id: str,
    *,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    """建立目前聊天 SSE 使用的 OpenAI 相容 chunk。"""
    chunk: dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk
