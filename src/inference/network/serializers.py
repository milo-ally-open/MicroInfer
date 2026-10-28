"""将共用推理结果序列化为目前兼容的 HTTP API 字段。"""

from __future__ import annotations

import time
import uuid
from typing import Any

from inference.engine.sequence import InferenceResult


def chat_completion(
    result: InferenceResult,
    model_id: str,
    *,
    video: bool = False,
    tool_calls: list[dict[str, Any]] | None = None,
    content: str | None = None,
) -> dict[str, Any]:
    """创建聊天或视频理解 endpoint 使用的 JSON response。"""
    response: dict[str, Any] = {
        "id": f"{'videocmpl' if video else 'chatcmpl'}-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.content if content is None else content},
            "finish_reason": result.finish_reason,
        }],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    }
    if tool_calls:
        response["choices"][0]["message"]["tool_calls"] = [
            call.as_dict() if hasattr(call, "as_dict") else call for call in tool_calls
        ]
        if content is None:
            response["choices"][0]["message"]["content"] = None
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
    """创建目前聊天 SSE 使用的 OpenAI 兼容 chunk。"""
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
