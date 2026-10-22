"""OpenAI 風格聊天請求及 SSE 回應路由。"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import anyio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from api.schemas import ChatCompletionRequest, ChatCompletionResponse
from api.serializers import chat_completion, chat_completion_chunk
from engine.errors import InferenceError


router = APIRouter()
_STREAM_END = object()
logger = logging.getLogger(__name__)


class DisconnectAwareStreamingResponse(StreamingResponse):
    """同時送出 SSE 並監聽客戶端斷線，及時關閉上游事件 iterator。"""

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        async with anyio.create_task_group() as task_group:

            async def stream_response() -> None:
                await self.stream_response(send)
                task_group.cancel_scope.cancel()

            async def listen_for_disconnect() -> None:
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        logger.info("SSE client disconnected")
                        task_group.cancel_scope.cancel()
                        return

            task_group.start_soon(stream_response)
            task_group.start_soon(listen_for_disconnect)


def _next_stream_event(stream: Any) -> Any:
    """從同步 Engine stream 取得下一事件，以 sentinel 表示正常結束。"""
    try:
        return next(stream)
    except StopIteration:
        return _STREAM_END


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(body: ChatCompletionRequest, request: Request) -> Any:
    if body.model not in {None, request.app.state.model_id}:
        raise HTTPException(status_code=404, detail=f"Unknown model: {body.model}")
    request_body = body.model_dump(exclude_none=True)
    if body.stream:
        async def event_stream():
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            stream = request.app.state.engine.stream_chat(request_body)
            try:
                while True:
                    event = await run_in_threadpool(_next_stream_event, stream)
                    if event is _STREAM_END:
                        break
                    if event.kind == "start":
                        chunk = chat_completion_chunk(
                            request_id, created, request.app.state.model_id, delta={"role": "assistant"}
                        )
                    elif event.kind == "text":
                        chunk = chat_completion_chunk(
                            request_id, created, request.app.state.model_id, delta={"content": event.text}
                        )
                    elif event.kind == "finish":
                        prompt_tokens = int(event.prompt_tokens or 0)
                        completion_tokens = int(event.completion_tokens or 0)
                        chunk = chat_completion_chunk(
                            request_id,
                            created,
                            request.app.state.model_id,
                            delta={},
                            finish_reason=event.finish_reason,
                            usage={
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens,
                            },
                        )
                    else:
                        continue
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            except InferenceError as exc:
                error = {"error": {"message": exc.detail, "type": "invalid_request_error"}}
                yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
            except Exception as exc:
                error = {"error": {"message": str(exc), "type": "server_error"}}
                yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
            finally:
                cancel = getattr(stream, "cancel", None)
                with anyio.CancelScope(shield=True):
                    await run_in_threadpool(cancel if callable(cancel) else stream.close)
            yield "data: [DONE]\n\n"

        return DisconnectAwareStreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    result = await run_in_threadpool(request.app.state.engine.generate_chat, request_body)
    return chat_completion(result, request.app.state.model_id)
