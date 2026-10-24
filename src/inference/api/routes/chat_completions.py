"""OpenAI 风格聊天请求及 SSE 回应路由。"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import replace
from typing import Any

import anyio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from api.schemas import ChatCompletionRequest, ChatCompletionResponse, ImageContentPart
from api.serializers import chat_completion, chat_completion_chunk
from engine.errors import InferenceError


router = APIRouter()
_STREAM_END = object()
logger = logging.getLogger(__name__)


class DisconnectAwareStreamingResponse(StreamingResponse):
    """同时送出 SSE 并监听客户端断线，及时关闭上游事件 iterator。"""

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
    """从同步 Engine stream 取得下一事件，以 sentinel 表示正常结束。"""
    try:
        return next(stream)
    except StopIteration:
        return _STREAM_END


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(body: ChatCompletionRequest, request: Request) -> Any:
    if body.model not in {None, request.app.state.model_id}:
        raise HTTPException(status_code=404, detail=f"Unknown model: {body.model}")
    modalities = request.app.state.chat_input_modalities
    if modalities is not None:
        image_input = any(
            isinstance(part, ImageContentPart)
            for message in body.messages
            if isinstance(message.content, list)
            for part in message.content
        )
        modality = "image" if image_input else "text"
        if modality not in modalities:
            kind = "image input" if image_input else "text-only chat input"
            raise HTTPException(status_code=400, detail=f"This model does not accept {kind}.")
    if body.stream and not request.app.state.streaming_enabled:
        raise HTTPException(status_code=400, detail="This model does not support streaming output.")
    request_body = body.model_dump(exclude_none=True)
    input_processor = request.app.state.input_processor
    # Tokenizer、Chat Template 与多模态前处理可能耗时，避免同步阻塞 ASGI event loop。
    try:
        prepared_input = (
            await run_in_threadpool(input_processor.prepare_chat, request_body)
            if input_processor is not None
            else None
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if body.stream:
        output_processor = (
            input_processor.create_stream_output_processor() if input_processor is not None else None
        )

        async def event_stream():
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            if prepared_input is None:
                stream = request.app.state.engine.stream_chat(request_body)
            else:
                stream = request.app.state.engine.stream_chat(request_body, prepared_input=prepared_input)
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
                    elif event.kind == "token" and event.token_id is not None and output_processor is not None:
                        text_delta = output_processor.push(event.token_id)
                        if not text_delta:
                            continue
                        chunk = chat_completion_chunk(
                            request_id, created, request.app.state.model_id, delta={"content": text_delta}
                        )
                    elif event.kind == "finish":
                        if output_processor is not None:
                            final_text = output_processor.finish()
                            if final_text:
                                final_chunk = chat_completion_chunk(
                                    request_id,
                                    created,
                                    request.app.state.model_id,
                                    delta={"content": final_text},
                                )
                                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
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
            except Exception:
                logger.exception("Streaming inference request failed")
                error = {"error": {"message": "Inference request failed.", "type": "server_error"}}
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
    if prepared_input is None:
        result = await run_in_threadpool(request.app.state.engine.generate_chat, request_body)
    else:
        result = await run_in_threadpool(
            request.app.state.engine.generate_chat, request_body, prepared_input
        )
    if result.token_ids and input_processor is not None:
        result = replace(result, content=input_processor.decode_tokens(result.token_ids))
    return chat_completion(result, request.app.state.model_id)
