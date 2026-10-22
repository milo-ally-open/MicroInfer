"""OpenAI 風格聊天請求及 SSE 回應路由。"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from api.schemas import ChatCompletionRequest, ChatCompletionResponse
from api.serializers import chat_completion, chat_completion_chunk
from engine.errors import InferenceError


router = APIRouter()


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(body: ChatCompletionRequest, request: Request) -> Any:
    if body.model not in {None, request.app.state.model_id}:
        raise HTTPException(status_code=404, detail=f"Unknown model: {body.model}")
    request_body = body.model_dump(exclude_none=True)
    if body.stream:
        def event_stream():
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            stream = request.app.state.engine.stream_chat(request_body)
            try:
                for event in stream:
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
                stream.close()
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    result = await run_in_threadpool(request.app.state.engine.generate_chat, request_body)
    return chat_completion(result, request.app.state.model_id)
