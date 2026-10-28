"""OpenAI 风格聊天请求及 SSE 响应路由。"""

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

from inference.network.schemas import ChatCompletionRequest, ChatCompletionResponse, ImageContentPart
from inference.network.serializers import chat_completion, chat_completion_chunk
from inference.network.tool_calling import ChatOutputDelta, validate_tool_definitions
from inference.engine.sequence import InferenceError


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


def _validate_tool_request(body: ChatCompletionRequest, request: Request) -> tuple[list[dict[str, Any]], str]:
    raw_tools = body.model_dump(exclude_none=True).get("tools")
    try:
        tools = validate_tool_definitions(raw_tools) if raw_tools is not None else []
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    choice = body.tool_choice if body.tool_choice is not None else ("auto" if tools else "none")
    if not isinstance(choice, str) or choice not in {"auto", "none"}:
        raise HTTPException(status_code=400, detail="tool_choice supports only 'auto' and 'none'.")
    if choice == "auto" and not tools:
        raise HTTPException(status_code=400, detail="tool_choice='auto' requires at least one tool definition.")
    if body.parallel_tool_calls is True:
        raise HTTPException(status_code=400, detail="parallel_tool_calls=true is not supported; calls are limited to one per turn.")
    enabled = bool(request.app.state.tool_calling_enabled)
    if tools and not enabled:
        raise HTTPException(status_code=400, detail="The selected model does not support tool calling.")
    if tools and request.app.state.tool_call_adapter is None:
        raise HTTPException(status_code=503, detail="The model tool-call adapter is not configured.")
    active_tools = tools if choice == "auto" else []
    return active_tools, choice


def _tool_deltas_as_chunks(
    deltas: list[ChatOutputDelta], request_id: str, created: int, model_id: str
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for output in deltas:
        delta: dict[str, Any] = {}
        if output.content:
            delta["content"] = output.content
        if output.tool_call is not None:
            delta["tool_calls"] = [output.tool_call.as_dict()]
        if delta:
            chunks.append(chat_completion_chunk(request_id, created, model_id, delta=delta))
    return chunks


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
    active_tools, tool_choice = _validate_tool_request(body, request)
    request_body["tool_choice"] = tool_choice
    if body.tools is not None:
        request_body["tools"] = active_tools
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
        tool_stream_parser = None
        tool_adapter = request.app.state.tool_call_adapter
        if tool_adapter is not None:
            tool_stream_parser = tool_adapter.create_stream_parser(active_tools, allow_parallel=False)

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
                        deltas = tool_stream_parser.feed(event.text or "") if tool_stream_parser is not None else [ChatOutputDelta(content=event.text)]
                        for rendered in _tool_deltas_as_chunks(deltas, request_id, created, request.app.state.model_id):
                            yield f"data: {json.dumps(rendered, ensure_ascii=False)}\n\n"
                        continue
                    elif event.kind == "token" and event.token_id is not None and output_processor is not None:
                        text_delta = output_processor.push(event.token_id)
                        if not text_delta:
                            continue
                        deltas = tool_stream_parser.feed(text_delta) if tool_stream_parser is not None else [ChatOutputDelta(content=text_delta)]
                        for rendered in _tool_deltas_as_chunks(deltas, request_id, created, request.app.state.model_id):
                            yield f"data: {json.dumps(rendered, ensure_ascii=False)}\n\n"
                        continue
                    elif event.kind == "finish":
                        if output_processor is not None:
                            final_text = output_processor.finish()
                            if final_text:
                                deltas = tool_stream_parser.feed(final_text) if tool_stream_parser is not None else [ChatOutputDelta(content=final_text)]
                                for rendered in _tool_deltas_as_chunks(deltas, request_id, created, request.app.state.model_id):
                                    yield f"data: {json.dumps(rendered, ensure_ascii=False)}\n\n"
                        if tool_stream_parser is not None:
                            for rendered in _tool_deltas_as_chunks(
                                tool_stream_parser.finish(), request_id, created, request.app.state.model_id
                            ):
                                yield f"data: {json.dumps(rendered, ensure_ascii=False)}\n\n"
                        prompt_tokens = int(event.prompt_tokens or 0)
                        completion_tokens = int(event.completion_tokens or 0)
                        finish_reason = "tool_calls" if tool_stream_parser is not None and tool_stream_parser.tool_calls else event.finish_reason
                        chunk = chat_completion_chunk(
                            request_id,
                            created,
                            request.app.state.model_id,
                            delta={},
                            finish_reason=finish_reason,
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
            except ValueError as exc:
                logger.warning("Model returned an invalid tool call: %s", exc)
                error = {"error": {"message": "The model returned an invalid tool call.", "type": "server_error", "code": "invalid_tool_call"}}
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
    parsed_output = None
    if request.app.state.tool_call_adapter is not None:
        try:
            parsed_output = request.app.state.tool_call_adapter.parse_output(
                result.content, active_tools, allow_parallel=False
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=502, detail="The model returned an invalid tool call.") from exc
        result = replace(
            result,
            content=parsed_output.content or "",
            finish_reason="tool_calls" if parsed_output.tool_calls else result.finish_reason,
        )
    return chat_completion(
        result,
        request.app.state.model_id,
        tool_calls=None if parsed_output is None else [call.as_dict() for call in parsed_output.tool_calls],
        content=None if parsed_output is None else parsed_output.content,
    )
