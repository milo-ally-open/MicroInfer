"""MediaMTX 即時工作階段 API 與 SSE 事件路由。"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from dataclasses import replace
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from api.schemas import LiveSessionCreateRequest, LiveSessionResponse, LiveSessionStopResponse
from media.live_sessions import LiveSessionLimitError, LiveSessionNotFoundError


router = APIRouter()


@router.post("/v1/live/sessions", response_model=LiveSessionResponse)
async def create_session(request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    engine = request.app.state.engine
    settings = engine.live_settings()
    try:
        session_request = LiveSessionCreateRequest.model_validate(payload)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc
    question = session_request.question
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise HTTPException(status_code=400, detail="question must be a non-empty string of at most 2000 characters.")

    try:
        sample_interval = float(session_request.sample_interval_seconds if session_request.sample_interval_seconds is not None else settings.sample_interval_seconds)
        window_seconds = float(session_request.window_seconds if session_request.window_seconds is not None else settings.window_seconds)
        max_duration = float(session_request.max_duration_seconds if session_request.max_duration_seconds is not None else settings.max_duration_seconds)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Live timing options must be numbers.") from exc
    video_backend = session_request.video_backend or settings.video_backend
    codec_engine = session_request.codec_engine or settings.codec_engine
    if not 0.25 <= sample_interval <= 60:
        raise HTTPException(status_code=400, detail="sample_interval_seconds must be between 0.25 and 60.")
    if not 1 <= window_seconds <= 60 or (video_backend == "frames" and window_seconds < sample_interval):
        raise HTTPException(status_code=400, detail="window_seconds must be between 1 and 60; frame mode also requires it to be at least the sampling interval.")
    if not 1 <= max_duration <= settings.max_duration_seconds:
        raise HTTPException(status_code=400, detail=f"max_duration_seconds must be between 1 and {settings.max_duration_seconds:g}.")
    if video_backend == "codec" and (settings.live_codec_target_canvas <= 0 or settings.codec_max_pixels <= 0):
        raise HTTPException(status_code=500, detail="Codec target_canvas and max_pixels must be positive in the API YAML.")
    engine.validate_live_video_backend(video_backend, codec_engine)

    generation_request = session_request.model_dump(
        include={"max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k"},
        exclude_none=True,
    )
    try:
        generation_options = engine.generation_options(generation_request)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid generation option: {exc}") from exc

    session_settings = replace(
        settings,
        sample_interval_seconds=sample_interval,
        window_seconds=window_seconds,
        max_duration_seconds=max_duration,
        video_backend=video_backend,
        codec_engine=codec_engine,
    )
    session_id = secrets.token_urlsafe(24)
    try:
        session = engine.start_live_session(
            session_id=session_id,
            question=question.strip(),
            generation_options=generation_options,
            settings=session_settings,
        )
    except LiveSessionLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    public_host = settings.public_host or request.url.hostname or "127.0.0.1"
    return {
        "id": session_id,
        "status": session.status,
        "model": request.app.state.model_id,
        "publish": {
            "rtsp": f"rtsp://{public_host}:{settings.rtsp_port}/{session_id}",
            "rtmp": f"rtmp://{public_host}:{settings.rtmp_port}/{session_id}",
        },
        "events_url": f"{str(request.base_url).rstrip('/')}/v1/live/sessions/{session_id}/events",
        "stop_url": f"{str(request.base_url).rstrip('/')}/v1/live/sessions/{session_id}",
        "settings": {
            "video_backend": video_backend,
            "codec_engine": codec_engine,
            "codec_target_canvas": settings.live_codec_target_canvas if video_backend == "codec" else None,
            "sample_interval_seconds": sample_interval,
            "window_seconds": window_seconds,
            "segment_seconds": window_seconds if video_backend == "codec" else None,
            "max_duration_seconds": max_duration,
            "max_frames": settings.max_frames,
        },
    }


@router.get("/v1/live/sessions/{session_id}/events")
async def session_events(session_id: str, request: Request) -> StreamingResponse:
    try:
        session = request.app.state.engine.get_live_session(session_id)
    except LiveSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Live session was not found or has expired.") from exc
    try:
        last_event_id = max(0, int(request.headers.get("last-event-id", "0")))
    except ValueError:
        last_event_id = 0

    async def event_stream():
        nonlocal last_event_id
        last_keepalive = time.monotonic()
        while True:
            if await request.is_disconnected():
                return
            events, finished = session.events_after(last_event_id)
            for event in events:
                last_event_id = event["id"]
                data = json.dumps(event["data"], ensure_ascii=False)
                yield f"id: {event['id']}\nevent: {event['event']}\ndata: {data}\n\n"
            if finished and not events:
                return
            if time.monotonic() - last_keepalive >= 15:
                yield ": keep-alive\n\n"
                last_keepalive = time.monotonic()
            await asyncio.sleep(0.25)

    return StreamingResponse(
        event_stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.delete("/v1/live/sessions/{session_id}", response_model=LiveSessionStopResponse)
def stop_session(session_id: str, request: Request) -> dict[str, str]:
    try:
        session = request.app.state.engine.get_live_session(session_id)
    except LiveSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Live session was not found or has expired.") from exc
    request.app.state.engine.stop_live_session(session_id)
    return {"id": session_id, "status": "stopping"}


@router.post("/v1/live/mediamtx/auth")
async def authorize_mediamtx_client(request: Request) -> Response:
    """只授權仍有效工作階段使用的 RTSP／RTMP 路徑。"""
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    action, protocol, path = body.get("action"), body.get("protocol"), body.get("path")
    if action not in {"publish", "read"} or protocol not in {"rtsp", "rtmp"} or not isinstance(path, str):
        raise HTTPException(status_code=401, detail="MediaMTX action is not allowed.")
    try:
        session = request.app.state.engine.get_live_session(path.strip("/"))
    except LiveSessionNotFoundError as exc:
        raise HTTPException(status_code=401, detail="MediaMTX stream path is not an active live session.") from exc
    if not session.is_active() or session.stop_event.is_set():
        raise HTTPException(status_code=401, detail="MediaMTX stream path is not an active live session.")
    return Response(status_code=204)
