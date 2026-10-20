"""組裝 Mage-VL-4B API 應用程式及其 HTTP 路由。"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from . import service as inference

system_router = APIRouter()
chat_router = APIRouter()
video_router = APIRouter()


@system_router.get("/health", response_model=inference.HealthResponse)
def health() -> dict[str, Any]:
    """回報模型服務的健康狀態。"""
    return inference.INFERENCE_SERVICE.health()


@system_router.get("/v1/models", response_model=inference.ModelListResponse)
def models() -> dict[str, Any]:
    """回傳此服務目前提供的模型清單。"""
    return inference.INFERENCE_SERVICE.model_list()


@chat_router.post("/v1/chat/completions", response_model=inference.ChatCompletionResponse)
async def chat_completions(body: inference.ChatCompletionRequest) -> Any:
    """處理文字或圖片對話，並依請求設定選擇串流或一般回應。"""
    if body.model not in {None, inference.MODEL_ID}:
        raise HTTPException(status_code=404, detail=f"Unknown model: {body.model}")

    request_body = body.model_dump(exclude_none=True)
    if body.stream:
        return StreamingResponse(
            inference.INFERENCE_SERVICE.stream_chat(request_body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    return await run_in_threadpool(inference.INFERENCE_SERVICE.generate_chat, request_body)


@video_router.post("/v1/videos/understand", response_model=inference.VideoUnderstandingResponse)
def understand_uploaded_video(
    video: UploadFile = File(...),
    question: str = Form(...),
    video_backend: Literal["frames", "codec"] | None = Form(default=None),
    codec_engine: Literal["hevc", "dcvc-rt"] | None = Form(default=None),
    max_tokens: int | None = Form(default=None),
    temperature: float | None = Form(default=None),
    top_p: float | None = Form(default=None),
) -> Any:
    """接收上傳視訊與分析參數，並回傳視訊理解結果。"""
    if not question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty.")
    request = inference.VideoUnderstandingRequest(
        question=question,
        video_backend=video_backend,
        codec_engine=codec_engine,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return inference.INFERENCE_SERVICE.understand_video(video, request)


app = FastAPI(title="Mage-VL-4B API", version="0.1.0", lifespan=inference.lifespan)
app.include_router(system_router)
app.include_router(chat_router)
app.include_router(video_router)
app.include_router(inference.LIVE_VIDEO.router)


@app.middleware("http")
async def limit_api_request_content_length(request: Request, call_next: Any) -> Any:
    """在讀取 JSON 或 multipart 本文前，先拒絕超過限制的請求。"""
    limits = {
        "/v1/chat/completions": ("Chat request", inference.MAX_CHAT_REQUEST_BYTES),
        "/v1/videos/understand": ("Video upload request", inference.MAX_VIDEO_REQUEST_BYTES),
        "/v1/live/sessions": ("Live session request", inference.MAX_LIVE_REQUEST_BYTES),
        "/v1/live/mediamtx/auth": ("MediaMTX authorization request", inference.MAX_LIVE_REQUEST_BYTES),
    }
    configured_limit = limits.get(request.url.path) if request.method == "POST" else None
    if configured_limit is not None:
        # 先檢查本文長度，避免解析大型請求時占用過多記憶體。
        label, max_bytes = configured_limit
        content_length = request.headers.get("content-length")
        if content_length is None:
            return JSONResponse(status_code=411, content={"detail": "Content-Length is required for this endpoint."})
        try:
            request_bytes = int(content_length)
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
        if request_bytes < 0:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
        if request_bytes > max_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": f"{label} exceeds the configured {max_bytes}-byte limit."},
            )
    return await call_next(request)
