"""組裝 API Server、路由及 Inference Engine 生命週期。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.routes import chat_completions, live_sessions, video_understanding
from api.schemas import HealthResponse, ModelListResponse
from engine.errors import InferenceError


def create_app(
    *,
    engine: Any,
    model_id: str,
    request_limits: dict[str, tuple[str, int]],
    title: str | None = None,
) -> FastAPI:
    """建立不依賴特定模型套件的 FastAPI app。"""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        engine.start()
        try:
            yield
        finally:
            engine.close()

    app = FastAPI(title=title or f"{model_id} API", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine
    app.state.model_id = model_id
    app.state.request_limits = request_limits

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> dict[str, Any]:
        return request.app.state.engine.health()

    @app.get("/v1/models", response_model=ModelListResponse)
    def models(request: Request) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": request.app.state.model_id, "object": "model", "created": 0, "owned_by": "local"}],
        }

    @app.middleware("http")
    async def limit_api_request_content_length(request: Request, call_next: Any) -> Any:
        configured_limit = request.app.state.request_limits.get(request.url.path) if request.method == "POST" else None
        if configured_limit is not None:
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
                return JSONResponse(status_code=413, content={"detail": f"{label} exceeds the configured {max_bytes}-byte limit."})
        return await call_next(request)

    @app.exception_handler(InferenceError)
    async def inference_error_handler(_: Request, exc: InferenceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    app.include_router(chat_completions.router)
    app.include_router(video_understanding.router)
    app.include_router(live_sessions.router)
    return app
