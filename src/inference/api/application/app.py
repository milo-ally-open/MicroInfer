"""组装 API Server、路由及 Inference Engine 生命周期。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from inference.api.routes import chat_completions, live_sessions, video_understanding
from inference.api.contracts.schemas import HealthResponse, ModelListResponse
from inference.engine.contracts import InferenceError


class ContentLengthLimitMiddleware:
    """以纯 ASGI middleware 验证本文长度，避免包装 SSE response iterator。"""

    def __init__(self, app: Any, *, request_limits: dict[str, tuple[str, int]]) -> None:
        self.app = app
        for path, policy in request_limits.items():
            if (
                not isinstance(path, str)
                or not path.startswith("/")
                or not isinstance(policy, (tuple, list))
                or len(policy) != 2
                or not isinstance(policy[0], str)
                or not policy[0].strip()
                or not isinstance(policy[1], int)
                or isinstance(policy[1], bool)
                or policy[1] < 1
            ):
                raise ValueError(f"Invalid request size policy for {path!r}.")
        self.request_limits = dict(request_limits)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        configured_limit = self.request_limits.get(str(scope.get("path", "")))
        if configured_limit is None:
            await self.app(scope, receive, send)
            return
        label, max_bytes = configured_limit
        content_lengths = [
            value for key, value in scope.get("headers", ()) if key.lower() == b"content-length"
        ]
        if not content_lengths:
            response = JSONResponse(
                status_code=411,
                content={"detail": "Content-Length is required for this endpoint."},
            )
        elif len(content_lengths) != 1 or not content_lengths[0].isdigit():
            response = JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
        else:
            try:
                request_bytes = int(content_lengths[0])
            except ValueError:
                response = JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
            else:
                if request_bytes < 0:
                    response = JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
                elif request_bytes > max_bytes:
                    response = JSONResponse(
                        status_code=413,
                        content={"detail": f"{label} exceeds the configured {max_bytes}-byte limit."},
                    )
                else:
                    await self.app(scope, receive, send)
                    return
        await response(scope, receive, send)


def create_app(
    *,
    engine: Any,
    model_id: str,
    request_limits: dict[str, tuple[str, int]],
    input_processor: Any | None = None,
    tool_call_adapter: Any | None = None,
    tool_calling_enabled: bool = False,
    chat_input_modalities: frozenset[str] | None = None,
    enable_streaming: bool = True,
    title: str | None = None,
    enable_video: bool = True,
    enable_live_sessions: bool = True,
) -> FastAPI:
    """建立不依赖特定模型套件的 FastAPI app。"""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        engine_started = False
        try:
            if input_processor is not None:
                input_processor.start()
            engine.start()
            engine_started = True
            yield
        finally:
            try:
                if engine_started:
                    engine.close()
            finally:
                if input_processor is not None:
                    input_processor.close()

    app = FastAPI(title=title or f"{model_id} API", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine
    app.state.model_id = model_id
    app.state.request_limits = request_limits
    app.state.input_processor = input_processor
    app.state.tool_call_adapter = tool_call_adapter
    app.state.tool_calling_enabled = tool_calling_enabled
    app.state.chat_input_modalities = chat_input_modalities
    app.state.streaming_enabled = enable_streaming

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> dict[str, Any]:
        return request.app.state.engine.health()

    @app.get("/v1/models", response_model=ModelListResponse)
    def models(request: Request) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": request.app.state.model_id, "object": "model", "created": 0, "owned_by": "local"}],
        }

    app.add_middleware(ContentLengthLimitMiddleware, request_limits=request_limits)

    @app.exception_handler(InferenceError)
    async def inference_error_handler(_: Request, exc: InferenceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    app.include_router(chat_completions.router)
    if enable_video:
        app.include_router(video_understanding.router)
    if enable_live_sessions:
        app.include_router(live_sessions.router)
    return app
