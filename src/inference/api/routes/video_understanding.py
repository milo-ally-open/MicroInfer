"""上传影片理解 endpoint。"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from inference.api.contracts.schemas import VideoUnderstandingResponse
from inference.api.contracts.serializers import chat_completion


router = APIRouter()


@router.post("/v1/videos/understand", response_model=VideoUnderstandingResponse)
def understand_uploaded_video(
    request: Request,
    video: UploadFile = File(...),
    question: str = Form(...),
    video_backend: Literal["frames", "codec"] | None = Form(default=None),
    codec_engine: Literal["hevc", "dcvc-rt"] | None = Form(default=None),
    max_tokens: int | None = Form(default=None),
    temperature: float | None = Form(default=None),
    top_p: float | None = Form(default=None),
) -> Any:
    if not question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty.")
    request_body = {
        "question": question,
        "video_backend": video_backend,
        "codec_engine": codec_engine,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    try:
        result = request.app.state.engine.understand_video(video.file, video.filename, request_body)
        return chat_completion(result, request.app.state.model_id, video=True)
    finally:
        video.file.close()
