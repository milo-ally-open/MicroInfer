"""对外推理 API 的 Pydantic schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

class ImageURL(APIModel):
    url: str
    detail: str | None = None

class TextContentPart(APIModel):
    type: Literal["text"]
    text: str

class ImageContentPart(APIModel):
    type: Literal["image_url"]
    image_url: ImageURL

class FunctionToolCall(APIModel):
    name: str
    arguments: str

class AssistantToolCall(APIModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionToolCall

class FunctionToolDefinition(APIModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    strict: bool | None = None

class ToolDefinition(APIModel):
    type: Literal["function"]
    function: FunctionToolDefinition

class ChatMessage(APIModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[TextContentPart | ImageContentPart] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[AssistantToolCall] | None = None

class GenerationRequest(APIModel):
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    min_p: float | None = Field(default=None, ge=0, le=1)
    repetition_penalty: float | None = Field(default=None, gt=0)
    no_repeat_ngram_size: int | None = Field(default=None, ge=0)
    num_beams: int | None = Field(default=None, ge=1)

class ChatCompletionRequest(GenerationRequest):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None

class AssistantMessage(APIModel):
    role: Literal["assistant"] = "assistant"
    content: str | None
    tool_calls: list[AssistantToolCall] | None = None

class TokenUsage(APIModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CompletionChoice(APIModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: Literal["stop", "length", "tool_calls"]

class ChatCompletionResponse(APIModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: TokenUsage

class StreamDelta(APIModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

class StreamChoice(APIModel):
    index: int = 0
    delta: StreamDelta
    finish_reason: Literal["stop", "length", "tool_calls"] | None = None

class ChatCompletionChunk(APIModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]
    usage: TokenUsage | None = None

class VideoUnderstandingRequest(GenerationRequest):
    question: str = Field(min_length=1)
    video_backend: Literal["frames", "codec"] | None = None
    codec_engine: Literal["hevc", "dcvc-rt"] | None = None

class VideoMetadata(APIModel):
    backend: Literal["frames", "codec"] = "codec"
    codec_engine: str | None = None
    duration_seconds: float
    source_fps: float
    source_frame_count: int
    source_width: int
    source_height: int
    sampled_frame_count: int | None = None
    sampled_timestamps_seconds: list[float] = Field(default_factory=list)
    codec_canvas_count: int | None = None
    upload_bytes: int

class RequestTimings(APIModel):
    staging_and_decode: float
    processor: float
    generation: float
    server_total: float

class GPUMemoryUsage(APIModel):
    peak_allocated: int
    peak_reserved: int

class VideoUnderstandingResponse(ChatCompletionResponse):
    object: Literal["magevl4b.video.understanding"] = "magevl4b.video.understanding"
    video: VideoMetadata
    timing_seconds: RequestTimings
    gpu_memory_bytes: GPUMemoryUsage

class HealthResponse(APIModel):
    status: Literal["ok", "starting"]
    backend: str = "transformers"
    model: str
    revision: str
    quantization: str = "none"
    live_sessions: int
    live_video_enabled: bool = False
    api_pid: int | None = None
    engine_core_pid: int | None = None
    worker_pid: int | None = None
    scheduler: dict[str, int] = Field(default_factory=dict)
    kv_cache: dict[str, int] = Field(default_factory=dict)
    worker_kv_cache: dict[str, int] = Field(default_factory=dict)
    gpu_peak_memory: dict[str, int] = Field(default_factory=dict)
    gpu: str | None = None
    vram_free_bytes: int | None = None
    vram_total_bytes: int | None = None

class ModelCard(APIModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "local"

class ModelListResponse(APIModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]

class LiveSessionCreateRequest(GenerationRequest):
    question: str = Field(min_length=1, max_length=2000)
    video_backend: Literal["frames", "codec"] | None = None
    codec_engine: Literal["hevc", "dcvc-rt"] | None = None
    sample_interval_seconds: float | None = None
    window_seconds: float | None = None
    max_duration_seconds: float | None = None

class LivePublishURLs(APIModel):
    rtsp: str
    rtmp: str

class LiveSessionSettings(APIModel):
    video_backend: Literal["frames", "codec"]
    codec_engine: str | None = None
    codec_target_canvas: int | None = None
    sample_interval_seconds: float
    window_seconds: float
    segment_seconds: float | None = None
    max_duration_seconds: float
    max_frames: int

class LiveSessionResponse(APIModel):
    id: str
    status: str
    model: str
    publish: LivePublishURLs
    events_url: str
    stop_url: str
    settings: LiveSessionSettings

class LiveSessionStopResponse(APIModel):
    id: str
    status: Literal["stopping"]
