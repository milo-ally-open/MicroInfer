"""讀取 Mage-VL YAML，集中保存模型專屬設定。"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from media.live_sessions import LiveVideoSettings


CONFIG_PATH = os.environ.get("INFERENCE_CONFIG")
if not CONFIG_PATH:
    raise RuntimeError("Set INFERENCE_CONFIG to the selected model YAML before importing the API app.")
CONFIG = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
NETWORK_CONFIG_PATH = os.environ.get("INFERENCE_NETWORK_CONFIG")
if not NETWORK_CONFIG_PATH:
    raise RuntimeError("Set INFERENCE_NETWORK_CONFIG to the selected network YAML before importing the API app.")
NETWORK_CONFIG = yaml.safe_load(Path(NETWORK_CONFIG_PATH).read_text(encoding="utf-8"))
if not isinstance(NETWORK_CONFIG, dict):
    raise RuntimeError("The selected network YAML must contain a mapping at the top level.")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODEL_CONFIG = CONFIG["model"]
QUANTIZATION_CONFIG = MODEL_CONFIG["quantization"]
COMPONENTS_CONFIG = MODEL_CONFIG.get("components", {})
if not isinstance(COMPONENTS_CONFIG, dict):
    raise RuntimeError("model.components must be a mapping in the API YAML.")
COMPONENT_REPLACEMENTS = COMPONENTS_CONFIG.get("replacements", {})
if not isinstance(COMPONENT_REPLACEMENTS, dict):
    raise RuntimeError("model.components.replacements must be a mapping in the API YAML.")
LOCAL_FILES_ONLY = bool(CONFIG.get("runtime", {}).get("local_files_only", False)) or os.environ.get(
    "INFERENCE_OFFLINE", ""
).strip().lower() in {"1", "true", "yes"}
API_CONFIG = CONFIG["api"]
ENGINE_CONFIG = CONFIG.get("engine", {})
GENERATION_CONFIG = CONFIG["generation"]
MODEL_ID = MODEL_CONFIG["id"]
MODEL_REVISION = MODEL_CONFIG["revision"]


def resolve_model_load_path() -> str:
    """準備固定 revision 的模型快照並回傳快取路徑。"""
    local_model_path = CONFIG.get("runtime", {}).get("local_model_path")
    if local_model_path:
        path = Path(os.path.expandvars(os.path.expanduser(str(local_model_path))))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        if not path.is_dir() or not (path / "config.json").is_file():
            raise FileNotFoundError(f"Local model snapshot is incomplete or missing: {path}")
        return str(path)

    hf_home_value = os.environ.get("HF_HOME") or CONFIG.get("runtime", {}).get("hf_home", "checkpoints")
    hf_home = Path(os.path.expandvars(os.path.expanduser(str(hf_home_value))))
    if not hf_home.is_absolute():
        hf_home = PROJECT_ROOT / hf_home
    cache_dir = hf_home / "hub"
    try:
        from huggingface_hub import snapshot_download

        return str(
            snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=str(cache_dir),
                local_files_only=LOCAL_FILES_ONLY,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not prepare model revision {MODEL_REVISION} in the project cache {cache_dir}: {exc}"
        ) from exc


MAX_IMAGE_SIDE = int(API_CONFIG["max_image_side"])
IMAGE_RESAMPLING = str(API_CONFIG["image_resampling"]).lower()
MAX_CONTEXT_TOKENS = int(API_CONFIG["max_context_tokens"])
MAX_NEW_TOKENS = int(API_CONFIG["max_new_tokens"])
MAX_NUM_SEQS = int(ENGINE_CONFIG.get("max_num_seqs", API_CONFIG.get("max_concurrent_requests", 4)))
MAX_WAITING_REQUESTS = int(ENGINE_CONFIG.get("max_waiting_requests", 16))
MAX_NUM_SCHEDULED_TOKENS = int(ENGINE_CONFIG.get("max_num_scheduled_tokens", MAX_CONTEXT_TOKENS))
BATCH_WAIT_SECONDS = float(ENGINE_CONFIG.get("batch_wait_seconds", 0.01))
if MAX_NUM_SCHEDULED_TOKENS < MAX_CONTEXT_TOKENS:
    raise RuntimeError(
        "engine.max_num_scheduled_tokens must be at least api.max_context_tokens "
        "while chunked prefill is disabled."
    )
KV_CACHE_CONFIG = ENGINE_CONFIG.get("kv_cache", {})
KV_CACHE_BLOCK_SIZE_TOKENS = int(KV_CACHE_CONFIG.get("block_size_tokens", 16))
KV_CACHE_CAPACITY_TOKENS = int(KV_CACHE_CONFIG.get("capacity_tokens", MAX_CONTEXT_TOKENS * MAX_NUM_SEQS))
MAX_IMAGES_PER_REQUEST = int(API_CONFIG["max_images_per_request"])
MAX_IMAGE_BYTES = int(API_CONFIG["max_image_bytes"])
MAX_IMAGE_PIXELS = int(API_CONFIG["max_image_pixels"])
MAX_CHAT_REQUEST_BYTES = int(API_CONFIG["max_chat_request_bytes"])
MAX_LIVE_REQUEST_BYTES = int(API_CONFIG["max_live_request_bytes"])
MAX_VIDEO_BYTES = int(API_CONFIG["max_video_bytes"])
MAX_VIDEO_REQUEST_BYTES = int(API_CONFIG["max_video_request_bytes"])
MAX_VIDEO_DURATION_SECONDS = float(API_CONFIG["max_video_duration_seconds"])
VIDEO_MAX_FRAMES = int(API_CONFIG["video_max_frames"])
VIDEO_MAX_FRAME_SIDE = int(API_CONFIG["video_max_frame_side"])
VIDEO_BACKEND = str(API_CONFIG.get("video_backend", "codec")).lower()
if VIDEO_BACKEND not in {"frames", "codec"}:
    raise RuntimeError("api.video_backend must be either 'frames' or 'codec'.")
CODEC_CONFIG = API_CONFIG.get("codec", {})
if not isinstance(CODEC_CONFIG, dict):
    raise RuntimeError("api.codec must be a mapping in the API YAML.")
CODEC_ENGINE = str(CODEC_CONFIG["engine"]).lower() if CODEC_CONFIG.get("engine") else None
CODEC_TARGET_CANVAS = int(CODEC_CONFIG.get("target_canvas", 32))
CODEC_MAX_PIXELS = int(CODEC_CONFIG.get("max_pixels", 150000))
CODEC_FFMPEG_BIN = str(CODEC_CONFIG.get("ffmpeg_bin", "ffmpeg"))
CODEC_FFPROBE_BIN = str(CODEC_CONFIG.get("ffprobe_bin", "ffprobe"))
CODEC_TIMEOUT_SECONDS = int(CODEC_CONFIG.get("timeout_seconds", 900))
if CODEC_ENGINE not in {None, "hevc", "dcvc-rt"}:
    raise RuntimeError("api.codec.engine must be 'hevc', 'dcvc-rt', or null.")
if CODEC_TARGET_CANVAS <= 0 or CODEC_MAX_PIXELS <= 0 or CODEC_TIMEOUT_SECONDS <= 0:
    raise RuntimeError("api.codec target_canvas, max_pixels, and timeout_seconds must be positive.")
IMAGE_URL_TIMEOUT = float(API_CONFIG["image_url_timeout_seconds"])
STREAM_TIMEOUT = float(API_CONFIG["stream_timeout_seconds"])
WORKER_STARTUP_TIMEOUT = float(CONFIG.get("runtime", {}).get("worker_startup_timeout_seconds", 2400))
WORKER_REQUEST_TIMEOUT = float(CONFIG.get("runtime", {}).get("worker_request_timeout_seconds", 1800))
ENGINE_CORE_STARTUP_TIMEOUT = float(
    CONFIG.get("runtime", {}).get("engine_core_startup_timeout_seconds", WORKER_STARTUP_TIMEOUT + 30)
)
ENGINE_CORE_REQUEST_TIMEOUT = float(
    CONFIG.get("runtime", {}).get("engine_core_request_timeout_seconds", WORKER_REQUEST_TIMEOUT + 30)
)
if ENGINE_CORE_STARTUP_TIMEOUT <= 0 or ENGINE_CORE_REQUEST_TIMEOUT <= 0:
    raise RuntimeError("Engine Core startup and request timeouts must be positive.")
LIVE_CONFIG = API_CONFIG.get("live_stream", {})
if not isinstance(LIVE_CONFIG, dict):
    raise RuntimeError("api.live_stream must be a mapping in the API YAML.")
LIVE_NETWORK_CONFIG = NETWORK_CONFIG.get("live_stream", {})
if not isinstance(LIVE_NETWORK_CONFIG, dict):
    raise RuntimeError("network.live_stream must be a mapping in the network YAML.")
LIVE_SETTINGS = LiveVideoSettings(
    rtsp_host=str(LIVE_NETWORK_CONFIG.get("mediamtx_rtsp_host", "127.0.0.1")),
    rtsp_port=int(LIVE_NETWORK_CONFIG.get("mediamtx_rtsp_port", 8554)),
    rtmp_port=int(LIVE_NETWORK_CONFIG.get("mediamtx_rtmp_port", 1935)),
    public_host=str(
        LIVE_NETWORK_CONFIG.get("public_host")
        or os.environ.get("MAGEVL4B_LIVE_PUBLIC_HOST")
        or os.environ.get("MAGEVL_LIVE_PUBLIC_HOST", "")
    ).strip(),
    sample_interval_seconds=float(LIVE_CONFIG.get("sample_interval_seconds", 2.0)),
    window_seconds=float(LIVE_CONFIG.get("window_seconds", 8.0)),
    max_duration_seconds=float(LIVE_CONFIG.get("max_duration_seconds", 3600)),
    max_sessions=int(LIVE_CONFIG.get("max_sessions", 1)),
    open_timeout_ms=int(float(LIVE_CONFIG.get("open_timeout_seconds", 10)) * 1000),
    read_timeout_ms=int(float(LIVE_CONFIG.get("read_timeout_seconds", 5)) * 1000),
    reconnect_seconds=float(LIVE_CONFIG.get("reconnect_interval_seconds", 2.0)),
    max_frames=VIDEO_MAX_FRAMES,
    max_frame_side=VIDEO_MAX_FRAME_SIDE,
    video_backend=str(LIVE_CONFIG.get("video_backend", VIDEO_BACKEND)).lower(),
    codec_engine=CODEC_ENGINE,
    codec_target_canvas=CODEC_TARGET_CANVAS,
    live_codec_target_canvas=int(LIVE_CONFIG.get("codec_target_canvas", 16)),
    codec_max_pixels=CODEC_MAX_PIXELS,
    codec_max_pending_segments=int(LIVE_CONFIG.get("codec_max_pending_segments", 2)),
    ffmpeg_bin=CODEC_FFMPEG_BIN,
)
if LIVE_SETTINGS.video_backend not in {"frames", "codec"}:
    raise RuntimeError("api.live_stream.video_backend must be either 'frames' or 'codec'.")
if LIVE_SETTINGS.codec_max_pending_segments < 1:
    raise RuntimeError("api.live_stream.codec_max_pending_segments must be positive.")
