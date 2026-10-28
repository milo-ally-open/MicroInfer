"""读取 Mage-VL YAML，集中保存模型专属配置。"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

from inference.engine.config import PROJECT_ROOT, selected_config

from inference.processors.live_sessions import LiveVideoSettings


logger = logging.getLogger(__name__)
CONFIG = selected_config("INFERENCE_CONFIG")  # 模型与推理参数。
NETWORK_CONFIG = selected_config("INFERENCE_NETWORK_CONFIG")  # API 与媒体网络参数。

MODEL_CONFIG = CONFIG["model"]  # 上游模型 ID及加载选项。
QUANTIZATION_CONFIG = MODEL_CONFIG["quantization"]  # 权重量化方式与设备内存限制。
COMPONENT_REPLACEMENTS = MODEL_CONFIG["components"]["replacements"]  # 模型组件替换登记。
RUNTIME_CONFIG = CONFIG["runtime"]  # 权重快照与程序超时配置。
LOCAL_FILES_ONLY = RUNTIME_CONFIG["local_files_only"]  # 仅使用本地 Hugging Face 快照。
API_CONFIG = CONFIG["api"]  # 请求限制与多模态处理参数。
ENGINE_CONFIG = CONFIG["engine"]  # 调度与 KV 容量参数。
GENERATION_CONFIG = CONFIG["generation"]  # 文字生成默认值。
MODEL_ID = MODEL_CONFIG["id"]  # Hugging Face repo_id。
MODEL_REVISION = MODEL_CONFIG["revision"]  # 固定的上游模型 revision。
OPTIMIZATION_CONFIG = CONFIG["optimization"]["pytorch"]  # PyTorch 执行优化参数。

MAX_IMAGE_SIDE = int(API_CONFIG["max_image_side"])  # 输入图片最大边长。
IMAGE_RESAMPLING = str(API_CONFIG["image_resampling"]).lower()  # 图片缩放滤波器。
MAX_CONTEXT_TOKENS = int(API_CONFIG["max_context_tokens"])  # 单次请求的上下文上限。
MAX_NEW_TOKENS = int(API_CONFIG["max_new_tokens"])  # 单次生成长度上限。
MAX_NUM_SEQS = int(ENGINE_CONFIG["max_num_seqs"])  # 同时执行的序列数上限。
MAX_WAITING_REQUESTS = int(ENGINE_CONFIG["max_waiting_requests"])  # 等候队列上限。
MAX_NUM_SCHEDULED_TOKENS = int(ENGINE_CONFIG["max_num_scheduled_tokens"])  # 单轮调度 Token 预算。
BATCH_WAIT_SECONDS = float(ENGINE_CONFIG["batch_wait_seconds"])  # 合并批次的等待时间。
PREFILL_CHUNK_SIZE_TOKENS = int(ENGINE_CONFIG["prefill_chunk_size_tokens"])  # 分块 Prefill 大小。
KV_CACHE_CONFIG = ENGINE_CONFIG["kv_cache"]  # Engine 逻辑 KV 容量配置。
KV_CACHE_BLOCK_SIZE_TOKENS = int(KV_CACHE_CONFIG["block_size_tokens"])  # 每个逻辑区块的 Token 数。
KV_CACHE_CAPACITY_TOKENS = int(KV_CACHE_CONFIG["capacity_tokens"])  # KV 逻辑预留总容量。
MAX_IMAGES_PER_REQUEST = int(API_CONFIG["max_images_per_request"])  # 每个请求可输入的图片数。
MAX_IMAGE_BYTES = int(API_CONFIG["max_image_bytes"])  # 单张图片大小上限。
MAX_IMAGE_PIXELS = int(API_CONFIG["max_image_pixels"])  # 单张图片像素上限。
MAX_CHAT_REQUEST_BYTES = int(API_CONFIG["max_chat_request_bytes"])  # 聊天请求正文大小上限。
MAX_LIVE_REQUEST_BYTES = int(API_CONFIG["max_live_request_bytes"])  # Live API 请求正文大小上限。
MAX_VIDEO_BYTES = int(API_CONFIG["max_video_bytes"])  # 上传视频文件大小上限。
MAX_VIDEO_REQUEST_BYTES = int(API_CONFIG["max_video_request_bytes"])  # 视频理解请求正文大小上限。
MAX_VIDEO_DURATION_SECONDS = float(API_CONFIG["max_video_duration_seconds"])  # 视频时长上限。
VIDEO_MAX_FRAMES = int(API_CONFIG["video_max_frames"])  # 单次视频抽取帧上限。
VIDEO_MAX_FRAME_SIDE = int(API_CONFIG["video_max_frame_side"])  # 视频帧最大边长。
VIDEO_BACKEND = str(API_CONFIG["video_backend"]).lower()  # 视频输入解码后端。
CODEC_CONFIG = API_CONFIG["codec"]  # Codec 视频路径参数。
CODEC_ENGINE = CODEC_CONFIG["engine"]  # Codec 使用的编码器。
CODEC_TARGET_CANVAS = int(CODEC_CONFIG["target_canvas"])  # Codec 帧目标尺寸倍率。
CODEC_MAX_PIXELS = int(CODEC_CONFIG["max_pixels"])  # Codec 帧像素上限。
CODEC_FFMPEG_BIN = str(CODEC_CONFIG["ffmpeg_bin"])  # FFmpeg 执行文件路径。
CODEC_FFPROBE_BIN = str(CODEC_CONFIG["ffprobe_bin"])  # FFprobe 执行文件路径。
CODEC_TIMEOUT_SECONDS = int(CODEC_CONFIG["timeout_seconds"])  # Codec 处理超时。
IMAGE_URL_TIMEOUT = float(API_CONFIG["image_url_timeout_seconds"])  # 下载远端图片的超时。
STREAM_TIMEOUT = float(API_CONFIG["stream_timeout_seconds"])  # 文本流式超时。
WORKER_STARTUP_TIMEOUT = float(RUNTIME_CONFIG["worker_startup_timeout_seconds"])  # Worker 启动超时。
WORKER_REQUEST_TIMEOUT = float(RUNTIME_CONFIG["worker_request_timeout_seconds"])  # Worker 请求超时。
ENGINE_CORE_STARTUP_TIMEOUT = float(RUNTIME_CONFIG["engine_core_startup_timeout_seconds"])  # Engine Core 启动超时。
ENGINE_CORE_REQUEST_TIMEOUT = float(RUNTIME_CONFIG["engine_core_request_timeout_seconds"])  # Engine Core 请求超时。
LIVE_CONFIG = API_CONFIG["live_stream"]  # 实时视频取样与工作会话参数。
LIVE_NETWORK_CONFIG = NETWORK_CONFIG["live_stream"]  # MediaMTX 及公开媒体端点。
LIVE_PUBLIC_HOST = str(LIVE_NETWORK_CONFIG["public_host"]).strip()  # 返回给推流用户端的主机名称。
LIVE_SETTINGS = LiveVideoSettings(
    rtsp_host=str(LIVE_NETWORK_CONFIG["mediamtx_rtsp_host"]),  # 推理服务连接的 RTSP 主机。
    rtsp_port=int(LIVE_NETWORK_CONFIG["mediamtx_rtsp_port"]),  # MediaMTX RTSP 端口。
    rtmp_port=int(LIVE_NETWORK_CONFIG["mediamtx_rtmp_port"]),  # MediaMTX RTMP 端口。
    public_host=LIVE_PUBLIC_HOST,  # 返回给外部推流端的公开主机。
    sample_interval_seconds=float(LIVE_CONFIG["sample_interval_seconds"]),  # Live 帧取样间隔。
    window_seconds=float(LIVE_CONFIG["window_seconds"]),  # 每次理解的滑动时间窗。
    max_duration_seconds=float(LIVE_CONFIG["max_duration_seconds"]),  # Live 工作会话时长上限。
    max_sessions=int(LIVE_CONFIG["max_sessions"]),  # 同时 Live 工作会话上限。
    open_timeout_ms=int(float(LIVE_CONFIG["open_timeout_seconds"]) * 1000),  # 创建流式连接的超时毫秒数。
    read_timeout_ms=int(float(LIVE_CONFIG["read_timeout_seconds"]) * 1000),  # 读取帧超时毫秒数。
    reconnect_seconds=float(LIVE_CONFIG["reconnect_interval_seconds"]),  # 断线重连间隔。
    max_frames=VIDEO_MAX_FRAMES,  # Live 单次取样帧上限。
    max_frame_side=VIDEO_MAX_FRAME_SIDE,  # Live 帧最大边长。
    video_backend=str(LIVE_CONFIG["video_backend"]).lower(),  # Live 视频解码后端。
    codec_engine=CODEC_ENGINE,  # Live Codec 编码器。
    codec_target_canvas=CODEC_TARGET_CANVAS,  # Live 共用 Codec 尺寸倍率。
    live_codec_target_canvas=int(LIVE_CONFIG["codec_target_canvas"]),  # Live 专用尺寸倍率。
    codec_max_pixels=CODEC_MAX_PIXELS,  # Live Codec 帧像素上限。
    codec_max_pending_segments=int(LIVE_CONFIG["codec_max_pending_segments"]),  # 等待处理的片段上限。
    ffmpeg_bin=CODEC_FFMPEG_BIN,  # Live Codec 使用的 FFmpeg 执行文件。
)


def resolve_model_load_path() -> str:
    """准备固定 revision 的模型快照并返回缓存路径。"""
    local_model_path = RUNTIME_CONFIG["local_model_path"]
    if local_model_path:
        path = Path(os.path.expandvars(os.path.expanduser(str(local_model_path))))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        if not path.is_dir() or not (path / "config.json").is_file():
            raise FileNotFoundError(f"Local model snapshot is incomplete or missing: {path}")
        return str(path)

    hf_home_value = os.environ.get("HF_HOME") or RUNTIME_CONFIG["hf_home"]
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


def _build_quantization_config(torch: Any, bits_and_bytes_config: Any) -> Any:
    """依 YAML 的量化方法创建 Transformers 量化配置。"""
    quantization = QUANTIZATION_CONFIG
    method = quantization["method"]
    quantization_builders = {
        "nf4": lambda: bits_and_bytes_config(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quantization["quant_type"]),
            bnb_4bit_use_double_quant=bool(quantization["double_quant"]),
            bnb_4bit_compute_dtype=getattr(torch, quantization["compute_dtype"]),
            bnb_4bit_quant_storage=getattr(torch, quantization["quant_storage"])
            if quantization["quant_storage"]
            else None,
        ),
        "int8": lambda: bits_and_bytes_config(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=bool(quantization["llm_int8_enable_fp32_cpu_offload"]),
            llm_int8_threshold=float(quantization["llm_int8_threshold"]),
            llm_int8_has_fp16_weight=bool(quantization["llm_int8_has_fp16_weight"]),
            llm_int8_skip_modules=quantization["llm_int8_skip_modules"],
        ),
        "none": lambda: None,
        "prequantized": lambda: None,
    }
    try:
        return quantization_builders[method]()
    except KeyError as exc:
        raise RuntimeError(f"Unsupported Transformers quantization method in config: {method}") from exc


def _build_model_kwargs(torch: Any, kernel_config_type: Any, bits_and_bytes_config: Any) -> dict[str, Any]:
    """将 YAML 模型加载选项转为 Transformers 权重加载参数。"""
    kernel_config = MODEL_CONFIG["kernel_config"]
    offload_folder = MODEL_CONFIG["offload_folder"]
    max_memory = {
        device: str(memory)
        for device, memory in {
            0: QUANTIZATION_CONFIG["gpu_max_memory"],
            "cpu": QUANTIZATION_CONFIG["cpu_max_memory"],
        }.items()
        if memory
    }
    resolved_offload_folder = (
        Path(os.path.expandvars(os.path.expanduser(offload_folder)))
        if offload_folder
        else None
    )
    if resolved_offload_folder is not None and not resolved_offload_folder.is_absolute():
        resolved_offload_folder = PROJECT_ROOT / resolved_offload_folder
    if resolved_offload_folder is not None:
        resolved_offload_folder.mkdir(parents=True, exist_ok=True)
    cuda_available = torch.cuda.is_available()
    return {
        key: value
        for key, value in {
            "device_map": MODEL_CONFIG["device_map"] if cuda_available else "cpu",
            "revision": MODEL_REVISION,
            "dtype": getattr(torch, MODEL_CONFIG["dtype"]),
            "local_files_only": LOCAL_FILES_ONLY,
            "attn_implementation": MODEL_CONFIG["attn_implementation"],
            "experts_implementation": MODEL_CONFIG["experts_implementation"],
            "use_kernels": MODEL_CONFIG["use_kernels"],
            "use_safetensors": MODEL_CONFIG["use_safetensors"],
            "weights_only": MODEL_CONFIG["weights_only"],
            "offload_buffers": MODEL_CONFIG["offload_buffers"],
            "disable_mmap": MODEL_CONFIG["disable_mmap"],
            "kernel_config": kernel_config_type(**kernel_config) if kernel_config else None,
            "offload_folder": str(resolved_offload_folder) if resolved_offload_folder else None,
            "quantization_config": _build_quantization_config(torch, bits_and_bytes_config) if cuda_available else None,
            "max_memory": (max_memory or None) if cuda_available else None,
        }.items()
        if value is not None
    }


def load_mage_vl() -> tuple[str, Any, Any]:
    """依模型 YAML 加载模型和 Processor，并返回快照路径。"""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, KernelConfig

    from ..model import MAGE_VL_SOURCE_REVISION
    from ..model.conditional_generation import MageVLForConditionalGeneration
    from ..model.configuration_mage_vl import MageVLConfig

    if MODEL_REVISION != MAGE_VL_SOURCE_REVISION:
        raise RuntimeError(
            "The local Mage-VL model implementation is pinned to revision "
            f"{MAGE_VL_SOURCE_REVISION}, but inference.yml selects {MODEL_REVISION}."
        )
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(OPTIMIZATION_CONFIG["allow_tf32"])
        torch.backends.cudnn.allow_tf32 = bool(OPTIMIZATION_CONFIG["cudnn_allow_tf32"])
        torch.backends.cudnn.benchmark = bool(OPTIMIZATION_CONFIG["cudnn_benchmark"])
    else:
        logger.warning("CUDA is unavailable; Mage-VL will load on CPU without GPU quantization.")
    matmul_precision = OPTIMIZATION_CONFIG["float32_matmul_precision"]
    if matmul_precision not in {"highest", "high", "medium"}:
        raise RuntimeError("optimization.pytorch.float32_matmul_precision must be highest, high, or medium.")
    torch.set_float32_matmul_precision(matmul_precision)
    model_path = resolve_model_load_path()
    model_kwargs = _build_model_kwargs(torch, KernelConfig, BitsAndBytesConfig)

    processor_config = CONFIG["processor"]
    processor_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "revision": MODEL_REVISION,
        "local_files_only": LOCAL_FILES_ONLY,
    }
    processor_kwargs.update(
        {"use_fast": bool(processor_config["use_fast"])}
        if processor_config["use_fast"] is not None
        else {}
    )

    model_config_object = MageVLConfig.from_pretrained(
        model_path,
        revision=MODEL_REVISION,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model_config_object.text_config.attention_query_chunk_size = int(
        MODEL_CONFIG["attention"]["query_chunk_size"]
    )
    processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
    model = MageVLForConditionalGeneration.from_pretrained(
        model_path,
        config=model_config_object,
        **model_kwargs,
    ).eval()
    return model_path, model, processor
