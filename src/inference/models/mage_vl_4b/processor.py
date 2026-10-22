"""Mage-VL-4B 專屬的圖片、影片與生成輸入適配。"""

from __future__ import annotations

import base64
import binascii
import importlib
import importlib.metadata
import inspect
import io
import os
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request as URLRequest, urlopen

import torch
from PIL import Image, UnidentifiedImageError

from engine.errors import InferenceError
from media.video import probe_video_metadata, sample_video_frames, save_upload, uploaded_video_path
from . import config as settings
from .config import (
    API_CONFIG, CODEC_ENGINE, CODEC_FFMPEG_BIN, CODEC_FFPROBE_BIN, CODEC_MAX_PIXELS,
    CODEC_TARGET_CANVAS, CODEC_TIMEOUT_SECONDS, GENERATION_CONFIG, IMAGE_RESAMPLING,
    IMAGE_URL_TIMEOUT, LOCAL_FILES_ONLY, MAX_CONTEXT_TOKENS, MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS, MAX_IMAGE_SIDE, MAX_IMAGES_PER_REQUEST, MAX_NEW_TOKENS,
    MAX_VIDEO_BYTES, MAX_VIDEO_DURATION_SECONDS, STREAM_TIMEOUT,
)


_RESAMPLING_FILTERS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}

def _resize_image(image: Image.Image, max_side: int) -> Image.Image:
    if IMAGE_RESAMPLING not in _RESAMPLING_FILTERS:
        raise RuntimeError(f"Unsupported image resampling mode in config: {IMAGE_RESAMPLING}")
    image.thumbnail((max_side, max_side), _RESAMPLING_FILTERS[IMAGE_RESAMPLING])
    return image


def _load_image(source: str) -> Image.Image:
    if source.startswith("data:"):
        header, separator, payload = source.partition(",")
        if not separator or ";base64" not in header.lower():
            raise InferenceError(status_code=400, detail="Image data URLs must use base64 encoding.")
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InferenceError(status_code=400, detail="Invalid base64 image payload.") from exc
    else:
        parsed = urlparse(source)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise InferenceError(status_code=400, detail="image_url must be an http(s) URL or a base64 data URL.")
        request = URLRequest(source, headers={"User-Agent": "MageVLDeployment/0.1"})
        try:
            with urlopen(request, timeout=IMAGE_URL_TIMEOUT) as response:
                if response.status != 200:
                    raise InferenceError(status_code=400, detail="Image URL did not return HTTP 200.")
                raw = response.read(MAX_IMAGE_BYTES + 1)
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(status_code=400, detail=f"Could not fetch image_url: {exc}") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise InferenceError(status_code=413, detail=f"Image exceeds MAX_IMAGE_BYTES={MAX_IMAGE_BYTES}.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as opened:
                if opened.width * opened.height > MAX_IMAGE_PIXELS:
                    raise InferenceError(
                        status_code=413,
                        detail=f"Image exceeds max_image_pixels={MAX_IMAGE_PIXELS}.",
                    )
                opened.load()
                image = opened.convert("RGB")
    except InferenceError:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise InferenceError(
            status_code=413,
            detail=f"Image exceeds max_image_pixels={MAX_IMAGE_PIXELS}.",
        ) from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise InferenceError(status_code=400, detail="Payload is not a supported image.") from exc
    return _resize_image(image, MAX_IMAGE_SIDE)


def _move_inputs_to_model(inputs: Any, model_runner: Any) -> dict[str, Any]:
    input_device = model_runner.model.get_input_embeddings().weight.device
    model_dtype = model_runner.model.dtype
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if key in {"pixel_values", "pixel_values_videos"} and value.is_floating_point():
                moved[key] = value.to(device=input_device, dtype=model_dtype)
            else:
                moved[key] = value.to(device=input_device)
        else:
            moved[key] = value
    return moved


def _require_codec_runtime(engine: str | None) -> None:
    """選用編碼器缺少主機端前處理工具時，提供明確錯誤訊息。"""
    for name, executable in (("ffmpeg", CODEC_FFMPEG_BIN), ("ffprobe", CODEC_FFPROBE_BIN)):
        if shutil.which(executable) is None and not Path(executable).is_file():
            raise InferenceError(status_code=503, detail=f"Codec video mode requires {name}; executable not found: {executable}")
    try:
        importlib.metadata.version("codec-video-prep")
    except importlib.metadata.PackageNotFoundError as exc:
        raise InferenceError(
            status_code=503,
            detail="Codec video mode requires the Python package codec-video-prep; run the Transformers init.sh environment setup.",
        ) from exc
    if engine == "hevc":
        cv_preinfer = os.environ.get("CV_PREINFER_BIN", "cv-preinfer")
        if shutil.which(cv_preinfer) is None and not Path(cv_preinfer).is_file():
            raise InferenceError(
                status_code=503,
                detail=f"Codec engine 'hevc' requires cv-preinfer on PATH (or CV_PREINFER_BIN): {cv_preinfer}",
            )


def _codec_processor_config(
    engine: str | None,
    target_canvas: int,
    cache_root: Path | None,
    model_runner: Any,
) -> dict[str, Any]:
    config: dict[str, Any] = {"target_canvas": target_canvas, "timeout_seconds": CODEC_TIMEOUT_SECONDS}
    if engine:
        config["engine"] = engine
    image_processor = getattr(model_runner.processor, "image_processor", None)
    patch_size = getattr(image_processor, "patch_size", None)
    if patch_size is not None:
        try:
            patch_size = int(patch_size)
        except (TypeError, ValueError):
            patch_size = 0
        if patch_size > 0:
            # 將編碼 patch 座標對齊目前載入權重使用的影像網格。
            config["patch"] = patch_size
    if cache_root is not None:
        config["cache_root"] = cache_root
    if engine == "dcvc-rt":
        model_path = Path(model_runner.model_path or "")
        if not model_path.is_dir():
            try:
                from huggingface_hub import snapshot_download

                model_path = Path(snapshot_download(
                    repo_id=settings.MODEL_ID,
                    revision=settings.MODEL_REVISION,
                    local_files_only=LOCAL_FILES_ONLY,
                ))
            except Exception as exc:
                raise InferenceError(status_code=503, detail=f"Could not locate the loaded model snapshot for DCVC-RT: {exc}") from exc
        neural_codec_path = model_path / "neural_codec"
        if not neural_codec_path.is_dir():
            raise InferenceError(status_code=503, detail=f"DCVC-RT files are missing from the model snapshot: {neural_codec_path}")
        device = model_runner.model.get_input_embeddings().weight.device
        config["dcvc"] = {"pkg_dir": str(neural_codec_path), "device": str(device)}
    # 模型處理器的遠端程式碼會提供編碼設定結構；依實際載入的結構篩選
    # 服務選項，避免 API 綁定特定版權重使用的 dataclass 欄位。
    try:
        processor_module = importlib.import_module(model_runner.processor.__class__.__module__)
        codec_module = importlib.import_module(
            ".codec_video_processing_mage_vl",
            package=processor_module.__package__,
        )
        codec_config_type = getattr(codec_module, "CodecConfig")
        supported = set(inspect.signature(codec_config_type).parameters)
    except (ImportError, AttributeError, TypeError, ValueError):
        supported = {"engine", "target_canvas", "dcvc"}
    return {key: value for key, value in config.items() if key in supported}


def _require_processor_codec_support(model_runner: Any) -> None:
    """檢查已載入模型的處理器介面，不將程式固定綁定至特定版本。"""
    if model_runner.processor is None:
        raise InferenceError(status_code=503, detail="The Mage-VL-4B processor is not ready yet.")
    try:
        parameters = inspect.signature(model_runner.processor.__call__).parameters
    except (TypeError, ValueError):
        return
    if not {"video_backend", "codec_config"}.issubset(parameters):
        raise InferenceError(
            status_code=501,
            detail="The loaded Mage-VL-4B processor does not expose the codec video backend; use video_backend=frames or load a model snapshot that includes it.",
        )


def _video_inputs(
    question: str,
    video: list[Image.Image] | Path,
    *,
    backend: str = "frames",
    codec_engine: str | None = None,
    target_canvas: int = CODEC_TARGET_CANVAS,
    max_pixels: int = CODEC_MAX_PIXELS,
    codec_cache_root: Path | None = None,
    model_runner: Any,
) -> tuple[Any, int, int | None]:
    messages = [{"role": "user", "content": [
        {"type": "video"},
        {"type": "text", "text": question},
    ]}]
    prompt = model_runner.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if backend == "frames":
        inputs = model_runner.processor(text=[prompt], videos=[video], return_tensors="pt", padding=True)
    elif backend == "codec":
        _require_codec_runtime(codec_engine)
        _require_processor_codec_support(model_runner)
        if not isinstance(video, Path):
            raise ValueError("Codec video processing requires a local video file path.")
        codec_config = _codec_processor_config(codec_engine, target_canvas, codec_cache_root, model_runner)
        try:
            inputs = model_runner.processor(
                text=[prompt],
                videos=[str(video)],
                video_backend="codec",
                max_pixels=max_pixels,
                codec_config=codec_config,
                return_tensors="pt",
                padding=True,
            )
        except TypeError as exc:
            raise InferenceError(
                status_code=501,
                detail="The loaded Mage-VL-4B processor does not expose the codec video backend; use video_backend=frames or update the model processor files.",
            ) from exc
        except (ImportError, ModuleNotFoundError, FileNotFoundError) as exc:
            raise InferenceError(status_code=503, detail=f"Codec preprocessing dependency is unavailable: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise InferenceError(status_code=504, detail="Codec video preprocessing exceeded its configured timeout.") from exc
        except (RuntimeError, ValueError) as exc:
            raise InferenceError(status_code=422, detail=f"Codec video preprocessing failed: {exc}") from exc
    else:
        raise InferenceError(status_code=400, detail=f"Unsupported video_backend: {backend}")
    input_tokens = int(inputs["input_ids"].shape[-1])
    grids = inputs.get("image_grid_thw")
    canvas_count = int(grids.shape[0]) if backend == "codec" and isinstance(grids, torch.Tensor) else None
    return _move_inputs_to_model(inputs, model_runner), input_tokens, canvas_count


def _normalize_messages(raw_messages: Any) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """驗證對話內容並載入圖片，整理成模型處理器需要的格式。"""
    if not isinstance(raw_messages, list) or not raw_messages:
        raise InferenceError(status_code=400, detail="messages must be a non-empty array.")
    normalized: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    for item in raw_messages:
        if not isinstance(item, dict) or item.get("role") not in {"system", "user", "assistant"}:
            raise InferenceError(status_code=400, detail="Each message needs a supported role and content.")
        content = item.get("content")
        if isinstance(content, str):
            normalized.append({"role": item["role"], "content": content})
            continue
        if not isinstance(content, list):
            raise InferenceError(status_code=400, detail="Message content must be text or an array of text/image parts.")
        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                raise InferenceError(status_code=400, detail="Each content part must be an object.")
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url":
                image_url = part.get("image_url")
                source = image_url.get("url") if isinstance(image_url, dict) else None
                if not isinstance(source, str):
                    raise InferenceError(status_code=400, detail="image_url.url must be a string.")
                images.append(_load_image(source))
                parts.append({"type": "image"})
                if len(images) > MAX_IMAGES_PER_REQUEST:
                    raise InferenceError(status_code=413, detail=f"At most {MAX_IMAGES_PER_REQUEST} image(s) are allowed per request.")
            else:
                raise InferenceError(status_code=400, detail="Only text and image_url content parts are supported.")
        normalized.append({"role": item["role"], "content": parts})
    has_image = any(
        part.get("type") == "image"
        for message in normalized
        for part in message.get("content", []) if isinstance(message.get("content"), list)
    )
    if not has_image:
        raise InferenceError(status_code=400, detail="This initial API accepts image requests; include at least one image_url.")
    return normalized, images


def _prepare_inputs(messages: list[dict[str, Any]], images: list[Image.Image], model_runner: Any) -> tuple[Any, int]:
    """套用對話模板、準備模型輸入張量，並計算提示詞 token 數。"""
    prompt = model_runner.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = model_runner.processor(text=[prompt], images=images, return_tensors="pt")
    input_tokens = int(inputs["input_ids"].shape[-1])
    return _move_inputs_to_model(inputs, model_runner), input_tokens


def _generation_args(request_body: dict[str, Any]) -> dict[str, Any]:
    """合併模型預設值與請求參數，產生受 API 限制的生成設定。"""
    max_tokens = int(request_body.get("max_tokens", request_body.get("max_completion_tokens", GENERATION_CONFIG["max_tokens_default"])))
    if not 1 <= max_tokens <= MAX_NEW_TOKENS:
        raise InferenceError(status_code=400, detail=f"max_tokens must be between 1 and {MAX_NEW_TOKENS}.")
    temperature = float(request_body.get("temperature", GENERATION_CONFIG["temperature"]))
    top_p = float(request_body.get("top_p", GENERATION_CONFIG["top_p"]))
    if not 0 <= temperature <= 2 or not 0 < top_p <= 1:
        raise InferenceError(status_code=400, detail="temperature must be 0..2 and top_p must be in (0,1].")
    args: dict[str, Any] = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
    if temperature > 0:
        args.update(temperature=temperature, top_p=top_p)
        top_k = request_body.get("top_k", GENERATION_CONFIG.get("top_k"))
        min_p = request_body.get("min_p", GENERATION_CONFIG.get("min_p"))
        if top_k is not None:
            top_k = int(top_k)
            if top_k < 0:
                raise InferenceError(status_code=400, detail="top_k must be a non-negative integer.")
            args["top_k"] = top_k
        if min_p is not None:
            min_p = float(min_p)
            if not 0 <= min_p <= 1:
                raise InferenceError(status_code=400, detail="min_p must be in [0,1].")
            args["min_p"] = min_p

    repetition_penalty = float(
        request_body.get("repetition_penalty", GENERATION_CONFIG.get("repetition_penalty", 1.0))
    )
    if repetition_penalty <= 0:
        raise InferenceError(status_code=400, detail="repetition_penalty must be greater than zero.")
    args["repetition_penalty"] = repetition_penalty

    no_repeat_ngram_size = int(
        request_body.get("no_repeat_ngram_size", GENERATION_CONFIG.get("no_repeat_ngram_size", 0))
    )
    if no_repeat_ngram_size < 0:
        raise InferenceError(status_code=400, detail="no_repeat_ngram_size must be non-negative.")
    if no_repeat_ngram_size:
        args["no_repeat_ngram_size"] = no_repeat_ngram_size

    num_beams = int(request_body.get("num_beams", GENERATION_CONFIG.get("num_beams", 1)))
    if num_beams < 1:
        raise InferenceError(status_code=400, detail="num_beams must be at least 1.")
    if num_beams > 1:
        raise InferenceError(status_code=400, detail="num_beams greater than 1 is not supported by the stepwise scheduler.")

    args["use_cache"] = bool(GENERATION_CONFIG.get("use_cache", True))
    cache_implementation = GENERATION_CONFIG.get("cache_implementation")
    if cache_implementation:
        raise RuntimeError("Stepwise generation currently uses the model's DynamicCache; clear generation.cache_implementation.")
    cache_config = GENERATION_CONFIG.get("cache_config")
    if cache_config is not None:
        raise RuntimeError("Stepwise generation does not accept generation.cache_config yet.")

    advanced_kwargs = GENERATION_CONFIG.get("advanced_kwargs", {})
    if not isinstance(advanced_kwargs, dict):
        raise RuntimeError("generation.advanced_kwargs must be a mapping in the API YAML.")
    reserved_kwargs = {"input_ids", "inputs", "max_new_tokens", "streamer", "return_dict_in_generate"}
    if reserved_kwargs.intersection(advanced_kwargs):
        raise RuntimeError(
            "generation.advanced_kwargs cannot override API-managed generation arguments: "
            + ", ".join(sorted(reserved_kwargs.intersection(advanced_kwargs)))
        )
    if advanced_kwargs:
        raise RuntimeError("generation.advanced_kwargs is not compatible with the stepwise token runner yet.")
    return args


def _check_context(prompt_tokens: int, generation_args: dict[str, Any]) -> None:
    if prompt_tokens + generation_args["max_new_tokens"] > MAX_CONTEXT_TOKENS:
        raise InferenceError(status_code=413, detail=f"Prompt plus requested output exceeds {MAX_CONTEXT_TOKENS} tokens.")


def _analyze_live_frames(
    question: str,
    frames: list[Image.Image],
    generation_args: dict[str, Any],
    model_runner: Any,
) -> dict[str, Any]:
    """將連續影格結果轉換為共用的即時工作階段格式。"""
    started = time.perf_counter()
    inputs, prompt_tokens, _ = _video_inputs(question, frames, backend="frames", model_runner=model_runner)
    _check_context(prompt_tokens, generation_args)
    output = model_runner.generate(inputs, generation_args)
    generated_ids = output[0, inputs["input_ids"].shape[-1] :]
    answer = model_runner.decode(generated_ids)
    return {
        "answer": answer,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(generated_ids.numel()),
        "generation_seconds": round(time.perf_counter() - started, 3),
    }


def _analyze_live_video(
    question: str,
    video_path: Path,
    generation_args: dict[str, Any],
    codec_engine: str | None,
    target_canvas: int,
    max_pixels: int,
    codec_cache_root: Path | None,
    model_runner: Any,
) -> dict[str, Any]:
    """對一段已封裝完成的即時視訊執行編碼前處理與文字生成。"""
    started = time.perf_counter()
    inputs, prompt_tokens, canvas_count = _video_inputs(
        question,
        video_path,
        backend="codec",
        codec_engine=codec_engine,
        target_canvas=target_canvas,
        max_pixels=max_pixels,
        codec_cache_root=codec_cache_root,
        model_runner=model_runner,
    )
    _check_context(prompt_tokens, generation_args)
    processor_finished = time.perf_counter()
    output = model_runner.generate(inputs, generation_args)
    generated_ids = output[0, inputs["input_ids"].shape[-1] :]
    answer = model_runner.decode(generated_ids)
    generated_at = time.perf_counter()
    return {
        "answer": answer,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(generated_ids.numel()),
        "processor_seconds": round(processor_finished - started, 3),
        "generation_seconds": round(generated_at - processor_finished, 3),
        "codec_canvas_count": canvas_count,
    }
