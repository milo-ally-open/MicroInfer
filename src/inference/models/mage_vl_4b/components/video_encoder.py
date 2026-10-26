"""Mage-VL 视频输入编码：取帧／Codec 预处理并交给模型 Processor。"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from inference.engine.sequence import InferenceError
from ..utils import loader as settings
from ..utils.loader import (
    CODEC_FFMPEG_BIN,
    CODEC_FFPROBE_BIN,
    CODEC_MAX_PIXELS,
    CODEC_TARGET_CANVAS,
    CODEC_TIMEOUT_SECONDS,
    LOCAL_FILES_ONLY,
)
from .image_encoder import _apply_chat_template, _move_inputs_to_model


def _require_codec_runtime(engine: str | None) -> None:
    """选用编码器缺少主机端前处理工具时，提供明确错误讯息。"""
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
    """检查已载入模型的处理器介面，不将程式固定绑定至特定版本。"""
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
    prompt = _apply_chat_template(model_runner, messages)
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


class MageVLVideoEncoder:
    """单独负责影片帧或 Codec 输入的格式转换与张量生成。"""

    def __init__(self, model_runner: Any) -> None:
        self.model_runner = model_runner

    def encode(
        self,
        question: str,
        video: list[Image.Image] | Path,
        **options: Any,
    ) -> tuple[dict[str, Any], int, int | None]:
        return _video_inputs(question, video, model_runner=self.model_runner, **options)

    @staticmethod
    def require_runtime(engine: str | None) -> None:
        _require_codec_runtime(engine)

    def require_processor_support(self) -> None:
        _require_processor_codec_support(self.model_runner)

    def processor_config(
        self, engine: str | None, target_canvas: int, cache_root: Path | None
    ) -> dict[str, Any]:
        return _codec_processor_config(engine, target_canvas, cache_root, self.model_runner)
