"""组合 Mage-VL 图片、视频与生成参数编码组件。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from inference.engine.sequence import InferenceError
from ..utils.loader import GENERATION_CONFIG, MAX_CONTEXT_TOKENS, MAX_NEW_TOKENS
from .image_encoder import (
    MageVLImageEncoder,
    _apply_chat_template,
    _load_image,
    _move_inputs_to_model,
    _normalize_messages,
    _prepare_inputs,
    _resize_image,
)
from .video_encoder import (
    MageVLVideoEncoder,
    _codec_processor_config,
    _require_codec_runtime,
    _require_processor_codec_support,
    _video_inputs,
)


def _generation_args(request_body: dict[str, Any]) -> dict[str, Any]:
    """验证请求生成选项，并合并 Mage-VL 模型设定。"""
    max_tokens = int(
        request_body.get(
            "max_tokens",
            request_body.get("max_completion_tokens", GENERATION_CONFIG["max_tokens_default"]),
        )
    )
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
        raise InferenceError(
            status_code=400,
            detail="num_beams greater than 1 is not supported by the stepwise scheduler.",
        )

    args["use_cache"] = bool(GENERATION_CONFIG.get("use_cache", True))
    if GENERATION_CONFIG.get("cache_implementation"):
        raise RuntimeError("Stepwise generation currently uses the model's DynamicCache; clear generation.cache_implementation.")
    if GENERATION_CONFIG.get("cache_config") is not None:
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


class MageVLEncoder:
    """将服务请求分发给文字／图片 Tokenizer 和视频 Processor 编码组件。"""

    def __init__(self, model_runner: Any) -> None:
        self.image_encoder = MageVLImageEncoder(model_runner)
        self.video_encoder = MageVLVideoEncoder(model_runner)

    def encode_chat(self, request_body: dict[str, Any]) -> tuple[dict[str, Any], int, dict[str, Any]]:
        model_inputs, prompt_tokens = self.image_encoder.encode_chat(request_body.get("messages"))
        generation_args = _generation_args(request_body)
        _check_context(prompt_tokens, generation_args)
        return model_inputs, prompt_tokens, generation_args

    def encode_video(
        self,
        question: str,
        video: list[Image.Image] | Path,
        **options: Any,
    ) -> tuple[dict[str, Any], int, int | None]:
        return self.video_encoder.encode(question, video, **options)

    @staticmethod
    def generation_args(request_body: dict[str, Any]) -> dict[str, Any]:
        return _generation_args(request_body)

    @staticmethod
    def check_context(prompt_tokens: int, generation_args: dict[str, Any]) -> None:
        _check_context(prompt_tokens, generation_args)

    @staticmethod
    def resize_image(image: Image.Image, max_side: int) -> Image.Image:
        return _resize_image(image, max_side)

    @staticmethod
    def require_codec_runtime(engine: str | None) -> None:
        _require_codec_runtime(engine)

    def require_processor_codec_support(self) -> None:
        self.video_encoder.require_processor_support()

    def codec_processor_config(
        self, engine: str | None, target_canvas: int, cache_root: Path | None
    ) -> dict[str, Any]:
        return self.video_encoder.processor_config(engine, target_canvas, cache_root)
