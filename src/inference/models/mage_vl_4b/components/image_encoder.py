"""Mage-VL 图片与聊天内容编码：校验输入、加载图片并调用多模态 Processor。"""

from __future__ import annotations

import base64
import binascii
import io
import warnings
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request as URLRequest, urlopen

import torch
from PIL import Image, UnidentifiedImageError

from inference.engine.sequence import InferenceError
from ..utils.loader import (
    IMAGE_RESAMPLING,
    IMAGE_URL_TIMEOUT,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_SIDE,
    MAX_IMAGES_PER_REQUEST,
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


def _normalize_messages(raw_messages: Any) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """验证 API 消息并将图片引用解码为 RGB 图片。"""
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
                    raise InferenceError(status_code=413, detail=f"At most {MAX_IMAGES_PER_REQUEST} image(s) are allowed.")
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


def _apply_chat_template(model_runner: Any, messages: list[dict[str, Any]]) -> str:
    """先由文本 Tokenizer 应用模板，再交给多模态 Processor 编码张量。"""
    tokenizer = getattr(model_runner, "tokenizer", None) or model_runner.processor
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _prepare_inputs(messages: list[dict[str, Any]], images: list[Image.Image], model_runner: Any) -> tuple[Any, int]:
    prompt = _apply_chat_template(model_runner, messages)
    # 上游 Processor 在一次调用中对 prompt tokenize，并生成视觉张量和图像网格信息。
    inputs = model_runner.processor(text=[prompt], images=images, return_tensors="pt")
    input_tokens = int(inputs["input_ids"].shape[-1])
    return _move_inputs_to_model(inputs, model_runner), input_tokens


class MageVLImageEncoder:
    """单独负责聊天消息与图片输入的解析和编码。"""

    def __init__(self, model_runner: Any) -> None:
        self.model_runner = model_runner

    def encode_chat(self, raw_messages: Any) -> tuple[dict[str, Any], int]:
        messages, images = _normalize_messages(raw_messages)
        return _prepare_inputs(messages, images, self.model_runner)

    @staticmethod
    def resize_image(image: Image.Image, max_side: int) -> Image.Image:
        return _resize_image(image, max_side)
