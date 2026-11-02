"""安全读取 OpenAI 图片内容并将其转换为 RGB PIL 图像。"""

from __future__ import annotations

import base64
import binascii
import io
import warnings
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError

from inference.engine.sequence import InferenceError
from ..utils import loader as config

_RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def load_image(source: str) -> Image.Image:
    if source.startswith("data:"):
        header, separator, payload = source.partition(",")
        if not separator or ";base64" not in header.lower():
            raise InferenceError(400, "Image data URLs must use base64 encoding.")
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InferenceError(400, "Invalid base64 image payload.") from exc
    else:
        parsed = urlparse(source)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise InferenceError(400, "image_url must be an http(s) URL or a base64 data URL.")
        request = Request(source, headers={"User-Agent": "Qwen3VLInference/0.1"})
        try:
            with urlopen(request, timeout= config.IMAGE_URL_TIMEOUT) as response:
                if response.status != 200:
                    raise InferenceError(400, "Image URL did not return HTTP 200.")
                raw = response.read(config.MAX_IMAGE_BYTES + 1)
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(400, f"Could not fetch image_url: {exc}") from exc
    if len(raw) > config.MAX_IMAGE_BYTES:
        raise InferenceError(413, f"Image exceeds max_image_bytes={config.MAX_IMAGE_BYTES}.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as opened:
                if opened.width * opened.height > config.MAX_IMAGE_PIXELS:
                    raise InferenceError(413, f"Image exceeds max_image_pixels={config.MAX_IMAGE_PIXELS}.")
                opened.load()
                image = opened.convert("RGB")
    except InferenceError:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise InferenceError(413, f"Image exceeds max_image_pixels={config.MAX_IMAGE_PIXELS}.") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise InferenceError(400, "Payload is not a supported image.") from exc
    filter_name = config.IMAGE_RESAMPLING.lower()
    if filter_name not in _RESAMPLING:
        raise RuntimeError(f"Unsupported image resampling mode: {filter_name}.")
    image.thumbnail((config.MAX_IMAGE_SIDE, config.MAX_IMAGE_SIDE), _RESAMPLING[filter_name])
    return image
