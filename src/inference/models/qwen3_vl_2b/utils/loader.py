"""读取启动器所选 Qwen3-VL 配置和本地模型快照。"""

from __future__ import annotations

import os
from pathlib import Path

from inference.engine.config import PROJECT_ROOT, selected_config

CONFIG = selected_config("INFERENCE_CONFIG")
NETWORK_CONFIG = selected_config("INFERENCE_NETWORK_CONFIG")
MODEL_CONFIG = CONFIG["model"]
RUNTIME_CONFIG = CONFIG["runtime"]
API_CONFIG = CONFIG["api"]
ENGINE_CONFIG = CONFIG["engine"]
GENERATION_CONFIG = CONFIG["generation"]
MODEL_ID = str(MODEL_CONFIG["id"])
MODEL_REVISION = str(MODEL_CONFIG["revision"])
MODEL_DTYPE = str(MODEL_CONFIG["dtype"])
ATTENTION_IMPLEMENTATION = str(MODEL_CONFIG["attn_implementation"])
MAX_CONTEXT_TOKENS = int(API_CONFIG["max_context_tokens"])
MAX_NEW_TOKENS = int(API_CONFIG["max_new_tokens"])
MAX_IMAGES_PER_REQUEST = int(API_CONFIG["max_images_per_request"])
MAX_IMAGE_BYTES = int(API_CONFIG["max_image_bytes"])
MAX_IMAGE_PIXELS = int(API_CONFIG["max_image_pixels"])
MAX_IMAGE_SIDE = int(API_CONFIG["max_image_side"])
IMAGE_RESAMPLING = str(API_CONFIG["image_resampling"])
IMAGE_URL_TIMEOUT = float(API_CONFIG["image_url_timeout_seconds"])
KV_CACHE_CAPACITY_TOKENS = int(ENGINE_CONFIG["kv_cache"]["capacity_tokens"])


def resolve_model_path() -> Path:
    """返回仓库中的固定本地 snapshot，并拒绝缺失或不完整的权重目录。"""
    value = os.path.expandvars(os.path.expanduser(str(RUNTIME_CONFIG["local_model_path"])))
    path = Path(value)
    path = path if path.is_absolute() else PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"Qwen3-VL snapshot is missing or incomplete: {path}")
    return path
