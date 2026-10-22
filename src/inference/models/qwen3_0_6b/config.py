"""載入 Qwen3-0.6B 的模型與服務 YAML 設定。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = os.environ.get("INFERENCE_CONFIG")
NETWORK_CONFIG_PATH = os.environ.get("INFERENCE_NETWORK_CONFIG")
if not CONFIG_PATH or not NETWORK_CONFIG_PATH:
    raise RuntimeError("Set INFERENCE_CONFIG and INFERENCE_NETWORK_CONFIG before importing the Qwen3 app.")

CONFIG: dict[str, Any] = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
NETWORK_CONFIG: dict[str, Any] = yaml.safe_load(Path(NETWORK_CONFIG_PATH).read_text(encoding="utf-8"))
PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODEL_CONFIG = CONFIG["model"]
RUNTIME_CONFIG = CONFIG.get("runtime", {})
API_CONFIG = CONFIG["api"]
ENGINE_CONFIG = CONFIG.get("engine", {})
GENERATION_CONFIG = CONFIG["generation"]

MODEL_ALIAS = "qwen3-0.6b"
MODEL_ID = str(MODEL_CONFIG["id"])
MODEL_REVISION = str(MODEL_CONFIG["revision"])
MODEL_DTYPE = str(MODEL_CONFIG.get("dtype", "bfloat16"))
MODEL_DEVICE = str(MODEL_CONFIG.get("device", "cuda:0"))
LOCAL_MODEL_PATH = Path(os.path.expandvars(os.path.expanduser(str(RUNTIME_CONFIG["local_model_path"]))))
if not LOCAL_MODEL_PATH.is_absolute():
    LOCAL_MODEL_PATH = PROJECT_ROOT / LOCAL_MODEL_PATH
LOCAL_MODEL_PATH = LOCAL_MODEL_PATH.resolve()

MAX_CONTEXT_TOKENS = int(API_CONFIG["max_context_tokens"])
MAX_NEW_TOKENS = int(GENERATION_CONFIG["max_new_tokens"])
MAX_CHAT_REQUEST_BYTES = int(API_CONFIG["max_chat_request_bytes"])
MAX_NUM_SEQS = int(ENGINE_CONFIG.get("max_num_seqs", 4))
MAX_WAITING_REQUESTS = int(ENGINE_CONFIG.get("max_waiting_requests", 16))
MAX_NUM_SCHEDULED_TOKENS = int(ENGINE_CONFIG.get("max_num_scheduled_tokens", MAX_CONTEXT_TOKENS))
BATCH_WAIT_SECONDS = float(ENGINE_CONFIG.get("batch_wait_seconds", 0.01))
_prefill_chunk_size = ENGINE_CONFIG.get("prefill_chunk_size_tokens")
PREFILL_CHUNK_SIZE_TOKENS = int(_prefill_chunk_size) if _prefill_chunk_size is not None else None
KV_CACHE_CONFIG = ENGINE_CONFIG.get("kv_cache", {})
KV_CACHE_BLOCK_SIZE_TOKENS = int(KV_CACHE_CONFIG.get("block_size_tokens", 16))
_configured_blocks = KV_CACHE_CONFIG.get("num_blocks")
_configured_capacity = KV_CACHE_CONFIG.get("capacity_tokens")
if _configured_blocks is None:
    KV_CACHE_CAPACITY_TOKENS = int(_configured_capacity or MAX_CONTEXT_TOKENS * MAX_NUM_SEQS)
    if KV_CACHE_CAPACITY_TOKENS % KV_CACHE_BLOCK_SIZE_TOKENS:
        raise RuntimeError("engine.kv_cache.capacity_tokens must be divisible by block_size_tokens.")
    KV_CACHE_NUM_BLOCKS = KV_CACHE_CAPACITY_TOKENS // KV_CACHE_BLOCK_SIZE_TOKENS
else:
    KV_CACHE_NUM_BLOCKS = int(_configured_blocks)
    KV_CACHE_CAPACITY_TOKENS = KV_CACHE_NUM_BLOCKS * KV_CACHE_BLOCK_SIZE_TOKENS
    if _configured_capacity is not None and int(_configured_capacity) != KV_CACHE_CAPACITY_TOKENS:
        raise RuntimeError("engine.kv_cache.num_blocks conflicts with capacity_tokens.")
WORKER_STARTUP_TIMEOUT = float(RUNTIME_CONFIG.get("worker_startup_timeout_seconds", 2400))
WORKER_REQUEST_TIMEOUT = float(RUNTIME_CONFIG.get("worker_request_timeout_seconds", 1800))
ENGINE_CORE_STARTUP_TIMEOUT = float(RUNTIME_CONFIG.get("engine_core_startup_timeout_seconds", WORKER_STARTUP_TIMEOUT + 30))
ENGINE_CORE_REQUEST_TIMEOUT = float(RUNTIME_CONFIG.get("engine_core_request_timeout_seconds", WORKER_REQUEST_TIMEOUT + 30))

if MAX_CONTEXT_TOKENS < 1 or MAX_NEW_TOKENS < 1:
    raise RuntimeError("Qwen3 context and generation limits must be positive.")
if KV_CACHE_NUM_BLOCKS < 1:
    raise RuntimeError("engine.kv_cache.num_blocks must be positive.")
if MAX_NUM_SCHEDULED_TOKENS < MAX_CONTEXT_TOKENS:
    raise RuntimeError("engine.max_num_scheduled_tokens must cover api.max_context_tokens.")
if PREFILL_CHUNK_SIZE_TOKENS is not None and PREFILL_CHUNK_SIZE_TOKENS < 1:
    raise RuntimeError("engine.prefill_chunk_size_tokens must be positive when configured.")
