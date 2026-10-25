"""将固定 checkpoint 的 Safetensors 权重安全映射到本地模型计算图。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase, Qwen3Config

from inference.engine.config.model_config import PROJECT_ROOT, selected_config


# 模型设定由 Transformers Config 载入；服务参数由 YAML 一次载入。
CONFIG: dict[str, Any] = selected_config("INFERENCE_CONFIG")  # 模型与推论服务设定。
NETWORK_CONFIG: dict[str, Any] = selected_config("INFERENCE_NETWORK_CONFIG")  # API 网路设定。
MODEL_CONFIG = CONFIG["model"]  # Transformers Qwen3Config 与模型权重参数。
RUNTIME_CONFIG = CONFIG["runtime"]  # 装置与 Worker 执行环境设定。
API_CONFIG = CONFIG["api"]  # 请求上下文与本文大小限制。
ENGINE_CONFIG = CONFIG["engine"]  # Engine 排程与 KV page pool 设定。
GENERATION_CONFIG = CONFIG["generation"]  # Token 生成预设参数。

MODEL_ALIAS = "qwen3-0.6b"  # 对外使用的专案模型别名。
MODEL_ID = str(MODEL_CONFIG["id"])  # Hugging Face repo_id。
MODEL_REVISION = str(MODEL_CONFIG["revision"])  # 固定的上游模型 revision。
MODEL_DTYPE = str(MODEL_CONFIG["dtype"])  # 权重与计算资料型别。
MODEL_DEVICE = str(MODEL_CONFIG["device"])  # 模型 Worker 使用的装置。
PLATFORM_NAME = str(RUNTIME_CONFIG["platform"])  # PyTorch 执行平台识别名称。
LOCAL_MODEL_PATH = (
    PROJECT_ROOT / os.path.expandvars(os.path.expanduser(RUNTIME_CONFIG["local_model_path"]))
).resolve()  # 固定 revision 的本机快照目录。

MAX_CONTEXT_TOKENS = int(API_CONFIG["max_context_tokens"])  # 单次请求的上下文上限。
MAX_NEW_TOKENS = int(GENERATION_CONFIG["max_new_tokens"])  # 单次生成长度上限。
MAX_CHAT_REQUEST_BYTES = int(API_CONFIG["max_chat_request_bytes"])  # 聊天请求本文大小上限。
MAX_NUM_SEQS = int(ENGINE_CONFIG["max_num_seqs"])  # 同时执行的序列数上限。
MAX_WAITING_REQUESTS = int(ENGINE_CONFIG["max_waiting_requests"])  # 等候伫列上限。
MAX_NUM_SCHEDULED_TOKENS = int(ENGINE_CONFIG["max_num_scheduled_tokens"])  # 单轮排程 Token 预算。
BATCH_WAIT_SECONDS = float(ENGINE_CONFIG["batch_wait_seconds"])  # 合并批次的等待时间。
PREFILL_CHUNK_SIZE_TOKENS = int(ENGINE_CONFIG["prefill_chunk_size_tokens"])  # 分块 Prefill 大小。
KV_CACHE_CONFIG = ENGINE_CONFIG["kv_cache"]  # 实体 KV page pool 容量设定。
KV_CACHE_BLOCK_SIZE_TOKENS = int(KV_CACHE_CONFIG["block_size_tokens"])  # 每个 KV page 的 Token 数。
KV_CACHE_NUM_BLOCKS = int(KV_CACHE_CONFIG["num_blocks"])  # 实体 KV page 数量。
KV_CACHE_CAPACITY_TOKENS = KV_CACHE_NUM_BLOCKS * KV_CACHE_BLOCK_SIZE_TOKENS  # KV page pool 可容纳的 Token 数。
WORKER_STARTUP_TIMEOUT = float(RUNTIME_CONFIG["worker_startup_timeout_seconds"])  # Worker 启动逾时。
WORKER_REQUEST_TIMEOUT = float(RUNTIME_CONFIG["worker_request_timeout_seconds"])  # Worker 请求逾时。
ENGINE_CORE_STARTUP_TIMEOUT = float(RUNTIME_CONFIG["engine_core_startup_timeout_seconds"])  # Engine Core 启动逾时。
ENGINE_CORE_REQUEST_TIMEOUT = float(RUNTIME_CONFIG["engine_core_request_timeout_seconds"])  # Engine Core 请求逾时。

def _weight_files(snapshot: Path) -> tuple[Path, ...]:
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid Safetensors index: {index_path}")
        names = sorted(set(str(name) for name in weight_map.values()))
        files = tuple((snapshot / name).resolve() for name in names)
        if any(path.parent != snapshot.resolve() for path in files):
            raise ValueError("Safetensors index entries must stay inside the local model snapshot.")
    else:
        files = tuple(sorted(snapshot.glob("*.safetensors")))
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError(f"No complete Safetensors checkpoint was found in {snapshot}.")
    return files


def load_safetensors(model: torch.nn.Module, snapshot: Path) -> int:
    """载入且严格核对所有权重名称与 shape，回传已载入参数量。"""
    from safetensors.torch import load_file

    state: dict[str, torch.Tensor] = {}
    for path in _weight_files(snapshot):
        shard = load_file(str(path), device="cpu")
        duplicate = state.keys() & shard.keys()
        if duplicate:
            raise ValueError(f"Duplicate checkpoint tensor: {sorted(duplicate)[0]}.")
        state.update(shard)

    expected = model.state_dict()
    missing = expected.keys() - state.keys()
    unexpected = state.keys() - expected.keys()
    if missing or unexpected:
        raise ValueError(
            "Qwen3 checkpoint keys do not match the local model: "
            f"missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]}."
        )
    for name, parameter in expected.items():
        if state[name].shape != parameter.shape:
            raise ValueError(
                f"Qwen3 checkpoint shape mismatch for {name}: "
                f"expected {tuple(parameter.shape)}, received {tuple(state[name].shape)}."
            )
    result: Any = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"Qwen3 checkpoint load was incomplete: {result}.")
    del expected, state
    model.tie_weights()
    return sum(parameter.numel() for parameter in model.parameters())


def load_model_config(snapshot: Path, revision: str) -> Qwen3Config:
    """使用 Transformers 的 Qwen3Config 载入已固定 revision 的本机设定。"""
    verify_snapshot_revision(snapshot, revision)
    config = Qwen3Config.from_pretrained(snapshot, revision=revision, local_files_only=True)
    required = (
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "rms_norm_eps",
    )
    missing = [name for name in required if getattr(config, name, None) is None]
    if missing:
        raise ValueError(f"Qwen3 config is missing required fields: {', '.join(missing)}.")
    if int(config.hidden_size) % int(config.num_attention_heads):
        raise ValueError("Qwen3 hidden_size must be divisible by num_attention_heads.")
    if int(config.num_attention_heads) % int(config.num_key_value_heads):
        raise ValueError("Qwen3 query heads must be divisible by key/value heads.")
    head_dim = getattr(config, "head_dim", None)
    head_dim = int(head_dim or (int(config.hidden_size) // int(config.num_attention_heads)))
    if head_dim < 1 or head_dim % 2:
        raise ValueError("Qwen3 rotary head dimension must be even.")
    config.head_dim = head_dim
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    rope_type = str(rope_parameters.get("rope_type", "default")) if isinstance(rope_parameters, dict) else "default"
    if rope_type != "default":
        raise ValueError("This Qwen3 runner supports the pinned default RoPE configuration only.")
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None and isinstance(rope_parameters, dict):
        rope_theta = rope_parameters.get("rope_theta")
    if rope_theta is None:
        raise ValueError("Qwen3 config is missing the default RoPE theta.")
    config.rope_theta = float(rope_theta)
    return config


def verify_snapshot_revision(snapshot: Path, revision: str) -> None:
    """确认本机 Hugging Face 快照及其档案 metadata 属于固定 commit。"""
    import re

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Qwen3 revision must be a full 40-character Hugging Face commit hash.")
    metadata_dir = snapshot / ".cache" / "huggingface" / "download"
    metadata_files = tuple(metadata_dir.glob("*.metadata"))
    if not metadata_files:
        raise FileNotFoundError(f"Hugging Face revision metadata is missing from the local snapshot: {metadata_dir}")
    required_files = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    required_files.update(path.name for path in snapshot.glob("*.safetensors"))
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        required_files.add(index_path.name)
    missing_assets = sorted(name for name in required_files if not (snapshot / name).is_file())
    if missing_assets:
        raise FileNotFoundError("Required files are missing from the local Qwen3 snapshot: " + ", ".join(missing_assets))
    metadata_by_name = {path.name.removesuffix(".metadata"): path for path in metadata_files}
    missing = sorted(required_files - metadata_by_name.keys())
    if missing:
        raise FileNotFoundError("Hugging Face revision metadata is missing for local snapshot assets: " + ", ".join(missing))
    for path in metadata_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        received = lines[0].strip() if lines else "<empty>"
        if received != revision:
            raise ValueError(
                f"Qwen3 snapshot asset {path.name.removesuffix('.metadata')} belongs to revision "
                f"{received}, expected {revision}."
            )


def load_tokenizer(snapshot: Path, revision: str) -> PreTrainedTokenizerBase:
    """依 Hugging Face tokenizer metadata 建立固定 revision 的 Fast Tokenizer。"""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        snapshot, revision=revision, local_files_only=True, use_fast=True, trust_remote_code=False
    )
