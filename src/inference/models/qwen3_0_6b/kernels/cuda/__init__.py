"""PyTorch C++/CUDA 推理算子；扩展在显式选择 CUDA 后端时才载入。"""

from __future__ import annotations

from importlib.machinery import EXTENSION_SUFFIXES
import math
from pathlib import Path
from typing import Any

import torch


def load_extension() -> Any:
    try:
        libraries = [
            path
            for path in Path(__file__).parent.glob("_C*")
            if path.is_file() and any(path.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES)
        ]
        if not libraries:
            raise FileNotFoundError("No in-package Qwen3 CUDA extension library was found.")
        torch.ops.load_library(str(libraries[0]))
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "CUDA attention was selected, but the Qwen3 CUDA extension is not built. "
            "Build the extension with the active PyTorch/CUDA toolchain first."
        ) from exc
    if not hasattr(torch.ops, "qwen3_cuda") or not hasattr(torch.ops.qwen3_cuda, "paged_decode"):
        raise RuntimeError("The Qwen3 CUDA extension did not register qwen3_cuda::paged_decode.")
    return torch.ops.qwen3_cuda


def paged_decode_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    *,
    layer_index: int,
    block_size_tokens: int,
    scale: float,
) -> torch.Tensor:
    """直接按每个请求的 block table 读取 pages，执行单 token decode Attention。"""
    if query.ndim != 3 or kv_cache.ndim != 6:
        raise ValueError("Paged decode expects query [batch, heads, head_dim] and a six-dimensional KV pool.")
    if not query.is_cuda or not kv_cache.is_cuda:
        raise ValueError("The native paged decode operator requires CUDA tensors.")
    if query.device != kv_cache.device or query.dtype != kv_cache.dtype:
        raise ValueError("Query and KV pool must use the same device and dtype.")
    if query.shape[0] < 1 or query.shape[1] < 1 or query.shape[2] < 1:
        raise ValueError("Paged decode query dimensions must be positive.")
    if query.shape[2] > 128 or query.shape[2] & (query.shape[2] - 1):
        raise ValueError("Paged decode head dimension must be a power of two no larger than 128.")
    if kv_cache.shape[1] != 2 or any(dim < 1 for dim in (kv_cache.shape[0], *kv_cache.shape[2:])):
        raise ValueError("Paged decode KV pool dimensions must be positive with key/value axis 1 of size 2.")
    if query.shape[1] % kv_cache.shape[4] or query.shape[2] != kv_cache.shape[5]:
        raise ValueError("Paged decode query and KV head dimensions are incompatible.")
    if block_size_tokens < 1 or kv_cache.shape[3] != block_size_tokens:
        raise ValueError("Paged decode block size does not match the KV pool.")
    if block_tables.dtype != torch.int32 or context_lengths.dtype != torch.int32:
        raise ValueError("CUDA block tables and context lengths must use int32.")
    if block_tables.ndim != 2 or context_lengths.ndim != 1:
        raise ValueError("Block tables must be rank two and context lengths rank one.")
    if block_tables.shape[0] != query.shape[0] or context_lengths.shape[0] != query.shape[0]:
        raise ValueError("Paged decode metadata batch dimensions must match the query.")
    if block_tables.shape[1] < 1:
        raise ValueError("Paged decode block tables must have at least one page column.")
    if block_tables.device != query.device or context_lengths.device != query.device:
        raise ValueError("CUDA attention metadata must be on the query device.")
    if not block_tables.is_contiguous() or not context_lengths.is_contiguous():
        raise ValueError("CUDA attention metadata must be contiguous.")
    if layer_index < 0 or layer_index >= kv_cache.shape[0]:
        raise ValueError("Paged decode layer index is outside the KV pool.")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Paged decode scale must be finite and positive.")
    return load_extension().paged_decode(
        query,
        kv_cache,
        block_tables,
        context_lengths,
        int(layer_index),
        int(block_size_tokens),
        float(scale),
    )
