"""在 CUDA 上对照 Qwen3 paged decode kernel 与 PyTorch 稠密参考。"""

from __future__ import annotations

import torch

from . import paged_decode_attention


if not torch.cuda.is_available():
    raise RuntimeError("Qwen3 native CUDA kernel check requires an available CUDA device.")

device = torch.device("cuda")
torch.manual_seed(173)
torch.cuda.manual_seed_all(173)
batch_size = 3
query_heads = 4
kv_heads = 2
head_dim = 32
block_size = 4
context_lengths = torch.tensor([1, 4, 9], device=device, dtype=torch.int32)
block_tables = torch.tensor(
    [
        [7, 2, 5],
        [0, 6, 3],
        [1, 4, 8],
    ],
    device=device,
    dtype=torch.int32,
)
query = torch.randn(batch_size, query_heads, head_dim, device=device, dtype=torch.bfloat16)
kv_cache = torch.randn(
    1,
    2,
    9,
    block_size,
    kv_heads,
    head_dim,
    device=device,
    dtype=torch.bfloat16,
)
scale = head_dim**-0.5

actual = paged_decode_attention(
    query,
    kv_cache,
    block_tables,
    context_lengths,
    layer_index=0,
    block_size_tokens=block_size,
    scale=scale,
)

group_size = query_heads // kv_heads
expected_batches = []
for batch_index, context_length in enumerate(context_lengths.tolist()):
    positions = torch.arange(context_length, device=device)
    logical_pages = positions // block_size
    page_offsets = positions % block_size
    physical_pages = block_tables[batch_index, logical_pages].long()
    keys = kv_cache[0, 0, physical_pages, page_offsets].float()
    values = kv_cache[0, 1, physical_pages, page_offsets].float()
    expected_heads = []
    for query_head in range(query_heads):
        kv_head = query_head // group_size
        scores = torch.mv(keys[:, kv_head, :], query[batch_index, query_head].float()) * scale
        probabilities = torch.softmax(scores, dim=0)
        expected_heads.append(torch.mv(values[:, kv_head, :].T, probabilities))
    expected_batches.append(torch.stack(expected_heads))
expected = torch.stack(expected_batches).to(dtype=query.dtype)

torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
if not bool(torch.isfinite(actual).all()):
    raise AssertionError("The CUDA paged decode kernel returned non-finite values.")

half_query = torch.randn(batch_size, query_heads, head_dim, device=device, dtype=torch.float16)
half_cache = torch.randn(
    1, 2, 9, block_size, kv_heads, head_dim, device=device, dtype=torch.float16
)
half_actual = paged_decode_attention(
    half_query,
    half_cache,
    block_tables,
    context_lengths,
    layer_index=0,
    block_size_tokens=block_size,
    scale=scale,
)
half_expected_batches = []
for batch_index, context_length in enumerate(context_lengths.tolist()):
    positions = torch.arange(context_length, device=device)
    physical_pages = block_tables[batch_index, positions // block_size].long()
    page_offsets = positions % block_size
    keys = half_cache[0, 0, physical_pages, page_offsets].float()
    values = half_cache[0, 1, physical_pages, page_offsets].float()
    half_expected_heads = []
    for query_head in range(query_heads):
        kv_head = query_head // group_size
        scores = torch.mv(keys[:, kv_head, :], half_query[batch_index, query_head].float()) * scale
        probabilities = torch.softmax(scores, dim=0)
        half_expected_heads.append(torch.mv(values[:, kv_head, :].T, probabilities))
    half_expected_batches.append(torch.stack(half_expected_heads))
half_expected = torch.stack(half_expected_batches).to(dtype=half_query.dtype)
torch.testing.assert_close(half_actual, half_expected, atol=5e-3, rtol=5e-3)
if not bool(torch.isfinite(half_actual).all()):
    raise AssertionError("The float16 CUDA paged decode kernel returned non-finite values.")

for invalid_call in (
    lambda: paged_decode_attention(
        query[:0],
        kv_cache,
        block_tables,
        context_lengths,
        layer_index=0,
        block_size_tokens=block_size,
        scale=scale,
    ),
    lambda: paged_decode_attention(
        query,
        kv_cache,
        block_tables,
        context_lengths,
        layer_index=1,
        block_size_tokens=block_size,
        scale=scale,
    ),
    lambda: paged_decode_attention(
        query,
        kv_cache,
        block_tables,
        context_lengths,
        layer_index=0,
        block_size_tokens=block_size + 1,
        scale=scale,
    ),
):
    try:
        invalid_call()
    except ValueError:
        continue
    raise AssertionError("The CUDA paged decode wrapper accepted invalid metadata.")

print(
    "Qwen3 CUDA paged decode matches the PyTorch reference on "
    f"{torch.cuda.get_device_name(device)}; contexts={context_lengths.tolist()}, "
    f"dtypes={query.dtype}/{half_query.dtype}, "
    f"max_abs_error={max((actual.float() - expected.float()).abs().max().item(), (half_actual.float() - half_expected.float()).abs().max().item()):.6g}."
)
