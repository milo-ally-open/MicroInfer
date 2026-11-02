#!/usr/bin/env python3
"""Profile Qwen3-VL-2B Hugging Face Prefill and one Decode step on a local image."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "checkpoints/Qwen3-VL-2B-Instruct-8964489"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
IMAGE_PATH = ROOT / "test.jpg"
PROMPT = "请简要描述图片中的主要内容。"
DTYPE = torch.bfloat16
ATTENTION = "sdpa"
MAX_NEW_TOKENS = 2  # Prefill 产生首 token，再对下一 token 执行一次 Decode。
MAX_OPERATORS = 30
TRACE_DIR = ROOT / "artifacts/qwen3_vl_2b/traces"
OUTPUT_PATH = ROOT / "artifacts/qwen3-vl-2b-transformers-forward.json"
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if CUDA_AVAILABLE else "cpu")
ACTIVITIES = [torch.profiler.ProfilerActivity.CPU]
if CUDA_AVAILABLE:
    ACTIVITIES.append(torch.profiler.ProfilerActivity.CUDA)

TRACE_DIR.mkdir(parents=True, exist_ok=True)
if not MODEL_PATH.is_dir() or not (MODEL_PATH / "model.safetensors").is_file():
    raise FileNotFoundError(f"固定 revision 的模型快照尚未完整下载：{MODEL_PATH}")
if not IMAGE_PATH.is_file():
    raise FileNotFoundError(f"图文测试图片不存在：{IMAGE_PATH}")

#  ========================================= Transformers 基线 =========================================
print("==================== transformers ====================")
print("model:", MODEL_PATH)
print("revision:", REVISION)
print("device:", torch.cuda.get_device_name(0) if CUDA_AVAILABLE else DEVICE)
print("image:", IMAGE_PATH)

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    revision=REVISION,
    local_files_only=True,
)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    revision=REVISION,
    local_files_only=True,
    use_safetensors=True,
    dtype=DTYPE,
    device_map=0 if CUDA_AVAILABLE else "cpu",
    attn_implementation=ATTENTION,
).eval()

image = Image.open(IMAGE_PATH).convert("RGB")
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT},
    ],
}]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
model_inputs = {}
for name, value in inputs.items():
    if isinstance(value, torch.Tensor):
        if name in {"pixel_values", "pixel_values_videos"} and value.is_floating_point():
            model_inputs[name] = value.to(device=model.device, dtype=model.dtype)
        else:
            model_inputs[name] = value.to(model.device)
    else:
        model_inputs[name] = value
model_inputs.pop("token_type_ids", None)
model_inputs.setdefault("attention_mask", torch.ones_like(model_inputs["input_ids"]))

# Prefill：一次处理文本与图片，生成视觉特征、prompt logits 和可续接的 KV Cache。
if CUDA_AVAILABLE:
    torch.cuda.reset_peak_memory_stats(DEVICE)
prefill_start = torch.cuda.Event(enable_timing=True) if CUDA_AVAILABLE else None
prefill_end = torch.cuda.Event(enable_timing=True) if CUDA_AVAILABLE else None
prefill_wall_start = time.perf_counter()
with torch.profiler.profile(
    activities=ACTIVITIES,
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prefill_profile:
    if prefill_start is not None:
        prefill_start.record()
    with torch.inference_mode():
        prefill = model(**model_inputs, use_cache=True, return_dict=True)
    if prefill_end is not None:
        prefill_end.record()
        prefill_end.synchronize()
prefill_profile.export_chrome_trace(str(TRACE_DIR / "transformers-prefill.json"))
if prefill.past_key_values is None:
    raise RuntimeError("Qwen3-VL Prefill 没有返回 past_key_values。")
next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
prefill_logits = prefill.logits[:, -1, :].detach().float().cpu()
prefill_ms = (
    float(prefill_start.elapsed_time(prefill_end))
    if CUDA_AVAILABLE
    else (time.perf_counter() - prefill_wall_start) * 1000
)

# Decode：把 Prefill 采样出的 token 接到原 prompt 后，并从 HF Cache 续算一个位置。
decode_ids = torch.cat((model_inputs["input_ids"], next_token), dim=-1)
decode_kwargs = {
    name: value for name, value in model_inputs.items()
    if name not in {"input_ids", "attention_mask"}
}
decode_attention_mask = torch.ones_like(decode_ids)
generation_kwargs = {
    **decode_kwargs,
    "attention_mask": decode_attention_mask,
    "past_key_values": prefill.past_key_values,
}
# Qwen3-VL overrides this hook to create MRoPE position IDs and apply its saved rope_deltas.
# GenerationMixin normally calls this before prepare_inputs_for_generation; a manual step must do so too.
decode_position_ids = model._prepare_position_ids_for_generation(decode_ids, generation_kwargs)
decode_inputs = model.prepare_inputs_for_generation(
    input_ids=decode_ids,
    next_sequence_length=1,
    past_key_values=prefill.past_key_values,
    attention_mask=decode_attention_mask,
    position_ids=decode_position_ids,
    use_cache=True,
    is_first_iteration=False,
    **decode_kwargs,
)
if decode_inputs["input_ids"].shape[-1] != 1:
    raise AssertionError(
        "Decode Profile must forward exactly one new token; "
        f"got query length {decode_inputs['input_ids'].shape[-1]}."
    )
prefill_cache_tokens = int(prefill.past_key_values.get_seq_length())
decode_start = torch.cuda.Event(enable_timing=True) if CUDA_AVAILABLE else None
decode_end = torch.cuda.Event(enable_timing=True) if CUDA_AVAILABLE else None
decode_wall_start = time.perf_counter()
with torch.profiler.profile(
    activities=ACTIVITIES,
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as decode_profile:
    if decode_start is not None:
        decode_start.record()
    with torch.inference_mode():
        decode = model(**decode_inputs, return_dict=True)
    if decode_end is not None:
        decode_end.record()
        decode_end.synchronize()
decode_profile.export_chrome_trace(str(TRACE_DIR / "transformers-decode.json"))
decode_logits = decode.logits[:, -1, :].detach().float().cpu()
decode_cache_tokens = int(decode.past_key_values.get_seq_length())
decode_ms = (
    float(decode_start.elapsed_time(decode_end))
    if CUDA_AVAILABLE
    else (time.perf_counter() - decode_wall_start) * 1000
)

# 保存机器可读算子记录；报告中的 shape、调用次数和耗时全部来自 Profiler。
operator_records = []
for phase, profile in (("prefill", prefill_profile), ("decode", decode_profile)):
    averages = profile.key_averages(group_by_input_shape=True)
    sort_key = "self_device_time_total" if CUDA_AVAILABLE else "self_cpu_time_total"
    print(f"{phase}_top_operators:")
    print(averages.table(sort_by=sort_key, row_limit=MAX_OPERATORS))
    for event in averages:
        operator_records.append({
            "phase": phase,
            "operator": str(event.key),
            "input_shapes": event.input_shapes,
            "call_count": int(event.count),
            "self_device_time_total_us": float(getattr(event, "self_device_time_total", 0.0) or 0.0),
            "self_cpu_time_total_us": float(event.self_cpu_time_total),
            "self_device_memory_usage_bytes": int(getattr(event, "self_device_memory_usage", 0) or 0),
            "self_cpu_memory_usage_bytes": int(event.self_cpu_memory_usage),
        })

assert prefill_logits.shape == decode_logits.shape
assert torch.isfinite(prefill_logits).all() and torch.isfinite(decode_logits).all()
assert prefill_cache_tokens == int(model_inputs["input_ids"].shape[-1])
assert decode_cache_tokens == prefill_cache_tokens + 1
result = {
    "model_alias": "qwen3-vl-2b",
    "repo_id": "Qwen/Qwen3-VL-2B-Instruct",
    "revision": REVISION,
    "implementation": "transformers",
    "gpu_name": torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU",
    "dtype": str(model.dtype),
    "attention": ATTENTION,
    "image_path": str(IMAGE_PATH),
    "image_width": image.width,
    "image_height": image.height,
    "prompt_tokens": int(model_inputs["input_ids"].shape[-1]),
    "decode_query_tokens": int(decode_inputs["input_ids"].shape[-1]),
    "prefill_cache_tokens": prefill_cache_tokens,
    "decode_cache_tokens_after_forward": decode_cache_tokens,
    "prefill_ms": prefill_ms,
    "decode_ms": decode_ms,
    "prefill_logits_shape": list(prefill_logits.shape),
    "decode_logits_shape": list(decode_logits.shape),
    "next_token_id": int(next_token.item()),
    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(DEVICE) if CUDA_AVAILABLE else 0,
    "profiled_at_utc": datetime.now(timezone.utc).isoformat(),
    "operator_records": operator_records,
}
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("prefill_ms:", round(prefill_ms, 3))
print("decode_ms:", round(decode_ms, 3))
print("prompt_tokens:", result["prompt_tokens"])
print("peak_cuda_memory_bytes:", result["peak_cuda_memory_bytes"])
print("profile_json:", OUTPUT_PATH)
print("prefill_trace:", TRACE_DIR / "transformers-prefill.json")
print("decode_trace:", TRACE_DIR / "transformers-decode.json")
