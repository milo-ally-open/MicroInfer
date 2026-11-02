#!/usr/bin/env python3
"""Profile Qwen3-VL-2B Hugging Face Prefill and one Decode step on a local image."""

from __future__ import annotations

import json
import gc
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_CONFIG = ROOT / "scripts/qwen3_vl_2b/profiles/inference.yml"
NETWORK_CONFIG = ROOT / "scripts/qwen3_vl_2b/profiles/network.yml"
MANIFEST_PATH = ROOT / "scripts/qwen3_vl_2b/profiles/model.yml"
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
BENCHMARK_PATH = ROOT / "artifacts/qwen3-vl-2b-transformers-benchmark.json"
INFERENCE_OUTPUT_PATH = ROOT / "artifacts/qwen3-vl-2b-inference-forward.json"
INFERENCE_BENCHMARK_PATH = ROOT / "artifacts/qwen3-vl-2b-inference-benchmark.json"
WARMUP_REQUESTS = 1
BENCHMARK_ITERATIONS = 3
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if CUDA_AVAILABLE else "cpu")
ACTIVITIES = [torch.profiler.ProfilerActivity.CPU]
if CUDA_AVAILABLE:
    ACTIVITIES.append(torch.profiler.ProfilerActivity.CUDA)

# 让同一模型目录的 Runtime 工厂读取 scripts/ 下的独立配置副本。
os.environ["INFERENCE_PROJECT_ROOT"] = str(ROOT)
os.environ["INFERENCE_MODEL_ALIAS"] = "qwen3-vl-2b"
os.environ["INFERENCE_MODEL_MANIFEST"] = str(MANIFEST_PATH)
os.environ["INFERENCE_CONFIG"] = str(INFERENCE_CONFIG)
os.environ["INFERENCE_NETWORK_CONFIG"] = str(NETWORK_CONFIG)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from inference.engine.sequence import ExecutionPhase, ScheduledWorkItem
from inference.models.qwen3_vl_2b.runtime import create_runtime

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

processor = AutoProcessor.from_pretrained(MODEL_PATH, revision=REVISION, local_files_only=True)
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
second_token = decode.logits[:, -1, :].argmax(dim=-1, keepdim=True)
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
    "second_token_id": int(second_token.item()),
    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(DEVICE) if CUDA_AVAILABLE else 0,
    "profiled_at_utc": datetime.now(timezone.utc).isoformat(),
    "operator_records": operator_records,
}
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

# Model-level Bench：完整图文请求固定生成两个 token；独立统计重复请求延迟与吞吐。
expected_generated_ids = torch.cat((next_token, second_token), dim=-1).detach().cpu().reshape(-1)
bench_latencies_seconds = []
bench_output_tokens = []
for bench_index in range(WARMUP_REQUESTS + BENCHMARK_ITERATIONS):
    if CUDA_AVAILABLE:
        torch.cuda.synchronize(DEVICE)
    started_at = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**model_inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
    if CUDA_AVAILABLE:
        torch.cuda.synchronize(DEVICE)
    elapsed = time.perf_counter() - started_at
    generated_ids = generated[:, model_inputs["input_ids"].shape[-1]:].detach().cpu()
    if bench_index == 0:
        torch.testing.assert_close(generated_ids.reshape(-1), expected_generated_ids)
    if bench_index >= WARMUP_REQUESTS:
        bench_latencies_seconds.append(elapsed)
        bench_output_tokens.append(int(generated_ids.shape[-1]))
transformers_benchmark = {
    "model_alias": "qwen3-vl-2b",
    "revision": REVISION,
    "device": torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "cpu",
    "dtype": str(model.dtype),
    "implementation": "transformers",
    "input_tokens": int(model_inputs["input_ids"].shape[-1]),
    "latency_seconds_median": statistics.median(bench_latencies_seconds),
    "tokens_per_second_median": statistics.median(
        tokens / latency for tokens, latency in zip(bench_output_tokens, bench_latencies_seconds)
    ),
    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(DEVICE) if CUDA_AVAILABLE else 0,
    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(DEVICE) if CUDA_AVAILABLE else 0,
    "warmup_requests": WARMUP_REQUESTS,
    "iterations": BENCHMARK_ITERATIONS,
}
BENCHMARK_PATH.write_text(json.dumps(transformers_benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
print("Test: Transformers generate matches manual Prefill/Decode token IDs.")
print("Bench: Transformers median latency:", round(transformers_benchmark["latency_seconds_median"] * 1000, 3), "ms")
print("Bench: Transformers median throughput:", round(transformers_benchmark["tokens_per_second_median"], 3), "tokens/s")
print("benchmark_json:", BENCHMARK_PATH)
print("prefill_ms:", round(prefill_ms, 3))
print("decode_ms:", round(decode_ms, 3))
print("prompt_tokens:", result["prompt_tokens"])
print("peak_cuda_memory_bytes:", result["peak_cuda_memory_bytes"])
print("profile_json:", OUTPUT_PATH)
print("prefill_trace:", TRACE_DIR / "transformers-prefill.json")
print("decode_trace:", TRACE_DIR / "transformers-decode.json")

# ==================== inference ====================
# 释放 Transformers 权重后加载项目 Runtime，避免同一张 GPU 同时驻留两份 2B 权重。
expected_generated_ids = expected_generated_ids.cpu()
transformer_benchmark = dict(transformers_benchmark)
del model, processor, prefill, decode, prefill_profile, decode_profile
gc.collect()
if CUDA_AVAILABLE:
    torch.cuda.empty_cache()

print("==================== inference ====================")
runtime = create_runtime()
try:
    runtime.load()
    runner = runtime.model_runner
    if runner.model is None:
        raise RuntimeError("Inference Runtime 没有加载 Qwen3-VL 模型。")
    runner_inputs = {}
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if name in {"pixel_values", "pixel_values_videos"} and value.is_floating_point():
                runner_inputs[name] = value.to(device=runner.input_device, dtype=runner.model.dtype)
            else:
                runner_inputs[name] = value.to(runner.input_device)
        else:
            runner_inputs[name] = value
    runner_inputs.pop("token_type_ids", None)
    runner_inputs.setdefault("attention_mask", torch.ones_like(runner_inputs["input_ids"]))
    prompt_tokens = int(runner_inputs["input_ids"].shape[-1])
    block_size = int(yaml.safe_load(INFERENCE_CONFIG.read_text(encoding="utf-8"))["engine"]["kv_cache"]["block_size_tokens"])
    block_count = (prompt_tokens + MAX_NEW_TOKENS + block_size - 1) // block_size
    options = runtime.model_adapter.generation_options({"max_tokens": MAX_NEW_TOKENS, "temperature": 0})
    request_id = "qwen3-vl-forward-profile"
    runner.begin_generation(request_id, runner_inputs, options)
    runner.bind_generation_cache(request_id, tuple(range(block_count)))

    # Inference Prefill/Decode 各记录一次算子 Profile；Cache 断言检查实际 Query 与续算长度。
    inference_profiles = []
    phase_latencies_ms = {}
    for phase, token_budget in ((ExecutionPhase.PREFILL, prompt_tokens), (ExecutionPhase.DECODE, 1)):
        phase_name = phase.value
        started = torch.cuda.Event(enable_timing=True) if CUDA_AVAILABLE else None
        ended = torch.cuda.Event(enable_timing=True) if CUDA_AVAILABLE else None
        wall_started = time.perf_counter()
        with torch.profiler.profile(activities=ACTIVITIES, record_shapes=True, profile_memory=True, with_stack=True) as phase_profile:
            if started is not None:
                started.record()
            phase_result = runner.execute_batch((ScheduledWorkItem(request_id, phase, token_budget),))[0]
            if ended is not None:
                ended.record()
                ended.synchronize()
        phase_profile.export_chrome_trace(str(TRACE_DIR / f"inference-{phase_name}.json"))
        phase_latencies_ms[phase_name] = float(started.elapsed_time(ended)) if CUDA_AVAILABLE else (time.perf_counter() - wall_started) * 1000
        inference_profiles.append((phase_name, phase_profile))
        print(f"{phase_name}_ms:", round(phase_latencies_ms[phase_name], 3))
    inference_generated_ids, finish_reason = runner.finish_generation(request_id)
    inference_generated_ids = inference_generated_ids.detach().cpu()
    torch.testing.assert_close(inference_generated_ids.reshape(-1), expected_generated_ids)

    inference_operator_records = []
    for phase_name, phase_profile in inference_profiles:
        averages = phase_profile.key_averages(group_by_input_shape=True)
        sort_key = "self_device_time_total" if CUDA_AVAILABLE else "self_cpu_time_total"
        print(f"inference_{phase_name}_top_operators:")
        print(averages.table(sort_by=sort_key, row_limit=MAX_OPERATORS))
        for event in averages:
            inference_operator_records.append({
                "phase": phase_name,
                "operator": str(event.key),
                "input_shapes": event.input_shapes,
                "call_count": int(event.count),
                "self_device_time_total_us": float(getattr(event, "self_device_time_total", 0.0) or 0.0),
                "self_cpu_time_total_us": float(event.self_cpu_time_total),
                "self_device_memory_usage_bytes": int(getattr(event, "self_device_memory_usage", 0) or 0),
                "self_cpu_memory_usage_bytes": int(event.self_cpu_memory_usage),
            })

    # Model-level Bench：每轮独立创建/释放请求状态，计时完整的图文 Prefill 与单 token Decode。
    inference_latencies = []
    for bench_index in range(WARMUP_REQUESTS + BENCHMARK_ITERATIONS):
        bench_request_id = f"qwen3-vl-benchmark-{bench_index}"
        runner.begin_generation(bench_request_id, runner_inputs, options)
        runner.bind_generation_cache(bench_request_id, tuple(range(block_count)))
        if CUDA_AVAILABLE:
            torch.cuda.synchronize(DEVICE)
        bench_started = time.perf_counter()
        runner.execute_batch((ScheduledWorkItem(bench_request_id, ExecutionPhase.PREFILL, prompt_tokens),))
        runner.execute_batch((ScheduledWorkItem(bench_request_id, ExecutionPhase.DECODE, 1),))
        if CUDA_AVAILABLE:
            torch.cuda.synchronize(DEVICE)
        elapsed = time.perf_counter() - bench_started
        bench_ids, _ = runner.finish_generation(bench_request_id)
        torch.testing.assert_close(bench_ids.detach().cpu().reshape(-1), expected_generated_ids)
        if bench_index >= WARMUP_REQUESTS:
            inference_latencies.append(elapsed)

    inference_benchmark = {
        "model_alias": "qwen3-vl-2b",
        "revision": REVISION,
        "device": torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "cpu",
        "dtype": str(runner.model.dtype),
        "implementation": "inference",
        "input_tokens": prompt_tokens,
        "latency_seconds_median": statistics.median(inference_latencies),
        "tokens_per_second_median": statistics.median(
            int(inference_generated_ids.numel()) / latency for latency in inference_latencies
        ),
        "peak_cuda_memory_bytes": runner.peak_memory()["peak_allocated"],
        "peak_cuda_reserved_bytes": runner.peak_memory()["peak_reserved"],
        "warmup_requests": WARMUP_REQUESTS,
        "iterations": BENCHMARK_ITERATIONS,
    }
    INFERENCE_BENCHMARK_PATH.write_text(json.dumps(inference_benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
    inference_result = {
        "model_alias": "qwen3-vl-2b",
        "revision": REVISION,
        "implementation": "inference",
        "gpu_name": torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU",
        "dtype": str(runner.model.dtype),
        "prompt_tokens": prompt_tokens,
        "decode_query_tokens": 1,
        "generated_token_ids": [int(token) for token in inference_generated_ids.reshape(-1).tolist()],
        "finish_reason": finish_reason,
        "prefill_ms": phase_latencies_ms["prefill"],
        "decode_ms": phase_latencies_ms["decode"],
        "peak_cuda_memory_bytes": runner.peak_memory()["peak_allocated"],
        "operator_records": inference_operator_records,
        "benchmark": inference_benchmark,
    }
    INFERENCE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    INFERENCE_OUTPUT_PATH.write_text(json.dumps(inference_result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Test: Inference generated token IDs match Transformers.")
    print("Bench: Inference median latency:", round(inference_benchmark["latency_seconds_median"] * 1000, 3), "ms")
    print("Bench: Inference median throughput:", round(inference_benchmark["tokens_per_second_median"], 3), "tokens/s")
    print("inference_profile_json:", INFERENCE_OUTPUT_PATH)
    print("inference_benchmark_json:", INFERENCE_BENCHMARK_PATH)
finally:
    runtime.close()
