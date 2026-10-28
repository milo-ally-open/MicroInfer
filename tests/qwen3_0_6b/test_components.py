#!/usr/bin/env python3
"""检查 Qwen3 计算图、模块入口，并可选分析真实 Attention 算子。"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True, help="明确指定测试 YAML。")
parser.add_argument(
    "--profile-attention",
    action="store_true",
    help="使用本地 Qwen3 第 0 层 Attention 权重分析算子；优先 CUDA，仅在 CUDA 不可用时回退 CPU。",
)
parser.add_argument(
    "--operator-report",
    type=Path,
    help="将算子测量值写入机器可读 JSON；须同时指定 --profile-attention。",
)
args = parser.parse_args()
if args.operator_report and not args.profile_attention:
    parser.error("--operator-report 必须与 --profile-attention 一起使用。")

config_path = args.config.resolve()
settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if not isinstance(settings, dict):
    raise ValueError("测试 YAML 最上层必须是 mapping。")
snapshot = (config_path.parent / settings["snapshot"]).resolve()
if not (snapshot / "config.json").is_file():
    raise FileNotFoundError(f"本地模型配置不存在：{snapshot / 'config.json'}")
inference_config_path = (config_path.parent / settings["inference_config"]).resolve()
inference_settings = yaml.safe_load(inference_config_path.read_text(encoding="utf-8"))
inference_root = (config_path.parent / settings["component"]["engine_module"]).resolve()
repository_root = config_path.parents[2]
os.environ["INFERENCE_CONFIG"] = str(inference_config_path)
os.environ["INFERENCE_NETWORK_CONFIG"] = str(inference_config_path.with_name("network.yml"))

from inference.models.qwen3_0_6b.utils.loader import load_model_config
from inference.models.qwen3_0_6b.model.qwen3 import Qwen3ForCausalLM
from inference.engine.block_manager import BlockManager
from inference.engine.sequence import ExecutionPhase, SequenceState, WorkerResult
from inference.engine.scheduler import TokenScheduler

sequence = SequenceState(
    request_id="engine-sequence-check",
    prompt_tokens_total=2,
    max_output_tokens=2,
)
assert sequence.prompt_tokens_remaining == 2 and not sequence.waiting

capacity_manager = BlockManager(block_size_tokens=4, capacity_tokens=8, bytes_per_token=16)
first_lease = capacity_manager.reserve(
    "block-manager-first", prompt_tokens=4, max_output_tokens=4, initial_tokens=4
)
second_lease = capacity_manager.reserve(
    "block-manager-second", prompt_tokens=4, max_output_tokens=4, initial_tokens=4
)
assert first_lease is not None and second_lease is not None
assert capacity_manager.ensure_capacity("block-manager-first", 5) is None
assert capacity_manager.get("block-manager-first").block_ids == first_lease.block_ids
capacity_manager.release("block-manager-second")
capacity_manager.release("block-manager-second")
extended_table = capacity_manager.ensure_capacity("block-manager-first", 5)
assert extended_table is not None and len(extended_table) == 2
capacity_manager.release("block-manager-first")
capacity_manager.release("block-manager-first")
assert capacity_manager.snapshot()["used_blocks"] == 0

scheduler_manager = BlockManager(block_size_tokens=4, capacity_tokens=16)
scheduler_lease = scheduler_manager.reserve(
    "scheduler-sequence", prompt_tokens=2, max_output_tokens=2
)
assert scheduler_lease is not None
scheduler = TokenScheduler(
    max_num_seqs=1,
    max_waiting_requests=0,
    max_num_scheduled_tokens=8,
    batch_wait_seconds=0,
)
scheduler.add_request(
    "scheduler-sequence",
    prompt_tokens=2,
    max_output_tokens=2,
    cache_block_ids=scheduler_lease.block_ids,
)
prefill_batch = scheduler.schedule_batch(scheduler_manager)
assert prefill_batch is not None
prefill_item = prefill_batch.work_items[0]
assert prefill_item.phase is ExecutionPhase.PREFILL
assert prefill_item.cache_block_ids == scheduler_lease.block_ids
scheduler.complete_batch(
    (WorkerResult("scheduler-sequence", token_id=17, generated_tokens=1, prompt_tokens_processed=2),)
)
decode_batch = scheduler.schedule_batch(scheduler_manager)
assert decode_batch is not None
decode_item = decode_batch.work_items[0]
assert decode_item.phase is ExecutionPhase.DECODE
assert decode_item.cache_block_ids == scheduler_lease.block_ids
scheduler.complete_batch(
    (WorkerResult("scheduler-sequence", token_id=23, generated_tokens=1, finished=True),)
)
scheduler.wait_idle()
scheduler_manager.release("scheduler-sequence")
assert scheduler.snapshot() == {"running": 0, "waiting": 0, "scheduled_rounds": 2}
print("Qwen3 Engine Sequence, BlockManager and Scheduler lifecycle passed.")

revision = str(inference_settings["model"]["revision"])
config = load_model_config(snapshot, revision)
dtype = getattr(torch, settings["component"].get("dtype", "bfloat16"))
with torch.device("meta"):
    engine_model = Qwen3ForCausalLM(config, dtype=dtype, device="meta")
    baseline_model = AutoModelForCausalLM.from_config(config)

print("==========engine===========")
print(f"Qwen3 本地 Runner 计算图（meta device，不加载权重）: {type(engine_model).__module__}.{type(engine_model).__name__}")
for name, module in engine_model.named_modules():
    if not name or name.count(".") <= 2:
        print(f"{name or '<root>'}: {type(module).__module__}.{type(module).__name__}")
print("==========baseline==========")
print(f"Transformers 原生模型（meta device，不加载权重）: {type(baseline_model).__module__}.{type(baseline_model).__name__}")
for name, module in baseline_model.named_modules():
    if not name or name.count(".") <= 2:
        print(f"{name or '<root>'}: {type(module).__module__}.{type(module).__name__}")

module_entrypoints = [
    ("inference.models.qwen3_0_6b.model.qwen3", inference_root / "models/qwen3_0_6b/model/qwen3.py"),
    ("inference.models.qwen3_0_6b.layers.attention", inference_root / "models/qwen3_0_6b/layers/attention.py"),
    ("inference.models.qwen3_0_6b.layers.layers", inference_root / "models/qwen3_0_6b/layers/layers.py"),
    ("inference.models.qwen3_0_6b.layers.rope", inference_root / "models/qwen3_0_6b/layers/rope.py"),
    ("inference.models.qwen3_0_6b.components.model_runner", inference_root / "models/qwen3_0_6b/components/model_runner.py"),
    ("inference.models.qwen3_0_6b.components.paged_kv_cache", inference_root / "models/qwen3_0_6b/components/paged_kv_cache.py"),
    ("inference.models.qwen3_0_6b.components.sampler", inference_root / "models/qwen3_0_6b/components/sampler.py"),
    ("inference.models.qwen3_0_6b.components.input_batch", inference_root / "models/qwen3_0_6b/components/input_batch.py"),
]
module_environment = os.environ.copy()
for module_name, entrypoint in module_entrypoints:
    print(f"==========module entrypoint: {entrypoint.relative_to(inference_root)}==========")
    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=repository_root,
        env=module_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if result.returncode:
        raise SystemExit(f"模块入口失败：{entrypoint}（exit={result.returncode}）")
print(f"Qwen3 module entrypoints passed: {len(module_entrypoints)}")

if args.profile_attention:
    cuda_available = torch.cuda.is_available()

    from inference.models.qwen3_0_6b.components.paged_kv_cache import PagedKVCache
    from inference.models.qwen3_0_6b.layers.attention import Qwen3Attention

    operator_settings = settings.get("operator_analysis", {})
    batch_size = int(operator_settings.get("batch_size", 1))
    prompt_tokens = int(operator_settings.get("prompt_tokens", 512))
    warmup_iterations = int(operator_settings.get("warmup_iterations", 2))
    iterations = int(operator_settings.get("iterations", 5))
    top_operators = int(operator_settings.get("top_operators", 20))
    if min(batch_size, prompt_tokens, iterations, top_operators) < 1 or warmup_iterations < 0:
        raise ValueError("operator_analysis 的 batch、token、iteration 与 operator 数必须有效。")

    device = torch.device(inference_settings["model"].get("device", "cuda:0")) if cuda_available else torch.device("cpu")
    attention = Qwen3Attention(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=float(config.rope_parameters["rope_theta"]),
        device=device,
        dtype=dtype,
    ).eval()

    checkpoint_path = snapshot / "model.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Qwen3 单档 Safetensors 权重不存在：{checkpoint_path}")
    attention_prefix = "model.layers.0.self_attn."
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        attention_weights = {
            name.removeprefix(attention_prefix): checkpoint.get_tensor(name)
            for name in checkpoint.keys()
            if name.startswith(attention_prefix)
        }
    expected_attention_weights = {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "q_norm.weight",
        "k_norm.weight",
    }
    if set(attention_weights) != expected_attention_weights:
        raise ValueError(f"第 0 层 Attention 权重不完整：{sorted(attention_weights)}")
    attention.load_state_dict(attention_weights, strict=True)
    attention = attention.to(device=device, dtype=dtype)

    block_size = int(inference_settings["engine"]["kv_cache"]["block_size_tokens"])
    blocks_per_request = (prompt_tokens + 1 + block_size - 1) // block_size
    request_ids = tuple(f"operator-analysis-{index}" for index in range(batch_size))
    block_tables = tuple(
        tuple(range(index * blocks_per_request, (index + 1) * blocks_per_request))
        for index in range(batch_size)
    )
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=batch_size * blocks_per_request,
        block_size_tokens=block_size,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        device=device,
        dtype=dtype,
    )
    prefill_hidden_states = torch.randn(
        batch_size, prompt_tokens, config.hidden_size, device=device, dtype=dtype
    )
    decode_hidden_states = torch.randn(batch_size, 1, config.hidden_size, device=device, dtype=dtype)
    token_lengths = tuple(prompt_tokens for _ in range(batch_size))
    position_offsets = tuple(0 for _ in range(batch_size))
    decode_lengths = tuple(1 for _ in range(batch_size))
    decode_offsets = tuple(prompt_tokens for _ in range(batch_size))
    prefill_samples_ms = []
    decode_samples_ms = []
    prefill_start = torch.cuda.Event(enable_timing=True) if cuda_available else None
    prefill_end = torch.cuda.Event(enable_timing=True) if cuda_available else None
    decode_start = torch.cuda.Event(enable_timing=True) if cuda_available else None
    decode_end = torch.cuda.Event(enable_timing=True) if cuda_available else None
    if cuda_available:
        torch.cuda.reset_peak_memory_stats(device)

    for iteration in range(warmup_iterations + iterations):
        for request_id in request_ids:
            cache.release(request_id)
        for request_id, block_table in zip(request_ids, block_tables, strict=True):
            cache.bind(request_id, block_table)
        prefill_cpu_start = time.perf_counter()
        if prefill_start is not None:
            prefill_start.record()
        with torch.inference_mode():
            attention.forward_batch(
                prefill_hidden_states,
                layer_index=0,
                request_ids=request_ids,
                token_lengths=token_lengths,
                position_offsets=position_offsets,
                block_tables=block_tables,
                cache=cache,
            )
        if prefill_end is not None:
            prefill_end.record()
            prefill_end.synchronize()
        if iteration >= warmup_iterations:
            prefill_samples_ms.append(prefill_start.elapsed_time(prefill_end) if cuda_available else (time.perf_counter() - prefill_cpu_start) * 1000)

    for iteration in range(warmup_iterations + iterations):
        for request_id in request_ids:
            cache.release(request_id)
        for request_id, block_table in zip(request_ids, block_tables, strict=True):
            cache.bind(request_id, block_table)
        with torch.inference_mode():
            attention.forward_batch(
                prefill_hidden_states,
                layer_index=0,
                request_ids=request_ids,
                token_lengths=token_lengths,
                position_offsets=position_offsets,
                block_tables=block_tables,
                cache=cache,
            )
        decode_cpu_start = time.perf_counter()
        if decode_start is not None:
            decode_start.record()
        with torch.inference_mode():
            attention.forward_batch(
                decode_hidden_states,
                layer_index=0,
                request_ids=request_ids,
                token_lengths=decode_lengths,
                position_offsets=decode_offsets,
                block_tables=block_tables,
                cache=cache,
            )
        if decode_end is not None:
            decode_end.record()
            decode_end.synchronize()
        if iteration >= warmup_iterations:
            decode_samples_ms.append(decode_start.elapsed_time(decode_end) if cuda_available else (time.perf_counter() - decode_cpu_start) * 1000)

    for request_id in request_ids:
        cache.release(request_id)
    for request_id, block_table in zip(request_ids, block_tables, strict=True):
        cache.bind(request_id, block_table)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if cuda_available else []),
        record_shapes=True,
        profile_memory=True,
    ) as prefill_profile:
        with torch.inference_mode(), torch.profiler.record_function("Qwen3Attention.prefill"):
            attention.forward_batch(
                prefill_hidden_states,
                layer_index=0,
                request_ids=request_ids,
                token_lengths=token_lengths,
                position_offsets=position_offsets,
                block_tables=block_tables,
                cache=cache,
            )
    for request_id in request_ids:
        cache.release(request_id)
    for request_id, block_table in zip(request_ids, block_tables, strict=True):
        cache.bind(request_id, block_table)
    with torch.inference_mode():
        attention.forward_batch(
            prefill_hidden_states,
            layer_index=0,
            request_ids=request_ids,
            token_lengths=token_lengths,
            position_offsets=position_offsets,
            block_tables=block_tables,
            cache=cache,
        )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if cuda_available else []),
        record_shapes=True,
        profile_memory=True,
    ) as decode_profile:
        with torch.inference_mode(), torch.profiler.record_function("Qwen3Attention.decode"):
            attention.forward_batch(
                decode_hidden_states,
                layer_index=0,
                request_ids=request_ids,
                token_lengths=decode_lengths,
                position_offsets=decode_offsets,
                block_tables=block_tables,
                cache=cache,
            )

    operator_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run": {
        "model_alias": settings["model_alias"],
        "revision": revision,
        "gpu_name": torch.cuda.get_device_name(device) if cuda_available else "CPU fallback",
        "execution_device": str(device),
        "execution_device": "cuda" if cuda_available else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": str(dtype),
        "batch_size": batch_size,
        "query_tokens": prompt_tokens,
        "kv_tokens": prompt_tokens + 1,
        "block_size_tokens": block_size,
        "cache_allocated_bytes": cache.bytes_allocated,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": iterations,
        "layer_index": 0,
        "prefill_cuda_event_ms": {
            "samples": prefill_samples_ms,
            "median": statistics.median(prefill_samples_ms),
            "mean": statistics.fmean(prefill_samples_ms),
        },
        "decode_cuda_event_ms": {
            "samples": decode_samples_ms,
            "median": statistics.median(decode_samples_ms),
            "mean": statistics.fmean(decode_samples_ms),
        },
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if cuda_available else 0,
        },
        "operator_records": [],
    }
    for phase, profile in (("prefill", prefill_profile), ("decode", decode_profile)):
        phase_records = []
        for event in profile.key_averages(group_by_input_shape=True):
            cuda_time = getattr(event, "self_device_time_total", None) if cuda_available else 0.0
            if cuda_available and cuda_time is None:
                cuda_time = getattr(event, "self_cuda_time_total", 0.0)
            cuda_memory = getattr(event, "self_device_memory_usage", None) if cuda_available else 0
            if cuda_available and cuda_memory is None:
                cuda_memory = getattr(event, "self_cuda_memory_usage", 0)
            phase_records.append({
                "phase": phase,
                "operator": str(event.key),
                "input_shapes": event.input_shapes,
                "call_count": int(event.count),
                "self_cuda_time_total_us": float(cuda_time),
                "self_cpu_time_total_us": float(event.self_cpu_time_total),
                "self_cuda_memory_usage_bytes": int(cuda_memory),
                "self_cpu_memory_usage_bytes": int(event.self_cpu_memory_usage),
            })
        phase_records.sort(key=lambda record: record["self_cuda_time_total_us"], reverse=True)
        operator_report["operator_records"].extend(phase_records[:top_operators])

    print("==========Qwen3 Attention operator analysis==========")
    print(json.dumps(operator_report, ensure_ascii=False, indent=2, default=str))
    if args.operator_report:
        report_path = args.operator_report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(operator_report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"operator report: {report_path}")
    print("==========prefill top operators (" + ("CUDA" if cuda_available else "CPU") + " self time)==========")
    print(prefill_profile.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_time_total" if cuda_available else "self_cpu_time_total", row_limit=top_operators
    ))
    print("==========decode top operators (" + ("CUDA" if cuda_available else "CPU") + " self time)==========")
    print(decode_profile.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_time_total" if cuda_available else "self_cpu_time_total", row_limit=top_operators
    ))
