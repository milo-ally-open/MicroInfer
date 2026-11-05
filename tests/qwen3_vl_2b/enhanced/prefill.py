"""Inference 实现 Prefill：只采集完整图文 Prompt 的前向算子数据。"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "src"))
config_path = root / "tests/qwen3_vl_2b/enhanced/profiles/inference.yml"
model_config_path = root / "src/inference/models/qwen3_vl_2b/profiles/model.yml"
network_config_path = root / "src/inference/models/qwen3_vl_2b/profiles/network.yml"
os.environ.update({"INFERENCE_PROJECT_ROOT": str(root), "INFERENCE_MODEL_ALIAS": "qwen3-vl-2b", "INFERENCE_MODEL_MANIFEST": str(model_config_path), "INFERENCE_CONFIG": str(config_path), "INFERENCE_NETWORK_CONFIG": str(network_config_path), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor

from inference.engine.sequence import ExecutionPhase, ScheduledWorkItem
from inference.models.qwen3_vl_2b.runtime import create_runtime

config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
model_path = (root / config["runtime"]["local_model_path"]).resolve()
revision = str(config["model"]["revision"])
alias = "qwen3-vl-2b"
image_path = root / "test.jpg"
prompt = "请简要描述图片中的主要内容。"
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
activities = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if cuda else [])
max_operators = 30

print("================= enhanced =================")
processor = AutoProcessor.from_pretrained(model_path, revision=revision, local_files_only=True)
image = Image.open(image_path).convert("RGB")
messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
cpu_inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
cpu_inputs = {key: value for key, value in cpu_inputs.items() if key != "token_type_ids"}
cpu_inputs.setdefault("attention_mask", torch.ones_like(cpu_inputs["input_ids"]))
runtime = create_runtime()
try:
    runtime.load()
    runner = runtime.model_runner
    inputs = {key: value.to(device=runner.input_device, dtype=runner.model.dtype) if key in {"pixel_values", "pixel_values_videos"} and value.is_floating_point() else value.to(runner.input_device) if isinstance(value, torch.Tensor) else value for key, value in cpu_inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    block_size = int(config["engine"]["kv_cache"]["block_size_tokens"])
    block_count = (prompt_tokens + 2 + block_size - 1) // block_size
    request_id = "qwen3-vl-enhanced-prefill"
    options = runtime.model_adapter.generation_options({"max_tokens": 2, "temperature": 0})
    runner.begin_generation(request_id, inputs, options)
    runner.bind_generation_cache(request_id, tuple(range(block_count)))
    runner.reset_peak_memory()
    trace_path = root / "artifacts/qwen3_vl_2b/traces/enhanced-prefill.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    start_event = torch.cuda.Event(enable_timing=True) if cuda else None
    end_event = torch.cuda.Event(enable_timing=True) if cuda else None
    started = time.perf_counter()
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True) as profile:
        if start_event:
            start_event.record()
        with torch.inference_mode():
            prefill_result = runner.execute_batch((ScheduledWorkItem(request_id, ExecutionPhase.PREFILL, prompt_tokens),))[0]
        if end_event:
            end_event.record()
            end_event.synchronize()
    prefill_ms = float(start_event.elapsed_time(end_event)) if cuda else (time.perf_counter() - started) * 1000
    profile.export_chrome_trace(str(trace_path))
    # 单独执行一次 Decode 以释放请求 Cache；Decode 不写入本脚本的 Profile 数据。
    with torch.inference_mode():
        runner.execute_batch((ScheduledWorkItem(request_id, ExecutionPhase.DECODE, 1),))
    generated, _ = runner.finish_generation(request_id)
    operators = [{
        "phase": "prefill", "operator": str(event.key), "input_shapes": event.input_shapes, "call_count": int(event.count),
        "self_device_time_total_us": float(getattr(event, "self_device_time_total", 0) or 0), "self_cpu_time_total_us": float(event.self_cpu_time_total),
        "self_device_memory_usage_bytes": int(getattr(event, "self_device_memory_usage", 0) or 0), "self_cpu_memory_usage_bytes": int(event.self_cpu_memory_usage),
    } for event in profile.key_averages(group_by_input_shape=True)]
    operators.sort(key=lambda row: row["self_device_time_total_us"], reverse=True)
    print(profile.key_averages(group_by_input_shape=True).table(sort_by="self_device_time_total" if cuda else "self_cpu_time_total", row_limit=max_operators))
    report = {
        "schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"name": "qwen3_vl_2b", "alias": alias, "repo_id": "Qwen/Qwen3-VL-2B-Instruct", "revision": revision},
        "implementation": "enhanced", "task": "prefill",
        "environment": {"device": str(device), "gpu_name": torch.cuda.get_device_name(0) if cuda else "CPU", "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "dtype": str(runner.model.dtype)},
        "workload": {"input_type": "image", "image_width": image.width, "image_height": image.height, "prompt": prompt, "prompt_tokens": prompt_tokens},
        "result": {"status": "passed", "query_tokens": prompt_tokens, "decode_query_tokens": 1, "generation_tokens_after_cleanup": int(generated.numel()), "prefill_result_type": type(prefill_result).__name__},
        "metrics": {"prefill_latency_ms": prefill_ms, "peak_cuda_memory_bytes": runner.peak_memory()["peak_allocated"]},
        "operator_records": operators, "trace_path": str(trace_path),
    }
    output_path = root / "artifacts/qwen3_vl_2b/enhanced/prefill.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print("prefill_ms:", round(prefill_ms, 3), "json:", output_path)
finally:
    runtime.close()
