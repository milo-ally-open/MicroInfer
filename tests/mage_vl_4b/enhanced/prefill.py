"""Inference 实现 Prefill：只采集完整图片 Prompt 的前向算子数据。"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "src"))
config_path = root / "tests/mage_vl_4b/enhanced/profiles/inference.yml"
model_config_path = root / "src/inference/models/mage_vl_4b/profiles/model.yml"
network_config_path = root / "src/inference/models/mage_vl_4b/profiles/network.yml"
os.environ.update({"INFERENCE_PROJECT_ROOT": str(root), "INFERENCE_MODEL_ALIAS": "mage-vl-4b", "INFERENCE_MODEL_MANIFEST": str(model_config_path), "INFERENCE_CONFIG": str(config_path), "INFERENCE_NETWORK_CONFIG": str(network_config_path), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor

from inference.engine.sequence import ExecutionPhase, ScheduledWorkItem
from inference.models.mage_vl_4b.architecture.image_encoder import _resize_image
from inference.models.mage_vl_4b.runtime import create_runtime

config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
model_settings = config["model"]
model_path = Path(os.path.expandvars(os.path.expanduser(config["runtime"]["local_model_path"])))
model_path = (root / model_path).resolve() if not model_path.is_absolute() else model_path.resolve()
revision, alias = str(model_settings["revision"]), "mage-vl-4b"
prompt, max_new_tokens = "请简要描述图片中的主要内容。", 2
image_path = root / "test.jpg"
cuda = torch.cuda.is_available()
profile_mode = os.environ.get("INFERENCE_PROFILE_MODE", "torch").strip().lower()
if profile_mode not in {"torch", "external"}: raise ValueError("INFERENCE_PROFILE_MODE 只支持 torch 或 external。")
activities = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if cuda and profile_mode == "torch" else [])
trace_suffix = "-external" if profile_mode == "external" else ""

print("================= enhanced =================")
processor_kwargs = {"trust_remote_code": True, "revision": revision, "local_files_only": True}
if config.get("processor", {}).get("use_fast") is not None: processor_kwargs["use_fast"] = bool(config["processor"]["use_fast"])
processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
with Image.open(image_path) as opened:
    if opened.width * opened.height > int(config["api"]["max_image_pixels"]): raise ValueError("图片像素超过推理配置上限。")
    image = opened.convert("RGB")
image = _resize_image(image, int(config["api"]["max_image_side"]))
prompt_text = processor.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}], tokenize=False, add_generation_prompt=True)
encoded = processor(text=[prompt_text], images=[image], return_tensors="pt")
encoded = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in encoded.items()}
encoded.setdefault("attention_mask", torch.ones_like(encoded["input_ids"]))
runtime = create_runtime()
try:
    runtime.load()
    runner = runtime.model_runner
    input_device = runner.model.get_input_embeddings().weight.device
    inputs = {key: value.to(device=input_device, dtype=runner.model.dtype) if key in {"pixel_values", "pixel_values_videos"} and isinstance(value, torch.Tensor) and value.is_floating_point() else value.to(input_device) if isinstance(value, torch.Tensor) else value for key, value in encoded.items()}
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    block_size = int(config["engine"]["kv_cache"]["block_size_tokens"])
    block_count = (prompt_tokens + max_new_tokens + block_size - 1) // block_size
    request_id = "mage-vl-enhanced-prefill"
    options = {"max_new_tokens": max_new_tokens, "do_sample": False, "temperature": 0.0, "use_cache": True, "eos_token_id": ()}
    runner.begin_generation(request_id, inputs, options)
    runner.bind_generation_cache(request_id, tuple(range(block_count)))
    runner.reset_peak_memory()
    trace_path = root / f"artifacts/mage_vl_4b/traces/enhanced-prefill{trace_suffix}.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    start_event = torch.cuda.Event(enable_timing=True) if cuda else None
    end_event = torch.cuda.Event(enable_timing=True) if cuda else None
    started = time.perf_counter()
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True) as profile:
        if start_event: start_event.record()
        with torch.inference_mode(): prefill_result = runner.execute_batch((ScheduledWorkItem(request_id, ExecutionPhase.PREFILL, prompt_tokens),))[0]
        if end_event:
            end_event.record()
            end_event.synchronize()
    prefill_ms = float(start_event.elapsed_time(end_event)) if cuda else (time.perf_counter() - started) * 1000
    profile.export_chrome_trace(str(trace_path))
    # 补做一次 Decode 以释放请求 Cache；报告内只包含 Prefill 的算子。
    with torch.inference_mode(): runner.execute_batch((ScheduledWorkItem(request_id, ExecutionPhase.DECODE, 1),))
    generated, _ = runner.finish_generation(request_id)
    operators = [{"phase": "prefill", "operator": str(event.key), "input_shapes": event.input_shapes, "call_count": int(event.count), "self_device_time_total_us": float(getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0)) or 0), "self_cpu_time_total_us": float(event.self_cpu_time_total), "self_device_memory_usage_bytes": int(getattr(event, "self_device_memory_usage", getattr(event, "self_cuda_memory_usage", 0)) or 0), "self_cpu_memory_usage_bytes": int(event.self_cpu_memory_usage)} for event in profile.key_averages(group_by_input_shape=True)]
    operators.sort(key=lambda row: row["self_device_time_total_us"], reverse=True)
    print(profile.key_averages(group_by_input_shape=True).table(sort_by="self_device_time_total" if cuda else "self_cpu_time_total", row_limit=20))
    report = {
        "schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"name": "mage_vl_4b", "alias": alias, "repo_id": str(model_settings["id"]), "revision": revision},
        "implementation": "enhanced", "task": "prefill",
        "environment": {"device": "cuda" if cuda else "cpu", "gpu_name": ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())) if cuda else "CPU", "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "dtype": str(runner.model.dtype), "profile_mode": profile_mode},
        "workload": {"input_type": "image", "image_width": image.width, "image_height": image.height, "prompt": prompt, "prompt_tokens": prompt_tokens},
        "result": {"status": "passed", "query_tokens": prompt_tokens, "decode_query_tokens": 1, "generation_tokens_after_cleanup": int(generated.numel()), "prefill_result_type": type(prefill_result).__name__},
        "metrics": {"prefill_latency_ms": prefill_ms, "peak_cuda_memory_bytes": runner.peak_memory()["peak_allocated"], "peak_cuda_reserved_bytes": runner.peak_memory()["peak_reserved"]},
        "operator_records": operators, "trace_path": str(trace_path),
    }
    output_path = root / "artifacts/mage_vl_4b/enhanced/prefill.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print("prefill_ms:", round(prefill_ms, 3), "operator_records:", len(operators), "json:", output_path)
finally:
    runtime.close()
