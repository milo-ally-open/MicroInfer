"""Transformers 基线 Prefill：记录整段图片 Prompt 的前向算子。"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

root = Path(__file__).resolve().parents[3]
config_path = root / "tests/qwen3_vl_2b/baseline/profiles/inference.yml"
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
model_path = (root / config["runtime"]["local_model_path"]).resolve()
revision = str(config["model"]["revision"])
alias = "qwen3-vl-2b"
image_path = root / "test.jpg"
prompt = "请简要描述图片中的主要内容。"
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
dtype = getattr(torch, str(config["model"]["dtype"]))
activities = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if cuda else [])

print("================= baseline =================")
processor = AutoProcessor.from_pretrained(model_path, revision=revision, local_files_only=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path, revision=revision, local_files_only=True, use_safetensors=True,
    dtype=dtype, device_map=0 if cuda else "cpu", attn_implementation=str(config["model"]["attn_implementation"]),
).eval()
image = Image.open(image_path).convert("RGB")
messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {key: value.to(model.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items() if key != "token_type_ids"}
inputs.setdefault("attention_mask", torch.ones_like(inputs["input_ids"]))
if cuda:
    torch.cuda.reset_peak_memory_stats(device)
trace_path = root / "artifacts/qwen3_vl_2b/traces/baseline-prefill.json"
trace_path.parent.mkdir(parents=True, exist_ok=True)
start_event = torch.cuda.Event(enable_timing=True) if cuda else None
end_event = torch.cuda.Event(enable_timing=True) if cuda else None
started = time.perf_counter()
with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True) as profile:
    if start_event:
        start_event.record()
    with torch.inference_mode():
        output = model(**inputs, use_cache=True, return_dict=True)
    if end_event:
        end_event.record()
        end_event.synchronize()
prefill_ms = float(start_event.elapsed_time(end_event)) if cuda else (time.perf_counter() - started) * 1000
if output.past_key_values is None:
    raise RuntimeError("Prefill 没有返回 KV Cache。")
cache_tokens = int(output.past_key_values.get_seq_length())
prompt_tokens = int(inputs["input_ids"].shape[-1])
if cache_tokens != prompt_tokens:
    raise AssertionError(f"Prefill Cache 长度 {cache_tokens} 与输入长度 {prompt_tokens} 不一致。")
profile.export_chrome_trace(str(trace_path))
operators = [{
    "phase": "prefill", "operator": str(event.key), "input_shapes": event.input_shapes,
    "call_count": int(event.count), "self_device_time_total_us": float(getattr(event, "self_device_time_total", 0) or 0),
    "self_cpu_time_total_us": float(event.self_cpu_time_total),
    "self_device_memory_usage_bytes": int(getattr(event, "self_device_memory_usage", 0) or 0),
    "self_cpu_memory_usage_bytes": int(event.self_cpu_memory_usage),
} for event in profile.key_averages(group_by_input_shape=True)]
operators.sort(key=lambda row: row["self_device_time_total_us"], reverse=True)
print(profile.key_averages(group_by_input_shape=True).table(sort_by="self_device_time_total" if cuda else "self_cpu_time_total", row_limit=30))
report = {
    "schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "model": {"name": "qwen3_vl_2b", "alias": alias, "repo_id": "Qwen/Qwen3-VL-2B-Instruct", "revision": revision},
    "implementation": "baseline", "task": "prefill",
    "environment": {"device": str(device), "gpu_name": torch.cuda.get_device_name(0) if cuda else "CPU", "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "dtype": str(model.dtype)},
    "workload": {"input_type": "image", "image_width": image.width, "image_height": image.height, "prompt": prompt, "prompt_tokens": prompt_tokens},
    "result": {"status": "passed", "query_tokens": prompt_tokens, "cache_tokens_after_prefill": cache_tokens, "logits_shape": list(output.logits.shape)},
    "metrics": {"prefill_latency_ms": prefill_ms, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if cuda else 0},
    "operator_records": operators, "trace_path": str(trace_path),
}
output_path = root / "artifacts/qwen3_vl_2b/baseline/prefill.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
print("prefill_ms:", round(prefill_ms, 3), "json:", output_path)
