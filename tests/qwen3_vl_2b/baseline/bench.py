"""Transformers 基线 Bench：测量固定图文请求的延迟与吞吐。"""

import json
import statistics
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
max_new_tokens, warmups, iterations = 2, 1, 3
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
dtype = getattr(torch, str(config["model"]["dtype"]))

print("================= baseline =================")
reference = json.loads((root / "artifacts/qwen3_vl_2b/baseline/test.json").read_text(encoding="utf-8"))["result"]["generated_token_ids"]
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
latencies_ms, token_counts, token_matches = [], [], []
for index in range(warmups + iterations):
    if cuda:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    if cuda:
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - started) * 1000
    token_ids = output[:, inputs["input_ids"].shape[-1]:].detach().cpu().reshape(-1).tolist()
    token_matches.append(token_ids == reference)
    if index >= warmups:
        latencies_ms.append(elapsed)
        token_counts.append(len(token_ids))
metrics = {
    "warmup_iterations": warmups, "measured_iterations": iterations,
    "latency_ms_samples": latencies_ms, "latency_ms_median": statistics.median(latencies_ms),
    "tokens_per_second_median": statistics.median(count / (ms / 1000) for count, ms in zip(token_counts, latencies_ms)),
    "baseline_test_token_match_all_runs": all(token_matches),
    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if cuda else 0,
    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else 0,
}
report = {
    "schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "model": {"name": "qwen3_vl_2b", "alias": alias, "repo_id": "Qwen/Qwen3-VL-2B-Instruct", "revision": revision},
    "implementation": "baseline", "task": "bench",
    "environment": {"device": str(device), "gpu_name": torch.cuda.get_device_name(0) if cuda else "CPU", "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "dtype": str(model.dtype)},
    "workload": {"input_type": "image", "prompt_tokens": int(inputs["input_ids"].shape[-1]), "max_new_tokens": max_new_tokens},
    "result": {"status": "passed" if all(token_matches) else "failed", "token_match_all_runs": all(token_matches)}, "metrics": metrics, "operator_records": [], "trace_path": None,
}
output_path = root / "artifacts/qwen3_vl_2b/baseline/bench.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"latency_median_ms: {metrics['latency_ms_median']:.3f}")
print(f"throughput_median_tokens_s: {metrics['tokens_per_second_median']:.3f}")
print("json:", output_path)
if not all(token_matches):
    raise AssertionError(f"Bench 输出与 baseline Test token 不一致，记录见 {output_path}")
