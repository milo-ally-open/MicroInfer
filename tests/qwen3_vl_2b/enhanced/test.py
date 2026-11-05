"""Inference 实现 Test：执行完整生成并与 Transformers token 逐个核对。"""

import json
import os
import sys
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
max_new_tokens = 2
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
baseline_path = root / "artifacts/qwen3_vl_2b/baseline/test.json"

print("================= enhanced =================")
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["result"]["generated_token_ids"]
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
    block_count = (prompt_tokens + max_new_tokens + block_size - 1) // block_size
    request_id = "qwen3-vl-enhanced-test"
    options = runtime.model_adapter.generation_options({"max_tokens": max_new_tokens, "temperature": 0})
    runner.begin_generation(request_id, inputs, options)
    runner.bind_generation_cache(request_id, tuple(range(block_count)))
    with torch.inference_mode():
        runner.execute_batch((ScheduledWorkItem(request_id, ExecutionPhase.PREFILL, prompt_tokens),))
        runner.execute_batch((ScheduledWorkItem(request_id, ExecutionPhase.DECODE, 1),))
    generated, finish_reason = runner.finish_generation(request_id)
    generated_ids = generated.detach().cpu().reshape(-1).tolist()
    match = generated_ids == baseline
    text = runner.decode(generated.detach().cpu()).strip()
    report = {
        "schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"name": "qwen3_vl_2b", "alias": alias, "repo_id": "Qwen/Qwen3-VL-2B-Instruct", "revision": revision},
        "implementation": "enhanced", "task": "test",
        "environment": {"device": str(device), "gpu_name": torch.cuda.get_device_name(0) if cuda else "CPU", "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "dtype": str(runner.model.dtype)},
        "workload": {"input_type": "image", "image_width": image.width, "image_height": image.height, "prompt": prompt, "prompt_tokens": prompt_tokens, "max_new_tokens": max_new_tokens},
        "result": {"status": "passed" if match else "failed", "generated_token_ids": generated_ids, "generated_text": text, "baseline_token_match": match, "finish_reason": finish_reason},
        "metrics": {}, "operator_records": [], "trace_path": None,
    }
    output_path = root / "artifacts/qwen3_vl_2b/enhanced/test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("status:", report["result"]["status"], "baseline_token_match:", match)
    print("json:", output_path)
    if not match:
        raise AssertionError(f"生成结果与 baseline token 不一致，完整记录见 {output_path}")
finally:
    runtime.close()
