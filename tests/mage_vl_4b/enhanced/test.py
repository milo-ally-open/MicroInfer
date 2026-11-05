"""Inference 实现 Test：验证 Mage-VL Runtime 图片生成并对齐 Transformers。"""

import json
import os
import sys
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
baseline_path = root / "artifacts/mage_vl_4b/baseline/test.json"

print("================= enhanced =================")
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["result"]["generated_token_ids"]
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
    request_id = "mage-vl-enhanced-test"
    options = {"max_new_tokens": max_new_tokens, "do_sample": False, "temperature": 0.0, "use_cache": True, "eos_token_id": ()}
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
        "model": {"name": "mage_vl_4b", "alias": alias, "repo_id": str(model_settings["id"]), "revision": revision},
        "implementation": "enhanced", "task": "test",
        "environment": {"device": "cuda" if cuda else "cpu", "gpu_name": torch.cuda.get_device_name(0) if cuda else "CPU", "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "dtype": str(runner.model.dtype)},
        "workload": {"input_type": "image", "image_width": image.width, "image_height": image.height, "prompt": prompt, "prompt_tokens": prompt_tokens, "max_new_tokens": max_new_tokens},
        "result": {"status": "passed" if match else "failed", "generated_token_ids": generated_ids, "generated_text": text, "baseline_token_match": match, "finish_reason": finish_reason},
        "metrics": {}, "operator_records": [], "trace_path": None,
    }
    output_path = root / "artifacts/mage_vl_4b/enhanced/test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print("status:", report["result"]["status"], "baseline_token_match:", match, "json:", output_path)
    if not match: raise AssertionError(f"Inference 输出与 Transformers 不一致，记录见 {output_path}")
finally:
    runtime.close()
