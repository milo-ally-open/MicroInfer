#!/usr/bin/env python3
"""Test pinned Qwen3-VL-2B Transformers loading, image preprocessing and generation."""

from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from inference.engine.sequence import ExecutionPhase
from inference.models.qwen3_vl_2b.components.stepwise_generation import Qwen3VLStepwiseGeneration

TEST_CONFIG = Path(__file__).resolve().with_name("test.yml")
settings = yaml.safe_load(TEST_CONFIG.read_text(encoding="utf-8"))
snapshot = (TEST_CONFIG.parent / settings["snapshot"]).resolve()
image_path = (TEST_CONFIG.parent / settings["image"]).resolve()
revision = str(settings["revision"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = getattr(torch, settings["dtype"])

assert (snapshot / "config.json").is_file(), f"Model config missing: {snapshot}"
assert (snapshot / "model.safetensors").is_file(), f"Model weights missing: {snapshot}"
assert image_path.is_file(), f"Image fixture missing: {image_path}"
processor = AutoProcessor.from_pretrained(snapshot, revision=revision, local_files_only=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    snapshot,
    revision=revision,
    local_files_only=True,
    use_safetensors=True,
    dtype=dtype,
    device_map=0 if device.type == "cuda" else "cpu",
    attn_implementation=settings["attention"],
).eval()

# 验证固定 checkpoint 确实是 Qwen3-VL，并且同时含 Vision/Text 子配置。
assert model.config.model_type == "qwen3_vl"
assert model.config.vision_config.depth > 0
assert model.config.text_config.num_hidden_layers > 0
assert model.config.text_config.num_key_value_heads > 0

# 用仓库固定图片经过官方 Processor 和 Chat Template，确认文字、图像张量与网格信息齐全。
image = Image.open(image_path).convert("RGB")
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": settings["prompt"]},
    ],
}]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
assert inputs["input_ids"].ndim == 2 and inputs["input_ids"].shape[0] == 1
patch_size = int(processor.image_processor.patch_size)
temporal_patch_size = int(processor.image_processor.temporal_patch_size)
assert inputs["pixel_values"].ndim == 2
assert inputs["pixel_values"].shape[-1] == 3 * patch_size * patch_size * temporal_patch_size
assert "image_grid_thw" in inputs and inputs["image_grid_thw"].numel() == 3

# 执行真实图文 Prefill，检查 logits 和 HF DynamicCache，再执行一次官方 generate。
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
with torch.inference_mode():
    prefill = model(**model_inputs, use_cache=True, return_dict=True)
assert prefill.logits.ndim == 3
assert prefill.logits.shape[0] == 1 and prefill.logits.shape[-1] == model.config.text_config.vocab_size
assert prefill.past_key_values is not None
assert torch.isfinite(prefill.logits[:, -1, :]).all()
image_input_tokens = int(model_inputs["input_ids"].shape[-1])
pixel_values_shape = list(model_inputs["pixel_values"].shape)
image_grid_thw = model_inputs["image_grid_thw"].detach().cpu().tolist()
image_logits_shape = list(prefill.logits.shape)
with torch.inference_mode():
    generated_ids = model.generate(**model_inputs, max_new_tokens=4, do_sample=False)
reference_generated_ids = generated_ids[0, model_inputs["input_ids"].shape[-1]:].tolist()
answer = processor.batch_decode(
    [generated_ids[0, model_inputs["input_ids"].shape[-1]:]],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]
assert answer.strip(), "Qwen3-VL generated an empty answer for the image fixture."

# 用新模型包的逐步执行器对齐 HF generate，覆盖真实图像 Prefill、MRoPE 和单 token Cache Decode。
stepwise = Qwen3VLStepwiseGeneration(block_size_tokens=16)
stepwise.bind(model, processor.tokenizer)
stepwise_request_id = "qwen3-vl-stepwise-image-test"
stepwise_options = {
    "max_new_tokens": 2,
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "use_cache": True,
}
stepwise.begin(stepwise_request_id, model_inputs, stepwise_options)
required_blocks = (int(model_inputs["input_ids"].shape[-1]) + 2 + 15) // 16
stepwise.bind_cache(stepwise_request_id, tuple(range(required_blocks)))
stepwise.execute(stepwise_request_id, ExecutionPhase.PREFILL.value, int(model_inputs["input_ids"].shape[-1]))
stepwise.execute(stepwise_request_id, ExecutionPhase.DECODE.value, 1)
stepwise_ids, stepwise_finish = stepwise.finish(stepwise_request_id)
assert stepwise_ids.tolist() == reference_generated_ids[:2]
assert stepwise_finish == "length"

# 同一模型也必须接收纯文本对话；移除图像前向 cache 后再测，降低 8 GB GPU 峰值。
del prefill, generated_ids, inputs, model_inputs, image
text_inputs = processor.apply_chat_template(
    [{"role": "user", "content": "用一句话介绍推理服务。"}],
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
text_inputs = {
    name: value.to(model.device) if isinstance(value, torch.Tensor) else value
    for name, value in text_inputs.items()
}
text_inputs.pop("token_type_ids", None)
with torch.inference_mode():
    text_output_ids = model.generate(**text_inputs, max_new_tokens=3, do_sample=False)
assert text_output_ids.shape[-1] > text_inputs["input_ids"].shape[-1]
print({
    "status": "passed",
    "device": str(model.device),
    "dtype": str(model.dtype),
    "image_input_tokens": image_input_tokens,
    "pixel_values_shape": pixel_values_shape,
    "image_grid_thw": image_grid_thw,
    "image_logits_shape": image_logits_shape,
    "text_only_generated_tokens": int(text_output_ids.shape[-1] - text_inputs["input_ids"].shape[-1]),
    "image_answer": answer,
})
