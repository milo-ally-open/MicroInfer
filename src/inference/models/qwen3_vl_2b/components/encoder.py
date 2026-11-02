"""校验文本/图片消息并创建 Qwen3-VL 模型张量与采样参数。"""

from __future__ import annotations

from typing import Any

import torch

if __name__ == "__main__":
    import os
    from pathlib import Path

    _model_dir = Path(__file__).resolve().parents[1]
    _repo_root = _model_dir.parents[3]
    os.environ.setdefault("INFERENCE_PROJECT_ROOT", str(_repo_root))
    os.environ.setdefault("INFERENCE_MODEL_ALIAS", "qwen3-vl-2b")
    os.environ.setdefault("INFERENCE_MODEL_MANIFEST", str(_model_dir / "profiles/model.yml"))
    os.environ.setdefault("INFERENCE_CONFIG", str(_model_dir / "profiles/inference.yml"))
    os.environ.setdefault("INFERENCE_NETWORK_CONFIG", str(_model_dir / "profiles/network.yml"))

from inference.engine.sequence import InferenceError
from ..architecture.image_encoder import load_image
from .tokenizer import Qwen3VLTokenizer
from .. import loader as config


class Qwen3VLEncoder:
    def __init__(self, model_runner: Any) -> None:
        self.model_runner = model_runner

    def encode_chat(self, request_body: dict[str, Any]) -> tuple[dict[str, Any], int, dict[str, Any]]:
        messages, image_count = self._normalize_messages(request_body.get("messages"))
        tokenizer = self.model_runner.tokenizer
        if tokenizer is None or self.model_runner.model is None:
            raise RuntimeError("Qwen3-VL model and Processor must be loaded before request encoding.")
        try:
            model_inputs = tokenizer.tokenize_chat(messages)
        except (TypeError, ValueError) as exc:
            raise InferenceError(400, f"Could not encode Qwen3-VL chat input: {exc}") from exc
        prompt_tokens = int(model_inputs["input_ids"].shape[-1])
        options = self.generation_options(request_body)
        if prompt_tokens + options["max_new_tokens"] > config.MAX_CONTEXT_TOKENS:
            raise InferenceError(413, f"Prompt plus requested output exceeds {config.MAX_CONTEXT_TOKENS} tokens.")
        if prompt_tokens > config.KV_CACHE_CAPACITY_TOKENS:
            raise InferenceError(413, "Prompt exceeds the configured Engine KV capacity.")
        model = self.model_runner.model
        input_device = model.get_input_embeddings().weight.device
        model_dtype = model.dtype
        prepared: dict[str, Any] = {}
        for name, value in model_inputs.items():
            if isinstance(value, torch.Tensor):
                prepared[name] = (
                    value.to(device=input_device, dtype=model_dtype)
                    if name in {"pixel_values", "pixel_values_videos"} and value.is_floating_point()
                    else value.to(input_device)
                )
            else:
                prepared[name] = value
        prepared.pop("token_type_ids", None)
        prepared.setdefault("attention_mask", torch.ones_like(prepared["input_ids"]))
        return prepared, prompt_tokens, options

    @staticmethod
    def generation_options(request_body: dict[str, Any]) -> dict[str, Any]:
        settings = config.GENERATION_CONFIG
        max_tokens = int(request_body.get("max_tokens", request_body.get("max_completion_tokens", settings["max_tokens_default"])))
        if not 1 <= max_tokens <= config.MAX_NEW_TOKENS:
            raise InferenceError(400, f"max_tokens must be between 1 and {config.MAX_NEW_TOKENS}.")
        temperature = float(request_body.get("temperature", settings["temperature"]))
        top_p = float(request_body.get("top_p", settings["top_p"]))
        top_k = int(request_body.get("top_k", settings["top_k"]))
        penalty = float(request_body.get("repetition_penalty", settings["repetition_penalty"]))
        if not 0 <= temperature <= 2 or not 0 < top_p <= 1 or top_k < 0 or penalty <= 0:
            raise InferenceError(400, "Invalid temperature, top_p, top_k, or repetition_penalty.")
        if int(request_body.get("num_beams", settings["num_beams"])) != 1:
            raise InferenceError(400, "Qwen3-VL Engine currently supports num_beams=1 only.")
        if not bool(settings.get("use_cache", True)):
            raise RuntimeError("Qwen3-VL stepwise decoding requires generation.use_cache=true.")
        if settings.get("advanced_kwargs"):
            raise RuntimeError("generation.advanced_kwargs is not supported by the Qwen3-VL stepwise runner.")
        return {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": penalty,
            "no_repeat_ngram_size": int(request_body.get("no_repeat_ngram_size", settings["no_repeat_ngram_size"])),
        }

    @staticmethod
    def _normalize_messages(raw_messages: Any) -> tuple[list[dict[str, Any]], int]:
        if not isinstance(raw_messages, list) or not raw_messages:
            raise InferenceError(400, "messages must be a non-empty array.")
        messages = []
        image_count = 0
        for message in raw_messages:
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
                raise InferenceError(400, "Each message needs a supported role and content.")
            content = message.get("content")
            if isinstance(content, str):
                messages.append({"role": message["role"], "content": content})
                continue
            if not isinstance(content, list):
                raise InferenceError(400, "Message content must be text or an array of text/image parts.")
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    raise InferenceError(400, "Each content part must be an object.")
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url":
                    image_url = part.get("image_url")
                    source = image_url.get("url") if isinstance(image_url, dict) else None
                    if not isinstance(source, str):
                        raise InferenceError(400, "image_url.url must be a string.")
                    parts.append({"type": "image", "image": load_image(source)})
                    image_count += 1
                    if image_count > config.MAX_IMAGES_PER_REQUEST:
                        raise InferenceError(413, f"At most {config.MAX_IMAGES_PER_REQUEST} images are allowed.")
                else:
                    raise InferenceError(400, "Only text and image_url content parts are supported.")
            messages.append({"role": message["role"], "content": parts})
        return messages, image_count


if __name__ == "__main__":
    import time
    from types import SimpleNamespace

    class FakeTokenizer:
        def tokenize_chat(self, messages):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }

    fake_model = SimpleNamespace(
        dtype=torch.float32,
        get_input_embeddings=lambda: SimpleNamespace(weight=torch.empty(1, device="cpu")),
    )
    encoder = Qwen3VLEncoder(SimpleNamespace(model=fake_model, tokenizer=FakeTokenizer()))
    request = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 2}
    encoded, token_count, options = encoder.encode_chat(request)
    assert token_count == 3
    assert encoded["input_ids"].shape == (1, 3)
    assert options["max_new_tokens"] == 2
    try:
        encoder.encode_chat({"messages": [], "max_tokens": 2})
    except InferenceError:
        pass
    else:
        raise AssertionError("Encoder must reject an empty messages array.")
    print("Test: Qwen3VLEncoder text encoding and invalid-message validation passed.")

    started_at = time.perf_counter()
    for _ in range(100):
        encoder.encode_chat(request)
    print(f"Bench: text request encoding={(time.perf_counter() - started_at) * 1e3 / 100:.4f} ms/request.")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as profile:
        encoder.encode_chat(request)
    print("Profile: Qwen3VLEncoder")
    print(profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))
