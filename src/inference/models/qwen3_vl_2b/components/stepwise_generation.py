"""用 Transformers DynamicCache 按 Engine 的 Prefill/单 token Decode 步进生成。"""

from __future__ import annotations

import time
from typing import Any

import torch

from inference.network.output_processor import IncrementalDetokenizer


class Qwen3VLStepwiseGeneration:
    """Qwen3-VL 的 Cache、请求状态和 MRoPE Decode 输入准备。"""

    def __init__(self, block_size_tokens: int = 16) -> None:
        if block_size_tokens < 1:
            raise ValueError("block_size_tokens must be positive.")
        self.block_size_tokens = block_size_tokens
        self.model: Any = None
        self.tokenizer: Any = None
        self._states: dict[str, dict[str, Any]] = {}

    def bind(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def begin(self, request_id: str, model_inputs: dict[str, Any], options: dict[str, Any]) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Qwen3-VL model and tokenizer are not loaded.")
        if request_id in self._states:
            raise ValueError(f"Generation request already exists: {request_id}")
        input_ids = model_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Qwen3-VL stepwise generation requires input_ids with shape [1, tokens].")
        if not options.get("use_cache", True):
            raise ValueError("Qwen3-VL stepwise generation requires use_cache=true.")
        eos_token_id = getattr(self.model.generation_config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(self.model.config.text_config, "eos_token_id", None)
        eos_token_ids = set(
            eos_token_id
            if isinstance(eos_token_id, (tuple, list, set))
            else ([] if eos_token_id is None else [eos_token_id])
        )
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0:
            eos_token_ids.add(im_end_id)
        inputs = dict(model_inputs)
        inputs.pop("input_ids")
        inputs.setdefault("attention_mask", torch.ones_like(input_ids))
        self._states[request_id] = {
            "input_ids": input_ids,
            "model_inputs": inputs,
            "prompt_tokens": int(input_ids.shape[-1]),
            "prefill_processed_tokens": 0,
            "past_key_values": None,
            "logical_block_ids": (),
            "options": dict(options),
            "generated_ids": [],
            "eos_token_ids": {int(token_id) for token_id in eos_token_ids},
            "done": False,
            "finish_reason": "stop",
            "detokenizer": IncrementalDetokenizer(self.tokenizer),
        }

    def bind_cache(self, request_id: str, block_ids: tuple[int, ...]) -> None:
        state = self._state(request_id)
        block_ids = tuple(int(block_id) for block_id in block_ids)
        if not block_ids or len(block_ids) != len(set(block_ids)):
            raise ValueError("Engine must bind a non-empty, unique KV block table.")
        state["logical_block_ids"] = block_ids

    def prefill_min_chunk_tokens(self, request_id: str) -> int:
        return int(self._state(request_id)["prompt_tokens"])

    def execute(
        self,
        request_id: str,
        phase: str,
        token_budget: int,
        prompt_offset: int = 0,
    ) -> dict[str, Any]:
        state = self._state(request_id)
        if phase == "prefill":
            if prompt_offset != 0 or token_budget != state["prompt_tokens"]:
                raise ValueError("Qwen3-VL image and text prompts must be processed in one complete Prefill.")
            return self._prefill(state)
        if phase == "decode":
            if token_budget != 1:
                raise ValueError("Qwen3-VL Decode must process exactly one new token.")
            return self._decode(state)
        raise ValueError(f"Unsupported Qwen3-VL execution phase: {phase}.")

    def finish(self, request_id: str) -> tuple[torch.Tensor, str]:
        state = self._state(request_id)
        generated = torch.tensor(state["generated_ids"], dtype=torch.long)
        finish_reason = str(state["finish_reason"])
        self._states.pop(request_id, None)
        return generated, finish_reason

    def cancel(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    def clear(self) -> None:
        self._states.clear()
        self.model = None
        self.tokenizer = None

    def cache_snapshot(self) -> dict[str, int]:
        populated = [state for state in self._states.values() if state["past_key_values"] is not None]
        return {
            "active_requests": len(self._states),
            "populated_caches": len(populated),
            "cached_tokens": sum(self._cache_length(state["past_key_values"]) for state in populated),
            "logical_blocks": sum(len(state["logical_block_ids"]) for state in self._states.values()),
        }

    def _prefill(self, state: dict[str, Any]) -> dict[str, Any]:
        inputs = {"input_ids": state["input_ids"], **state["model_inputs"]}
        with torch.inference_mode():
            output = self.model(**inputs, use_cache=True, return_dict=True)
        cache = getattr(output, "past_key_values", None)
        if cache is None:
            raise RuntimeError("Qwen3-VL Prefill did not return a Transformers Cache.")
        cache_length = self._cache_length(cache)
        if cache_length != state["prompt_tokens"]:
            raise RuntimeError(
                "Qwen3-VL Prefill Cache length mismatch: "
                f"prompt={state['prompt_tokens']}, cache={cache_length}."
            )
        state["past_key_values"] = cache
        state["prefill_processed_tokens"] = state["prompt_tokens"]
        # Image and MRoPE prompt metadata are consumed by Prefill; later steps use only Cache and mask.
        state["model_inputs"] = {"attention_mask": state["model_inputs"]["attention_mask"]}
        return self._record_token(state, output.logits[:, -1, :])

    def _decode(self, state: dict[str, Any]) -> dict[str, Any]:
        if state["done"] or state["past_key_values"] is None or not state["generated_ids"]:
            raise RuntimeError("Qwen3-VL request has not completed Prefill and produced a token.")
        input_ids = state["input_ids"]
        past_key_values = state["past_key_values"]
        past_length = self._cache_length(past_key_values)
        if past_length + 1 > len(state["logical_block_ids"]) * self.block_size_tokens:
            raise RuntimeError("Qwen3-VL Decode exceeded the Engine-reserved KV blocks.")
        attention_mask = torch.ones(
            (1, past_length + 1), dtype=torch.long, device=input_ids.device
        )
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }
        # This Qwen3-VL hook adds saved rope_deltas to text positions and returns 3-axis MRoPE IDs.
        position_ids = self.model._prepare_position_ids_for_generation(input_ids, generation_kwargs)
        prepared = self.model.prepare_inputs_for_generation(
            input_ids=input_ids,
            next_sequence_length=1,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            is_first_iteration=False,
        )
        if prepared["input_ids"].shape[-1] != 1:
            raise RuntimeError("Qwen3-VL Decode prepared more than one query token.")
        with torch.inference_mode():
            output = self.model(**prepared, return_dict=True)
        cache = getattr(output, "past_key_values", None)
        if cache is None or self._cache_length(cache) != past_length + 1:
            raise RuntimeError("Qwen3-VL Decode did not append exactly one token to its Cache.")
        state["past_key_values"] = cache
        state["model_inputs"]["attention_mask"] = attention_mask
        return self._record_token(state, output.logits[:, -1, :])

    def _record_token(self, state: dict[str, Any], logits: torch.Tensor) -> dict[str, Any]:
        input_ids = state["input_ids"]
        token_id = self._sample(logits, input_ids[0].tolist(), state["options"])
        state["generated_ids"].append(token_id)
        state["input_ids"] = torch.cat(
            [input_ids, input_ids.new_tensor([[token_id]])], dim=-1
        )
        reached_limit = len(state["generated_ids"]) >= int(state["options"]["max_new_tokens"])
        is_eos = token_id in state["eos_token_ids"]
        state["done"] = reached_limit or is_eos
        state["finish_reason"] = "length" if reached_limit and not is_eos else "stop"
        detokenizer = state["detokenizer"]
        text = detokenizer.push(token_id)
        if state["done"]:
            text += detokenizer.finish()
        return {
            "token_id": token_id,
            "text": text,
            "generated_tokens": 1,
            "done": state["done"],
            "finish_reason": state["finish_reason"],
            "prompt_tokens_processed": state["prefill_processed_tokens"],
        }

    @staticmethod
    def _sample(logits: torch.Tensor, previous_tokens: list[int], options: dict[str, Any]) -> int:
        scores = logits.float().clone()
        penalty = float(options["repetition_penalty"])
        if penalty != 1:
            previous = torch.tensor(sorted(set(previous_tokens)), device=scores.device, dtype=torch.long)
            selected = scores[:, previous]
            scores[:, previous] = torch.where(selected < 0, selected * penalty, selected / penalty)
        ngram_size = int(options["no_repeat_ngram_size"])
        if ngram_size and len(previous_tokens) >= ngram_size - 1:
            prefix = tuple(previous_tokens[-(ngram_size - 1):]) if ngram_size > 1 else ()
            for start in range(max(1, len(previous_tokens) - ngram_size + 1)):
                ngram = previous_tokens[start : start + ngram_size]
                if len(ngram) == ngram_size and (ngram_size == 1 or tuple(ngram[:-1]) == prefix):
                    scores[:, ngram[-1]] = -torch.inf
        if options["do_sample"]:
            scores /= max(float(options["temperature"]), 1e-5)
            top_k = int(options["top_k"])
            if 0 < top_k < scores.shape[-1]:
                threshold = torch.topk(scores, top_k, dim=-1).values[:, -1, None]
                scores = scores.masked_fill(scores < threshold, -torch.inf)
            top_p = float(options["top_p"])
            if top_p < 1:
                sorted_scores, indices = torch.sort(scores, descending=True, dim=-1)
                remove = torch.cumsum(torch.softmax(sorted_scores, dim=-1), dim=-1) > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                scores = scores.masked_fill(torch.zeros_like(remove).scatter(1, indices, remove), -torch.inf)
            token_id = torch.multinomial(torch.softmax(scores, dim=-1), 1)
        else:
            token_id = torch.argmax(scores, dim=-1, keepdim=True)
        return int(token_id.item())

    @staticmethod
    def _cache_length(cache: Any) -> int:
        getter = getattr(cache, "get_seq_length", None)
        if callable(getter):
            return int(getter())
        return int(cache[0][0].shape[-2])

    def _state(self, request_id: str) -> dict[str, Any]:
        try:
            return self._states[request_id]
        except KeyError as exc:
            raise RuntimeError(f"Unknown Qwen3-VL request: {request_id}") from exc


if __name__ == "__main__":
    # 用单层小模型覆盖 Prefill、单 token Decode、Cache 追加和采样路径。
    test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(101)
    from types import SimpleNamespace

    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration as TransformersQwen3VLForConditionalGeneration,
    )

    from inference.engine.sequence import ExecutionPhase, ScheduledWorkItem
    from ..modeling import Qwen3VLForConditionalGeneration

    text_config = Qwen3VLTextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rope_parameters={
            "rope_theta": 500000.0,
            "rope_type": "default",
            "mrope_section": [1, 1, 2],
        },
        eos_token_id=None,
    )
    vision_config = Qwen3VLVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        depth=1,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=32,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "axial"},
    )
    text_config._attn_implementation = "sdpa"
    vision_config._attn_implementation = "sdpa"
    model_config = Qwen3VLConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=126,
        video_token_id=127,
    )
    reference_model = TransformersQwen3VLForConditionalGeneration(model_config).eval().to(test_device)
    engine_model = Qwen3VLForConditionalGeneration(model_config).eval().to(test_device)
    engine_model.load_state_dict(reference_model.state_dict(), strict=True)
    reference_model.generation_config.eos_token_id = None
    engine_model.generation_config.eos_token_id = None

    # 最小假 tokenizer 只负责结束标记查询与增量文字输出，不引入本地权重依赖。
    tokenizer = SimpleNamespace(
        convert_tokens_to_ids=lambda token: -1,
        decode=lambda token_ids, **kwargs: "".join(f"token-{token_id}" for token_id in token_ids),
    )
    stepwise = Qwen3VLStepwiseGeneration(block_size_tokens=8)
    stepwise.bind(engine_model, tokenizer)
    input_ids = torch.tensor([[3, 7, 11, 19]], device=test_device)
    attention_mask = torch.ones_like(input_ids)
    options = {
        "max_new_tokens": 2,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "do_sample": False,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "use_cache": True,
    }
    request_id = "qwen3-vl-stepwise-forward-check"
    stepwise.begin(
        request_id,
        {"input_ids": input_ids, "attention_mask": attention_mask},
        options,
    )
    stepwise.bind_cache(request_id, (0,))

    # Test：用 Transformers 独立模型检查 Prefill logits、Decode token 与 Cache 长度。
    with torch.inference_mode():
        reference_prefill = reference_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        engine_prefill = stepwise.execute(request_id, "prefill", input_ids.shape[-1])
        expected_first = int(torch.argmax(reference_prefill.logits[0, -1]).item())
        assert engine_prefill["token_id"] == expected_first
        assert stepwise.cache_snapshot()["cached_tokens"] == input_ids.shape[-1]

        full_input_ids = torch.cat((input_ids, input_ids.new_tensor([[expected_first]])), dim=-1)
        decode_mask = torch.ones_like(full_input_ids)
        generation_kwargs = {
            "input_ids": full_input_ids,
            "attention_mask": decode_mask,
            "past_key_values": reference_prefill.past_key_values,
        }
        position_ids = reference_model._prepare_position_ids_for_generation(
            full_input_ids,
            generation_kwargs,
        )
        prepared = reference_model.prepare_inputs_for_generation(
            input_ids=full_input_ids,
            next_sequence_length=1,
            past_key_values=reference_prefill.past_key_values,
            attention_mask=decode_mask,
            position_ids=position_ids,
            use_cache=True,
            is_first_iteration=False,
        )
        assert prepared["input_ids"].shape[-1] == 1
        reference_decode = reference_model(**prepared, return_dict=True)
        expected_second = int(torch.argmax(reference_decode.logits[0, -1]).item())
        engine_decode = stepwise.execute(request_id, "decode", 1)
        assert engine_decode["token_id"] == expected_second
        generated_ids, finish_reason = stepwise.finish(request_id)
    torch.testing.assert_close(
        generated_ids,
        torch.tensor([expected_first, expected_second], device="cpu"),
    )
    assert finish_reason == "length"
    print(f"Qwen3VLStepwiseGeneration Prefill/Decode parity passed on {test_device}.")

    # Bench：每轮创建新 Cache，计量完整 Prefill + 单 token Decode 请求。
    for warmup_index in range(2):
        warmup_id = f"stepwise-warmup-{warmup_index}"
        stepwise.begin(warmup_id, {"input_ids": input_ids, "attention_mask": attention_mask}, options)
        stepwise.bind_cache(warmup_id, (0,))
        stepwise.execute(warmup_id, "prefill", input_ids.shape[-1])
        stepwise.execute(warmup_id, "decode", 1)
        stepwise.finish(warmup_id)
    if test_device.type == "cuda":
        torch.cuda.synchronize(test_device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for run_index in range(10):
            benchmark_id = f"stepwise-bench-{run_index}"
            stepwise.begin(benchmark_id, {"input_ids": input_ids, "attention_mask": attention_mask}, options)
            stepwise.bind_cache(benchmark_id, (0,))
            stepwise.execute(benchmark_id, "prefill", input_ids.shape[-1])
            stepwise.execute(benchmark_id, "decode", 1)
            stepwise.finish(benchmark_id)
        end_event.record()
        end_event.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event) / 10
    else:
        started_at = time.perf_counter()
        for run_index in range(10):
            benchmark_id = f"stepwise-bench-{run_index}"
            stepwise.begin(benchmark_id, {"input_ids": input_ids, "attention_mask": attention_mask}, options)
            stepwise.bind_cache(benchmark_id, (0,))
            stepwise.execute(benchmark_id, "prefill", input_ids.shape[-1])
            stepwise.execute(benchmark_id, "decode", 1)
            stepwise.finish(benchmark_id)
        elapsed_ms = (time.perf_counter() - started_at) * 1000 / 10
    print(f"Bench: Qwen3VLStepwiseGeneration Prefill+Decode={elapsed_ms:.3f} ms/request on {test_device}.")

    # Profile：从组件入口采集实际 Prefill/Decode 算子、输入形状和显存分配。
    profile_id = "qwen3-vl-stepwise-profile"
    stepwise.begin(profile_id, {"input_ids": input_ids, "attention_mask": attention_mask}, options)
    stepwise.bind_cache(profile_id, (0,))
    profile_activities = [torch.profiler.ProfilerActivity.CPU]
    if test_device.type == "cuda":
        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.inference_mode():
        with torch.profiler.profile(
            activities=profile_activities,
            record_shapes=True,
            profile_memory=True,
        ) as stepwise_profile:
            stepwise.execute(profile_id, "prefill", input_ids.shape[-1])
            stepwise.execute(profile_id, "decode", 1)
    stepwise.finish(profile_id)
    if test_device.type == "cuda":
        torch.cuda.synchronize(test_device)
        profile_sort = "self_cuda_time_total"
    else:
        profile_sort = "self_cpu_time_total"
    print("Profile: Qwen3VLStepwiseGeneration Prefill + Decode operators")
    print(stepwise_profile.key_averages().table(sort_by=profile_sort, row_limit=20))
