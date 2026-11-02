"""用 Transformers DynamicCache 按 Engine 的 Prefill/单 token Decode 步进生成。"""

from __future__ import annotations

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
        eos_token_ids = set(eos_token_id if isinstance(eos_token_id, (tuple, list, set)) else ([] if eos_token_id is None else [eos_token_id]))
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

    def execute(self, request_id: str, phase: str, token_budget: int, prompt_offset: int = 0) -> dict[str, Any]:
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
                f"Qwen3-VL Prefill Cache length mismatch: prompt={state['prompt_tokens']}, cache={cache_length}."
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
