"""管理 Mage-VL 的逐步 prefill、decode、cache 狀態與抽樣。"""

from __future__ import annotations

import inspect
from typing import Any

import torch


class MageVLStepwiseGeneration:
    """保存每個 request 的 Transformers Cache 並逐 token 執行模型。"""

    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self._states: dict[str, dict[str, Any]] = {}

    def bind(self, model: Any, processor: Any) -> None:
        self.model = model
        self.processor = processor

    def begin(self, request_id: str, inputs: dict[str, Any], options: dict[str, Any]) -> None:
        if self.model is None or self.processor is None:
            raise RuntimeError("Mage-VL model is not loaded.")
        unsupported = set(options) - {
            "max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "min_p",
            "repetition_penalty", "no_repeat_ngram_size", "use_cache", "eos_token_id", "pad_token_id",
        }
        if unsupported:
            raise ValueError("Stepwise decoding does not support generation options: " + ", ".join(sorted(unsupported)))
        if not options.get("use_cache", True):
            raise ValueError("Stepwise decoding requires generation.use_cache=true.")
        if int(options.get("num_beams", 1)) != 1:
            raise ValueError("Stepwise decoding currently requires num_beams=1.")
        if request_id in self._states:
            raise ValueError(f"Generation request already exists: {request_id}")

        model_inputs = dict(inputs)
        input_ids = model_inputs.pop("input_ids", None)
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Mage-VL stepwise decoding currently requires a single input sequence.")
        if "attention_mask" not in model_inputs:
            model_inputs["attention_mask"] = torch.ones_like(input_ids)
        eos_token_id = options.get("eos_token_id")
        if eos_token_id is None:
            eos_token_id = getattr(self.model.generation_config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        self._states[request_id] = {
            "input_ids": input_ids,
            "model_inputs": model_inputs,
            "options": dict(options),
            "past_key_values": None,
            "generated_ids": [],
            "eos_token_ids": set(
                eos_token_id if isinstance(eos_token_id, (list, tuple, set))
                else ([] if eos_token_id is None else [eos_token_id])
            ),
            "finish_reason": "stop",
            "done": False,
            "logical_block_ids": (),
        }

    def bind_cache(self, request_id: str, block_ids: tuple[int, ...]) -> None:
        self._state(request_id)["logical_block_ids"] = tuple(block_ids)

    def prefill(self, request_id: str) -> dict[str, Any]:
        return self._step(request_id, first=True)

    def decode(self, request_id: str) -> dict[str, Any]:
        return self._step(request_id, first=False)

    def finish(self, request_id: str) -> tuple[Any, str]:
        state = self._state(request_id)
        generated_ids = torch.tensor(
            state["generated_ids"], dtype=torch.long, device=state["input_ids"].device
        )
        finish_reason = state["finish_reason"]
        self._states.pop(request_id, None)
        return generated_ids, finish_reason

    def cancel(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    def clear(self) -> None:
        self._states.clear()

    def _state(self, request_id: str) -> dict[str, Any]:
        try:
            return self._states[request_id]
        except KeyError as exc:
            raise RuntimeError(f"Unknown generation request: {request_id}") from exc

    def _step(self, request_id: str, *, first: bool) -> dict[str, Any]:
        state = self._state(request_id)
        options = state["options"]
        if state["done"]:
            return {"text": "", "generated_tokens": 0, "done": True, "finish_reason": state["finish_reason"]}

        input_ids = state["input_ids"]
        past_key_values = state["past_key_values"]
        model_inputs = dict(state["model_inputs"])
        if state["generated_ids"]:
            generated_mask = torch.ones_like(input_ids[:, -len(state["generated_ids"]):])
            model_inputs["attention_mask"] = torch.cat([model_inputs["attention_mask"], generated_mask], dim=-1)
        prepare = getattr(self.model, "prepare_inputs_for_generation", None)
        if prepare is None:
            raise RuntimeError("The loaded Mage-VL model does not expose prepare_inputs_for_generation().")
        try:
            parameters = inspect.signature(prepare).parameters
        except (TypeError, ValueError):
            parameters = {}
        kwargs: dict[str, Any] = {
            **model_inputs,
            "past_key_values": past_key_values,
            "attention_mask": model_inputs["attention_mask"],
            "use_cache": bool(options.get("use_cache", True)),
        }
        if "cache_position" in parameters:
            cache_length = self._cache_length(past_key_values)
            kwargs["cache_position"] = torch.arange(cache_length, input_ids.shape[-1], device=input_ids.device)
        if "is_first_iteration" in parameters:
            kwargs["is_first_iteration"] = first
        prepared = prepare(input_ids, **kwargs)
        prepared["use_cache"] = bool(options.get("use_cache", True))
        with torch.inference_mode():
            output = self.model(**prepared, return_dict=True)
        if getattr(output, "past_key_values", None) is None:
            raise RuntimeError("The loaded Mage-VL model did not return past_key_values for stepwise decoding.")
        state["past_key_values"] = output.past_key_values
        token_id = self._sample_next_token(output.logits[:, -1, :], input_ids[0].tolist(), options)
        state["generated_ids"].append(token_id)
        token_tensor = torch.tensor([[token_id]], dtype=input_ids.dtype, device=input_ids.device)
        state["input_ids"] = torch.cat([input_ids, token_tensor], dim=-1)
        max_tokens = int(options["max_new_tokens"])
        done = token_id in state["eos_token_ids"] or len(state["generated_ids"]) >= max_tokens
        if done:
            state["done"] = True
            state["finish_reason"] = (
                "length" if len(state["generated_ids"]) >= max_tokens and token_id not in state["eos_token_ids"]
                else "stop"
            )
        try:
            text = self.processor.tokenizer.decode(
                [token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        except TypeError:
            text = self.processor.tokenizer.decode([token_id], skip_special_tokens=True)
        return {"text": text, "generated_tokens": 1, "done": done, "finish_reason": state["finish_reason"]}

    @staticmethod
    def _cache_length(cache: Any) -> int:
        if cache is None:
            return 0
        getter = getattr(cache, "get_seq_length", None)
        if callable(getter):
            return int(getter())
        return int(cache[0][0].shape[-2])

    @staticmethod
    def _sample_next_token(logits: Any, input_ids: list[int], options: dict[str, Any]) -> int:
        scores = logits.float().clone()
        penalty = float(options.get("repetition_penalty", 1.0))
        if penalty != 1.0:
            previous = torch.tensor(sorted(set(input_ids)), dtype=torch.long, device=scores.device)
            selected = scores[:, previous]
            scores[:, previous] = torch.where(selected < 0, selected * penalty, selected / penalty)
        ngram_size = int(options.get("no_repeat_ngram_size", 0))
        if ngram_size > 0 and len(input_ids) >= ngram_size - 1:
            prefix = tuple(input_ids[-(ngram_size - 1):]) if ngram_size > 1 else ()
            for index in range(max(1, len(input_ids) - ngram_size + 1)):
                ngram = input_ids[index:index + ngram_size]
                if len(ngram) == ngram_size and (ngram_size == 1 or tuple(ngram[:-1]) == prefix):
                    scores[:, ngram[-1]] = -torch.inf
        if options.get("do_sample", False):
            scores /= max(float(options.get("temperature", 1.0)), 1e-5)
            top_k = int(options.get("top_k", 0) or 0)
            if top_k > 0 and top_k < scores.shape[-1]:
                cutoff = torch.topk(scores, top_k, dim=-1).values[:, -1, None]
                scores = scores.masked_fill(scores < cutoff, -torch.inf)
            top_p = float(options.get("top_p", 1.0))
            if top_p < 1:
                sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
                cumulative = torch.cumsum(torch.softmax(sorted_scores, dim=-1), dim=-1)
                remove = cumulative > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                indices_to_remove = torch.zeros_like(remove).scatter(1, sorted_indices, remove)
                scores = scores.masked_fill(indices_to_remove, -torch.inf)
            min_p = options.get("min_p")
            if min_p is not None and float(min_p) > 0:
                probabilities = torch.softmax(scores, dim=-1)
                scores = scores.masked_fill(
                    probabilities < probabilities.max(dim=-1, keepdim=True).values * float(min_p), -torch.inf
                )
            token_id = torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1)
        else:
            token_id = torch.argmax(scores, dim=-1, keepdim=True)
        return int(token_id.item())
