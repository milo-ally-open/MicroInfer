"""管理 Mage-VL 的逐步 prefill、decode、cache 狀態與抽樣。"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
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

    def execute_batch(self, requests: Sequence[tuple[str, bool]]) -> tuple[dict[str, Any], ...]:
        """以一次模型 forward 執行相容請求，並維持各請求獨立的 cache 與輸出。"""
        if not requests:
            return ()
        request_ids = [request_id for request_id, _ in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("Stepwise batch request IDs must be unique.")
        phases = {first for _, first in requests}
        if len(phases) != 1:
            raise ValueError("Stepwise batch requests must use the same execution phase.")
        first = requests[0][1]
        states = [self._state(request_id) for request_id in request_ids]
        if any(state["done"] for state in states):
            if len(states) == 1:
                state = states[0]
                return ({"text": "", "generated_tokens": 0, "done": True, "finish_reason": state["finish_reason"]},)
            return tuple(self._step(request_id, first=first) for request_id in request_ids)
        if len(states) == 1:
            return (self._step(request_ids[0], first=first),)

        if first:
            prepared_batch = self._combine_prefill_inputs(states)
            if prepared_batch is None:
                return tuple(self._step(request_id, first=True) for request_id in request_ids)
            input_ids, model_inputs, attention_masks, pad_offsets = prepared_batch
            past_key_values = None
            cache_position_start = 0
        else:
            past_key_values = self._combine_caches([state["past_key_values"] for state in states])
            if past_key_values is None:
                return tuple(self._step(request_id, first=False) for request_id in request_ids)
            cache_lengths = {self._cache_length(state["past_key_values"]) for state in states}
            if len(cache_lengths) != 1:
                return tuple(self._step(request_id, first=False) for request_id in request_ids)
            cache_position_start = cache_lengths.pop()
            input_ids = torch.cat([state["input_ids"][:, -1:] for state in states], dim=0)
            attention_masks = [
                torch.cat(
                    [state["model_inputs"]["attention_mask"], state["model_inputs"]["attention_mask"].new_ones((1, 1))],
                    dim=-1,
                )
                for state in states
            ]
            model_inputs = {"attention_mask": torch.cat(attention_masks, dim=0)}
            pad_offsets = [0] * len(states)

        output = self._batch_forward(
            input_ids,
            model_inputs,
            past_key_values=past_key_values,
            first=first,
            cache_position_start=cache_position_start,
        )
        output_cache = getattr(output, "past_key_values", None)
        if output_cache is None:
            raise RuntimeError("The loaded Mage-VL model did not return past_key_values for batch decoding.")
        split_caches = self._split_cache(output_cache, len(states), pad_offsets)
        results: list[dict[str, Any]] = []
        for index, (request_id, state) in enumerate(zip(request_ids, states)):
            state["past_key_values"] = split_caches[index]
            if first:
                # Cache rows have their left-padding positions removed, so each request keeps its true prompt length.
                state["model_inputs"] = {"attention_mask": attention_masks[index][:, pad_offsets[index] :]}
            else:
                state["model_inputs"] = {"attention_mask": attention_masks[index]}
            results.append(
                self._record_token(state, output.logits[index : index + 1, -1, :])
            )
        return tuple(results)

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
        if not first:
            target_mask_length = self._cache_length(past_key_values) + 1
            attention_mask = model_inputs["attention_mask"]
            missing_tokens = target_mask_length - attention_mask.shape[-1]
            if missing_tokens < 0:
                raise RuntimeError("The request attention mask is longer than its actual Cache.")
            if missing_tokens:
                model_inputs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((1, missing_tokens))], dim=-1
                )
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
        state["model_inputs"] = {"attention_mask": model_inputs["attention_mask"]}
        return self._record_token(state, output.logits[:, -1, :])

    def _record_token(self, state: dict[str, Any], logits: Any) -> dict[str, Any]:
        input_ids = state["input_ids"]
        options = state["options"]
        token_id = self._sample_next_token(logits, input_ids[0].tolist(), options)
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

    def _batch_forward(
        self,
        input_ids: Any,
        model_inputs: dict[str, Any],
        *,
        past_key_values: Any,
        first: bool,
        cache_position_start: int,
    ) -> Any:
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
            "use_cache": True,
        }
        if "cache_position" in parameters:
            sequence_length = input_ids.shape[-1]
            kwargs["cache_position"] = torch.arange(
                cache_position_start,
                cache_position_start + sequence_length,
                device=input_ids.device,
            )
        if "is_first_iteration" in parameters:
            kwargs["is_first_iteration"] = first
        prepared = prepare(input_ids, **kwargs)
        prepared["use_cache"] = True
        with torch.inference_mode():
            return self.model(**prepared, return_dict=True)

    def _combine_prefill_inputs(
        self, states: Sequence[dict[str, Any]]
    ) -> tuple[Any, dict[str, Any], list[Any], list[int]] | None:
        input_ids = [state["input_ids"] for state in states]
        if any(ids.ndim != 2 or ids.shape[0] != 1 for ids in input_ids):
            return None
        max_length = max(ids.shape[-1] for ids in input_ids)
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.processor.tokenizer, "eos_token_id", 0)
        padded_ids: list[Any] = []
        attention_masks: list[Any] = []
        pad_offsets: list[int] = []
        for state, ids in zip(states, input_ids):
            offset = max_length - ids.shape[-1]
            pad_offsets.append(offset)
            padded_ids.append(torch.nn.functional.pad(ids, (offset, 0), value=int(pad_id)))
            attention = state["model_inputs"].get("attention_mask", torch.ones_like(ids))
            attention_masks.append(torch.nn.functional.pad(attention, (offset, 0), value=0))

        model_inputs: dict[str, Any] = {"attention_mask": torch.cat(attention_masks, dim=0)}
        visual_keys = {
            "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw",
            "second_per_grid_ts", "patch_positions",
        }
        all_keys = set().union(*(state["model_inputs"].keys() for state in states)) - {"attention_mask"}
        for key in all_keys:
            values = [state["model_inputs"].get(key) for state in states]
            present = [value for value in values if value is not None]
            if not present:
                continue
            if key in visual_keys:
                if not all(value is None or isinstance(value, torch.Tensor) for value in values):
                    return None
                merged = self._concat_visual_values(key, present)
                if merged is None:
                    return None
                model_inputs[key] = merged
                continue
            if len(present) != len(values):
                return None
            if all(isinstance(value, torch.Tensor) for value in values):
                if all(value.ndim == 2 and value.shape[0] == 1 and value.shape[-1] == ids.shape[-1]
                       for value, ids in zip(values, input_ids)):
                    model_inputs[key] = torch.cat([
                        torch.nn.functional.pad(value, (offset, 0), value=0)
                        for value, offset in zip(values, pad_offsets)
                    ], dim=0)
                else:
                    merged = self._concat_visual_values(key, values)
                    if merged is None:
                        return None
                    model_inputs[key] = merged
            elif all(value == values[0] for value in values):
                model_inputs[key] = values[0]
            else:
                return None
        return torch.cat(padded_ids, dim=0), model_inputs, attention_masks, pad_offsets

    @staticmethod
    def _concat_visual_values(key: str, values: Sequence[Any]) -> Any | None:
        if not values or any(not isinstance(value, torch.Tensor) for value in values):
            return None
        first = values[0]
        if key == "patch_positions" and first.ndim == 3 and first.shape[0] == 1:
            if any(value.ndim != 3 or value.shape[0] != 1 or value.shape[2:] != first.shape[2:] for value in values):
                return None
            return torch.cat(values, dim=1)
        if any(value.ndim != first.ndim or value.shape[1:] != first.shape[1:] for value in values):
            return None
        return torch.cat(values, dim=0)

    def _combine_caches(self, caches: Sequence[Any]) -> Any | None:
        if not caches or any(cache is None or type(cache) is not type(caches[0]) for cache in caches):
            return None
        cache_lengths = {self._cache_length(cache) for cache in caches}
        if len(cache_lengths) != 1:
            return None
        layer_pairs = [self._cache_layer_pairs(cache) for cache in caches]
        if any(pairs is None for pairs in layer_pairs):
            return None
        layer_count = len(layer_pairs[0])
        if any(len(pairs) != layer_count for pairs in layer_pairs):
            return None
        merged_pairs = []
        for layer_index in range(layer_count):
            key_values = [pairs[layer_index] for pairs in layer_pairs]
            keys = [pair[0] for pair in key_values]
            values = [pair[1] for pair in key_values]
            if not self._can_concat_cache_tensors(keys) or not self._can_concat_cache_tensors(values):
                return None
            merged_pairs.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
        return self._new_cache(caches[0], merged_pairs)

    def _split_cache(self, cache: Any, batch_size: int, pad_offsets: Sequence[int]) -> list[Any]:
        layer_pairs = self._cache_layer_pairs(cache)
        if layer_pairs is not None:
            split: list[Any] = []
            for row, offset in enumerate(pad_offsets):
                row_pairs = []
                for key, value in layer_pairs:
                    key_length = key.shape[-2]
                    value_length = value.shape[-2]
                    if offset > key_length or offset > value_length:
                        raise RuntimeError("Cannot remove prefill padding from the returned cache.")
                    row_pairs.append((
                        key[row : row + 1, ..., offset:, :].clone(),
                        value[row : row + 1, ..., offset:, :].clone(),
                    ))
                split.append(self._new_cache(cache, row_pairs))
            return split
        batch_split = getattr(cache, "batch_split", None)
        if callable(batch_split) and not any(pad_offsets):
            return list(batch_split(full_batch_size=batch_size, split_size=1))
        raise RuntimeError("The loaded Mage-VL cache type cannot be split into per-request caches.")

    @staticmethod
    def _cache_layer_pairs(cache: Any) -> list[tuple[Any, Any]] | None:
        layers = getattr(cache, "layers", None)
        if layers is not None:
            pairs = []
            for layer in layers:
                key, value = getattr(layer, "keys", None), getattr(layer, "values", None)
                if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                    return None
                pairs.append((key, value))
            return pairs or None
        if isinstance(cache, (tuple, list)) and cache:
            pairs = []
            for layer in cache:
                if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                    return None
                pairs.append((layer[0], layer[1]))
            return pairs
        return None

    def _new_cache(self, template: Any, layer_pairs: Sequence[tuple[Any, Any]]) -> Any:
        if isinstance(template, tuple):
            return tuple(tuple(pair) for pair in layer_pairs)
        if isinstance(template, list):
            return [tuple(pair) for pair in layer_pairs]
        cache_type = type(template)
        try:
            parameters = inspect.signature(cache_type).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "ddp_cache_data" not in parameters:
            raise RuntimeError(f"Cache type {cache_type.__name__} cannot be rebuilt for batch execution.")
        kwargs: dict[str, Any] = {"ddp_cache_data": layer_pairs}
        if "config" in parameters:
            config = getattr(getattr(self.model, "config", None), "text_config", None)
            kwargs["config"] = config or getattr(self.model, "config", None)
        return cache_type(**kwargs)

    @staticmethod
    def _can_concat_cache_tensors(tensors: Sequence[Any]) -> bool:
        if not tensors or any(not isinstance(tensor, torch.Tensor) or tensor.ndim < 1 for tensor in tensors):
            return False
        first = tensors[0]
        return all(
            tensor.ndim == first.ndim
            and tensor.shape[1:] == first.shape[1:]
            and tensor.device == first.device
            and tensor.dtype == first.dtype
            for tensor in tensors
        )

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
