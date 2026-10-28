"""管理 Mage-VL 的逐步 prefill、decode、cache 状态与抽样。"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import torch

from .cache_manager import RequestCacheManager
from .detokenizer import MageVLDetokenizer
from .tokenizer import MageVLTokenizer


class MageVLStepwiseGeneration:
    """保存每个 request 的 Transformers Cache 并逐 token 执行模型。"""

    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None
        self._states: dict[str, dict[str, Any]] = {}
        self.cache_manager = RequestCacheManager()

    def bind(self, model: Any, processor: Any, tokenizer: MageVLTokenizer | None = None) -> None:
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer or MageVLTokenizer(processor.tokenizer)

    def begin(self, request_id: str, inputs: dict[str, Any], options: dict[str, Any]) -> None:
        if self.model is None or self.processor is None or self.tokenizer is None:
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
            eos_token_id = self.tokenizer.eos_token_id
        self.cache_manager.register(request_id)
        self._states[request_id] = {
            "input_ids": input_ids,
            "prompt_attention_mask": model_inputs["attention_mask"],
            "prefill_processed_tokens": 0,
            "model_inputs": model_inputs,
            "options": dict(options),
            "generated_ids": [],
            "eos_token_ids": set(
                eos_token_id if isinstance(eos_token_id, (list, tuple, set))
                else ([] if eos_token_id is None else [eos_token_id])
            ),
            "finish_reason": "stop",
            "done": False,
            "detokenizer": MageVLDetokenizer(self.tokenizer),
        }

    def bind_cache(self, request_id: str, block_ids: tuple[int, ...]) -> None:
        self._state(request_id)
        self.cache_manager.bind_blocks(request_id, block_ids)

    def prefill_min_chunk_tokens(self, request_id: str) -> int:
        """回报首轮必须涵盖的视觉 token 前缀长度。"""
        state = self._state(request_id)
        if self._has_visual_inputs(state):
            # Visual tensors can expand prompt markers into model-specific positions.
            # Keep the complete prompt in one forward so text offsets never split that expansion.
            return int(state["input_ids"].shape[-1])
        config = getattr(self.model, "config", None)
        visual_ids = {
            value for value in (
                getattr(config, "image_token_id", None),
                getattr(config, "video_token_id", None),
            ) if isinstance(value, int)
        }
        if not visual_ids:
            return 1
        input_ids = state["input_ids"]
        positions = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in visual_ids:
            positions |= input_ids == token_id
        visual_positions = torch.nonzero(positions[0], as_tuple=False)
        if visual_positions.numel() == 0:
            return 1
        return int(visual_positions[-1, 0].item()) + 1

    def execute_work_batch(self, requests: Sequence[Any]) -> tuple[dict[str, Any], ...]:
        """在一个 Worker 请求中混排 prefill/decode，并按原顺序返回逐请求结果。"""
        if not requests:
            return ()
        ids = [item.request_id for item in requests]
        if len(ids) != len(set(ids)):
            raise ValueError("Stepwise work batch request IDs must be unique.")
        results: dict[str, dict[str, Any]] = {}
        unsupported = [item.phase.value for item in requests if item.phase.value not in {"prefill", "decode"}]
        if unsupported:
            raise ValueError("Unsupported Mage-VL execution phase: " + ", ".join(sorted(set(unsupported))))
        for phase in dict.fromkeys(item.phase.value for item in requests):
            group = [item for item in requests if item.phase.value == phase]
            if phase == "prefill":
                steps = self.execute_prefill_chunks(tuple(
                    (item.request_id, item.prompt_offset, item.token_budget) for item in group
                ))
            else:
                steps = self._execute_decode_batch(tuple(item.request_id for item in group))
            results.update((item.request_id, step) for item, step in zip(group, steps))
        return tuple(results[request_id] for request_id in ids)

    def execute_prefill_chunks(
        self, requests: Sequence[tuple[str, int, int]]
    ) -> tuple[dict[str, Any], ...]:
        """将提示词片段追加至 Cache；仅最后片段取样并产生第一个 token。"""
        if not requests:
            return ()
        request_ids = [request_id for request_id, _, _ in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("Prefill chunk request IDs must be unique.")
        for request_id, offset, token_budget in requests:
            state = self._state(request_id)
            total = int(state["prompt_attention_mask"].shape[-1])
            if offset != state["prefill_processed_tokens"]:
                raise RuntimeError(
                    f"Prefill offset mismatch for {request_id}: expected "
                    f"{state['prefill_processed_tokens']}, received {offset}."
                )
            if token_budget < 1 or offset + token_budget > total:
                raise ValueError(f"Prefill chunk exceeds prompt length for {request_id}.")

        # Fused chunk forward is safe when all rows have the same prompt length, cache offset,
        # and chunk width. Otherwise, execute each chunk separately rather than pad Cache rows.
        first = requests[0]
        states = [self._state(request_id) for request_id, _, _ in requests]
        compatible = (
            all(offset == first[1] and budget == first[2] for _, offset, budget in requests)
            and len({int(state["input_ids"].shape[-1]) for state in states}) == 1
        )
        if compatible and len(requests) > 1:
            combined = self._execute_compatible_prefill_chunks(requests, states)
            if combined is not None:
                return combined
        return tuple(
            self._execute_prefill_chunk(request_id, offset, token_budget)
            for request_id, offset, token_budget in requests
        )

    def _execute_prefill_chunk(
        self, request_id: str, offset: int, token_budget: int
    ) -> dict[str, Any]:
        state = self._state(request_id)
        end = offset + token_budget
        prompt_mask = state["prompt_attention_mask"][:, :end]
        model_inputs = dict(state["model_inputs"])
        model_inputs["attention_mask"] = prompt_mask
        past_key_values = self.cache_manager.get(request_id)
        output = self._batch_forward(
            state["input_ids"][:, offset:end],
            model_inputs,
            past_key_values=past_key_values,
            first=offset == 0,
            cache_position_start=offset,
            cache_position_length=token_budget,
        )
        output_cache = getattr(output, "past_key_values", None)
        if output_cache is None:
            raise RuntimeError("The loaded Mage-VL model did not append a Cache for the prefill chunk.")
        self._update_prefill_cache(request_id, output_cache, end)
        state["prefill_processed_tokens"] = end
        # Visual tensors are only sent with offset zero. The next chunk uses the Cache and text IDs.
        state["model_inputs"] = {"attention_mask": prompt_mask}
        if end < int(state["prompt_attention_mask"].shape[-1]):
            return {
                "text": "",
                "generated_tokens": 0,
                "done": False,
                "prompt_tokens_processed": end,
            }
        step = self._record_token(state, output.logits[:, -1, :])
        step["prompt_tokens_processed"] = end
        return step

    def _execute_compatible_prefill_chunks(
        self,
        requests: Sequence[tuple[str, int, int]],
        states: Sequence[dict[str, Any]],
    ) -> tuple[dict[str, Any], ...] | None:
        request_id, offset, token_budget = requests[0]
        end = offset + token_budget
        if any(self.cache_manager.length(item[0]) != offset for item in requests):
            return None
        input_ids = torch.cat([state["input_ids"][:, offset:end] for state in states], dim=0)
        attention_mask = torch.cat([
            state["prompt_attention_mask"][:, :end] for state in states
        ], dim=0)
        model_inputs: dict[str, Any] = {"attention_mask": attention_mask}
        if offset == 0:
            visual_keys = {
                "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw",
                "second_per_grid_ts", "patch_positions",
            }
            keys = set().union(*(state["model_inputs"].keys() for state in states)) - {"attention_mask"}
            for key in keys:
                values = [state["model_inputs"].get(key) for state in states]
                present = [value for value in values if value is not None]
                if not present:
                    continue
                if len(present) != len(values):
                    return None
                if all(isinstance(value, torch.Tensor) for value in values):
                    merged = self._concat_visual_values(key, values)
                    if merged is None:
                        return None
                    model_inputs[key] = merged
                elif key in visual_keys:
                    return None
                elif all(value == values[0] for value in values):
                    model_inputs[key] = values[0]
                else:
                    return None
        caches = [self.cache_manager.get(item[0]) for item in requests]
        past_key_values = None if offset == 0 else self._combine_caches(caches)
        if offset and past_key_values is None:
            return None
        output = self._batch_forward(
            input_ids,
            model_inputs,
            past_key_values=past_key_values,
            first=offset == 0,
            cache_position_start=offset,
            cache_position_length=token_budget,
        )
        output_cache = getattr(output, "past_key_values", None)
        if output_cache is None:
            raise RuntimeError("The loaded Mage-VL model did not append a Cache for the prefill chunk.")
        split_caches = self._split_cache(output_cache, len(states))
        results: list[dict[str, Any]] = []
        for index, ((request_id, _, _), state) in enumerate(zip(requests, states)):
            self._update_prefill_cache(request_id, split_caches[index], end)
            state["prefill_processed_tokens"] = end
            state["model_inputs"] = {"attention_mask": state["prompt_attention_mask"][:, :end]}
            if end < int(state["prompt_attention_mask"].shape[-1]):
                results.append({
                    "text": "", "generated_tokens": 0, "done": False,
                    "prompt_tokens_processed": end,
                })
            else:
                step = self._record_token(state, output.logits[index : index + 1, -1, :])
                step["prompt_tokens_processed"] = end
                results.append(step)
        return tuple(results)

    def _execute_decode_batch(self, request_ids: Sequence[str]) -> tuple[dict[str, Any], ...]:
        """批次执行兼容的 decode 请求，并维持各请求独立的 cache 与输出。"""
        if not request_ids:
            return ()
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("Stepwise batch request IDs must be unique.")
        states = [self._state(request_id) for request_id in request_ids]
        if any(state["done"] for state in states):
            if len(states) == 1:
                state = states[0]
                return ({"text": "", "generated_tokens": 0, "done": True, "finish_reason": state["finish_reason"]},)
            return tuple(self._decode_one(request_id) for request_id in request_ids)
        if len(states) == 1:
            return (self._decode_one(request_ids[0]),)
        past_key_values = self._combine_caches([self.cache_manager.get(request_id) for request_id in request_ids])
        if past_key_values is None:
            return tuple(self._decode_one(request_id) for request_id in request_ids)
        cache_lengths = {self.cache_manager.length(request_id) for request_id in request_ids}
        if len(cache_lengths) != 1:
            return tuple(self._decode_one(request_id) for request_id in request_ids)
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

        output = self._batch_forward(
            input_ids,
            model_inputs,
            past_key_values=past_key_values,
            first=False,
            cache_position_start=cache_position_start,
        )
        output_cache = getattr(output, "past_key_values", None)
        if output_cache is None:
            raise RuntimeError("The loaded Mage-VL model did not return a Cache for batched decoding.")
        split_caches = self._split_cache(output_cache, len(states))
        expected_cache_length = cache_position_start + 1
        results: list[dict[str, Any]] = []
        for index, (request_id, state) in enumerate(zip(request_ids, states)):
            state["model_inputs"] = {"attention_mask": attention_masks[index]}
            self.cache_manager.update(
                request_id, split_caches[index], expected_length=expected_cache_length
            )
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
        self.cache_manager.release(request_id)
        return generated_ids, finish_reason

    def cancel(self, request_id: str) -> None:
        self._states.pop(request_id, None)
        self.cache_manager.release(request_id)

    def clear(self) -> None:
        self._states.clear()
        self.cache_manager.clear()
        self.model = None
        self.processor = None
        self.tokenizer = None

    def cache_snapshot(self) -> dict[str, int]:
        return self.cache_manager.snapshot()

    def _state(self, request_id: str) -> dict[str, Any]:
        try:
            return self._states[request_id]
        except KeyError as exc:
            raise RuntimeError(f"Unknown generation request: {request_id}") from exc

    def _decode_one(self, request_id: str) -> dict[str, Any]:
        state = self._state(request_id)
        options = state["options"]
        if state["done"]:
            return {"text": "", "generated_tokens": 0, "done": True, "finish_reason": state["finish_reason"]}

        input_ids = state["input_ids"]
        decode_input_ids = input_ids[:, -1:]
        past_key_values = self.cache_manager.get(request_id)
        model_inputs = dict(state["model_inputs"])
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
            kwargs["cache_position"] = torch.arange(
                cache_length, cache_length + 1, device=decode_input_ids.device
            )
        if "is_first_iteration" in parameters:
            kwargs["is_first_iteration"] = False
        prepared = prepare(decode_input_ids, **kwargs)
        prepared["use_cache"] = bool(options.get("use_cache", True))
        with torch.inference_mode():
            output = self.model(**prepared, return_dict=True)
        if getattr(output, "past_key_values", None) is None:
            raise RuntimeError("The loaded Mage-VL model did not return past_key_values for stepwise decoding.")
        expected_cache_length = self.cache_manager.length(request_id) + 1
        self.cache_manager.update(
            request_id, output.past_key_values, expected_length=expected_cache_length
        )
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
        detokenizer = state["detokenizer"]
        text = detokenizer.push(token_id)
        if done:
            text += detokenizer.finish()
        return {"text": text, "generated_tokens": 1, "done": done, "finish_reason": state["finish_reason"]}

    def _batch_forward(
        self,
        input_ids: Any,
        model_inputs: dict[str, Any],
        *,
        past_key_values: Any,
        first: bool,
        cache_position_start: int,
        cache_position_length: int | None = None,
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
            sequence_length = cache_position_length or input_ids.shape[-1]
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

    def _split_cache(self, cache: Any, batch_size: int) -> list[Any]:
        layer_pairs = self._cache_layer_pairs(cache)
        if layer_pairs is not None:
            split: list[Any] = []
            for row in range(batch_size):
                row_pairs = []
                for key, value in layer_pairs:
                    row_pairs.append((
                        key[row : row + 1].clone(),
                        value[row : row + 1].clone(),
                    ))
                split.append(self._new_cache(cache, row_pairs))
            return split
        batch_split = getattr(cache, "batch_split", None)
        if callable(batch_split):
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
    def _has_visual_inputs(state: dict[str, Any]) -> bool:
        model_inputs = state.get("model_inputs", {})
        return any(
            model_inputs.get(key) is not None
            for key in (
                "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw",
                "second_per_grid_ts", "patch_positions",
            )
        )

    def _update_prefill_cache(self, request_id: str, cache: Any, minimum_length: int) -> None:
        """允许视觉模型展开 Cache 位置，但不接受少于已处理 token 的 Cache。"""
        actual_length = self.cache_manager.sequence_length(cache)
        if actual_length < minimum_length:
            raise RuntimeError(
                f"Model Cache is shorter than completed prefill for {request_id}: "
                f"expected at least {minimum_length}, received {actual_length}."
            )
        self.cache_manager.update(request_id, cache)

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


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
    from types import SimpleNamespace

    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3ForCausalLM as TransformersQwen3ForCausalLM,
    )

    from inference.engine.sequence import ExecutionPhase, ScheduledWorkItem
    from ..model.configuration_mage_vl import MageVLConfig, MageVLVisionConfig
    from ..model.conditional_generation import MageVLForConditionalGeneration

    torch.manual_seed(79)
    text_config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        eos_token_id=None,
    )
    text_config._attn_implementation = "eager"
    vision_config = MageVLVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        patch_size=2,
        out_hidden_size=32,
    )
    config = MageVLConfig(text_config=text_config, vision_config=vision_config)
    engine_model = MageVLForConditionalGeneration(config).eval()
    baseline_model = TransformersQwen3ForCausalLM(text_config).eval()
    baseline_model.model.load_state_dict(engine_model.model.language_model.state_dict(), strict=True)
    baseline_model.lm_head.load_state_dict(engine_model.lm_head.state_dict(), strict=True)
    tokenizer = SimpleNamespace(
        eos_token_id=None,
        decode=lambda token_ids, **kwargs: "".join(f"token-{token_id}" for token_id in token_ids),
    )
    processor = SimpleNamespace(tokenizer=tokenizer)
    engine = MageVLStepwiseGeneration()
    engine.bind(engine_model, processor)
    request_id = "stepwise-forward-check"
    input_ids = torch.tensor([[5, 7, 9, 11]])
    attention_mask = torch.ones_like(input_ids)
    options = {
        "max_new_tokens": 2,
        "temperature": 0.0,
        "do_sample": False,
        "use_cache": True,
        "eos_token_id": None,
    }
    engine.begin(
        request_id,
        {"input_ids": input_ids, "attention_mask": attention_mask},
        options,
    )
    try:
        with torch.inference_mode():
            baseline_prefill = baseline_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            engine_prefill = engine.execute_work_batch(
                (ScheduledWorkItem(request_id, ExecutionPhase.PREFILL, input_ids.shape[-1]),)
            )[0]
            baseline_first_token = int(torch.argmax(baseline_prefill.logits[0, -1]).item())
            assert engine._states[request_id]["generated_ids"] == [baseline_first_token]
            decode_item = ScheduledWorkItem(request_id, ExecutionPhase.DECODE, 1)
            engine_decode = engine.execute_work_batch((decode_item,))[0]
            baseline_decode = baseline_model(
                input_ids=torch.tensor([[baseline_first_token]]),
                attention_mask=torch.ones((1, input_ids.shape[-1] + 1), dtype=torch.long),
                past_key_values=baseline_prefill.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            baseline_second_token = int(torch.argmax(baseline_decode.logits[0, -1]).item())
            assert engine._states[request_id]["generated_ids"][-1] == baseline_second_token
        print("==========engine===========")
        print(f"prefill text={engine_prefill['text']}, decode text={engine_decode['text']}")
        print("==========baseline==========")
        print(f"prefill token={baseline_first_token}, decode token={baseline_second_token}")
        assert engine_prefill["text"] == f"token-{baseline_first_token}"
        assert engine_decode["text"] == f"token-{baseline_second_token}"
    finally:
        if request_id in engine._states:
            engine.cancel(request_id)
    print(f"Mage-VL stepwise text streaming matches the Transformers prefill/decode baseline on {test_device}.")
