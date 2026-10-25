"""提供符合逐步 Cache 契约之标准文字因果语言模型 Runner／Adapter。"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from typing import Any

import torch

from inference.engine.sequence import (
    ExecutionPhase,
    InferenceResult,
    PreparedChatInput,
    ScheduledWorkItem,
    WorkerResult,
)
from inference.engine.sequence import InferenceError
from inference.engine.config import model_source, selected_config, selected_model
from inference.engine.platform import Platform, create_platform


logger = logging.getLogger(__name__)


class StandardTextModelRunner:
    """使用上游 Transformers causal LM forward 的单装置逐步 Runner。

    Runner 每次 Worker batch 依序执行各 work item；Cache 由模型提供并按 request ID
    持有。Engine KV block 在此路径仅代表容量租约，并非实体 paged attention。
    """

    def __init__(self, platform: Platform | None = None) -> None:
        self.platform = platform
        self.model: Any = None
        self.model_config: Any = None
        self.device: Any = None
        self._states: dict[str, dict[str, Any]] = {}
        self._forward_parameters: set[str] = set()
        self._model_id = ""
        self._revision = ""
        self._block_size_tokens = 16

    def load(self) -> None:
        """依模型 YAML 载入单装置 AutoModelForCausalLM 与设定。"""
        config = selected_config("INFERENCE_CONFIG")
        model_settings = config.get("model", {})
        runtime = config.get("runtime", {})
        engine = config.get("engine", {})
        if not all(isinstance(value, dict) for value in (model_settings, runtime, engine)):
            raise ValueError("model, runtime and engine settings must be mappings.")
        if model_settings.get("device_map") not in (None, ""):
            raise ValueError("The standard text Runner supports one configured device; use custom profile for device_map.")

        from transformers import AutoModelForCausalLM

        self._model_id = str(model_settings["id"])
        self._revision = str(model_settings["revision"])
        self._block_size_tokens = int(engine.get("kv_cache", {}).get("block_size_tokens", 16))
        if self._block_size_tokens < 1:
            raise ValueError("engine.kv_cache.block_size_tokens must be positive.")
        platform_name = str(runtime.get("platform", "torch_cuda"))
        self.platform = self.platform or create_platform(platform_name)
        requested_device = str(model_settings.get("device", runtime.get("device", "cuda:0")))
        self.device = self.platform.resolve_device(requested_device)
        dtype_name = str(model_settings.get("dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name, None)
        if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(f"Unsupported standard model dtype: {dtype_name}.")

        source = model_source(config)
        runtime_local_only = runtime.get("local_files_only", True)
        if not isinstance(runtime_local_only, bool):
            raise ValueError("runtime.local_files_only must be a boolean.")
        load_options: dict[str, Any] = {
            "revision": self._revision,
            "local_files_only": runtime_local_only,
            "trust_remote_code": bool(model_settings.get("trust_remote_code", False)),
            "dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        attn_implementation = model_settings.get("attn_implementation")
        if attn_implementation:
            load_options["attn_implementation"] = str(attn_implementation)
        try:
            model = AutoModelForCausalLM.from_pretrained(source, **load_options)
            if bool(getattr(model.config, "is_encoder_decoder", False)):
                raise ValueError("The standard text Runner only supports causal decoder models.")
            self.model = model.to(self.device).eval()
            self.model_config = self.model.config
            parameters = inspect.signature(self.model.forward).parameters
            self._forward_parameters = set(parameters)
        except BaseException:
            try:
                self.close()
            except BaseException:
                logger.exception("Could not release standard text model resources after loading failed")
            raise

    def begin_generation(self, request_id: str, inputs: dict[str, Any], options: dict[str, Any]) -> None:
        if self.model is None or self.device is None:
            raise RuntimeError("The standard text model is not loaded.")
        if request_id in self._states:
            raise ValueError(f"Generation request already exists: {request_id}")
        input_ids = torch.as_tensor(inputs.get("input_ids"), dtype=torch.long, device="cpu")
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[-1] < 1:
            raise ValueError("The standard text Runner requires one non-empty, unpadded prompt.")
        max_new_tokens = int(options.get("max_new_tokens", 0))
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        eos_token_ids = options.get("eos_token_ids")
        if eos_token_ids is None:
            eos_token_ids = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if eos_token_ids is None:
            eos_token_ids = getattr(self.model_config, "eos_token_id", None)
        if not isinstance(eos_token_ids, (list, tuple, set)):
            eos_token_ids = [] if eos_token_ids is None else [eos_token_ids]
        self._states[request_id] = {
            "input_ids": input_ids,
            "options": dict(options),
            "generated_ids": [],
            "past_key_values": None,
            "prefill_processed_tokens": 0,
            "cache_length": 0,
            "block_ids": (),
            "eos_token_ids": {int(value) for value in eos_token_ids},
            "finish_reason": "stop",
            "done": False,
        }

    def bind_generation_cache(self, request_id: str, block_ids: tuple[int, ...]) -> None:
        state = self._state(request_id)
        required_tokens = int(state["input_ids"].shape[-1]) + int(state["options"]["max_new_tokens"])
        if len(block_ids) * self._block_size_tokens < required_tokens:
            raise RuntimeError("Engine KV lease is smaller than the generation's maximum sequence length.")
        state["block_ids"] = tuple(block_ids)

    def prefill_min_chunk_tokens(self, request_id: str) -> int:
        self._state(request_id)
        return 1

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> tuple[WorkerResult, ...]:
        """依照共用 Worker batch 契约执行，保留每个 request 的独立 Cache。"""
        if not requests:
            raise ValueError("The standard text Runner requires at least one work item.")
        ids = [item.request_id for item in requests]
        if len(ids) != len(set(ids)):
            raise ValueError("A standard text batch cannot contain duplicate request IDs.")
        return tuple(self._execute_item(item) for item in requests)

    def _execute_item(self, item: ScheduledWorkItem) -> WorkerResult:
        state = self._state(item.request_id)
        if item.phase is ExecutionPhase.PREFILL:
            offset = int(item.prompt_offset)
            processed = int(state["prefill_processed_tokens"])
            if offset != processed or offset != int(state["cache_length"]):
                raise RuntimeError(f"Prefill offset is out of sequence for {item.request_id}.")
            end = offset + int(item.token_budget)
            prompt_length = int(state["input_ids"].shape[-1])
            if end > prompt_length:
                raise ValueError("Prefill work item exceeds the remaining prompt.")
            logits = self._forward(state, state["input_ids"][0, offset:end], offset)
            state["prefill_processed_tokens"] = end
            state["cache_length"] = end
            if end < prompt_length:
                return WorkerResult(request_id=item.request_id, prompt_tokens_processed=end)
            return self._sample_result(item.request_id, state, logits, prompt_tokens_processed=end)

        if item.phase is not ExecutionPhase.DECODE or item.token_budget != 1:
            raise ValueError("The standard text Runner supports prefill and one-token decode work items only.")
        if state["done"] or not state["generated_ids"]:
            raise RuntimeError(f"Request is not ready for decode: {item.request_id}.")
        reserved_tokens = len(state["block_ids"]) * self._block_size_tokens
        if int(state["cache_length"]) + 1 > reserved_tokens:
            raise RuntimeError("Decode exceeded the request's logical KV reservation.")
        token = torch.tensor([state["generated_ids"][-1]], dtype=torch.long)
        logits = self._forward(state, token, int(state["cache_length"]))
        state["cache_length"] += 1
        return self._sample_result(item.request_id, state, logits)

    def _forward(self, state: dict[str, Any], token_ids: torch.Tensor, position: int) -> torch.Tensor:
        if self.model is None or self.device is None:
            raise RuntimeError("The standard text model is not loaded.")
        tokens = token_ids.to(device=self.device, dtype=torch.long).reshape(1, -1)
        kwargs: dict[str, Any] = {
            "input_ids": tokens,
            "use_cache": True,
            "return_dict": True,
        }
        past = state["past_key_values"]
        if past is not None:
            kwargs["past_key_values"] = past
        full_length = position + int(tokens.shape[-1])
        kwargs["attention_mask"] = torch.ones((1, full_length), dtype=torch.long, device=self.device)
        if "cache_position" in self._forward_parameters:
            kwargs["cache_position"] = torch.arange(position, full_length, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(**kwargs)
        cache = getattr(output, "past_key_values", None)
        logits = getattr(output, "logits", None)
        if cache is None or logits is None:
            raise RuntimeError("The standard model must return logits and a reusable past_key_values Cache.")
        state["past_key_values"] = cache
        return logits[0, -1, :]

    def _sample_result(
        self,
        request_id: str,
        state: dict[str, Any],
        logits: torch.Tensor,
        *,
        prompt_tokens_processed: int | None = None,
    ) -> WorkerResult:
        scores = logits.float().clone()
        history = [int(value) for value in state["input_ids"][0].tolist()] + state["generated_ids"]
        penalty = float(state["options"].get("repetition_penalty", 1.0))
        if penalty != 1.0 and history:
            token_indices = torch.tensor(sorted(set(history)), dtype=torch.long, device=scores.device)
            selected = scores[token_indices]
            scores[token_indices] = torch.where(selected < 0, selected * penalty, selected / penalty)
        self._apply_no_repeat_ngram(
            scores,
            history,
            int(state["options"].get("no_repeat_ngram_size", 0)),
        )
        temperature = float(state["options"].get("temperature", 0.0))
        if temperature <= 0:
            token_id = int(torch.argmax(scores).item())
        else:
            scores /= temperature
            top_k = int(state["options"].get("top_k", 0))
            if 0 < top_k < scores.numel():
                cutoff = torch.topk(scores, top_k).values[-1]
                scores[scores < cutoff] = -torch.inf
            probabilities = torch.softmax(scores, dim=-1)
            top_p = float(state["options"].get("top_p", 1.0))
            if 0.0 < top_p < 1.0:
                sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
                remove = torch.cumsum(sorted_probabilities, dim=-1) > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                probabilities[sorted_indices[remove]] = 0
            min_p = state["options"].get("min_p")
            if min_p is not None and float(min_p) > 0:
                probabilities[probabilities < probabilities.max() * float(min_p)] = 0
            probability_sum = probabilities.sum()
            if not torch.isfinite(probability_sum) or probability_sum <= 0:
                token_id = int(torch.argmax(scores).item())
                probabilities = None
            else:
                probabilities = probabilities / probability_sum
            if probabilities is not None:
                token_id = int(torch.multinomial(probabilities, num_samples=1).item())
        state["generated_ids"].append(token_id)
        reached_limit = len(state["generated_ids"]) >= int(state["options"]["max_new_tokens"])
        stopped = token_id in state["eos_token_ids"]
        state["done"] = stopped or reached_limit
        if state["done"]:
            state["finish_reason"] = "length" if reached_limit and not stopped else "stop"
        return WorkerResult(
            request_id=request_id,
            token_id=token_id,
            generated_tokens=1,
            finished=state["done"],
            finish_reason=state["finish_reason"],
            prompt_tokens_processed=prompt_tokens_processed,
        )

    @staticmethod
    def _apply_no_repeat_ngram(scores: torch.Tensor, tokens: list[int], ngram_size: int) -> None:
        if ngram_size < 1 or len(tokens) < ngram_size - 1:
            return
        prefix = tuple(tokens[-(ngram_size - 1):]) if ngram_size > 1 else ()
        banned = {
            tokens[index + ngram_size - 1]
            for index in range(len(tokens) - ngram_size + 1)
            if ngram_size == 1 or tuple(tokens[index:index + ngram_size - 1]) == prefix
        }
        if banned:
            scores[list(banned)] = -torch.inf

    def finish_generation(self, request_id: str) -> tuple[torch.Tensor, str]:
        state = self._state(request_id)
        generated = torch.tensor(state["generated_ids"], dtype=torch.long)
        finish_reason = str(state["finish_reason"])
        self._states.pop(request_id, None)
        return generated, finish_reason

    def cancel_generation(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    def kv_cache_profile(self) -> dict[str, int]:
        if self.model is None or self.model_config is None:
            return {"bytes_per_token": 0}
        config = getattr(self.model_config, "text_config", self.model_config)
        layers = getattr(config, "num_hidden_layers", None)
        key_value_heads = getattr(config, "num_key_value_heads", None) or getattr(config, "num_attention_heads", None)
        hidden_size = getattr(config, "hidden_size", None)
        attention_heads = getattr(config, "num_attention_heads", None)
        head_dim = getattr(config, "head_dim", None) or (
            int(hidden_size) // int(attention_heads) if hidden_size and attention_heads else None
        )
        if not all(isinstance(value, int) and value > 0 for value in (layers, key_value_heads, head_dim)):
            return {"bytes_per_token": 0}
        element_bytes = next(self.model.parameters()).element_size()
        return {"bytes_per_token": int(2 * layers * key_value_heads * head_dim * element_bytes)}

    def cache_snapshot(self) -> dict[str, int]:
        return {
            "active_requests": len(self._states),
            "populated_caches": sum(state["cache_length"] > 0 for state in self._states.values()),
            "cached_tokens": sum(int(state["cache_length"]) for state in self._states.values()),
        }

    def gpu_status(self) -> dict[str, Any]:
        if self.model is None or self.platform is None or self.device is None:
            return {}
        return self.platform.gpu_status(self.device)

    def peak_memory(self) -> dict[str, int]:
        if self.platform is None or self.device is None:
            return {"peak_allocated": 0, "peak_reserved": 0}
        return self.platform.peak_memory(self.device)

    def reset_peak_memory(self) -> None:
        if self.platform is not None and self.device is not None:
            self.platform.reset_peak_memory(self.device)

    def close(self) -> None:
        self._states.clear()
        self.model = None
        self.model_config = None
        self.device = None
        if self.platform is not None:
            self.platform.release_cached_memory()

    def _state(self, request_id: str) -> dict[str, Any]:
        try:
            return self._states[request_id]
        except KeyError as exc:
            raise RuntimeError(f"Unknown standard text generation request: {request_id}") from exc


class StandardTextModelAdapter:
    """将共用 Engine 工作适配至标准文字 Runner。"""

    def __init__(self, model_runner: StandardTextModelRunner) -> None:
        self.model_runner = model_runner
        self._tasks: dict[str, int] = {}

    def health_info(self) -> dict[str, Any]:
        return {
            "backend": "standard_text",
            "model": self.model_runner._model_id,
            "revision": self.model_runner._revision,
            "kv_cache_profile": self.model_runner.kv_cache_profile(),
            "worker_kv_cache": self.model_runner.cache_snapshot(),
            "gpu_peak_memory": self.model_runner.peak_memory(),
        }

    def generation_options(self, request_body: dict[str, Any]) -> dict[str, Any]:
        return dict(request_body)

    def prepare_chat_generation(
        self,
        request_id: str,
        request_body: dict[str, Any],
        prepared_input: PreparedChatInput | None = None,
    ) -> dict[str, int]:
        del request_body
        if prepared_input is None:
            raise InferenceError(400, "The standard text profile requires its registered chat input processor.")
        alias, _ = selected_model()
        if prepared_input.model_alias != alias:
            raise InferenceError(400, "Prepared input model does not match the selected model.")
        options = dict(prepared_input.sampling_params)
        prompt_tokens = len(prepared_input.input_ids)
        self.model_runner.begin_generation(
            request_id,
            {"input_ids": prepared_input.input_ids},
            options,
        )
        self._tasks[request_id] = prompt_tokens
        return {
            "prompt_tokens": prompt_tokens,
            "max_new_tokens": int(options["max_new_tokens"]),
            "prefill_min_chunk_tokens": 1,
        }

    def bind_generation_cache(self, request_id: str, block_ids: tuple[int, ...]) -> None:
        self.model_runner.bind_generation_cache(request_id, block_ids)

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> tuple[WorkerResult, ...]:
        return self.model_runner.execute_batch(requests)

    def finish_chat_generation(self, request_id: str) -> InferenceResult:
        try:
            prompt_tokens = self._tasks[request_id]
        except KeyError as exc:
            raise RuntimeError(f"Unknown standard text chat request: {request_id}") from exc
        generated_ids, finish_reason = self.model_runner.finish_generation(request_id)
        result = InferenceResult(
            content="",
            prompt_tokens=prompt_tokens,
            completion_tokens=int(generated_ids.numel()),
            finish_reason=finish_reason,
            token_ids=tuple(int(value) for value in generated_ids.tolist()),
        )
        self._tasks.pop(request_id, None)
        return result

    def cancel_generation(self, request_id: str) -> None:
        self.model_runner.cancel_generation(request_id)
        self._tasks.pop(request_id, None)
