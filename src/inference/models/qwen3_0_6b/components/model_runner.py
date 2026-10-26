"""协调本地 Qwen3 计算图、批次排程、实体 KV pages 与逐 token 输出。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase, Qwen3Config

from inference.engine.sequence import ExecutionPhase, ScheduledWorkItem, WorkerResult
from inference.engine.platform import Platform, create_platform
from ..utils import loader as config
from ..utils.loader import load_model_config, load_safetensors, load_tokenizer
from .input_batch import InputBatch
from .paged_kv_cache import PagedKVCache
from ..model.qwen3 import Qwen3ForCausalLM
from .sampler import Qwen3Sampler


class Qwen3ModelRunner:
    """GPU Worker 内的 Qwen3 batch runner 与 request 状态拥有者。"""

    def __init__(self, platform: Platform | None = None) -> None:
        self.platform = platform or create_platform(config.PLATFORM_NAME)
        self.model: Qwen3ForCausalLM | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.model_config: Qwen3Config | None = None
        self.cache: PagedKVCache | None = None
        self._states: dict[str, dict[str, Any]] = {}
        self._sampler = Qwen3Sampler()

    def load(self) -> None:
        snapshot = config.LOCAL_MODEL_PATH
        if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
            raise FileNotFoundError(f"Qwen3 local snapshot is incomplete: {snapshot}")
        device = self.platform.resolve_device(config.MODEL_DEVICE)
        dtype = getattr(torch, config.MODEL_DTYPE, None)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"Unsupported Qwen3 model dtype: {config.MODEL_DTYPE}.")

        self.model_config = load_model_config(snapshot, config.MODEL_REVISION)
        self.tokenizer = load_tokenizer(snapshot, config.MODEL_REVISION)
        # 直接在目标设备建立空权重，再逐个 Safetensors tensor 载入，避免 CPU 留一整份模型副本。
        model = Qwen3ForCausalLM(self.model_config, dtype=dtype, device="meta").to_empty(device=device)
        load_safetensors(model, snapshot)
        self.model = model.eval()

        head_dim = int(
            getattr(
                self.model_config,
                "head_dim",
                int(self.model_config.hidden_size) // int(self.model_config.num_attention_heads),
            )
        )
        self.cache = PagedKVCache(
            num_layers=int(self.model_config.num_hidden_layers),
            num_blocks=config.KV_CACHE_NUM_BLOCKS,
            block_size_tokens=config.KV_CACHE_BLOCK_SIZE_TOKENS,
            num_key_value_heads=int(self.model_config.num_key_value_heads),
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )

    @property
    def input_device(self) -> torch.device:
        if self.model is None:
            raise RuntimeError("Qwen3 model is not loaded.")
        return self.model.get_input_embeddings().weight.device

    def begin_generation(self, request_id: str, inputs: dict[str, Any], options: dict[str, Any]) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Qwen3 model is not loaded.")
        if request_id in self._states:
            raise ValueError(f"Generation request already exists: {request_id}")
        raw_input_ids = inputs["input_ids"]
        input_ids = torch.as_tensor(raw_input_ids, dtype=torch.long, device="cpu")
        attention_mask = inputs.get("attention_mask")
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Qwen3 chat generation requires one unpadded prompt per request.")
        if attention_mask is not None and not bool(torch.all(attention_mask == 1)):
            raise ValueError("Qwen3 chat prompts must not contain padding tokens.")
        eos_ids = getattr(self.model_config, "eos_token_id", None)
        if eos_ids is None:
            eos_ids = self.tokenizer.eos_token_id
        if not isinstance(eos_ids, (list, tuple, set)):
            eos_ids = [] if eos_ids is None else [eos_ids]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0:
            eos_ids = [*eos_ids, im_end_id]
        self._states[request_id] = {
            "input_ids": input_ids,
            "generated_ids": [],
            "prefill_processed_tokens": 0,
            "cache_length": 0,
            "block_ids": (),
            "options": dict(options),
            "eos_token_ids": {int(token_id) for token_id in eos_ids},
            "done": False,
            "finish_reason": "stop",
        }

    def bind_generation_cache(self, request_id: str, block_ids: tuple[int, ...]) -> None:
        state = self._state(request_id)
        if self.cache is None:
            raise RuntimeError("Qwen3 physical KV cache is not initialized.")
        required_tokens = int(state["input_ids"].shape[-1])
        required_blocks = (required_tokens + config.KV_CACHE_BLOCK_SIZE_TOKENS - 1) // config.KV_CACHE_BLOCK_SIZE_TOKENS
        if len(block_ids) < required_blocks:
            raise RuntimeError("Engine KV lease is smaller than the Qwen3 prompt.")
        self.cache.bind(request_id, block_ids)
        state["block_ids"] = tuple(block_ids)

    def prefill_min_chunk_tokens(self, request_id: str) -> int:
        self._state(request_id)
        return 1

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> tuple[WorkerResult, ...]:
        """依阶段合并序列 forward，并依原排程顺序回传 request 结果。"""
        if not requests:
            return ()
        results: dict[str, WorkerResult] = {}
        prefill_items = [item for item in requests if item.phase is ExecutionPhase.PREFILL]
        decode_items = [item for item in requests if item.phase is ExecutionPhase.DECODE]
        unsupported = [item.phase.value for item in requests if item.phase not in {ExecutionPhase.PREFILL, ExecutionPhase.DECODE}]
        if unsupported:
            raise ValueError(f"Unsupported Qwen3 execution phase: {unsupported[0]}")
        if self.cache is None:
            raise RuntimeError("Qwen3 physical KV cache is not initialized.")
        for item in requests:
            state = self._state(item.request_id)
            blocks = tuple(item.cache_block_ids)
            if blocks:
                self.cache.extend(item.request_id, blocks)
                state["block_ids"] = blocks
        if prefill_items:
            results.update(self._execute_prefill_batch(prefill_items))
        if decode_items:
            results.update(self._execute_decode_batch(decode_items))
        return tuple(results[item.request_id] for item in requests)

    def _execute_prefill_batch(self, items: Sequence[ScheduledWorkItem]) -> dict[str, WorkerResult]:
        input_sequences: list[torch.Tensor] = []
        lengths: list[int] = []
        offsets: list[int] = []
        for item in items:
            state = self._state(item.request_id)
            offset = int(item.prompt_offset)
            processed = int(state["prefill_processed_tokens"])
            if offset != processed or offset != int(state["cache_length"]):
                raise RuntimeError(f"Qwen3 prefill offset is out of sequence for {item.request_id}.")
            end = offset + int(item.token_budget)
            prompt_length = int(state["input_ids"].shape[-1])
            if item.token_budget < 1 or end > prompt_length:
                raise ValueError("Qwen3 prefill chunk is outside the remaining prompt.")
            input_sequences.append(state["input_ids"][0, offset:end])
            lengths.append(int(item.token_budget))
            offsets.append(offset)
        logits = self._forward_batch(items, input_sequences, lengths, offsets)
        results: dict[str, WorkerResult] = {}
        for index, item in enumerate(items):
            state = self._state(item.request_id)
            end = offsets[index] + lengths[index]
            state["prefill_processed_tokens"] = end
            state["cache_length"] = end
            prompt_length = int(state["input_ids"].shape[-1])
            if end < prompt_length:
                results[item.request_id] = WorkerResult(
                    request_id=item.request_id,
                    prompt_tokens_processed=end,
                )
                continue
            results[item.request_id] = self._sample_result(
                item.request_id,
                state,
                logits[index],
                prompt_tokens_processed=end,
            )
        return results

    def _execute_decode_batch(self, items: Sequence[ScheduledWorkItem]) -> dict[str, WorkerResult]:
        if any(item.token_budget != 1 for item in items):
            raise ValueError("Qwen3 decode steps require a one-token budget.")
        input_sequences: list[torch.Tensor] = []
        lengths: list[int] = []
        offsets: list[int] = []
        for item in items:
            state = self._state(item.request_id)
            if state["done"] or not state["generated_ids"]:
                raise RuntimeError(f"Qwen3 request is not ready to decode: {item.request_id}.")
            offset = int(state["cache_length"])
            if offset + 1 > len(state["block_ids"]) * config.KV_CACHE_BLOCK_SIZE_TOKENS:
                raise RuntimeError("Qwen3 decode exceeded its reserved physical KV pages.")
            input_sequences.append(torch.tensor([state["generated_ids"][-1]], dtype=torch.long))
            lengths.append(1)
            offsets.append(offset)
        logits = self._forward_batch(items, input_sequences, lengths, offsets)
        results = {}
        for index, item in enumerate(items):
            state = self._state(item.request_id)
            state["cache_length"] = offsets[index] + 1
            results[item.request_id] = self._sample_result(item.request_id, state, logits[index])
        return results

    def _forward_batch(
        self,
        items: Sequence[ScheduledWorkItem],
        input_sequences: Sequence[torch.Tensor],
        lengths: Sequence[int],
        offsets: Sequence[int],
    ) -> torch.Tensor:
        if self.model is None or self.cache is None:
            raise RuntimeError("Qwen3 model and physical KV cache must be loaded before execution.")
        if not (len(items) == len(input_sequences) == len(lengths) == len(offsets)):
            raise ValueError("Qwen3 batch tensors and work items must have matching lengths.")
        batch = InputBatch.build(
            items,
            input_sequences,
            lengths,
            offsets,
            [self._state(item.request_id)["block_ids"] for item in items],
            device=self.input_device,
        )
        with torch.inference_mode():
            return self.model.forward_batch(
                batch.input_ids,
                token_lengths=batch.token_lengths,
                position_offsets=batch.position_offsets,
                request_ids=batch.request_ids,
                block_tables=batch.block_tables,
                cache=self.cache,
            )

    def _sample_result(
        self,
        request_id: str,
        state: dict[str, Any],
        logits: torch.Tensor,
        *,
        prompt_tokens_processed: int | None = None,
    ) -> WorkerResult:
        prompt_tokens = state["input_ids"][0].tolist()
        token_id = self._sampler.sample(
            logits,
            prompt_tokens=prompt_tokens,
            generated_tokens=state["generated_ids"],
            options=state["options"],
        )
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

    def finish_generation(self, request_id: str) -> tuple[torch.Tensor, str]:
        state = self._state(request_id)
        generated = torch.tensor(state["generated_ids"], dtype=torch.long)
        finish_reason = str(state["finish_reason"])
        if self.cache is not None:
            self.cache.release(request_id)
        self._states.pop(request_id, None)
        return generated, finish_reason

    def cancel_generation(self, request_id: str) -> None:
        if self.cache is not None:
            self.cache.release(request_id)
        self._states.pop(request_id, None)

    def kv_cache_profile(self) -> dict[str, int]:
        if self.model_config is None or self.cache is None:
            return {"bytes_per_token": 0}
        head_dim = int(
            getattr(
                self.model_config,
                "head_dim",
                int(self.model_config.hidden_size) // int(self.model_config.num_attention_heads),
            )
        )
        bytes_per_token = (
            2
            * int(self.model_config.num_hidden_layers)
            * int(self.model_config.num_key_value_heads)
            * head_dim
            * self.cache.storage.element_size()
        )
        return {
            "bytes_per_token": bytes_per_token,
            "physical_blocks": self.cache.num_blocks,
            "block_size_tokens": self.cache.block_size_tokens,
            "allocated_bytes": self.cache.bytes_allocated,
        }

    def cache_snapshot(self) -> dict[str, int]:
        snapshot = {
            "active_requests": len(self._states),
            "populated_caches": sum(state["cache_length"] > 0 for state in self._states.values()),
            "cached_tokens": sum(int(state["cache_length"]) for state in self._states.values()),
        }
        if self.cache is not None:
            snapshot.update(self.cache.snapshot())
        return snapshot

    def gpu_status(self) -> dict[str, Any]:
        if self.model is None:
            return {}
        return self.platform.gpu_status(self.input_device)

    def peak_memory(self) -> dict[str, int]:
        if self.model is None:
            return {"peak_allocated": 0, "peak_reserved": 0}
        return self.platform.peak_memory(self.input_device)

    def reset_peak_memory(self) -> None:
        if self.model is not None:
            self.platform.reset_peak_memory(self.input_device)

    def close(self) -> None:
        if self.cache is not None:
            for request_id in tuple(self._states):
                self.cache.release(request_id)
        self._states.clear()
        self.cache = None
        self.model = None
        self.model_config = None
        self.tokenizer = None
        self.platform.release_cached_memory()

    def _state(self, request_id: str) -> dict[str, Any]:
        try:
            return self._states[request_id]
        except KeyError as exc:
            raise RuntimeError(f"Unknown Qwen3 generation request: {request_id}") from exc


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
    from types import SimpleNamespace

    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3ForCausalLM as TransformersQwen3ForCausalLM,
    )

    torch.manual_seed(73)
    test_config = Qwen3Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        eos_token_id=None,
    )
    test_config.rope_theta = 1_000_000.0
    test_config.rope_parameters = {"rope_type": "default", "rope_theta": 1_000_000.0}
    test_config._attn_implementation = "eager"
    baseline = TransformersQwen3ForCausalLM(test_config).eval()
    engine_model = Qwen3ForCausalLM(test_config, dtype=torch.float32, device=test_device).eval()
    engine_model.load_state_dict(baseline.state_dict(), strict=True)
    runner = Qwen3ModelRunner(platform=SimpleNamespace())
    runner.model = engine_model
    runner.model_config = test_config
    runner.tokenizer = SimpleNamespace(eos_token_id=None, convert_tokens_to_ids=lambda _: -1)
    runner.cache = PagedKVCache(
        num_layers=test_config.num_hidden_layers,
        num_blocks=2,
        block_size_tokens=config.KV_CACHE_BLOCK_SIZE_TOKENS,
        num_key_value_heads=test_config.num_key_value_heads,
        head_dim=test_config.head_dim,
        device=test_device,
        dtype=torch.float32,
    )
    prompt = torch.tensor([[4, 8, 15, 16, 23]])
    options = {"max_new_tokens": 2, "temperature": 0.0, "repetition_penalty": 1.0}
    request_id = "runner-forward-check"
    runner.begin_generation(
        request_id,
        {"input_ids": prompt, "attention_mask": torch.ones_like(prompt)},
        options,
    )
    runner.bind_generation_cache(request_id, (0, 1))
    try:
        with torch.inference_mode():
            baseline_prefill = baseline(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                use_cache=True,
                return_dict=True,
            )
            engine_prefill = runner.execute_batch(
                (ScheduledWorkItem(request_id, ExecutionPhase.PREFILL, prompt.shape[-1]),)
            )[0]
            expected_prefill_token = int(torch.argmax(baseline_prefill.logits[0, -1]).item())
            torch.testing.assert_close(
                torch.tensor(engine_prefill.token_id), torch.tensor(expected_prefill_token)
            )
            next_token = torch.tensor([[engine_prefill.token_id]])
            baseline_decode = baseline(
                input_ids=next_token,
                attention_mask=torch.ones((1, prompt.shape[-1] + 1), dtype=torch.long),
                past_key_values=baseline_prefill.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            engine_decode = runner.execute_batch(
                (ScheduledWorkItem(request_id, ExecutionPhase.DECODE, 1),)
            )[0]
            expected_decode_token = int(torch.argmax(baseline_decode.logits[0, -1]).item())
            torch.testing.assert_close(
                torch.tensor(engine_decode.token_id), torch.tensor(expected_decode_token)
            )
        print("==========engine===========")
        print(f"prefill token={engine_prefill.token_id}, decode token={engine_decode.token_id}")
        print("==========baseline==========")
        print(f"prefill token={expected_prefill_token}, decode token={expected_decode_token}")
    finally:
        runner.cancel_generation(request_id)
    print(f"Qwen3ModelRunner prefill/decode matches the Transformers baseline on {test_device}.")

