"""加载 Hugging Face Qwen3-VL 并实现 Engine Scheduler 的单序列执行入口。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from inference.engine.platform import Platform, create_platform
from inference.engine.sequence import ExecutionPhase, ScheduledWorkItem, WorkerResult
from .decoder import Qwen3VLDecoder
from .stepwise_generation import Qwen3VLStepwiseGeneration
from .tokenizer import Qwen3VLTokenizer
from ..utils import loader as config


class Qwen3VLModelRunner:
    """运行官方 Transformers 模型；Generation Cache 使用 HF DynamicCache。"""

    def __init__(self, platform: Platform | None = None) -> None:
        self.platform = platform or create_platform("torch_cuda")
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Qwen3VLTokenizer | None = None
        self.decoder: Qwen3VLDecoder | None = None
        block_size = int(config.ENGINE_CONFIG["kv_cache"]["block_size_tokens"])
        self._stepwise = Qwen3VLStepwiseGeneration(block_size_tokens=block_size)

    def load(self) -> None:
        if self.model is not None:
            return
        snapshot = config.resolve_model_path()
        dtype = getattr(torch, config.MODEL_DTYPE, None)
        if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(f"Unsupported Qwen3-VL dtype: {config.MODEL_DTYPE}.")
        device = self.platform.resolve_device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(
            snapshot,
            revision=config.MODEL_REVISION,
            local_files_only=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            snapshot,
            revision=config.MODEL_REVISION,
            local_files_only=True,
            use_safetensors=bool(config.MODEL_CONFIG["use_safetensors"]),
            weights_only=bool(config.MODEL_CONFIG["weights_only"]),
            dtype=dtype,
            device_map=0 if device.type == "cuda" else "cpu",
            attn_implementation=config.ATTENTION_IMPLEMENTATION,
        ).eval()
        self.tokenizer = Qwen3VLTokenizer(self.processor)
        self.decoder = Qwen3VLDecoder(self.tokenizer)
        self._stepwise.bind(self.model, self.tokenizer.tokenizer)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

    @property
    def input_device(self) -> torch.device:
        if self.model is None:
            raise RuntimeError("Qwen3-VL model is not loaded.")
        return self.model.get_input_embeddings().weight.device

    def begin_generation(self, request_id: str, inputs: dict[str, Any], options: dict[str, Any]) -> None:
        self._stepwise.begin(request_id, inputs, options)

    def bind_generation_cache(self, request_id: str, block_ids: tuple[int, ...]) -> None:
        self._stepwise.bind_cache(request_id, block_ids)

    def prefill_min_chunk_tokens(self, request_id: str) -> int:
        return self._stepwise.prefill_min_chunk_tokens(request_id)

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> tuple[WorkerResult, ...]:
        if len(requests) != 1:
            raise RuntimeError("Qwen3-VL currently requires max_num_seqs=1 to isolate its model-level MRoPE state.")
        item = requests[0]
        if item.phase not in {ExecutionPhase.PREFILL, ExecutionPhase.DECODE}:
            raise ValueError(f"Unsupported Qwen3-VL phase: {item.phase.value}.")
        result = self._stepwise.execute(
            item.request_id,
            item.phase.value,
            item.token_budget,
            item.prompt_offset,
        )
        return (WorkerResult(
            request_id=item.request_id,
            text_delta=str(result["text"]),
            generated_tokens=int(result["generated_tokens"]),
            finished=bool(result["done"]),
            finish_reason=result["finish_reason"] if result["done"] else None,
            prompt_tokens_processed=int(result["prompt_tokens_processed"]),
        ),)

    def finish_generation(self, request_id: str) -> tuple[torch.Tensor, str]:
        return self._stepwise.finish(request_id)

    def cancel_generation(self, request_id: str) -> None:
        self._stepwise.cancel(request_id)

    def decode(self, token_ids: Any) -> str:
        if self.decoder is None:
            raise RuntimeError("Qwen3-VL Decoder is not loaded.")
        return self.decoder.decode(token_ids)

    def kv_cache_profile(self) -> dict[str, int]:
        if self.model is None:
            return {"bytes_per_token": 0}
        text_config = self.model.config.text_config
        head_dim = int(getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads))
        bytes_per_token = (
            2 * int(text_config.num_hidden_layers) * int(text_config.num_key_value_heads)
            * head_dim * next(self.model.parameters()).element_size()
        )
        return {"bytes_per_token": bytes_per_token}

    def cache_snapshot(self) -> dict[str, int]:
        return self._stepwise.cache_snapshot()

    def gpu_status(self) -> dict[str, Any]:
        if not torch.cuda.is_available():
            return {"device": "cpu"}
        free, total = torch.cuda.mem_get_info()
        return {"gpu": torch.cuda.get_device_name(0), "vram_free_bytes": free, "vram_total_bytes": total}

    def peak_memory(self) -> dict[str, int]:
        if not torch.cuda.is_available():
            return {"peak_allocated": 0, "peak_reserved": 0}
        return {
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
        }

    def close(self) -> None:
        self._stepwise.clear()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.decoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
