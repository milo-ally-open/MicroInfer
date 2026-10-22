"""Mage-VL Model Runner：載入模型、執行生成並解碼結果。"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from transformers import TextIteratorStreamer

from engine.contracts import ExecutionPhase, ScheduledWorkItem, WorkerResult
from . import config
from .components.component_registry import MAGE_VL_COMPONENT_REGISTRY
from .components.model_loader import load_mage_vl
from .components.stepwise_generation import MageVLStepwiseGeneration


class MageVLModelRunner:
    """封裝 Mage-VL 的模型載入與底層生成呼叫。"""

    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.model_path: str | None = None
        self.component_inventory: dict[str, tuple[Any, ...]] = {}
        self._stepwise = MageVLStepwiseGeneration()

    def load(self) -> None:
        """載入固定 revision，檢查模型元件並套用明確登記的本地替換。"""
        optimization = config.CONFIG.get("optimization", {})
        try:
            # 設定錯誤或未登記的實作必須在下載／載入大型權重前拒絕。
            MAGE_VL_COMPONENT_REGISTRY.validate_configuration(config.COMPONENT_REPLACEMENTS)
            self.model_path, self.model, self.processor = load_mage_vl()
            self.component_inventory = MAGE_VL_COMPONENT_REGISTRY.inventory(self.model)
            components = config.COMPONENT_REPLACEMENTS
            self.model, _ = MAGE_VL_COMPONENT_REGISTRY.apply(self.model, components)
            self._stepwise.bind(self.model, self.processor)
            compile_settings = optimization.get("torch_compile", {})
            if compile_settings.get("enabled", False):
                self.model = torch.compile(
                    self.model,
                    backend=str(compile_settings.get("backend", "inductor")),
                    mode=str(compile_settings.get("mode", "default")),
                    fullgraph=bool(compile_settings.get("fullgraph", False)),
                    dynamic=bool(compile_settings.get("dynamic", True)),
                )
                self._stepwise.bind(self.model, self.processor)
        except BaseException:
            self.close()
            raise

    def generate(self, inputs: dict[str, Any], generation_options: dict[str, Any]) -> Any:
        if self.model is None:
            raise RuntimeError("Mage-VL model is not loaded.")
        with torch.inference_mode():
            return self.model.generate(**inputs, **generation_options)

    def begin_generation(self, request_id: str, inputs: dict[str, Any], generation_options: dict[str, Any]) -> None:
        self._stepwise.begin(request_id, inputs, generation_options)

    def bind_generation_cache(self, request_id: str, block_ids: tuple[int, ...]) -> None:
        """記錄 Engine 分配的邏輯 block 表，供健康資訊與執行追蹤使用。"""
        self._stepwise.bind_cache(request_id, block_ids)

    def prefill_generation(self, request_id: str) -> dict[str, Any]:
        return self._stepwise.prefill(request_id)

    def decode_generation_step(self, request_id: str) -> dict[str, Any]:
        return self._stepwise.decode(request_id)

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> tuple[WorkerResult, ...]:
        """逐請求執行一個同階段批次；張量級多序列單次 forward 尚未啟用。"""
        if not requests:
            raise ValueError("Mage-VL batch execution requires at least one request.")
        request_ids = [request.request_id for request in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("Mage-VL batch request IDs must be unique.")
        results: list[WorkerResult] = []
        for request in requests:
            if request.phase is ExecutionPhase.PREFILL:
                step = self.prefill_generation(request.request_id)
            elif request.phase is ExecutionPhase.DECODE:
                if request.token_budget != 1:
                    raise ValueError("Mage-VL decode execution requires a one-token budget.")
                step = self.decode_generation_step(request.request_id)
            else:
                raise ValueError(f"Mage-VL does not support execution phase: {request.phase.value}")
            results.append(
                WorkerResult(
                    request_id=request.request_id,
                    text_delta=str(step.get("text", "")),
                    generated_tokens=int(step.get("generated_tokens", 0)),
                    finished=bool(step.get("done", False)),
                    finish_reason=step.get("finish_reason"),
                )
            )
        return tuple(results)

    def finish_generation(self, request_id: str) -> tuple[Any, str]:
        return self._stepwise.finish(request_id)

    def cancel_generation(self, request_id: str) -> None:
        self._stepwise.cancel(request_id)

    def kv_cache_profile(self) -> dict[str, int]:
        """估算每個文字 token 的 KV bytes，供 Engine 做保守容量預留。"""
        if self.model is None:
            return {"bytes_per_token": 0}
        text_config = getattr(self.model.config, "text_config", self.model.config)
        layers = getattr(text_config, "num_hidden_layers", None)
        kv_heads = getattr(text_config, "num_key_value_heads", None)
        attention_heads = getattr(text_config, "num_attention_heads", None)
        hidden_size = getattr(text_config, "hidden_size", None)
        head_dim = getattr(text_config, "head_dim", None)
        if head_dim is None and attention_heads and hidden_size:
            head_dim = hidden_size // attention_heads
        if not all(isinstance(value, int) and value > 0 for value in (layers, kv_heads, head_dim)):
            return {"bytes_per_token": 0}
        dtype_bytes = next(self.model.parameters()).element_size()
        return {"bytes_per_token": int(2 * layers * kv_heads * head_dim * dtype_bytes)}

    def stream(self, inputs: dict[str, Any], generation_options: dict[str, Any], timeout: float) -> Iterator[str]:
        """在背景執行模型生成，並依序輸出 Processor streamer 的文字片段。"""
        if self.model is None or self.processor is None:
            raise RuntimeError("Mage-VL model is not loaded.")
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=timeout)
        worker_error: list[BaseException] = []

        def run_generation() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(**inputs, streamer=streamer, **generation_options)
            except BaseException as exc:
                worker_error.append(exc)
                streamer.end()

        worker = threading.Thread(target=run_generation, daemon=True)
        worker.start()
        try:
            yield from streamer
            worker.join()
            if worker_error:
                raise worker_error[0]
        finally:
            if worker.is_alive():
                worker.join()

    def decode(self, generated_ids: Any) -> str:
        if self.processor is None:
            raise RuntimeError("Mage-VL processor is not loaded.")
        return self.processor.batch_decode(generated_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()

    def token_count(self, text: str) -> int:
        if self.processor is None:
            raise RuntimeError("Mage-VL processor is not loaded.")
        return len(self.processor.tokenizer.encode(text, add_special_tokens=False))

    def gpu_status(self) -> dict[str, Any]:
        if self.model is None or not torch.cuda.is_available():
            return {}
        free, total = torch.cuda.mem_get_info()
        return {"gpu": torch.cuda.get_device_name(0), "vram_free_bytes": free, "vram_total_bytes": total}

    def peak_memory(self) -> dict[str, int]:
        return {"peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved()}

    def reset_peak_memory(self) -> None:
        torch.cuda.reset_peak_memory_stats()

    def close(self) -> None:
        self._stepwise.clear()
        self.model = None
        self.processor = None
        self.component_inventory.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
