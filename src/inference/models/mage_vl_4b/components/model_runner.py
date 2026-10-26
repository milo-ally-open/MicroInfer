"""Mage-VL Model Runner：载入模型、执行生成并解码结果。"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch

from inference.engine.sequence import ExecutionPhase, ScheduledWorkItem, WorkerResult
from ..utils import loader as config
from .component_registry import MAGE_VL_COMPONENT_REGISTRY
from ..utils.loader import load_mage_vl
from .stepwise_generation import MageVLStepwiseGeneration
from .decoder import MageVLDecoder
from .tokenizer import MageVLTokenizer


logger = logging.getLogger(__name__)


class MageVLModelRunner:
    """封装 Mage-VL 的模型载入与底层生成呼叫。"""

    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: MageVLTokenizer | None = None
        self.decoder: MageVLDecoder | None = None
        self.model_path: str | None = None
        self.component_inventory: dict[str, tuple[Any, ...]] = {}
        self._stepwise = MageVLStepwiseGeneration()

    def load(self) -> None:
        """载入固定 revision，检查模型元件并套用明确登记的本地替换。"""
        optimization = config.CONFIG.get("optimization", {})
        try:
            # 设定错误或未登记的实作必须在下载／载入大型权重前拒绝。
            MAGE_VL_COMPONENT_REGISTRY.validate_configuration(config.COMPONENT_REPLACEMENTS)
            self.model_path, self.model, self.processor = load_mage_vl()
            self.tokenizer = MageVLTokenizer(self.processor.tokenizer)
            self.decoder = MageVLDecoder(self.tokenizer)
            self.component_inventory = MAGE_VL_COMPONENT_REGISTRY.inventory(self.model)
            components = config.COMPONENT_REPLACEMENTS
            self.model, _ = MAGE_VL_COMPONENT_REGISTRY.apply(self.model, components)
            self._stepwise.bind(self.model, self.processor, self.tokenizer)
            compile_settings = optimization.get("torch_compile", {})
            if compile_settings.get("enabled", False):
                self.model = torch.compile(
                    self.model,
                    backend=str(compile_settings.get("backend", "inductor")),
                    mode=str(compile_settings.get("mode", "default")),
                    fullgraph=bool(compile_settings.get("fullgraph", False)),
                    dynamic=bool(compile_settings.get("dynamic", True)),
                )
                self._stepwise.bind(self.model, self.processor, self.tokenizer)
        except BaseException:
            try:
                self.close()
            except BaseException:
                logger.exception("Could not release Mage-VL resources after model loading failed")
            raise

    def begin_generation(self, request_id: str, inputs: dict[str, Any], generation_options: dict[str, Any]) -> None:
        self._stepwise.begin(request_id, inputs, generation_options)

    def bind_generation_cache(self, request_id: str, block_ids: tuple[int, ...]) -> None:
        """记录 Engine 分配的逻辑 block 表，供健康资讯与执行追踪使用。"""
        self._stepwise.bind_cache(request_id, block_ids)

    def prefill_min_chunk_tokens(self, request_id: str) -> int:
        return self._stepwise.prefill_min_chunk_tokens(request_id)

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> tuple[WorkerResult, ...]:
        """将同轮混合 work items 交给 Stepwise Runner 执行。"""
        if not requests:
            raise ValueError("Mage-VL batch execution requires at least one request.")
        request_ids = [request.request_id for request in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("Mage-VL batch request IDs must be unique.")
        for request in requests:
            if request.phase not in {ExecutionPhase.PREFILL, ExecutionPhase.DECODE}:
                raise ValueError(f"Mage-VL does not support execution phase: {request.phase.value}")
            if request.phase is ExecutionPhase.DECODE and request.token_budget != 1:
                raise ValueError("Mage-VL decode execution requires a one-token budget.")
        steps = self._stepwise.execute_work_batch(requests)
        if len(steps) != len(requests):
            raise RuntimeError("Mage-VL Stepwise Runner returned an incomplete batch result.")
        return tuple(
            WorkerResult(
                request_id=request.request_id,
                text_delta=str(step.get("text", "")),
                generated_tokens=int(step.get("generated_tokens", 0)),
                finished=bool(step.get("done", False)),
                finish_reason=step.get("finish_reason"),
                prompt_tokens_processed=(
                    int(step["prompt_tokens_processed"])
                    if step.get("prompt_tokens_processed") is not None else None
                ),
            )
            for request, step in zip(requests, steps)
        )

    def finish_generation(self, request_id: str) -> tuple[Any, str]:
        return self._stepwise.finish(request_id)

    def cancel_generation(self, request_id: str) -> None:
        self._stepwise.cancel(request_id)

    def kv_cache_profile(self) -> dict[str, int]:
        """估算每个文字 token 的 KV bytes，供 Engine 做保守容量预留。"""
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

    def kv_cache_snapshot(self) -> dict[str, int]:
        """回报 Worker 端实际持有的 Cache 物件、序列长度及关联逻辑 block 数。"""
        return self._stepwise.cache_snapshot()

    def decode(self, generated_ids: Any) -> str:
        if self.decoder is None:
            raise RuntimeError("Mage-VL Decoder is not loaded.")
        return self.decoder.decode(generated_ids)

    def gpu_status(self) -> dict[str, Any]:
        if self.model is None:
            return {}
        if not torch.cuda.is_available():
            return {"device": "cpu"}
        free, total = torch.cuda.mem_get_info()
        return {"gpu": torch.cuda.get_device_name(0), "vram_free_bytes": free, "vram_total_bytes": total}

    def peak_memory(self) -> dict[str, int]:
        if not torch.cuda.is_available():
            return {"peak_allocated": 0, "peak_reserved": 0}
        return {"peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved()}

    def reset_peak_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def close(self) -> None:
        self._stepwise.clear()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.decoder = None
        self.component_inventory.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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

    torch.manual_seed(83)
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
    model_config = MageVLConfig(text_config=text_config, vision_config=vision_config)
    engine_model = MageVLForConditionalGeneration(model_config).eval()
    baseline_model = TransformersQwen3ForCausalLM(text_config).eval()
    baseline_model.model.load_state_dict(engine_model.model.language_model.state_dict(), strict=True)
    baseline_model.lm_head.load_state_dict(engine_model.lm_head.state_dict(), strict=True)
    tokenizer = SimpleNamespace(
        eos_token_id=None,
        decode=lambda token_ids, **kwargs: "".join(f"token-{token_id}" for token_id in token_ids),
    )
    processor = SimpleNamespace(tokenizer=tokenizer)
    runner = MageVLModelRunner()
    runner.model = engine_model
    runner.processor = processor
    runner._stepwise.bind(engine_model, processor)
    request_id = "mage-runner-forward-check"
    prompt = torch.tensor([[2, 6, 10, 14]])
    options = {
        "max_new_tokens": 2,
        "temperature": 0.0,
        "do_sample": False,
        "use_cache": True,
        "eos_token_id": None,
    }
    runner.begin_generation(
        request_id,
        {"input_ids": prompt, "attention_mask": torch.ones_like(prompt)},
        options,
    )
    try:
        with torch.inference_mode():
            baseline_prefill = baseline_model(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                use_cache=True,
                return_dict=True,
            )
            engine_prefill = runner.execute_batch(
                (ScheduledWorkItem(request_id, ExecutionPhase.PREFILL, prompt.shape[-1]),)
            )[0]
            expected_first = int(torch.argmax(baseline_prefill.logits[0, -1]).item())
            assert engine_prefill.text_delta == f"token-{expected_first}"
            baseline_decode = baseline_model(
                input_ids=torch.tensor([[expected_first]]),
                attention_mask=torch.ones((1, prompt.shape[-1] + 1), dtype=torch.long),
                past_key_values=baseline_prefill.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            engine_decode = runner.execute_batch(
                (ScheduledWorkItem(request_id, ExecutionPhase.DECODE, 1),)
            )[0]
            expected_second = int(torch.argmax(baseline_decode.logits[0, -1]).item())
            assert engine_decode.text_delta == f"token-{expected_second}"
        print("==========engine===========")
        print(f"prefill delta={engine_prefill.text_delta}, decode delta={engine_decode.text_delta}")
        print("==========baseline==========")
        print(f"prefill token={expected_first}, decode token={expected_second}")
        assert engine_prefill.generated_tokens == engine_decode.generated_tokens == 1
        assert engine_decode.finished and engine_decode.finish_reason == "length"
    finally:
        runner.cancel_generation(request_id)
    print(f"Mage-VL ModelRunner text deltas match Transformers prefill/decode on {test_device}.")
