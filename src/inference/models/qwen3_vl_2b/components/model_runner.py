"""连接 Qwen3-VL 模型、Processor 与 Engine 的单序列执行入口。"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch
from transformers import AutoProcessor

from inference.engine.platform import Platform, create_platform
from inference.engine.sequence import ExecutionPhase, ScheduledWorkItem, WorkerResult

from ..modeling import Qwen3VLForConditionalGeneration
from .decoder import Qwen3VLDecoder
from .stepwise_generation import Qwen3VLStepwiseGeneration
from .tokenizer import Qwen3VLTokenizer


class Qwen3VLModelRunner:
    """管理模型运行生命周期；逐步生成使用 Transformers DynamicCache。"""

    def __init__(
        self,
        platform: Platform | None = None,
        block_size_tokens: int | None = None,
    ) -> None:
        self.platform = platform or create_platform("torch_cuda")
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Qwen3VLTokenizer | None = None
        self.decoder: Qwen3VLDecoder | None = None
        if block_size_tokens is None:
            # 服务启动时才读取模型 YAML；组件级小模型测试可直接注入 block size。
            from .. import loader as config

            block_size_tokens = int(config.ENGINE_CONFIG["kv_cache"]["block_size_tokens"])
        self._stepwise = Qwen3VLStepwiseGeneration(
            block_size_tokens=block_size_tokens,
        )

    def load(self) -> None:
        if self.model is not None:
            return

        from .. import loader as config

        snapshot = config.resolve_model_path()
        dtype = getattr(torch, config.MODEL_DTYPE, None)
        supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        if dtype not in supported_dtypes:
            raise ValueError(f"Unsupported Qwen3-VL dtype: {config.MODEL_DTYPE}.")

        device = self.platform.resolve_device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.processor = AutoProcessor.from_pretrained(
            snapshot,
            revision=config.MODEL_REVISION,
            local_files_only=True,
        )
        # 模型拓扑在同级 modeling.py 中显式由本地组件组装；这里仅加载权重。
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

    def begin_generation(
        self,
        request_id: str,
        inputs: dict[str, Any],
        options: dict[str, Any],
    ) -> None:
        self._stepwise.begin(request_id, inputs, options)

    def bind_generation_cache(
        self,
        request_id: str,
        block_ids: tuple[int, ...],
    ) -> None:
        self._stepwise.bind_cache(request_id, block_ids)

    def prefill_min_chunk_tokens(self, request_id: str) -> int:
        return self._stepwise.prefill_min_chunk_tokens(request_id)

    def execute_batch(
        self,
        requests: Sequence[ScheduledWorkItem],
    ) -> tuple[WorkerResult, ...]:
        if len(requests) != 1:
            raise RuntimeError(
                "Qwen3-VL currently requires max_num_seqs=1 "
                "to isolate its model-level MRoPE state."
            )

        item = requests[0]
        supported_phases = {ExecutionPhase.PREFILL, ExecutionPhase.DECODE}
        if item.phase not in supported_phases:
            raise ValueError(f"Unsupported Qwen3-VL phase: {item.phase.value}.")

        result = self._stepwise.execute(
            item.request_id,
            item.phase.value,
            item.token_budget,
            item.prompt_offset,
        )
        worker_result = WorkerResult(
            request_id=item.request_id,
            text_delta=str(result["text"]),
            generated_tokens=int(result["generated_tokens"]),
            finished=bool(result["done"]),
            finish_reason=result["finish_reason"] if result["done"] else None,
            prompt_tokens_processed=int(result["prompt_tokens_processed"]),
        )
        return (worker_result,)

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
        head_dim = int(
            getattr(
                text_config,
                "head_dim",
                text_config.hidden_size // text_config.num_attention_heads,
            )
        )
        bytes_per_token = (
            2
            * int(text_config.num_hidden_layers)
            * int(text_config.num_key_value_heads)
            * head_dim
            * next(self.model.parameters()).element_size()
        )
        return {"bytes_per_token": bytes_per_token}

    def cache_snapshot(self) -> dict[str, int]:
        return self._stepwise.cache_snapshot()

    def gpu_status(self) -> dict[str, Any]:
        if not torch.cuda.is_available():
            return {"device": "cpu"}

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "gpu": torch.cuda.get_device_name(0),
            "vram_free_bytes": free_bytes,
            "vram_total_bytes": total_bytes,
        }

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


if __name__ == "__main__":
    # 以小型 Qwen3-VL 和 Transformers 基线验证 Runner 对 Prefill/Decode 的编排。
    test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(103)
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

    # Runner 使用显式 block size，无需加载仓库模型权重或服务 YAML。
    runner = Qwen3VLModelRunner(block_size_tokens=8)
    runner.model = engine_model
    fake_tokenizer = SimpleNamespace(
        convert_tokens_to_ids=lambda token: -1,
        decode=lambda token_ids, **kwargs: "".join(f"token-{token_id}" for token_id in token_ids),
    )
    runner.processor = SimpleNamespace(tokenizer=fake_tokenizer)
    runner.tokenizer = Qwen3VLTokenizer(runner.processor)
    runner.decoder = Qwen3VLDecoder(runner.tokenizer)
    runner._stepwise.bind(engine_model, fake_tokenizer)

    prompt = torch.tensor([[3, 7, 11, 19]], device=test_device)
    attention_mask = torch.ones_like(prompt)
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
    request_id = "qwen3-vl-model-runner-forward-check"
    runner.begin_generation(
        request_id,
        {"input_ids": prompt, "attention_mask": attention_mask},
        options,
    )
    runner.bind_generation_cache(request_id, (0,))

    # Test：通过 Runner 的调度入口逐步执行，并和官方模型各自的前向结果核对。
    with torch.inference_mode():
        reference_prefill = reference_model(
            input_ids=prompt,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        engine_prefill = runner.execute_batch(
            (ScheduledWorkItem(request_id, ExecutionPhase.PREFILL, prompt.shape[-1]),)
        )[0]
        expected_first = int(torch.argmax(reference_prefill.logits[0, -1]).item())
        assert engine_prefill.generated_tokens == 1
        assert engine_prefill.text_delta == f"token-{expected_first}"
        assert runner.cache_snapshot()["cached_tokens"] == prompt.shape[-1]

        full_input_ids = torch.cat((prompt, prompt.new_tensor([[expected_first]])), dim=-1)
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
        engine_decode = runner.execute_batch(
            (ScheduledWorkItem(request_id, ExecutionPhase.DECODE, 1),)
        )[0]
        assert engine_decode.generated_tokens == 1
        assert engine_decode.text_delta == f"token-{expected_second}"
        generated_ids, finish_reason = runner.finish_generation(request_id)
    torch.testing.assert_close(
        generated_ids,
        torch.tensor([expected_first, expected_second], device="cpu"),
    )
    assert finish_reason == "length"
    print(f"Qwen3VLModelRunner Prefill/Decode parity passed on {test_device}.")

    # Bench：每轮走一次真实 Runner 请求生命周期，覆盖 Prefill 和单 token Decode。
    for warmup_index in range(2):
        warmup_id = f"runner-warmup-{warmup_index}"
        runner.begin_generation(warmup_id, {"input_ids": prompt, "attention_mask": attention_mask}, options)
        runner.bind_generation_cache(warmup_id, (0,))
        runner.execute_batch((ScheduledWorkItem(warmup_id, ExecutionPhase.PREFILL, prompt.shape[-1]),))
        runner.execute_batch((ScheduledWorkItem(warmup_id, ExecutionPhase.DECODE, 1),))
        runner.finish_generation(warmup_id)
    if test_device.type == "cuda":
        torch.cuda.synchronize(test_device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for run_index in range(10):
            benchmark_id = f"runner-bench-{run_index}"
            runner.begin_generation(
                benchmark_id,
                {"input_ids": prompt, "attention_mask": attention_mask},
                options,
            )
            runner.bind_generation_cache(benchmark_id, (0,))
            runner.execute_batch((ScheduledWorkItem(benchmark_id, ExecutionPhase.PREFILL, prompt.shape[-1]),))
            runner.execute_batch((ScheduledWorkItem(benchmark_id, ExecutionPhase.DECODE, 1),))
            runner.finish_generation(benchmark_id)
        end_event.record()
        end_event.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event) / 10
    else:
        started_at = time.perf_counter()
        for run_index in range(10):
            benchmark_id = f"runner-bench-{run_index}"
            runner.begin_generation(
                benchmark_id,
                {"input_ids": prompt, "attention_mask": attention_mask},
                options,
            )
            runner.bind_generation_cache(benchmark_id, (0,))
            runner.execute_batch((ScheduledWorkItem(benchmark_id, ExecutionPhase.PREFILL, prompt.shape[-1]),))
            runner.execute_batch((ScheduledWorkItem(benchmark_id, ExecutionPhase.DECODE, 1),))
            runner.finish_generation(benchmark_id)
        elapsed_ms = (time.perf_counter() - started_at) * 1000 / 10
    print(f"Bench: Qwen3VLModelRunner Prefill+Decode={elapsed_ms:.3f} ms/request on {test_device}.")

    # Profile：从 Worker Runner 公开执行接口采集实际模型算子和显存活动。
    profile_id = "qwen3-vl-model-runner-profile"
    runner.begin_generation(profile_id, {"input_ids": prompt, "attention_mask": attention_mask}, options)
    runner.bind_generation_cache(profile_id, (0,))
    profile_activities = [torch.profiler.ProfilerActivity.CPU]
    if test_device.type == "cuda":
        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.inference_mode():
        with torch.profiler.profile(
            activities=profile_activities,
            record_shapes=True,
            profile_memory=True,
        ) as runner_profile:
            runner.execute_batch((ScheduledWorkItem(profile_id, ExecutionPhase.PREFILL, prompt.shape[-1]),))
            runner.execute_batch((ScheduledWorkItem(profile_id, ExecutionPhase.DECODE, 1),))
    runner.finish_generation(profile_id)
    if test_device.type == "cuda":
        torch.cuda.synchronize(test_device)
        profile_sort = "self_cuda_time_total"
    else:
        profile_sort = "self_cpu_time_total"
    print("Profile: Qwen3VLModelRunner Prefill + Decode operators")
    print(runner_profile.key_averages().table(sort_by=profile_sort, row_limit=20))
