"""以共用 Engine Core 與 Worker 裝配登錄表中的模型元件。"""

from __future__ import annotations

from engine.engine import InferenceEngine
from engine.executor import LocalProcessExecutor
from engine.model_config import (
    load_factory,
    model_api_settings,
    model_component,
    selected_config,
    selected_model,
    validate_runtime_components,
)
from engine.worker import InProcessModelWorker


def create_model_worker() -> InProcessModelWorker:
    """只在 GPU Worker 子程序載入模型 Runner 與 Adapter。"""
    _, model_spec = selected_model()
    profile = str(model_spec.get("integration_profile", "custom"))
    runner_path = model_component(model_spec, "runner", required=profile != "standard")
    adapter_path = model_component(model_spec, "adapter", required=profile != "standard")
    if runner_path:
        runner = load_factory(runner_path)()
    elif profile == "standard":
        from engine.standard_text import StandardTextModelRunner

        runner = StandardTextModelRunner()
    else:
        raise ValueError("Custom model registration must define components.runner.")
    if adapter_path:
        adapter = load_factory(adapter_path)(runner)
    elif profile == "standard":
        from engine.standard_text import StandardTextModelAdapter

        adapter = StandardTextModelAdapter(runner)
    else:
        raise ValueError("Custom model registration must define components.adapter.")
    validate_runtime_components(model_spec, runner, adapter)
    return InProcessModelWorker(runner, adapter)


def create_engine_core() -> InferenceEngine:
    """由 YAML 建立共用 Scheduler、KV 管理器及獨立 Model Worker。"""
    model_alias, model_spec = selected_model()
    config = selected_config("INFERENCE_CONFIG")
    runtime = config.get("runtime", {})
    api_config = config.get("api", {})
    engine_config = config.get("engine", {})
    if not all(isinstance(section, dict) for section in (runtime, api_config, engine_config)):
        raise ValueError("runtime, api, engine and engine.kv_cache settings must be mappings.")
    cache_config = engine_config.get("kv_cache", {})
    if not isinstance(cache_config, dict):
        raise ValueError("engine.kv_cache settings must be a mapping.")

    block_size = int(cache_config.get("block_size_tokens", 16))
    num_blocks = cache_config.get("num_blocks")
    capacity = cache_config.get("capacity_tokens")
    if num_blocks is not None:
        block_capacity = int(num_blocks) * block_size
        if capacity is not None and int(capacity) != block_capacity:
            raise ValueError("engine.kv_cache.num_blocks conflicts with capacity_tokens.")
        capacity = block_capacity
    if capacity is None:
        capacity = int(api_config.get("max_context_tokens", 8192)) * int(
            engine_config.get("max_num_seqs", api_config.get("max_concurrent_requests", 4))
        )

    api_settings = model_api_settings(model_spec)
    executor = LocalProcessExecutor(
        "engine.model_bootstrap:create_model_worker",
        startup_timeout_seconds=float(runtime.get("worker_startup_timeout_seconds", 2400)),
        request_timeout_seconds=float(runtime.get("worker_request_timeout_seconds", 1800)),
    )
    return InferenceEngine(
        executor,
        model_id=str(api_settings.get("model_id") or model_alias),
        max_concurrent_requests=int(
            engine_config.get("max_num_seqs", api_config.get("max_concurrent_requests", 4))
        ),
        max_waiting_requests=int(engine_config.get("max_waiting_requests", 16)),
        max_num_scheduled_tokens=int(
            engine_config.get("max_num_scheduled_tokens", api_config.get("max_context_tokens", 8192))
        ),
        prefill_chunk_size_tokens=(
            int(engine_config["prefill_chunk_size_tokens"])
            if engine_config.get("prefill_chunk_size_tokens") is not None else None
        ),
        batch_wait_seconds=float(engine_config.get("batch_wait_seconds", 0.01)),
        kv_cache_block_size_tokens=block_size,
        kv_cache_capacity_tokens=int(capacity),
        max_video_bytes=int(api_config.get("max_video_bytes", 1)),
    )
