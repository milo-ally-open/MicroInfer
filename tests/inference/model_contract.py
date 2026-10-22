"""以 AST 檢查模型 Runner／Adapter 契約，不匯入模型程式或載入權重。"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest import TestCase


RUNNER_METHODS = {
    "load",
    "gpu_status",
    "close",
    "begin_generation",
    "bind_generation_cache",
    "execute_batch",
    "finish_generation",
    "cancel_generation",
}
ADAPTER_METHODS = {
    "health_info",
    "generation_options",
    "prepare_chat_generation",
    "bind_generation_cache",
    "execute_batch",
    "finish_chat_generation",
    "cancel_generation",
}


def _class_methods(source_path: Path, class_name: str) -> set[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"{class_name} is not declared in {source_path}")


def _factory_source(inference_root: Path, factory: str) -> tuple[Path, str]:
    module_name, separator, class_name = factory.partition(":")
    if not separator or not module_name or not class_name:
        raise AssertionError(f"invalid model factory path: {factory}")
    return inference_root.joinpath(*module_name.split(".")).with_suffix(".py"), class_name


def assert_model_contract(
    testcase: TestCase,
    inference_root: Path,
    *,
    profile: str,
    runner_factory: str | None,
    adapter_factory: str | None,
    input_processor_factory: str | None = None,
    output_mode: str = "text_delta",
    video_upload: bool,
    live_video: bool,
) -> None:
    """以同一份測試骨架檢查標準預設元件或自訂模型工廠。"""
    testcase.assertIn(profile, {"standard", "custom"})
    if profile == "standard":
        runner_factory = runner_factory or "engine.standard_text:StandardTextModelRunner"
        adapter_factory = adapter_factory or "engine.standard_text:StandardTextModelAdapter"
        if output_mode == "token_ids" and input_processor_factory is None:
            input_processor_factory = "api.default_chat_processor:DefaultChatInputProcessor"
    testcase.assertIsNotNone(runner_factory)
    testcase.assertIsNotNone(adapter_factory)
    runner_path, runner_class = _factory_source(inference_root, runner_factory)
    adapter_path, adapter_class = _factory_source(inference_root, adapter_factory)
    runner_methods = _class_methods(runner_path, runner_class)
    adapter_methods = _class_methods(adapter_path, adapter_class)

    testcase.assertFalse(RUNNER_METHODS - runner_methods, f"Runner missing methods: {RUNNER_METHODS - runner_methods}")
    testcase.assertFalse(ADAPTER_METHODS - adapter_methods, f"Adapter missing methods: {ADAPTER_METHODS - adapter_methods}")
    if video_upload:
        testcase.assertTrue({"prepare_video_generation", "finish_video_generation"} <= adapter_methods)
    if live_video:
        testcase.assertTrue(
            {"prepare_live_frames_generation", "prepare_live_video_generation", "finish_live_generation"}
            <= adapter_methods
        )
    if input_processor_factory:
        processor_path, processor_class = _factory_source(inference_root, input_processor_factory)
        processor_methods = _class_methods(processor_path, processor_class)
        required_processor_methods = {"start", "prepare_chat", "close"}
        if output_mode == "token_ids":
            required_processor_methods |= {"decode_tokens", "create_stream_output_processor"}
        testcase.assertFalse(
            required_processor_methods - processor_methods,
            f"InputProcessor missing methods: {required_processor_methods - processor_methods}",
        )
