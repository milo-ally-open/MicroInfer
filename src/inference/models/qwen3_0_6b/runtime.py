"""Qwen3 模型执行入口；内部运算元件留在本模型目录。"""

from inference.engine.worker import ModelRuntime

from .components.adapter import Qwen3ModelAdapter
from .components.model_runner import Qwen3ModelRunner


def create_runtime() -> ModelRuntime:
    """建立供共用 Worker 使用的单一 Qwen3 Runtime。"""
    runner = Qwen3ModelRunner()
    return ModelRuntime(runner, Qwen3ModelAdapter(runner))
