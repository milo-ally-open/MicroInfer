"""将 Qwen3-VL Transformers Runner 和 Engine Adapter 组合为 Runtime。"""

from inference.engine.worker import ModelRuntime

from .components.adapter import Qwen3VLModelAdapter
from .components.model_runner import Qwen3VLModelRunner


def create_runtime() -> ModelRuntime:
    runner = Qwen3VLModelRunner()
    return ModelRuntime(runner, Qwen3VLModelAdapter(runner))
