"""按生成顺序把 Qwen3 token IDs 增量还原为稳定文本片段。"""

from __future__ import annotations

from inference.network.output_processor import IncrementalDetokenizer


class Qwen3Detokenizer(IncrementalDetokenizer):
    """Qwen3 的流式输出边界；缓存由通用增量解码器管理。"""
