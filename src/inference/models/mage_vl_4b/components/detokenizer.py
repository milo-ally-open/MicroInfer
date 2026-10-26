"""按生成顺序把 Mage-VL token IDs 增量还原为稳定文本片段。"""

from __future__ import annotations

from inference.network.output_processor import IncrementalDetokenizer


class MageVLDetokenizer(IncrementalDetokenizer):
    """Mage-VL 的 SSE 文本边界，缓冲跨 token 的不完整字元。"""
