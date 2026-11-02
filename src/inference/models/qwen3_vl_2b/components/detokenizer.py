"""把连续生成的 Qwen3-VL token IDs 稳定地转换为文本 delta。"""

from inference.network.output_processor import IncrementalDetokenizer


class Qwen3VLDetokenizer(IncrementalDetokenizer):
    """缓冲跨 token 的不完整 UTF-8/Unicode 片段，避免 SSE 输出乱码。"""
