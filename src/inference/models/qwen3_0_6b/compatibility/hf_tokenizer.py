"""載入 Hugging Face Tokenizer，保留模型原生 chat template。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_tokenizer(snapshot: Path, revision: str) -> Any:
    """從本機快照建立 Fast Tokenizer，禁止執行遠端自訂程式碼。"""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        snapshot,
        revision=revision,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )

