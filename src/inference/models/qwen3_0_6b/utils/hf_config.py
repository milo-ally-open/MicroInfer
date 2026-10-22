"""讀取並驗證本機固定 revision 的 Qwen3 設定。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def verify_snapshot_revision(snapshot: Path, revision: str) -> None:
    """確認本機快照的 Hugging Face local-dir metadata 屬於固定 commit。"""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Qwen3 revision must be a full 40-character Hugging Face commit hash.")
    metadata_dir = snapshot / ".cache" / "huggingface" / "download"
    metadata_files = tuple(metadata_dir.glob("*.metadata"))
    if not metadata_files:
        raise FileNotFoundError(
            f"Hugging Face revision metadata is missing from the local snapshot: {metadata_dir}"
        )

    required_files = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    required_files.update(path.name for path in snapshot.glob("*.safetensors"))
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        required_files.add(index_path.name)
    missing_assets = sorted(name for name in required_files if not (snapshot / name).is_file())
    if missing_assets:
        raise FileNotFoundError(
            "Required files are missing from the local Qwen3 snapshot: " + ", ".join(missing_assets)
        )
    metadata_by_name = {path.name.removesuffix(".metadata"): path for path in metadata_files}
    missing = sorted(required_files - metadata_by_name.keys())
    if missing:
        raise FileNotFoundError(
            "Hugging Face revision metadata is missing for local snapshot assets: "
            + ", ".join(missing)
        )

    for path in metadata_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0].strip() != revision:
            received = lines[0].strip() if lines else "<empty>"
            raise ValueError(
                f"Qwen3 snapshot asset {path.name.removesuffix('.metadata')} belongs to revision "
                f"{received}, expected {revision}."
            )


def load_model_config(snapshot: Path, revision: str) -> Any:
    """透過 Hugging Face config API 載入本機 Qwen3 設定，不執行遠端程式碼。"""
    from transformers import AutoConfig

    verify_snapshot_revision(snapshot, revision)
    config = AutoConfig.from_pretrained(
        snapshot,
        revision=revision,
        local_files_only=True,
        trust_remote_code=False,
    )
    if getattr(config, "model_type", None) != "qwen3":
        raise ValueError(f"Expected model_type=qwen3, received {getattr(config, 'model_type', None)!r}.")
    required = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
    )
    missing = [name for name in required if getattr(config, name, None) is None]
    if missing:
        raise ValueError(f"Qwen3 config is missing required fields: {', '.join(missing)}.")
    if int(config.hidden_size) % int(config.num_attention_heads):
        raise ValueError("Qwen3 hidden_size must be divisible by num_attention_heads.")
    if int(config.num_attention_heads) % int(config.num_key_value_heads):
        raise ValueError("Qwen3 query heads must be divisible by key/value heads.")
    if int(config.hidden_size) // int(config.num_attention_heads) % 2:
        raise ValueError("Qwen3 rotary head dimension must be even.")
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    rope_type = str(rope_parameters.get("rope_type", "default")) if isinstance(rope_parameters, dict) else "default"
    if rope_type != "default":
        raise ValueError("This Qwen3 runner supports the pinned default RoPE configuration only.")
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None and isinstance(rope_parameters, dict):
        rope_theta = rope_parameters.get("rope_theta")
    if rope_theta is None:
        raise ValueError("Qwen3 config is missing the default RoPE theta.")
    config.rope_theta = float(rope_theta)
    return config

