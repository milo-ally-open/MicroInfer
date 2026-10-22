"""將固定 checkpoint 的 Safetensors 權重安全映射到本地模型計算圖。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _weight_files(snapshot: Path) -> tuple[Path, ...]:
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid Safetensors index: {index_path}")
        names = sorted(set(str(name) for name in weight_map.values()))
        files = tuple((snapshot / name).resolve() for name in names)
        if any(path.parent != snapshot.resolve() for path in files):
            raise ValueError("Safetensors index entries must stay inside the local model snapshot.")
    else:
        files = tuple(sorted(snapshot.glob("*.safetensors")))
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError(f"No complete Safetensors checkpoint was found in {snapshot}.")
    return files


def load_safetensors(model: torch.nn.Module, snapshot: Path) -> int:
    """載入且嚴格核對所有權重名稱與 shape，回傳已載入參數量。"""
    from safetensors.torch import load_file

    state: dict[str, torch.Tensor] = {}
    for path in _weight_files(snapshot):
        shard = load_file(str(path), device="cpu")
        duplicate = state.keys() & shard.keys()
        if duplicate:
            raise ValueError(f"Duplicate checkpoint tensor: {sorted(duplicate)[0]}.")
        state.update(shard)

    expected = model.state_dict()
    missing = expected.keys() - state.keys()
    unexpected = state.keys() - expected.keys()
    if missing or unexpected:
        raise ValueError(
            "Qwen3 checkpoint keys do not match the local model: "
            f"missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]}."
        )
    for name, parameter in expected.items():
        if state[name].shape != parameter.shape:
            raise ValueError(
                f"Qwen3 checkpoint shape mismatch for {name}: "
                f"expected {tuple(parameter.shape)}, received {tuple(state[name].shape)}."
            )
    result: Any = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"Qwen3 checkpoint load was incomplete: {result}.")
    del expected, state
    model.tie_weights()
    return sum(parameter.numel() for parameter in model.parameters())

