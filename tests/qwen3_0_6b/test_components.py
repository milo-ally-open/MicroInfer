#!/usr/bin/env python3
"""比较 Qwen3 本地模型计算图与 Transformers 原生模型结构。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM


def load_settings(path: Path) -> tuple[dict, Path]:
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("测试 YAML 最上层必须是 mapping。")
    snapshot = (path.parent / settings["snapshot"]).resolve()
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"本机模型设定不存在：{snapshot / 'config.json'}")
    return settings, snapshot


def show_tree(title: str, model: torch.nn.Module) -> None:
    print(title)
    print(f"model_class: {type(model).__module__}.{type(model).__name__}")
    for name, module in model.named_modules():
        if not name or name.count(".") <= 2:
            print(f"{name or '<root>'}: {type(module).__module__}.{type(module).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="明确指定测试 YAML。")
    args = parser.parse_args()
    settings, snapshot = load_settings(args.config.resolve())
    sys.path.insert(0, str((args.config.resolve().parent / settings["component"]["engine_module"]).resolve()))
    inference_config_path = (args.config.resolve().parent / settings["inference_config"]).resolve()
    inference_settings = yaml.safe_load(inference_config_path.read_text(encoding="utf-8"))
    os.environ["INFERENCE_CONFIG"] = str(inference_config_path)
    os.environ["INFERENCE_NETWORK_CONFIG"] = str(inference_config_path.with_name("network.yml"))
    dtype = getattr(torch, settings["component"].get("dtype", "bfloat16"))

    from models.qwen3_0_6b.utils.loader import load_model_config
    from models.qwen3_0_6b.model.qwen3 import Qwen3ForCausalLM

    config = load_model_config(snapshot, str(inference_settings["model"]["revision"]))
    with torch.device("meta"):
        engine_model = Qwen3ForCausalLM(config, dtype=dtype, device="meta")
        baseline_model = AutoModelForCausalLM.from_config(config)
    print("==========engine===========")
    show_tree("Qwen3 本地 Runner 计算图（meta device，不载入权重）", engine_model)
    print("==========baseline==========")
    show_tree("Transformers 原生 AutoModelForCausalLM（meta device，不载入权重）", baseline_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
