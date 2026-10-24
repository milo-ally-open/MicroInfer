#!/usr/bin/env python3
"""比较 Qwen3 Gateway-facing Inference API 与 Transformers 原生生成延迟。"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

import torch
import yaml
from transformers import AutoModelForCausalLM


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="明确指定测试 YAML。")
    args = parser.parse_args()
    config_path = args.config.resolve()
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    snapshot = (config_path.parent / settings["snapshot"]).resolve()
    inference_config_path = (config_path.parent / settings["inference_config"]).resolve()
    inference_settings = yaml.safe_load(inference_config_path.read_text(encoding="utf-8"))
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"本机模型快照不存在：{snapshot}")
    import sys
    inference_root = (config_path.parent / settings["component"]["engine_module"]).resolve()
    sys.path.insert(0, str(inference_root))
    os.environ["INFERENCE_CONFIG"] = str(inference_config_path)
    os.environ["INFERENCE_NETWORK_CONFIG"] = str(inference_config_path.with_name("network.yml"))
    from models.qwen3_0_6b.utils.loader import load_model_config, load_tokenizer

    revision = str(inference_settings["model"]["revision"])
    model_config = load_model_config(snapshot, revision)
    tokenizer = load_tokenizer(snapshot, revision)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, config=model_config, revision=revision, local_files_only=True,
        torch_dtype=getattr(torch, settings["component"]["dtype"]), attn_implementation="eager",
    ).to("cuda").eval()
    messages = [{"role": "user", "content": settings["prompt"]}]
    tokenized = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", enable_thinking=False
    )
    input_ids = tokenized.input_ids if hasattr(tokenized, "input_ids") else tokenized
    input_ids = input_ids.to(model.device)
    request_body = {
        "model": settings["model_alias"], "messages": messages, "stream": False,
        "max_tokens": int(settings["max_new_tokens"]), "temperature": 0,
    }

    def api_once() -> tuple[float, int]:
        request = Request(
            settings["inference_url"].rstrip("/") + "/v1/chat/completions",
            data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        started = time.perf_counter()
        with urlopen(request, timeout=float(settings["timeout_seconds"])) as response:
            result = json.loads(response.read())
        elapsed = time.perf_counter() - started
        tokens = int(result.get("usage", {}).get("completion_tokens", 0))
        return elapsed, tokens

    def baseline_once() -> tuple[float, int]:
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(input_ids, max_new_tokens=int(settings["max_new_tokens"]), do_sample=False)
        torch.cuda.synchronize()
        return time.perf_counter() - started, int(output.shape[-1] - input_ids.shape[-1])

    for _ in range(int(settings["warmup_requests"])):
        api_once()
        baseline_once()
    measurements = {"engine_api": [], "transformers_baseline": []}
    for _ in range(int(settings["iterations"])):
        measurements["engine_api"].append(api_once())
        measurements["transformers_baseline"].append(baseline_once())
    report = {"model_alias": settings["model_alias"], "iterations": settings["iterations"]}
    for name, values in measurements.items():
        latencies = [item[0] for item in values]
        token_rates = [item[1] / item[0] if item[0] else 0 for item in values]
        report[name] = {
            "latency_seconds_mean": statistics.fmean(latencies),
            "completion_tokens_per_second_mean": statistics.fmean(token_rates),
            "completion_tokens": [item[1] for item in values],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
