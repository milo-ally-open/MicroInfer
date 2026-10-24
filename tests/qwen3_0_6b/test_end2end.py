#!/usr/bin/env python3
"""从 Go Gateway 执行 Qwen3-0.6B 纯文字 SSE 验收。"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="明确指定测试 YAML。")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    api_key = os.environ.get(settings["api_key_env"], "")
    if not api_key:
        parser.error(f"请设定环境变数 {settings['api_key_env']}。")
    body = json.dumps({
        "model": settings["model_alias"], "stream": True,
        "messages": [{"role": "user", "content": settings["prompt"]}],
        "max_tokens": int(settings["max_new_tokens"]),
    }, ensure_ascii=False).encode("utf-8")
    request = Request(settings["gateway_url"].rstrip("/") + "/v1/chat/completions", data=body, headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "text/event-stream",
    }, method="POST")
    started = time.perf_counter()
    chunks = 0
    first_token = None
    returned_models = set()
    with urlopen(request, timeout=float(settings["timeout_seconds"])) as response:
        if response.status != 200:
            raise RuntimeError(f"Gateway 回传 HTTP {response.status}。")
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError("Gateway SSE 回传错误事件。")
            if event.get("model"):
                returned_models.add(event["model"])
            for choice in event.get("choices", []):
                if choice.get("delta", {}).get("content"):
                    chunks += 1
                    if first_token is None:
                        first_token = time.perf_counter() - started
    if chunks == 0 or returned_models - {settings["model_alias"]}:
        raise RuntimeError(f"SSE 无文字输出或公开模型别名不符：{returned_models}")
    print(json.dumps({
        "model_alias": settings["model_alias"], "status": "passed", "text_chunks": chunks,
        "ttft_seconds": first_token, "total_seconds": time.perf_counter() - started,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
