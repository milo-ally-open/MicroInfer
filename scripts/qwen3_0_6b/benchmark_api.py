"""以 OpenAI SSE API 量测 Qwen3 端到端生成延迟与吞吐。"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import platform
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="量测已启动 Qwen3 Chat Completions API 的文字 SSE 性能。")
    parser.add_argument("--base-url", required=True, help="推论 API 根网址，例如 http://127.0.0.1:8011")
    parser.add_argument("--model", required=True, help="公开模型别名，例如 qwen3-0.6b")
    parser.add_argument("--prompt", required=True, help="每一个请求使用的文字 prompt")
    parser.add_argument("--requests", required=True, type=int, help="测量请求数")
    parser.add_argument("--concurrency", required=True, type=int, help="同时执行的请求数")
    parser.add_argument("--max-tokens", required=True, type=int, help="每个回应允许生成的 token 上限")
    parser.add_argument("--output", required=True, type=Path, help="明确指定 JSON 报告路径")
    return parser.parse_args()


def _percentile(values: list[float], percentage: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((percentage / 100) * len(ordered) + 0.999999) - 1))
    return ordered[index]


def _get_json(url: str, *, timeout: float = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _stream_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    token_times: list[float] = []
    completion_tokens: int | None = None
    saw_done = False
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                saw_done = True
                break
            event = json.loads(data)
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if (choice.get("delta") or {}).get("content"):
                token_times.append(time.perf_counter())
            usage = event.get("usage")
            if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
    completed = time.perf_counter()
    if not saw_done:
        raise RuntimeError("SSE stream ended without the OpenAI [DONE] marker.")
    if completion_tokens is None:
        completion_tokens = len(token_times)
    inter_token_ms = [
        (right - left) * 1000 for left, right in zip(token_times, token_times[1:])
    ]
    return {
        "latency_seconds": completed - started,
        "ttft_seconds": token_times[0] - started if token_times else None,
        "inter_token_ms": inter_token_ms,
        "visible_text_chunks": len(token_times),
        "completion_tokens": completion_tokens,
    }


def _version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    args = _arguments()
    if args.requests < 1 or args.concurrency < 1 or args.max_tokens < 1:
        raise ValueError("requests, concurrency and max-tokens must be positive.")
    if args.concurrency > args.requests:
        raise ValueError("concurrency must not exceed requests.")
    base_url = args.base_url.rstrip("/")
    health_before = _get_json(base_url + "/health")
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(
            pool.map(
                lambda _: _stream_request(base_url, args.model, args.prompt, args.max_tokens),
                range(args.requests),
            )
        )
    elapsed = time.perf_counter() - started
    health_after = _get_json(base_url + "/health")

    ttft_ms = [row["ttft_seconds"] * 1000 for row in results if row["ttft_seconds"] is not None]
    latency_ms = [row["latency_seconds"] * 1000 for row in results]
    inter_token_ms = [value for row in results for value in row["inter_token_ms"]]
    completion_tokens = sum(int(row["completion_tokens"]) for row in results)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "revision": health_after.get("revision"),
        "backend": health_after.get("backend"),
        "hardware": health_after.get("gpu"),
        "software": {
            "python": platform.python_version(),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
        },
        "workload": {
            "requests": args.requests,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "prompt_characters": len(args.prompt),
            "visible_text_chunks": sum(int(row["visible_text_chunks"]) for row in results),
            "completion_tokens": completion_tokens,
        },
        "performance": {
            "wall_seconds": elapsed,
            "request_throughput_per_second": args.requests / max(elapsed, 1e-9),
            "token_throughput_per_second": completion_tokens / max(elapsed, 1e-9),
            "request_latency_ms_p50": statistics.median(latency_ms),
            "request_latency_ms_p95": _percentile(latency_ms, 95),
            "ttft_ms_p50": statistics.median(ttft_ms) if ttft_ms else None,
            "ttft_ms_p95": _percentile(ttft_ms, 95),
            "inter_token_ms_p50": statistics.median(inter_token_ms) if inter_token_ms else None,
            "inter_token_ms_p95": _percentile(inter_token_ms, 95),
        },
        "memory": {
            "vram_free_bytes_before": health_before.get("vram_free_bytes"),
            "vram_free_bytes_after": health_after.get("vram_free_bytes"),
            "peak_allocated_bytes": (health_after.get("gpu_peak_memory") or {}).get("peak_allocated"),
            "peak_reserved_bytes": (health_after.get("gpu_peak_memory") or {}).get("peak_reserved"),
            "worker_kv_cache": health_after.get("worker_kv_cache"),
        },
        "limitations": [
            "此报告量测单一 API 程序及本机 GPU 的短期基线，不构成 SLA。",
            "TTFT 以第一个非空 SSE content chunk 计算；inter-token latency 以相邻可见文字 chunk 计算。",
            "SDPA 目前会将 paged KV 收集至连续 batch tensor，报告不代表 PagedAttention kernel 效能。",
        ],
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
