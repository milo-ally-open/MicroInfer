#!/usr/bin/env python3
"""量測 Gateway 到推論服務的端到端性能；執行前須有可用的真實服務。"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from gateway_client import GatewayClient, MODEL_ALIAS, TEXT_MODEL_ALIAS  # noqa: E402


def percentile(values: list[float], percent: float) -> float | None:
    """回傳 nearest-rank percentile；空集合回傳 None。"""
    if not values:
        return None
    if not 0 < percent <= 1:
        raise ValueError("percent must be between 0 and 1")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percent) - 1))
    return ordered[index]


def distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def run_case(
    index: int,
    mode: str,
    client_args: tuple[str, str, float],
    model: str,
    image: Path,
    video: Path,
) -> dict:
    base_url, api_key, timeout = client_args
    client = GatewayClient(base_url, api_key, timeout=timeout)
    if mode == "text-sse":
        kind = "text_sse"
    elif mode == "chat-sse" or (mode == "mixed" and index % 2 == 0):
        kind = "chat_sse"
    else:
        kind = "video_upload"
    started = time.perf_counter()
    try:
        if kind == "text_sse":
            result = client.stream_chat_with_text("請用一句話說明推理引擎。", model=model)
            return {
                "kind": kind,
                "success": True,
                "latency_seconds": time.perf_counter() - started,
                "sse_ttft_seconds": result["first_token_seconds"],
            }
        if kind == "chat_sse":
            result = client.stream_chat_with_image(
                image, "請描述圖片中的主要內容。", model=model
            )
            return {
                "kind": kind,
                "success": True,
                "latency_seconds": time.perf_counter() - started,
                "sse_ttft_seconds": result["first_token_seconds"],
            }
        client.understand_video(video, "請簡要描述這段影片中發生的事情。")
        return {
            "kind": kind,
            "success": True,
            "latency_seconds": time.perf_counter() - started,
            "video_upload_seconds": client.last_video_upload_seconds,
        }
    except Exception as error:  # 報告只保留錯誤類別，不輸出可能含本文的錯誤內容。
        return {
            "kind": kind,
            "success": False,
            "latency_seconds": time.perf_counter() - started,
            "error_class": type(error).__name__,
        }


def build_report(
    results: list[dict], elapsed: float, mode: str, concurrency: int, model_alias: str
) -> dict:
    successes = [item for item in results if item["success"]]
    errors = [item for item in results if not item["success"]]
    latencies = [float(item["latency_seconds"]) for item in results]
    ttft = [float(item["sse_ttft_seconds"]) for item in successes if item.get("sse_ttft_seconds") is not None]
    upload = [float(item["video_upload_seconds"]) for item in successes if item.get("video_upload_seconds") is not None]
    error_classes = Counter(item["error_class"] for item in errors)
    return {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "model_alias": model_alias,
        "concurrency": concurrency,
        "request_count": len(results),
        "success_count": len(successes),
        "error_count": len(errors),
        "error_rate": len(errors) / len(results) if results else 0.0,
        "throughput_requests_per_second": len(results) / elapsed if elapsed > 0 else 0.0,
        "elapsed_seconds": elapsed,
        "latency_seconds": distribution(latencies),
        "sse_time_to_first_token_seconds": distribution(ttft),
        "video_upload_completion_seconds": distribution(upload),
        "error_classes": dict(error_classes),
        "sla": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", default=os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.environ.get("VIDEO_API_KEY", ""))
    parser.add_argument("--image", type=Path, default=Path(__file__).resolve().parents[2] / "test.jpg")
    parser.add_argument("--video", type=Path, default=Path(__file__).resolve().parents[2] / "test.mp4")
    parser.add_argument("--mode", choices=("text-sse", "chat-sse", "video", "mixed"), default="mixed")
    parser.add_argument("--model", help="聊天請求使用的公開模型別名；text-sse 預設 Qwen3。")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--output", type=Path, help="選用 JSON 報告輸出路徑；省略時寫至 stdout。")
    args = parser.parse_args(argv)
    if not args.api_key:
        parser.error("請透過 VIDEO_API_KEY 環境變數或 --api-key 提供 Gateway API Key。")
    if args.requests < 1 or args.concurrency < 1:
        parser.error("--requests 和 --concurrency 必須大於零。")
    if args.mode in {"chat-sse", "mixed"} and not args.image.is_file():
        parser.error(f"圖片不存在：{args.image}")
    if args.mode in {"video", "mixed"} and not args.video.is_file():
        parser.error(f"影片不存在：{args.video}")
    if args.mode in {"video", "mixed"} and args.model:
        parser.error("--model 僅適用於 text-sse 或 chat-sse；mixed/video 固定使用 Mage-VL。")
    model = args.model or (TEXT_MODEL_ALIAS if args.mode == "text-sse" else MODEL_ALIAS)

    client_args = (args.gateway_url, args.api_key, args.timeout_seconds)
    started = time.perf_counter()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                run_case, index, args.mode, client_args, model, args.image, args.video
            )
            for index in range(args.requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    report = build_report(
        results, time.perf_counter() - started, args.mode, args.concurrency, model
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"性能基線報告已寫入：{args.output}")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
