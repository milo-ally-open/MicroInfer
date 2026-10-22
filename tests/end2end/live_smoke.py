"""透過 Gateway、Inference 與 MediaMTX 驗證即時視訊分析鏈路。"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL_ALIAS = "mage-vl-4b"
MAX_RESPONSE_BYTES = 1024 * 1024


def request_json(url: str, api_key: str, *, method: str = "GET", payload: dict[str, Any] | None = None,
                 timeout: float = 15.0) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise RuntimeError("Gateway JSON 回應超過 1 MiB 測試上限。")
            value = json.loads(raw) if raw else {}
            if not isinstance(value, dict):
                raise RuntimeError("Gateway JSON 回應必須是物件。")
            return response.status, value
    except HTTPError as exc:
        detail = exc.read(MAX_RESPONSE_BYTES).decode("utf-8", errors="replace")
        raise RuntimeError(f"Gateway 回傳 HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"無法連線至 Gateway：{exc.reason}") from exc


def sse_events(response: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    event_name = "message"
    data_lines: list[bytes] = []
    for raw_line in response:
        line = raw_line.rstrip(b"\r\n")
        if not line:
            if data_lines:
                try:
                    value = json.loads(b"\n".join(data_lines))
                except json.JSONDecodeError:
                    value = {}
                if isinstance(value, dict):
                    yield event_name, value
            event_name = "message"
            data_lines = []
        elif line.startswith(b"event:"):
            event_name = line[6:].strip().decode("utf-8", errors="replace") or "message"
        elif line.startswith(b"data:"):
            data_lines.append(line[5:].lstrip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", default=os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key-env", default="VIDEO_API_KEY", help="含 Gateway API Key 的環境變數名稱。")
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG_BIN", "ffmpeg"))
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def stop_publisher(process: subprocess.Popen[bytes] | None) -> int | None:
    if process is None:
        return None
    if process.poll() is None:
        process.terminate()
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)
    return process.returncode


def safe_error(exc: Exception, sensitive_values: list[str]) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for value in sensitive_values:
        if value:
            message = message.replace(value, "[REDACTED]")
    message = re.sub(r"\b(?:https?|rtsp|rtmp)://[^\s]+", "[URL REDACTED]", message)
    return message[:1000]


def run() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"請先設定環境變數 {args.api_key_env}。")
    ffmpeg = shutil.which(args.ffmpeg)
    if not ffmpeg:
        raise RuntimeError(f"找不到 FFmpeg：{args.ffmpeg}")

    gateway_url = args.gateway_url.rstrip("/")
    report_path = args.output or (
        Path(__file__).resolve().parent
        / "results"
        / f"mage-vl-4b-live-smoke-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    )
    if report_path.exists():
        raise FileExistsError(f"拒絕覆寫既有測試報告：{report_path}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "test": "live-video-gateway-smoke",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_ALIAS,
        "input": "FFmpeg lavfi testsrc 320x240 at 2 fps, H.264 over RTSP/TCP",
        "settings": {
            "video_backend": "frames",
            "sample_interval_seconds": args.sample_interval_seconds,
            "window_seconds": args.window_seconds,
            "max_tokens": args.max_tokens,
        },
        "outcome": "failed",
        "events": [],
        "first_analysis_seconds": None,
        "analysis_frame_count": None,
        "cleanup_completed": False,
        "secret_fields_excluded": True,
    }
    session: dict[str, Any] | None = None
    publisher: subprocess.Popen[bytes] | None = None
    events_response: Any = None
    sensitive_values = [api_key]
    started = time.perf_counter()
    try:
        status, models = request_json(gateway_url + "/v1/models", api_key)
        report["models_http_status"] = status
        model_ids = [item.get("id") for item in models.get("data", []) if isinstance(item, dict)]
        if MODEL_ALIAS not in model_ids:
            raise RuntimeError(f"Gateway /v1/models 未列出 {MODEL_ALIAS}。")

        create_payload = {
            "model": MODEL_ALIAS,
            "question": "Describe the visible scene in one concise sentence.",
            "video_backend": "frames",
            "sample_interval_seconds": args.sample_interval_seconds,
            "window_seconds": args.window_seconds,
            "max_duration_seconds": min(120.0, args.timeout_seconds),
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
        }
        create_status, session = request_json(
            gateway_url + "/v1/live/sessions", api_key, method="POST", payload=create_payload
        )
        report["create_http_status"] = create_status
        publish_url = session.get("publish", {}).get("rtsp")
        events_url = session.get("events_url")
        stop_url = session.get("stop_url")
        sensitive_values.extend(
            value for value in (session.get("id"), publish_url, events_url, stop_url) if isinstance(value, str)
        )
        if not all(isinstance(value, str) and value for value in (publish_url, events_url, stop_url)):
            raise RuntimeError("Live session 回應缺少 publish.rtsp、events_url 或 stop_url。")

        event_request = Request(
            events_url,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "text/event-stream"},
        )
        events_response = urlopen(event_request, timeout=args.timeout_seconds)
        report["events_http_status"] = events_response.status
        publisher = subprocess.Popen(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-re",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=2",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-g",
                "2",
                "-f",
                "rtsp",
                "-rtsp_transport",
                "tcp",
                publish_url,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        deadline = started + args.timeout_seconds
        for event_name, data in sse_events(events_response):
            if time.perf_counter() >= deadline:
                raise TimeoutError(f"等待 Live analysis 超過 {args.timeout_seconds:g} 秒。")
            report["events"].append(event_name)
            if event_name == "analysis":
                report["first_analysis_seconds"] = round(time.perf_counter() - started, 3)
                report["analysis_frame_count"] = data.get("frame_count")
                report["outcome"] = "passed"
                break
            if event_name == "error":
                detail = data.get("message")
                if isinstance(detail, str) and detail:
                    report["inference_error"] = safe_error(RuntimeError(detail), sensitive_values)
                raise RuntimeError(f"Live 推理事件回報錯誤，phase={data.get('phase', 'unknown')}。")
            if event_name == "completed":
                raise RuntimeError(f"Live session 在產生 analysis 前結束，reason={data.get('reason', 'unknown')}。")
            if publisher.poll() is not None:
                raise RuntimeError(f"FFmpeg 推流程序提前結束，exit_code={publisher.returncode}。")

        if report["outcome"] != "passed":
            raise TimeoutError("SSE 串流結束前未收到 analysis 事件。")
    except Exception as exc:
        report.setdefault("error", safe_error(exc, sensitive_values))
    finally:
        if session and isinstance(session.get("stop_url"), str):
            try:
                stop_status, _ = request_json(session["stop_url"], api_key, method="DELETE", timeout=15.0)
                report["stop_http_status"] = stop_status
            except Exception as exc:
                report["cleanup_error"] = safe_error(exc, sensitive_values)
        publisher_status = stop_publisher(publisher)
        if publisher is not None:
            report["publisher_exit_code"] = publisher_status
        if events_response is not None:
            events_response.close()
        report["cleanup_completed"] = bool(report.get("stop_http_status", 0) in {200, 202})

    report["duration_seconds"] = round(time.perf_counter() - started, 3)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outcome": report["outcome"], "report": str(report_path), "duration_seconds": report["duration_seconds"]}, ensure_ascii=False))
    if report["outcome"] != "passed" or not report["cleanup_completed"]:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as exc:
        print(f"Live smoke 無法啟動：{exc}", file=sys.stderr)
        raise SystemExit(2)
