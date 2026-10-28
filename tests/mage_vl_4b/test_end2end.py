#!/usr/bin/env python3
"""从 Go Gateway 验收 Mage-VL 图片 SSE 与视频上传。"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import yaml

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True, help="明确指定测试 YAML。")
parser.add_argument("--output", type=Path, help="可选的机器可读 JSON 结果路径。")
args = parser.parse_args()

config_path = args.config.resolve()
settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if not isinstance(settings, dict):
    raise ValueError("测试 YAML 最上层必须是 mapping。")
api_key = os.environ.get(settings["api_key_env"], "")
if not api_key:
    parser.error(f"请设置环境变量 {settings['api_key_env']}。")
image = (config_path.parent / settings["image"]).resolve()
video = (config_path.parent / settings["video"]).resolve()
if not image.is_file() or not video.is_file():
    raise FileNotFoundError("YAML 指定的图片或视频不存在。")

encoded = base64.b64encode(image.read_bytes()).decode("ascii")
chat_body = json.dumps({
    "model": settings["model_alias"],
    "stream": True,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": settings["prompt"]},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
        ],
    }],
    "max_tokens": int(settings["max_new_tokens"]),
}, ensure_ascii=False).encode("utf-8")
request = Request(
    settings["gateway_url"].rstrip("/") + "/v1/chat/completions",
    data=chat_body,
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    },
    method="POST",
)
started = time.perf_counter()
chunks = 0
first_token = None
returned_models = set()
image_response_bytes = 0
with urlopen(request, timeout=float(settings["timeout_seconds"])) as response:
    image_http_status = response.status
    if response.status != 200:
        raise RuntimeError(f"Gateway 图片聊天返回 HTTP {response.status}。")
    for line in response:
        image_response_bytes += len(line)
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        event = json.loads(data)
        if event.get("error"):
            raise RuntimeError("Gateway 图片 SSE 返回错误事件。")
        if event.get("model"):
            returned_models.add(event["model"])
        if any(choice.get("delta", {}).get("content") for choice in event.get("choices", [])):
            chunks += 1
            if first_token is None:
                first_token = time.perf_counter() - started
if chunks == 0 or returned_models - {settings["model_alias"]}:
    raise RuntimeError(f"图片 SSE 无文字输出或公开模型别名不符：{returned_models}")
image_elapsed_seconds = time.perf_counter() - started

gateway = urlsplit(settings["gateway_url"])
boundary = "----ModelTest-" + uuid.uuid4().hex
filename = video.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
header = (
    f"--{boundary}\r\n"
    f"Content-Disposition: form-data; name=\"video\"; filename=\"{filename}\"\r\n"
    "Content-Type: video/mp4\r\n\r\n"
).encode()
question = (
    f"\r\n--{boundary}\r\n"
    "Content-Disposition: form-data; name=\"question\"\r\n\r\n"
    f"{settings['prompt']}\r\n"
    f"--{boundary}--\r\n"
).encode()
content_length = len(header) + video.stat().st_size + len(question)
path = (gateway.path.rstrip("/") or "") + "/v1/videos/understand?model=" + settings["model_alias"]
connection_type = http.client.HTTPSConnection if gateway.scheme == "https" else http.client.HTTPConnection
connection = connection_type(gateway.hostname, gateway.port, timeout=float(settings["timeout_seconds"]))
upload_started = time.perf_counter()
try:
    connection.putrequest("POST", path)
    connection.putheader("Authorization", f"Bearer {api_key}")
    connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
    connection.putheader("Content-Length", str(content_length))
    connection.endheaders()
    connection.send(header)
    with video.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            connection.send(chunk)
    connection.send(question)
    video_upload_complete_seconds = time.perf_counter() - upload_started
    response = connection.getresponse()
    video_http_status = response.status
    raw = response.read(10 * 1024 * 1024)
    if not 200 <= response.status < 300:
        raise RuntimeError(f"Gateway 视频理解返回 HTTP {response.status}。")
    video_result = json.loads(raw)
finally:
    connection.close()

report = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "model_alias": settings["model_alias"],
    "records": [
        {
            "input_type": "image",
            "endpoint": "/v1/chat/completions",
            "http_status": image_http_status,
            "request_bytes": len(chat_body),
            "response_bytes": image_response_bytes,
            "elapsed_seconds": image_elapsed_seconds,
            "first_text_chunk_seconds": first_token,
            "text_chunk_count": chunks,
        },
        {
            "input_type": "video",
            "endpoint": "/v1/videos/understand",
            "http_status": video_http_status,
            "request_bytes": content_length,
            "response_bytes": len(raw),
            "elapsed_seconds": time.perf_counter() - upload_started,
            "upload_complete_seconds": video_upload_complete_seconds,
            "first_text_chunk_seconds": None,
            "text_chunk_count": None,
        },
    ],
}
if args.output:
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report: {output_path}")
print(json.dumps(report, ensure_ascii=False, indent=2))
