#!/usr/bin/env python3
"""从 Go Gateway 验收 Mage-VL 图片 SSE 与影片上传。"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="明确指定测试 YAML。")
    args = parser.parse_args()
    config_path = args.config.resolve()
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    api_key = os.environ.get(settings["api_key_env"], "")
    if not api_key:
        parser.error(f"请设定环境变数 {settings['api_key_env']}。")
    image = (config_path.parent / settings["image"]).resolve()
    video = (config_path.parent / settings["video"]).resolve()
    if not image.is_file() or not video.is_file():
        raise FileNotFoundError("YAML 指定的图片或影片不存在。")

    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    chat_body = json.dumps({
        "model": settings["model_alias"], "stream": True,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": settings["prompt"]},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
        ]}], "max_tokens": int(settings["max_new_tokens"]),
    }, ensure_ascii=False).encode("utf-8")
    request = Request(settings["gateway_url"].rstrip("/") + "/v1/chat/completions", data=chat_body, headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "text/event-stream",
    }, method="POST")
    started = time.perf_counter()
    chunks = 0
    first_token = None
    returned_models = set()
    with urlopen(request, timeout=float(settings["timeout_seconds"])) as response:
        if response.status != 200:
            raise RuntimeError(f"Gateway 图片聊天回传 HTTP {response.status}。")
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError("Gateway 图片 SSE 回传错误事件。")
            if event.get("model"):
                returned_models.add(event["model"])
            if any(choice.get("delta", {}).get("content") for choice in event.get("choices", [])):
                chunks += 1
                if first_token is None:
                    first_token = time.perf_counter() - started
    if chunks == 0 or returned_models - {settings["model_alias"]}:
        raise RuntimeError(f"图片 SSE 无文字输出或公开模型别名不符：{returned_models}")

    gateway = urlsplit(settings["gateway_url"])
    boundary = "----ModelTest-" + uuid.uuid4().hex
    filename = video.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
    header = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; filename=\"{filename}\"\r\n"
              "Content-Type: video/mp4\r\n\r\n").encode()
    question = (f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"question\"\r\n\r\n"
                f"{settings['prompt']}\r\n--{boundary}--\r\n").encode()
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
        response = connection.getresponse()
        raw = response.read(10 * 1024 * 1024)
        if not 200 <= response.status < 300:
            raise RuntimeError(f"Gateway 影片理解回传 HTTP {response.status}。")
        video_result = json.loads(raw)
    finally:
        connection.close()
    print(json.dumps({
        "model_alias": settings["model_alias"], "status": "passed",
        "image_sse_text_chunks": chunks, "image_sse_ttft_seconds": first_token,
        "video_upload_and_inference_seconds": time.perf_counter() - upload_started,
        "video_response_fields": sorted(video_result.keys()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
