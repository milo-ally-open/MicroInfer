#!/usr/bin/env python3
"""验证 Qwen3-VL 服务的非流式文本输入与图文流式输出。"""

from pathlib import Path
import base64
import json
from urllib.error import URLError
from urllib.request import Request, urlopen

import yaml

TEST_CONFIG = Path(__file__).resolve().with_name("test.yml")
settings = yaml.safe_load(TEST_CONFIG.read_text(encoding="utf-8"))
base_url = settings["inference_url"].rstrip("/")
timeout = int(settings["timeout_seconds"])
image_path = (TEST_CONFIG.parent / settings["image"]).resolve()
image_data_url = "data:image/jpeg;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")

# 先发送纯文本非流式请求，确认模型别名、响应体和用量统计符合 OpenAI Chat Completions 协议。
text_body = {
    "model": settings["model_alias"],
    "messages": [{"role": "user", "content": "用一句话介绍推理服务。"}],
    "max_tokens": 12,
    "stream": False,
}
text_request = Request(
    f"{base_url}/v1/chat/completions",
    data=json.dumps(text_body, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urlopen(text_request, timeout=timeout) as response:
        assert response.status == 200
        text_result = json.loads(response.read())
except URLError as exc:
    raise RuntimeError(f"Could not reach Qwen3-VL API at {base_url}; start inference first.") from exc
assert text_result["model"] == settings["model_alias"]
assert text_result["choices"][0]["message"]["content"].strip()
assert text_result["usage"]["completion_tokens"] > 0

# 再发送包含真实图片 data URL 的流式请求，逐个检查增量、结束原因和 SSE [DONE]。
image_body = {
    "model": settings["model_alias"],
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text", "text": settings["prompt"]},
        ],
    }],
    "max_tokens": 16,
    "stream": True,
}
image_request = Request(
    f"{base_url}/v1/chat/completions",
    data=json.dumps(image_body, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    method="POST",
)
saw_text_delta = False
saw_finish_reason = False
saw_done = False
streamed_text = []
with urlopen(image_request, timeout=timeout) as response:
    assert response.status == 200
    assert "text/event-stream" in response.headers.get("Content-Type", "")
    for line in response:
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            saw_done = True
            break
        event = json.loads(data)
        if event.get("error"):
            raise RuntimeError(f"Qwen3-VL SSE returned an error: {event['error']}")
        for choice in event.get("choices", []):
            delta_text = choice.get("delta", {}).get("content")
            if delta_text:
                streamed_text.append(delta_text)
                saw_text_delta = True
            if choice.get("finish_reason") is not None:
                saw_finish_reason = True

assert saw_text_delta, "Image stream returned no text deltas."
assert saw_finish_reason, "Image stream returned no finish_reason."
assert saw_done, "Image stream did not terminate with [DONE]."
assert "".join(streamed_text).strip()
print({
    "status": "passed",
    "model": settings["model_alias"],
    "text_completion_tokens": text_result["usage"]["completion_tokens"],
    "image_stream_text": "".join(streamed_text),
    "image_stream_done": saw_done,
})
