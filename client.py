#!/usr/bin/env python3
"""使用 OpenAI Python SDK 调用本地 Gateway 的最小聊天和视频示例。"""

import os
from pathlib import Path

import httpx
from openai import OpenAI

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ["VIDEO_API_KEY"]

# 普通对话
client = OpenAI(api_key=API_KEY, base_url=GATEWAY_URL + "/v1")

response = client.chat.completions.create(
    model="qwen3-vl-2b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    stream=False,
)

print(response.choices[0].message.content)


# 视频理解
video_file = os.environ.get("VIDEO_FILE")
if video_file:
    video_path = Path(video_file).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"找不到 VIDEO_FILE 指定的视频：{video_path}")
    with video_path.open("rb") as video:
        video_response = httpx.post(
            GATEWAY_URL + "/v1/videos/understand",
            params={"model": os.environ.get("VIDEO_MODEL", "mage-vl-4b")},
            headers={"Authorization": f"Bearer {API_KEY}"},
            data={"question": os.environ.get("VIDEO_QUESTION", "请简要描述视频内容。")},
            files={"video": (video_path.name, video, "video/mp4")},
            timeout=float(os.environ.get("VIDEO_TIMEOUT_SECONDS", "900")),
        )
    video_response.raise_for_status()
    video_result = video_response.json()
    print(video_result["choices"][0]["message"].get("content"))
else:
    print("跳过视频理解：设置 VIDEO_FILE 后会调用现有的 /v1/videos/understand 接口。")
