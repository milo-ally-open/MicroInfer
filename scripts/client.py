#!/usr/bin/env python3
"""直接调用本地 Inference FastAPI 服务。"""

import os
from pathlib import Path

import httpx

INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://127.0.0.1:8012").rstrip("/")
MAGE_VL_URL = os.environ.get("MAGE_VL_URL", "http://127.0.0.1:8001").rstrip("/")

# 普通对话
response = httpx.post(
    INFERENCE_URL + "/v1/chat/completions",
    json={
        "model": "qwen3-vl-2b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        "stream": False,
    },
    timeout=300,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])


# 视频理解
video_file = os.environ.get("VIDEO_FILE")
if video_file:
    video_path = Path(video_file).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"找不到 VIDEO_FILE 指定的视频：{video_path}")
    with video_path.open("rb") as video:
        video_response = httpx.post(
            MAGE_VL_URL + "/v1/videos/understand",
            params={"model": os.environ.get("VIDEO_MODEL", "mage-vl-4b")},
            data={"question": os.environ.get("VIDEO_QUESTION", "请简要描述视频内容。")},
            files={"video": (video_path.name, video, "video/mp4")},
            timeout=float(os.environ.get("VIDEO_TIMEOUT_SECONDS", "900")),
        )
    video_response.raise_for_status()
    video_result = video_response.json()
    print(video_result["choices"][0]["message"].get("content"))
else:
    print("跳过视频理解：设置 VIDEO_FILE 后会直接调用 Mage-VL 的 /v1/videos/understand 接口。")
