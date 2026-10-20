#!/usr/bin/env python3
"""使用 OpenAI Python SDK 呼叫 Gateway 的圖片聊天 API。"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from openai import OpenAI


IMAGE_PATH = Path(__file__).resolve().with_name("test.jpg")

client = OpenAI(
    api_key=os.environ["VIDEO_API_KEY"],
    base_url=os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000").rstrip("/") + "/v1",
)

image = base64.b64encode(IMAGE_PATH.read_bytes()).decode("ascii")
response = client.chat.completions.create(
    model="mage-vl-4b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "請簡要描述圖片內容。"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                },
            ],
        }
    ],
)
print(response.choices[0].message.content)
