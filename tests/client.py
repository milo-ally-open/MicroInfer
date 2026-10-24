#!/usr/bin/env python3
"""使用 OpenAI Python SDK 呼叫本机 Gateway 的最小文字聊天范例。"""

import os

from openai import OpenAI


client = OpenAI(
    api_key=os.environ["VIDEO_API_KEY"],
    base_url=os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000").rstrip("/") + "/v1",
)

response = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    stream=False,
)

print(response.choices[0].message.content)
