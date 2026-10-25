#!/usr/bin/env python3
"""使用 OpenAI Python SDK 调用本机 Gateway 的最小聊天、工具和视频示例。"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
from openai import OpenAI

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ["VIDEO_API_KEY"]

# 普通对话
client = OpenAI(
    api_key=API_KEY,
    base_url=GATEWAY_URL + "/v1",
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


# 工具调用
tools = [{
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "查询当前 UTC 时间。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}]

def get_current_time(arguments):
    if arguments:
        raise ValueError("get_current_time 不接受参数。")
    return {"utc_time": datetime.now(timezone.utc).isoformat()}


tool_handlers = {"get_current_time": get_current_time}

messages = [
    {"role": "system", "content": "需要查询当前时间时，请调用提供的工具。工具结果返回后再用中文回答。"},
    {"role": "user", "content": "请调用工具查询当前 UTC 时间，并告诉我结果。"},
]
tool_call_count = 0
for _ in range(4):
    response = client.chat.completions.create(
        model="qwen3-0.6b",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        parallel_tool_calls=False,
        stream=False,
        max_tokens=128,
    )
    assistant_message = response.choices[0].message
    if not assistant_message.tool_calls:
        if tool_call_count == 0:
            raise RuntimeError("模型没有发起工具调用；请检查模型输出与 tool calling 配置。")
        print(assistant_message.content)
        break

    tool_call_count += len(assistant_message.tool_calls)
    messages.append(assistant_message.model_dump(exclude_none=True))
    for tool_call in assistant_message.tool_calls:
        handler = tool_handlers.get(tool_call.function.name)
        if handler is None:
            raise RuntimeError(f"客户端没有注册工具：{tool_call.function.name}")
        arguments = json.loads(tool_call.function.arguments or "{}")
        print(f"客户端执行工具：{tool_call.function.name}({arguments})")
        result = handler(arguments)
        print(f"工具结果：{result}")
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": json.dumps(result, ensure_ascii=False),
        })
else:
    raise RuntimeError("工具调用超过示例的最大轮数。")


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
