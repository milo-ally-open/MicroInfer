import base64
import mimetypes
from pathlib import Path

from inference.models.mage_vl_4b import create_engine_client

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
IMAGE_PATH = REPOSITORY_ROOT / "test.jpg"


# 检查示例图片，并编码成 OpenAI 图片输入支持的 data URL。
if not IMAGE_PATH.is_file():
    raise FileNotFoundError(f"Image for the Mage-VL example was not found: {IMAGE_PATH}")

image_type = mimetypes.guess_type(IMAGE_PATH.name)[0] or "image/jpeg"
image_data = base64.b64encode(IMAGE_PATH.read_bytes()).decode("ascii")
request = {
    "model": "mage-vl-4b",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "请简要描述图片中的主要内容。"},
            {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{image_data}"}},
        ],
    }],
    "max_tokens": 96,
}

# 启动 Engine；Mage-VL Worker 会在模型侧预处理图片输入。
engine = create_engine_client()
engine.start()

# Engine 按生成步返回已经解码的文字增量，收到一段就立即打印一段。
stream = engine.stream_chat(request)
for event in stream:
    if event.kind == "text" and event.text:
        print(event.text, end="", flush=True)
print()

# 正常结束后关闭串流和 Engine。
stream.close()
engine.close()
