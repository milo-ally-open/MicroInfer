"""通过 Engine Client 发送一张图片并流式打印 Qwen3-VL 回复。"""

import base64
from pathlib import Path

from inference.models.qwen3_vl_2b import create_engine_client

INFERENCE_CONFIG = Path(__file__).resolve().parent / "configs" / "inference.yml"
IMAGE_PATH = Path(__file__).resolve().parents[4] / "test.jpg"
IMAGE_DATA_URL = "data:image/jpeg;base64," + base64.b64encode(IMAGE_PATH.read_bytes()).decode("ascii")
if __name__ == "__main__":
    engine = create_engine_client(inference_config_path=INFERENCE_CONFIG)
    engine.start()

    request = {
        "model": "qwen3-vl-2b",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": IMAGE_DATA_URL}},
                {"type": "text", "text": "请简要描述图片中的主要内容。"},
            ],
        }],
        "max_tokens": 64,
    }

    # Engine 返回模型 Worker 生成的文本增量；图片的视觉编码和 tokenization 在模型 Encoder 中完成。
    stream = engine.stream_chat(request)
    for event in stream:
        if event.kind == "text" and event.text:
            print(event.text, end="", flush=True)
        elif event.kind == "finish":
            print()

    stream.close()
    engine.close()
