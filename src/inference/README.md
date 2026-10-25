# Inference 服务

Inference 是独立的 Python 模型服务，提供模型加载、请求预处理、生成和 GPU 资源管理。它与 Go Gateway 分开运行；Gateway 处理外部接入，Inference 不负责客户端 API Key 或租户政策。

## 架构

API Server、Engine Core 和 Model Worker 是独立进程。API Server 通过 `network/` 提供 HTTP/SSE；Engine Core 负责请求生命周期、逐轮调度和 KV block reservation；Worker 在自己的进程中载入模型并执行前向计算。模型专属 Runtime、Processor、Runner、拓扑和算子由模型包持有。

| 组件 | 位置 | 主要职责 |
| --- | --- | --- |
| Network | `network/` | FastAPI app、模型服务入口、HTTP schema、路由、序列化和 API 输入／输出处理 |
| 共用 Processors | `processors/` | 视频文件处理、媒体信息读取、Live session 管理等跨模型能力 |
| Engine | `engine/` | `inference_engine.py`、`scheduler.py`、`block_manager.py`、`worker.py`、IPC 和启动配置 |
| 模型插件 | `models/<model>/` | 模型 Runtime、Tokenizer／Processor、Runner、拓扑、权重映射和模型配置 |

模型专属 tokenizer、chat template、输入格式和视觉预处理继续放在模型包中。`processors/` 用于新增跨模型可复用的处理能力。

## 环境与安装

需要 Python 3.10 或更新版本，以及与目标设备匹配的 PyTorch/CUDA。PyTorch wheel 应先按机器环境安装。然后在所选模型环境中，从仓库根目录安装对应依赖：

```bash
# Mage-VL
python -m pip install -e "src/inference[mage-vl]"

# 或 Qwen3
python -m pip install -e "src/inference[qwen3]"
```

每个 extra 只安装对应模型的依赖，不会安装其他模型依赖。

## 检查与启动

从仓库根目录运行：

```bash
inference list
inference validate --model qwen3-0.6b
inference start --model qwen3-0.6b
```

`list` 显示自动发现的模型 manifest。`validate` 检查 manifest、YAML、Runtime 工厂声明和 API 能力，不导入模型工厂、不载入权重、不下载 snapshot，也不初始化 GPU。`start` 必须指定 `--model`。

服务监听地址来自模型的 `configs/network.yml`：Mage-VL 默认 `127.0.0.1:8001`，Qwen3 默认 `127.0.0.1:8011`。推理优先使用 CUDA；仅在 CUDA 不可用时回退到 CPU。

## 在服务端代码中使用 Engine

模型插件可以在 Python 服务端创建 Engine client，并显式指定推理与网络 YAML。以下示例直接导入 Qwen3 插件；插件仍从自己的 `model.yml` 读取模型别名和 Runtime 工厂，传入的 YAML 路径覆盖 manifest 中对应的 `inference_config` 和 `network_config`。Engine 会启动独立 Engine Core 和 Model Worker。

模型依赖、与环境匹配的 PyTorch/CUDA，以及 `configs/inference.yml` 指定的模型 checkpoint 必须已准备好。

```python
from pathlib import Path

from inference.models.qwen3_0_6b import create_engine_client


CONFIG_DIR = Path("src/inference/models/qwen3_0_6b/configs")


def main():
    engine = create_engine_client(
        inference_config_path=CONFIG_DIR / "inference.yml",
        network_config_path=CONFIG_DIR / "network.yml",
    )
    engine.start()
    try:
        result = engine.generate_chat({
            "model": "qwen3-0.6b",
            "messages": [{"role": "user", "content": "用一句话介绍分页 KV Cache。"}],
            "max_tokens": 128,
        })
        print(result.content)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
```

`generate_chat()` 返回 `InferenceResult`，文本位于 `result.content`。长驻服务应在启动和关闭钩子中分别调用 `engine.start()` 与 `engine.close()`，不要为每个请求创建 Engine。部署 wheel 时，可通过 `INFERENCE_PROJECT_ROOT` 指定 checkpoint 和运行配置所在目录。

`inference_config_path` 和 `network_config_path` 可接收字符串或 `Path`；显式传入的相对路径以当前工作目录为基准。省略某个参数时，该文件路径从模型 `model.yml` 的对应字段读取。YAML 顶层必须是 mapping；配置中的相对 checkpoint 路径仍以 `INFERENCE_PROJECT_ROOT` 为基准。长驻服务可以把环境专用配置放在部署目录，并将两个文件路径传给插件工厂。

## API 能力

Inference 提供 `GET /health`、`GET /v1/models` 和 `POST /v1/chat/completions`。模型已启用时，还会注册 `POST /v1/videos/understand` 或 Live session 路由。聊天接口支持 JSON 和文本 SSE；实际输入模态、串流和视频能力由模型 manifest 声明。

Qwen3 的聊天接口实现标准 function tool 消息字段，包括 `tools`、`tool_choice: "auto"/"none"`、`assistant.tool_calls`、`role: "tool"` 和 SSE `delta.tool_calls`。首版每轮只支持一个工具调用；不支持并行工具调用、指定函数的 `tool_choice` 对象或 `strict: true`。工具参数按 JSON Schema 验证。

Inference 只解析模型输出并返回标准 tool call，不执行工具。客户端／Agent 应用负责校验和执行工具，再将 `role: "tool"` 结果发回同一接口。当前仅 Qwen3 启用工具调用；Mage-VL 未宣告该能力。

## 配置

模型配置集中在 `models/<model>/configs/`：

| 文件 | 内容 |
| --- | --- |
| `model.yml` | 模型别名、API 能力、Runtime 与可选 API 处理器工厂 |
| `inference.yml` | 上游模型 revision、权重路径、生成、Scheduler/KV、请求限制和运行参数 |
| `network.yml` | API 监听地址及模型所需媒体端点 |

API Key 和其他密钥不写入模型 YAML；Gateway 认证配置见 [Gateway README](../gateway/README.md)。MediaMTX 是独立 sidecar，配置和启动方法见 [MediaMTX README](../../deploy/mediamtx/README.md)。

## 新模型接入

新模型应新增 `models/<model>/` 插件包和 `configs/model.yml` manifest。通常由模型包提供 `runtime.py`；按需增加 `processor.py`、`utils/loader.py`、`model/`、`layers/`、`components/` 和 `requirements.txt`。Launcher 会发现模型 manifest，不需要复制 API Server、Engine Core 或 Worker。

完整的文件职责、能力声明、配置格式和验证流程见[新模型接入指南](../../docs/Specifications/inference/model-integration.md)。

## 测试

每个模型维护 `test_components.py`、`test_performance.py`、`test_end2end.py` 和明确传入的 `test.yml`，位于 `tests/<model_alias>/`。组件测试不加载完整 checkpoint；性能测试需要模型权重、GPU 与 Inference API；端到端测试需要 Gateway、Inference、API Key 和适用素材。命令与结果限制见 [Tests README](../../tests/README.md)。

Qwen3 另有模型包内 CUDA decode attention prototype。它只用于单 token decode，当前测量慢于 PyTorch reference，默认仍关闭；构建要求和限制见 [CUDA kernel README](models/qwen3_0_6b/kernels/cuda/README.md)。项目不宣称 vLLM 兼容性，也没有 Prefix Caching。
