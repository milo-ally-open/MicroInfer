# MicroInfer

MicroInfer 提供 Python FastAPI 推理 API，负责加载模型并处理推理请求。当前仓库不包含 API Gateway 或控制面。

## 功能概览

- OpenAI 风格聊天 API，支持文本 SSE 输出。
- Mage-VL-4B 支持图片聊天、视频上传理解和 Live 视频会话。
- Qwen3-VL-2B 支持文字、图片聊天和流式文字输出。
- Inference 使用独立 Engine Core 与 Model Worker 进程；模型前向由各自 Transformers Runtime 执行。
- FastAPI 直接提供 OpenAI 兼容 API、SSE、视频上传和 Live API。

## 模型

| API model ID | 模型包 | 已实现能力 | 推理实现 |
| --- | --- | --- | --- |
| `mage-vl-4b` | `src/inference/models/mage_vl_4b/` | 图片聊天、文本 SSE、视频上传、Live | Mage-VL Transformers Runtime |
| `qwen3-vl-2b` | `src/inference/models/qwen3_vl_2b/` | 文字／图片聊天、文本 SSE | Qwen3-VL Transformers Runtime |

API model ID 使用仓库模型名，如 `mage-vl-4b`；Python 包目录和 Hugging Face `repo_id` 是模型加载标识。API 服务直接运行在模型对应的 FastAPI 进程中。

## 架构

```text
客户端／控制面 ─────────HTTP/SSE──────────> Python FastAPI Network
                                             ├── Engine Core
                                             └── Model Worker / 模型插件
客户端 ─────────────RTSP/RTMP──────────────> MediaMTX
```

`src/inference/network/` 提供共用 HTTP 服务；`engine/` 管理请求调度、KV block、Worker 与进程通信；`processors/` 承载跨模型视频和 Live 处理能力；模型专属 Runtime、Processor、Runner、拓扑和算子位于 `models/<model>/`。控制面由独立项目负责。

项目参考 vLLM 和 nano-vLLM 的职责划分，不提供 vLLM backend，也没有 Prefix Caching。当前模型使用 Transformers Cache；Engine KV block 是逻辑 reservation，不是实体 paged KV 内存池。

## 快速开始

以下命令适用于 Linux/WSL Bash，以 Qwen3-VL-2B 为例。需要 Python 3.10+ 和支持 BF16 的 NVIDIA GPU；先按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)安装匹配的 CUDA 版本。

```bash
git clone https://github.com/milo-ally-open/MicroInfer.git
cd MicroInfer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e "src/inference[qwen3-vl-2b]"

python scripts/download.py \
  --repo-id Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --local-dir checkpoints/Qwen3-VL-2B-Instruct-8964489

inference serve --model qwen3-vl-2b
```

服务启动后，在另一个终端发送请求：

```bash
curl http://127.0.0.1:8012/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-vl-2b","messages":[{"role":"user","content":"你好！"}]}'
```

模型权重下载到 `checkpoints/`，首次启动需要等待模型加载完成。服务不做 API Key 鉴权或租户限流；部署时请限制网络访问范围。

## 配置

| 配置 | 用途 |
| --- | --- |
| `src/inference/models/<model>/profiles/model.yml` | 模型别名、API 能力和 Runtime 工厂 |
| `src/inference/models/<model>/profiles/inference.yml` | 模型、Scheduler、KV、请求限制和 Runtime 参数 |
| `src/inference/models/<model>/profiles/network.yml` | Inference 监听地址及模型使用的媒体端点 |
| `deploy/mediamtx/mediamtx.yml` | 独立 MediaMTX sidecar 配置 |

模型权重和 Hub 缓存位于被忽略的 `checkpoints/` 目录。

## 文档

- [Inference 服务、模型引擎和模型接入](src/inference/README.md)
- [模型接入流程与固定版本下载](TUTORIAL.md)
- [Qwen3-VL-2B 模型实现说明](src/inference/models/qwen3_vl_2b/README.md) · [模型级验收](tests/qwen3_vl_2b/README.md)
- [Mage-VL-4B 模型实现说明](src/inference/models/mage_vl_4b/README.md) · [模型级验收](tests/mage_vl_4b/README.md)
- [MediaMTX 部署](deploy/mediamtx/README.md)
- [新模型接入流程](TUTORIAL.md)
- [文档目录与规格说明索引](docs/README.md)

## 开发与验证

模型层模块前向对齐位于各模型 layers/ 文件的 main 入口；每个模型的 Test、Bench、Profile 工作负载及测试说明位于 `tests/<model_name>/`，模型实现说明位于 `src/inference/models/<model_name>/README.md`。

现有测试不执行数据集精度评估。单层算子 profile、功能 smoke 和少量性能样本不构成整模型吞吐保证或 SLA。视频文字 SSE 表示生成文本以事件形式返回，不代表视频输入逐帧返回。
