# MicroInfer

MicroInfer 提供可直接对外服务的 Python FastAPI 推理 API。共用 Network 层负责 API Key 认证与限流；模型 Runtime 负责加载模型和 GPU 推理。当前仓库不包含 API Gateway 或控制面。

## 功能概览

- OpenAI 风格聊天 API，支持文本 SSE 输出。
- Mage-VL-4B 支持图片聊天、视频上传理解和 Live 视频会话。
- Qwen3-VL-2B 支持文字、图片聊天和流式文字输出。
- Inference 使用独立 Engine Core 与 Model Worker 进程；模型前向由各自 Transformers Runtime 执行。
- FastAPI 直接提供 OpenAI 兼容 API、API Key 认证、租户限流、SSE、视频上传和 Live API。

## 模型

| API model ID | 模型包 | 已实现能力 | 推理实现 |
| --- | --- | --- | --- |
| `mage-vl-4b` | `src/inference/models/mage_vl_4b/` | 图片聊天、文本 SSE、视频上传、Live | Mage-VL Transformers Runtime |
| `qwen3-vl-2b` | `src/inference/models/qwen3_vl_2b/` | 文字／图片聊天、文本 SSE | Qwen3-VL Transformers Runtime |

API model ID 使用仓库模型名，如 `mage-vl-4b`；Python 包目录和 Hugging Face `repo_id` 是模型加载标识。API 服务直接运行在模型对应的 FastAPI 进程中。

## 架构

```text
客户端／控制面 ──HTTP/SSE + Bearer API Key──> Python FastAPI Network
                                                   ├── API Key / tenant 限流
                                                   ├── Engine Core
                                                   └── Model Worker / 模型插件
客户端 ─────────────RTSP/RTMP──────────────> MediaMTX
```

`src/inference/network/` 提供直接对外的共用 HTTP 服务和认证；`engine/` 管理请求调度、KV block、Worker 与进程通信；`processors/` 承载跨模型视频和 Live 处理能力；模型专属 Runtime、Processor、Runner、拓扑和算子位于 `models/<model>/`。控制面由独立项目负责。

项目参考 vLLM 和 nano-vLLM 的职责划分，不提供 vLLM backend，也没有 Prefix Caching。当前模型使用 Transformers Cache；Engine KV block 是逻辑 reservation，不是实体 paged KV 内存池。

## 快速开始

### 环境准备

- Python 3.10 或更新版本。
- 与目标设备匹配的 PyTorch/CUDA 环境。PyTorch wheel 需单独安装；项目不会替用户选择 CUDA wheel。
- 当前模型配置使用本地固定 revision 快照并启用离线加载；启动服务前按[模型接入流程](TUTORIAL.md#2-配置环境并下载固定版本)准备 `checkpoints/` 中的权重。

### 安装并检查 Inference

在所选模型的 Python 环境中，从仓库根目录安装对应 extra：

```bash
# Mage-VL
python -m pip install -e "src/inference[mage-vl]"

# 或 Qwen3-VL
python -m pip install -e "src/inference[qwen3-vl-2b]"
```

检查模型并启动 Inference：

```bash
inference list
inference validate --model qwen3-vl-2b
export INFERENCE_API_KEYS_FILE="$HOME/.config/microinfer/api-keys.json"
inference serve --model qwen3-vl-2b --host 0.0.0.0
```

Inference 默认监听地址由模型 `profiles/network.yml` 决定：Mage-VL 为 `127.0.0.1:8001`，Qwen3-VL 为 `127.0.0.1:8012`。`validate` 不加载权重或初始化 GPU。推理优先使用 CUDA；仅在 CUDA 不可用时回退 CPU。对远程客户端开放时应设置 `--host 0.0.0.0` 并配置防火墙。API Key 文件示例和认证行为见 [Inference README](src/inference/README.md)。

## 配置

| 配置 | 用途 |
| --- | --- |
| `src/inference/models/<model>/profiles/model.yml` | 模型别名、API 能力和 Runtime 工厂 |
| `src/inference/models/<model>/profiles/inference.yml` | 模型、Scheduler、KV、请求限制和 Runtime 参数 |
| `src/inference/models/<model>/profiles/network.yml` | Inference 监听地址及模型使用的媒体端点 |
| `src/inference/models/<model>/profiles/network.yml` 的 `api.auth` | API Key 文件环境变量、tenant 限流和 Live HMAC 配置 |
| `deploy/mediamtx/mediamtx.yml` | 独立 MediaMTX sidecar 配置 |

密钥通过环境变量或仓库外的受保护文件提供，不要提交到 Git。模型权重和 Hub 缓存位于被忽略的 `checkpoints/` 目录。

## 文档

- [Inference 服务、模型引擎和模型接入](src/inference/README.md)
- [模型接入流程与固定版本下载](TUTORIAL.md)
- [Qwen3-VL-2B 模型实现说明](src/inference/models/qwen3_vl_2b/README.md) · [模型级验收](tests/qwen3_vl_2b/README.md)
- [Mage-VL-4B 模型实现说明](src/inference/models/mage_vl_4b/README.md) · [模型级验收](tests/mage_vl_4b/README.md)
- [MediaMTX 部署](deploy/mediamtx/README.md)
- [新模型接入流程](TUTORIAL.md)
- [文档目录与规格说明索引](docs/README.md)

## 开发与验证

FastAPI API Key/tenant/Live 句柄自检使用 `python -m inference.network.auth`。模型层模块前向对齐位于各模型 layers/ 文件的 main 入口；每个模型的 Test、Bench、Profile 工作负载及测试说明位于 `tests/<model_name>/`，模型实现说明位于 `src/inference/models/<model_name>/README.md`。

现有测试不执行数据集精度评估。单层算子 profile、功能 smoke 和少量性能样本不构成整模型吞吐保证或 SLA。视频文字 SSE 表示生成文本以事件形式返回，不代表视频输入逐帧返回。
