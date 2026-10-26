# MicroInfer

MicroInfer 提供视频理解推理服务和独立的 Go API Gateway。Inference 负责模型加载、输入处理与生成；Gateway 负责客户端接入、认证、限流、模型路由和请求转发。两者分别配置、启动和验收。

## 功能概览

- OpenAI 风格聊天 API，支持文本 SSE 输出。
- Mage-VL-4B 支持图片聊天、视频上传理解和 Live 视频会话。
- Qwen3-0.6B 支持纯文本聊天和标准 function tool calling 协议。
- Inference 使用独立 Engine Core 与 Model Worker 进程；Qwen3 支持批次 Runner 和模型包内可选 CUDA decode attention。
- Gateway 提供 API Key 认证、租户限流、能力检查、SSE 转发、健康检查和 Prometheus 格式指标。

## 模型

| 对外别名 | 模型包 | 已实现能力 | 推理实现 |
| --- | --- | --- | --- |
| `mage-vl-4b` | `src/inference/models/mage_vl_4b/` | 图片聊天、文本 SSE、视频上传、Live | Mage-VL Transformers Runtime |
| `qwen3-0.6b` | `src/inference/models/qwen3_0_6b/` | 文本聊天、文本 SSE、工具调用 | 模型包内 PyTorch 计算图与批次 Runner |

项目别名、Python 模型包名和上游 Hugging Face `repo_id` 是不同标识。别名由客户端选择；上游模型 ID 记录在模型配置中；Inference API 使用 manifest 中的 `api.model_id`。

## 架构

```text
客户端 ──HTTP/SSE──> Go Gateway ──HTTP/SSE──> Inference Network
                         │                          │
                         │                          ├── Engine Core
                         │                          └── Model Worker / 模型插件
                         └── Live 控制 API               │
客户端 ─────────────RTSP/RTMP──────────────> MediaMTX
```

`src/inference/network/` 提供共用 HTTP 服务和路由；`engine/` 管理请求调度、KV block、Worker 与进程通信；`processors/` 承载跨模型视频和 Live 处理能力；模型专属 Runtime、Processor、Runner、拓扑和算子位于 `models/<model>/`。Gateway 不加载模型，也不转发 RTSP/RTMP 媒体封包。

项目参考 vLLM 和 nano-vLLM 的职责划分，不提供 vLLM backend，也没有 Prefix Caching。Mage-VL 的 Engine KV 是逻辑 reservation。Qwen3 使用实体 KV page pool；PyTorch Attention 仍会 gather KV pages。Qwen3 模型包提供可选 native CUDA 单 token decode attention，但当前测量慢于 PyTorch 路径，默认配置仍使用 PyTorch。

## 快速开始

### 环境准备

- Python 3.10 或更新版本。
- Go 工具链，用于运行 Gateway。
- 与目标设备匹配的 PyTorch/CUDA 环境。PyTorch wheel 需单独安装；项目不会替用户选择 CUDA wheel。
- 模型权重可由 Inference 按固定 revision 获取，也可按模型配置准备在本地 `checkpoints/`。

### 安装并检查 Inference

在所选模型的 Python 环境中，从仓库根目录安装对应 extra：

```bash
# Mage-VL
python -m pip install -e "src/inference[mage-vl]"

# 或 Qwen3
python -m pip install -e "src/inference[qwen3]"
```

检查模型并启动 Inference：

```bash
inference list
inference validate --model qwen3-0.6b
inference start --model qwen3-0.6b
```

Inference 默认监听地址由模型的 `configs/network.yml` 决定：Mage-VL 为 `127.0.0.1:8001`，Qwen3 为 `127.0.0.1:8011`。`validate` 不载入权重或初始化 GPU。推理优先使用 CUDA；仅在 CUDA 不可用时回退 CPU。

### 启动 Gateway

Gateway 默认监听 `127.0.0.1:8000`。按 [Gateway README](src/gateway/README.md) 配置 API Key 和 HMAC 密钥，再从 `src/gateway/` 启动：

```bash
go run ./cmd/gateway --config configs/gateway.yml
```

当前 Gateway 配置包含两个模型的上游地址。需要 `/readyz` 返回就绪时，两个对应的 Inference 服务都必须已启动。

## 配置

| 配置 | 用途 |
| --- | --- |
| `src/inference/models/<model>/configs/model.yml` | 模型别名、API 能力和 Runtime 工厂 |
| `src/inference/models/<model>/configs/inference.yml` | 模型、Scheduler、KV、请求限制和 Runtime 参数 |
| `src/inference/models/<model>/configs/network.yml` | Inference 监听地址及模型使用的媒体端点 |
| `src/gateway/configs/gateway.yml` | Gateway 监听地址、模型路由、政策和上游设定 |
| `deploy/mediamtx/mediamtx.yml` | 独立 MediaMTX sidecar 配置 |

密钥通过环境变量或仓库外的受保护文件提供，不要提交到 Git。模型权重和 Hub 缓存位于被忽略的 `checkpoints/` 目录。

## 文档

- [Inference 服务、模型引擎和模型接入](src/inference/README.md)
- [Gateway 配置、认证和运行](src/gateway/README.md)
- [逐模型测试与算子分析](tests/README.md)
- [模型下载与离线分析工具](scripts/README.md)
- [MediaMTX 部署](deploy/mediamtx/README.md)
- [新模型接入指南](docs/Specifications/inference/model-integration.md)
- [Qwen3 CUDA attention 说明](src/inference/models/qwen3_0_6b/kernels/cuda/README.md)

## 开发与验证

逐模型测试入口位于 `tests/<model_alias>/`。组件检查不载入完整模型权重；性能测试需要模型权重、GPU 和 Inference API；端到端测试需要 Gateway、Inference、API Key 及相应素材。具体命令和前置条件见 [Tests README](tests/README.md)。

现有测试不执行数据集精度评估。单层算子 profile、功能 smoke 和少量性能样本不构成整模型吞吐保证或 SLA。视频文字 SSE 表示生成文本以事件形式返回，不代表视频输入逐帧回传。
