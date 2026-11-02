# Inference 服务

Inference 是独立的 Python 模型服务，负责模型加载、请求预处理、生成和 GPU 资源管理。Go Gateway 负责外部认证、路由和限流；模型能力与上游地址按模型登记。

## 当前模型

| Alias | 上游模型 | 能力 | 模型实现 |
| --- | --- | --- | --- |
| `mage-vl-4b` | `microsoft/Mage-VL` | 图片聊天、视频上传、Live、文字 SSE | Transformers Runtime 与模型专属多模态处理 |
| `qwen3-vl-2b` | `Qwen/Qwen3-VL-2B-Instruct` | 文字／图片聊天、文字 SSE | Transformers `Qwen3VLForConditionalGeneration`、`AutoProcessor`、`DynamicCache` |

Qwen3-VL 的 Runner 管理逐步生成和模型所需的 MRoPE 状态。视觉 Patch Embedding、Attention、MLP、RoPE、Merger，以及文字 Embedding、Attention、MLP、Norm、MRoPE 和 Decoder Layer 在模型包 layers/ 中独立实现，并通过固定组件槽位装配；每个层模块的 main 可对照 Transformers 前向。注意力 backend 仍由 Transformers 选择。Engine KV block 是调度层逻辑 reservation，并非实体 paged KV pool。

## 架构

API Server、Engine Core 和 Model Worker 分进程运行。API Server 通过 `network/` 提供 HTTP/SSE；Engine Core 负责请求生命周期、调度和 KV reservation；Worker 执行模型前向。模型专属 Runtime、Processor、Runner 和配置位于 `models/<model>/`。

| 组件 | 位置 | 职责 |
| --- | --- | --- |
| Network | `network/` | FastAPI app、HTTP schema、路由与序列化 |
| 共用 Processors | `processors/` | 跨模型的视频和 Live 处理能力 |
| Engine | `engine/` | Scheduler、KV block、Worker、IPC 与启动配置 |
| 模型插件 | `models/<model>/` | 模型输入、前向、Cache 和权重相关逻辑 |

模型自己的 tokenizer、chat template、视觉前处理留在模型包中。共用 `processors/` 仅存放跨模型复用的处理功能。

## 安装

需要 Python 3.10+ 和与目标设备匹配的 PyTorch/CUDA。PyTorch wheel 应先按目标机器单独安装。只部署 Qwen3-VL 时：

```bash
conda activate inference
python -m pip install -e "src/inference[qwen3-vl-2b]"
```

只部署 Mage-VL 时安装 `src/inference[mage-vl]`；同一环境同时部署两者时安装两个 extra。每个模型依赖单独声明。推理默认使用 CUDA，仅当 CUDA 不可用时回退 CPU。

## 检查与启动

从仓库根目录运行：

```bash
inference list
inference validate --model qwen3-vl-2b
inference start --model qwen3-vl-2b
```

`list` 显示自动发现的模型；`validate` 检查 manifest 和 YAML，不加载权重、不初始化 GPU；`start` 加载指定模型。监听地址在各模型 `profiles/network.yml` 中配置。Qwen3-VL 默认 `127.0.0.1:8012`，Mage-VL 默认 `127.0.0.1:8001`。

## Engine Client 示例

模型依赖和本地 checkpoint 必须已准备好。模型专属的完整流式示例位于 `models/<model_name>/example.py`；Qwen3-VL 图文示例可从仓库根目录运行：

```bash
python src/inference/models/qwen3_vl_2b/example.py
```

示例将仓库根目录的 `test.jpg` 发给模型，并逐段打印生成文本。模型 API 使用 OpenAI Chat Completions 兼容格式；工具调用是否可用以该模型的 manifest capability 为准，当前两个模型均未声明工具调用。

## 接入新模型

遵循仓库根目录的 [新模型接入流程](../../TUTORIAL.md)。模型目录顶层保留 `example.py`、`loader.py`、`modeling.py` 和 `runtime.py`；YAML 归档在 `profiles/`，模型依赖归档在 `requirements/`，专属模型能力实现放在 `architecture/`。运行组件和算子分别放在 `components/`、`layers/`。仓库根目录 `client.py` 仅经 Go Gateway 做请求冒烟；层模块对齐入口在模型包 layers/；Test、Bench、Profile 脚本和分析报告按模型能力放在 scripts/<alias>/ 与 analysis/<alias>/。

## 限制

当前模型使用 Transformers Runtime；项目不宣称 vLLM 兼容，也未提供 Prefix Caching。功能 smoke、模块 Profile 和少量性能样本不构成数据集精度评估或生产 SLA。
