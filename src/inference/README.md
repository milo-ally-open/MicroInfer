# Inference 服务

Inference 是可直接对外提供服务的 Python FastAPI 模型服务，负责 API Key 认证、tenant 限流、模型加载、请求预处理、生成和 GPU 资源管理。

## 当前模型

| API model ID | 上游权重仓库 | 能力 | 模型实现 |
| --- | --- | --- | --- |
| `mage-vl-4b` | `microsoft/Mage-VL` | 图片聊天、视频上传、Live、文字 SSE | Transformers Runtime 与模型专属多模态处理 |
| `qwen3-vl-2b` | `Qwen/Qwen3-VL-2B-Instruct` | 文字／图片聊天、文字 SSE | Transformers `Qwen3VLForConditionalGeneration`、`AutoProcessor`、`DynamicCache` |

Qwen3-VL 的 Runner 管理逐步生成和模型所需的 MRoPE 状态。视觉 Patch Embedding、Attention、MLP、RoPE、Merger，以及文字 Embedding、Attention、MLP、Norm、MRoPE 和 Decoder Layer 在模型包 `layers/` 中独立实现，并通过固定组件槽位装配；每个层模块的 main 可对照 Transformers 前向。注意力 backend 仍由 Transformers 选择。Engine KV block 是调度层逻辑 reservation，并非实体 paged KV pool。

## 架构

API Server、Engine Core 和 Model Worker 分进程运行。API Server 通过 `network/` 提供 HTTP/SSE；Engine Core 负责请求生命周期、调度和 KV reservation；Worker 执行模型前向。模型专属 Runtime、Processor、Runner 和配置位于 `models/<model>/`。

| 组件 | 位置 | 职责 |
| --- | --- | --- |
| Network | `network/` | FastAPI app、API Key/tenant 认证、限流、HTTP schema、路由与序列化 |
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
inference serve --model qwen3-vl-2b --host 0.0.0.0
```

`list` 显示自动发现的模型；`validate` 检查 manifest 和 YAML，不加载权重、不初始化 GPU；`serve` 加载并对外服务指定模型。`start` 仍是 `serve` 的命令别名。监听地址在各模型 `profiles/network.yml` 中配置。Qwen3-VL 默认 `127.0.0.1:8012`，Mage-VL 默认 `127.0.0.1:8001`；示例中的 `0.0.0.0` 允许受防火墙保护的远程客户端连接。

### API Key 配置

所有 `/v1/*` 模型 API（`/v1/models`、聊天、视频和 Live session）都要求 Bearer API Key。健康检查 `/health` 保持免认证；MediaMTX 回调 `/v1/live/mediamtx/auth` 为媒体服务专用例外，必须通过防火墙限制在 MediaMTX 可访问的内网范围。应用会进一步核对会话路径是否属于有效 Live session。

从仓库根目录创建一份仓库外的密钥文件，并将权限限制为当前服务用户可读：

```bash
mkdir -p "$HOME/.config/microinfer"
cp deploy/api-keys.example.json "$HOME/.config/microinfer/api-keys.json"
chmod 0600 "$HOME/.config/microinfer/api-keys.json"
export INFERENCE_API_KEYS_FILE="$HOME/.config/microinfer/api-keys.json"
```

把模板中的 `keys` 换成 `openssl rand -hex 32` 输出的值；每项使用 `ids`、`tenant_ids` 和 `keys` 字段。服务启动时读取文件并只在内存保留密钥摘要。Mage-VL 启用 Live 时还需设置 `INFERENCE_SESSION_HMAC_KEYS`，格式为逗号分隔的 Base64URL 随机密钥，解码后每个至少 32 bytes。不要把密钥写入模型 YAML 或 Git。若使用 systemd 等服务管理器，应将文件复制到服务账户可读的受保护路径，并将环境变量配置在服务单元中。

```bash
export INFERENCE_SESSION_HMAC_KEYS="$(python -c 'import base64,secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("="))')"
```

API Key 所属 tenant 使用 YAML 中的 token bucket 限流；默认每秒 5 次、突发 10 次。控制面仍负责用户 JWT、业务权限及任务配额；控制面在服务端持有推理 API Key，不把它发送给浏览器。

## Engine Client 示例

模型依赖和本地 checkpoint 必须已准备好。模型专属的完整流式示例位于 `models/<model_name>/example.py`；Qwen3-VL 图文示例可从仓库根目录运行：

```bash
python src/inference/models/qwen3_vl_2b/example.py
```

示例将仓库根目录的 `test.jpg` 发给模型，并逐段打印生成文本。模型 API 使用 OpenAI Chat Completions 兼容格式；工具调用是否可用以该模型的 manifest capability 为准，当前两个模型均未声明工具调用。

## 接入新模型

遵循仓库根目录的 [新模型接入流程](../../TUTORIAL.md)。模型目录顶层保留 `README.md`、`example.py`、`loader.py`、`modeling.py` 和 `runtime.py`；YAML 归档在 `profiles/`，模型依赖归档在 `requirements/`，专属模型能力实现放在 `architecture/`。运行组件和算子分别放在 `components/`、`layers/`。`scripts/client.py` 直接调用 FastAPI；层模块对齐入口在模型包 `layers/`；Test、Bench、Profile 脚本和分析报告按模型目录名放在 `tests/<model_name>/` 与 `analysis/<model_name>/`，模型实现说明放在 `src/inference/models/<model_name>/README.md`，测试说明分别放在各模型的 `tests/<model_name>/README.md`。

## 限制

当前模型使用 Transformers Runtime；项目不宣称 vLLM 兼容，也未提供 Prefix Caching。功能 smoke、模块 Profile 和少量性能样本不构成数据集精度评估或生产 SLA。
