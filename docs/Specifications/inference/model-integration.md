# 新模型接入操作指南

新模型的程式、设定和依赖放在 `src/inference/models/<model_alias>/`。一般接入需要新增模型目录及 `configs/model.yaml`，并按模型架构实作 `runtime.py` 和必要元件；不需在共用 API、Engine、Launcher 或根清单增加模型分支。

目前已登记 `mage-vl-4b` 和 `qwen3-0.6b`。Mage-VL 使用 Transformers Runtime，提供图片、影片及 Live 能力；Qwen3 使用模型包内 PyTorch 计算图，提供文字聊天。两者共用 API／Engine／Worker 契约，但不能互换模型计算图或 Cache。

## 建议目录

```text
src/inference/models/<model_alias>/
├── runtime.py                 # 必要：create_runtime() 工厂
├── processor.py               # 可选：Tokenizer、Chat Template、模型输入转换
├── model/                     # 可选：模型拓扑
├── components/                # 可选：Runner、Adapter、InputBatch、Cache 等执行元件
├── layers/                    # 可选：模型专属 Attention、Norm、Linear、RoPE
├── utils/                     # 可选：模型资产载入入口
│   └── loader.py              # Config、Tokenizer／Processor、snapshot 及权重载入
├── requirements.txt           # 可选：此模型所需相依套件
└── configs/
    ├── model.yaml             # 必要：模型登录与 API 能力
    ├── inference.yml          # 必要：模型与推论参数
    └── network.yml            # 必要：API 绑定位址
```

MediaMTX 是独立 sidecar，配置集中在专案根目录的 `deploy/mediamtx/mediamtx.yml`，不属于单一模型包。启动、端点同步与协定配置见 [`deploy/mediamtx/README.md`](../../../deploy/mediamtx/README.md)。

依实际需要建立 `model/`、`components/`、`layers/` 和 `utils/loader.py`。将 Transformers 提供的 `PretrainedConfig` 子类、`AutoTokenizer`／`AutoProcessor`、snapshot 准备及必要的权重映射集中在 loader；不要另建冗长 `config.py`、`hf_config.py`、`hf_tokenizer.py` 或 `model_loader.py`。模型服务参数放 YAML，避免把 Scheduler／GPU 服务设定混入 Transformers Config。不要建立空模组，也不要把模型拓扑、权重映射、Tokenizer 或模型 Cache 移进共用 `engine/`。

## 1. 建立 manifest

在 `configs/model.yaml` 登记模型 alias、Runtime 工厂、相对设定档名及对外能力：

```yaml
alias: example-model
display_name: Example Model
integration_profile: custom
backend: example_backend
runtime_factory: models.example_model.runtime:create_runtime
inference_config: inference.yml
network_config: network.yml
api:
  model_id: example-model
  input_processor: models.example_model.processor:ExampleInputProcessor
  capabilities:
    chat_inputs: [text]
    streaming: true
    output_mode: token_ids
    video_upload: false
    live_video: false
```

`model_id` 是推论 API 的识别码；上游 Hugging Face repo ID 设在 `inference.yml` 的 `model.id`，两者不得混用。所有 manifest 设定路径相对于该 manifest 所在的 `configs/` 目录。若输出 `token_ids`，需提供可在 API 程序工作的 `input_processor`，由 API 处理输入 tokenization 和输出 detokenization。只有使用 Live API 时才宣告 `live_video: true`，并登记 Live manager。MediaMTX 不会在模型 manifest 中登记或由 Launcher 启停；部署者使用独立 sidecar 配置启动它。模型 `network.yml` 保存推论服务连接 RTSP／RTMP 的端点和公开主机名称，MediaMTX 服务本身的连接埠及授权回呼由 `deploy/mediamtx/mediamtx.yml` 配置。

Launcher 会自动发现 `models/*/configs/model.yaml`；启动时以 `--model <alias>` 选择模型，不需要修改中央登录档。Gateway 若要公开新模型，另于 Gateway 设定登记路由和能力，并通过跨设定契约检查。

## 2. 建立 Runtime

`runtime.py` 提供 `create_runtime()`，在 GPU Worker 子程序建立单一 Runtime facade：

```python
from engine.worker import ModelRuntime
from models.example_model.components.adapter import ExampleModelAdapter
from models.example_model.components.model_runner import ExampleModelRunner


def create_runtime() -> ModelRuntime:
    runner = ExampleModelRunner()
    adapter = ExampleModelAdapter(runner)
    return ModelRuntime(runner, adapter)
```

Runner 负责模型载入、forward、Sampler 和模型 request/cache state；Adapter 将 Engine work item 转成模型呼叫。Runtime 必须符合 Worker 验证的生命周期、health、生成准备、Cache 绑定、`execute_batch()`、finish 和 cancel 方法。宣告影片或 Live 能力时，还必须实作相应模型操作。`execute_batch()` 按 request ID 回传独立结果；支援 batch API 不等于已实作跨序列单次 forward，能力应以真实 Runner 行为为准。

符合受限纯文字契约的模型可在 Runtime 工厂中包装共用 `StandardTextModelRunner`／`StandardTextModelAdapter`，省略模型专属 Runner 和 Adapter；仍需按现行 Worker 契约回传 Runtime。标准路径限定单装置 decoder-only causal LM、文字 Chat Template、可延续的 `past_key_values`。多模态、特殊 Cache、专属计算图或权重转换使用 `custom`。

## 3. 实作模型元件与 YAML

先固定模型 revision 并确认 Config、Tokenizer／Processor、权重格式及 forward/cache 行为，再依模型分层实作。大型拓扑可放 `model/`，Attention、Norm、RoPE 等层放 `layers/`，模型执行／Cache／批次元件放 `components/`；模型专属 `utils/loader.py` 负责呼叫 Transformers Config、`AutoTokenizer`／`AutoProcessor`，并完成 snapshot 与权重载入。模型参数、Sampler／Scheduler 可调值、记忆体容量和请求上限放 `configs/inference.yml`；API host／port 放 `configs/network.yml`。不要在 Python 程式散落实验参数。

新增模型不一定需要自订全部元件；保留能工作的最小 Runner，只有测量或架构需求证明有必要时才拆出模型层或最佳化替代实作。共用 Scheduler／KV Manager 管理 Engine 的排程与资源租约，模型 Runtime 明确负责其实际 Cache 物件与生命周期。没有 block-table 直接读写实体 page 的 kernel 时，不称为 PagedAttention。

## 4. 预检与测试

```bash
python src/inference/launcher.py list
python src/inference/launcher.py validate --model example-model
python tests/qwen3_0_6b/test_components.py --config tests/qwen3_0_6b/test.yaml
```

`launcher.py validate` 只检查 manifest、YAML、工厂路径／AST 方法和能力相依，不汇入模型工厂、不下载或载入权重、不启动 GPU。新增模型时至少增加以下无权重检查：

- manifest 能被发现且 alias 唯一；各相对设定档存在并可解析；
- Runtime factory 路径和必要方法符合契约；
- manifest 能力与输入 Processor、影片／Live 操作相符；
- SSE token 路径仍包含增量内容、finish usage 和 `[DONE]`，最终文字与非串流结果一致；
- finish、取消、超时和 Worker error 后 Scheduler／Engine lease／模型 Cache 可释放。

每个模型目录在 `tests/{model_alias}/` 维护三个测试入口：`test_components.py` 对照模组树、`test_performance.py` 比较 Inference API 与 Transformers 原生生成、`test_end2end.py` 从 Gateway 验证公开 API；参数和素材路径放该目录的 `test.yaml`，执行时必须明确以 `--config` 传入。元件对照不需权重或 GPU；性能测试需本机固定 snapshot、GPU 和已启动的 Inference API；端到端测试需 Gateway、Inference API、API Key 及模型需要的素材。实测 JSON 放在同目录 `results/`。只报告实际执行证据，不做精度资料集评估。

此外，每个独立计算层、模型组件和模型前向文件应提供文件内 CPU 对照入口：直接在 `if __name__ == "__main__":` 中建立输入、运行实现和基线、打印 `==========engine===========` 与 `==========baseline==========` 并断言结果；不要从入口调用另行封装的测试函数。纯配置／输出结构或无独立数值行为的文件不需要测试入口。带模型 loader 的 Runner 若需指定 manifest／YAML 环境，须在 Inference README 列出可复制的 `python -m` 命令。

## 目前模型包参考

| Alias | 模型包重点 | 能力 |
| --- | --- | --- |
| `mage-vl-4b` | `runtime.py`、`components/`、`processor.py`、`utils/loader.py`、根目录依赖清单 | 图片聊天、文字 SSE、影片上传、Live |
| `qwen3-0.6b` | `runtime.py`、`model/`、`layers/`、`components/`、`utils/loader.py`、`processor.py` | 文字聊天、token ID 输出及 API 端增量 SSE |

规格追踪见 [Inference Spec](spec.md) 中的 `REQ-12`；技术选择与目前结构见 [design.md](design.md)。
