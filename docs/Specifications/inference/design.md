# 技术设计：五层推理引擎

> 提案：[proposal.md](proposal.md)｜行为规格：[spec.md](spec.md)｜任务与验收：[tasks.md](tasks.md)

## 参考资料与本地证据

vLLM V1 官方架构将 API Server、Engine Core 和 GPU Worker 分成不同程序；API Server 执行输入处理并串流结果，Engine Core 执行 Scheduler、管理 KV Cache 并协调 Worker，Worker 载入模型并执行 forward。Model Runner 负责模型输入张量和模型执行细节。参考：[vLLM V1 Architecture Overview](https://docs.vllm.ai/en/v0.19.0/design/arch_overview/)。本设计采用这些责任边界，不复制 vLLM 的内部类别、IPC、Executor 或 Kernel，也不声称 vLLM 相容。

[nano-vLLM README](https://github.com/GeeeekExplorer/nano-vllm) 将自身描述为精简、可读的 vLLM 实作，展示其以较小程式结构承载 Engine、批次和最佳化概念。本专案参考其「用可读的核心流程学习／组织 Engine」方向；不把 README 中宣称的功能或 benchmark 当成本专案能力，也不将 nano-vLLM 加入执行依赖。

本地现况以提交 `0cb3c56` 的 Qwen3-0.6B 原生计算图为基线：固定 revision 的 config／tokenizer／Safetensors 载入、自有 Qwen3 PyTorch 模型、批次 prefill/decode、Worker 实体 KV pool 和文字 SSE 已存在。Qwen3 的 RTX 4060 算子测量保存在 `analysis/qwen3_0_6b/`；Mage-VL Gateway 历史记录保存在 `analysis/mage_vl_4b/`。这些记录不能替代本次 token event／Executor／InputBatch 的回归验收。Mage-VL 仍使用 Transformers 模型路径。此处不把 SGLang、vLLM backend、PagedAttention Kernel、Ray 或第二个原生模型写成已完成。

## 五层责任与相依

| 层 | 责任 | 本地程式边界 | 不拥有 |
|---|---|---|---|
| L1 接口与门面 | HTTP、OpenAI schema、模型输入 Tokenize、增量 Detokenize、SSE 和 Engine IPC client | `api/application/`、`api/routes/`、`api/contracts/`、`api/processors/`、`engine/transport/` | GPU tensor、Scheduler、KV pool、模型 forward |
| L2 编排与调度 | Engine Core 请求状态、准入、Scheduler、逻辑 Cache reservation、Worker 结果事件化 | `engine/core/`、`engine/scheduling/`、`engine/cache/` | HTTP/SSE schema、实体 GPU tensor、模型拓扑 |
| L3 执行抽象与 Worker | Executor 管理本机程序和 RPC；Platform 解析装置与资源观测；Worker 程序载入 Runner | `engine/core/ports.py`、`engine/execution/`、`engine/bootstrap/` | request policy、模型输入格式、HTTP |
| L4 模型运行时 | Runner 组装模型输入批次、forward、Cache 存取和 Sampling | 各模型 `components/model_runner.py`、模型批次及 Cache 元件 | OpenAI request／response、模型别名路由 |
| L5 模型与算子 | 模型拓扑、层、Attention、权重映射和上游相容 | 各模型 `model/`、`layers/`、`utils/` | 共用 HTTP API、Engine 伫列和租户政策 |

模型生态相容是跨层责任：API 端的 Chat Template／Tokenizer 在 L1 依模型建立输入；模型 config、Safetensors 权重映射和计算图在 L5。通用影片读取与 Live session 仍由既有 `media/` 持有；Mage-VL 的影片采样和 Processor 留在 Mage 模型包。五层以程式职责划界，不代表五个新 package 或五个 OS process。

## 共用 Inference package 边界

共用服务保持顶层 `api/` 与 `engine/` 两个可独立搬移的 Python package。文件按变化原因归档，模型包和 Gateway 不进入这两个 package：

```text
api/
├── application/       # create_app 与依 manifest 装配的 service_app
├── contracts/         # HTTP schemas、response serializers、共用 tool protocol
├── processors/        # InputProcessor、Detokenizer、standard chat processor
└── routes/            # HTTP endpoints，仅组合 API contracts/processors

engine/
├── core/              # InferenceEngine、EngineOutputProcessor、Executor port
├── contracts/         # API／Engine／Worker／模型 Adapter 间的可序列化类型与错误
├── scheduling/        # Token Scheduler 与 batch planner
├── cache/             # Engine 逻辑 KV reservation
├── execution/         # LocalProcessExecutor、Worker、Platform、standard runtime
├── transport/         # API Server 到 Engine Core 的 IPC client
├── bootstrap/         # Engine Core 与 Model Worker composition root
└── config/            # manifest、YAML 与 model factory 读取
```

依赖方向由组合入口向下：API application 可组装 routes、processors、Engine transport；API processors 通过 `engine/contracts/` 传递 Engine 输入／错误，不导入 Engine Core 或 Worker 实作。`engine/core/` 面向 contracts、Scheduler、Cache、`core/ports.py` 定义的 Executor Protocol 及 `media/` 共用上传档案辅助，不直接依赖具体本机 Executor；进程执行细节留在 `execution/`，实例组合留在 `bootstrap/`。模型包可使用稳定的 Engine contracts、配置读取和执行能力，但不反向依赖 API routes、Engine Core 实作或其他模型包。各 package 以 `__init__.py` 明确边界，不用根目录兼容 wrapper 维持过时的平面模块路径。

## 程序拓朴

```text
API Server process (L1)
  FastAPI routes / schemas / SSE serializer
  Qwen3InputProcessor + IncrementalDetokenizer
  EngineClient
       │ local duplex Pipe
       ▼
Engine Core process (L2)
  InferenceEngine → TokenScheduler → EngineOutputProcessor
       │
       └── LocalProcessExecutor (L3) ── local duplex Pipe
                                          │
                                          ▼
Model Worker process (L3)
  Qwen3ModelAdapter → Qwen3ModelRunner (L4) → Qwen3ForCausalLM (L5)
  CUDA-first Platform + selected-device KV page pool
```

仍为 API Server、Engine Core、单一 Model Worker 三个程序。`LocalProcessExecutor` 目前只管理一个 `spawn` 子程序；介面为未来更换执行拓朴留出位置，没有多 GPU、Ray 或 Worker pool 实作。Engine Core 不汇入模型包，模型包 factory 只在对应程序建立该模型所需元件。

## 请求与文字串流

1. Qwen app 将 `Qwen3InputProcessor` 注入共用 `create_app()`。API lifespan 先从 YAML 指定的本机 snapshot 载入 Tokenizer，再启动 EngineClient；此处只载入 CPU tokenizer，不载入权重或 GPU Runtime。
2. Chat route 验证 model alias，呼叫模型 `InputProcessor` 形成 `PreparedChatInput(model_alias, input_ids, sampling_params)`，再透过 EngineClient 传入 Engine Core。资料结构只包含可 pickle 的小型 Python 资料，不包含 tokenizer、GPU tensor 或模型 Cache。
3. Engine Core 将 PreparedChatInput 传给 Worker Adapter。Adapter 建立 request state，Engine 保留 token 容量 reservation，Scheduler 排入 prefill/decode 批次。
4. Worker 的 Qwen Model Runner 以 `InputBatch` 组装 token tensor、长度、position offset 和 block table，呼叫本地 Qwen forward 及 Sampler。生成步骤只回传 token ID；Mage-VL 仍回传文字增量。
5. Engine `EngineOutputProcessor` 依 request ID 将 token ID 或文字增量转成 Engine stream event。EngineClient 将事件送回 API process。
6. Qwen API 的 `IncrementalDetokenizer` 累积 token IDs 并对完整前缀解码；当前前缀若含不完整 Unicode replacement，先缓冲该位置之后的文字。可安全输出的 delta 立即序列化成 OpenAI SSE chunk；finish 时 flush 剩余内容，再送 finish usage 和 `[DONE]`。已送出的前缀若被 tokenizer 改写，输出处理器明确失败，避免静默重复／漏字。
7. HTTP SSE 的取消仍由 route 关闭 EngineClient stream，依 Engine cancellation path 停止 request 并释放 Scheduler／Cache／Worker 状态。一般非串流请求由 Worker 回传 `InferenceResult`；Qwen token IDs 在 API L1 解码成 `content` 后，再沿用现有 OpenAI response serializer。

InputProcessor／Tokenizer 位于 API process，因此 Qwen API 和 Worker 各保留一份小型 tokenizer instance；这是 Tokenize 与模型载入隔离的首版成本。固定 snapshot 验收需确认其 Chat Template 结果与 Worker 最终 decode 使用同一 revision。

## L2 排程与 KV 所有权

Engine Core 保持接收、准入、排程、Executor 执行、更新状态、发布事件和收尾流程。`EngineOutputProcessor` 只转换 Worker result；request terminal state、finish operation、取消和 Cache release 仍由 Engine Core 管理。

Token Scheduler 继续依 `max_num_seqs`、等待上限、`max_num_scheduled_tokens`、prefill chunk 和 batch wait 设定建批；批次按 request ID 回传结果。混合 prefill/decode 可同一次 Worker RPC 传输，Runner 仍按 phase 分批 forward。Sampler 保持在 L4 Worker，模型产生的 token 不交由 API 重新采样。

`KVCacheManager` 是 L2 逻辑资源管理器。Mage-VL 对应 Transformers Cache request state；Qwen Engine block ID 映射到 Worker 内的实体 page pool。Qwen Attention 目前 gather paged K/V 后以分组矩阵乘法执行 GQA，因此本地 pool 具有物理分页存储，但不是直接 block-table kernel。Engine 的 reservation／容量统计与 Worker 实际页面使用量分开回报。

## L3 Executor 与 Platform

`Executor` 是 Engine Core 使用的 process／execution contract。`LocalProcessExecutor` 拥有 spawn、Pipe RPC、ready handshake、request timeout、error transport 和 close。`worker.py` 保留子程序内执行模型 Adapter 的 `ModelWorker` contract 与 `InProcessModelWorker`；不得让 Engine Core直接依赖 Worker process 细节。

`Platform` 是 Worker Model Runner 使用的 device／memory contract。模型优先使用 CUDA，只有 CUDA 不可用时才回退到 CPU。Qwen YAML 的 `runtime.platform: torch_cuda` 由 `create_platform()` 建立 `TorchCudaPlatform`；装置选择、VRAM／peak memory 观测及 CUDA cache 清理经该介面执行。未知名称在 Runner 载入前明确失败；Ray、NPU 和多 GPU 不在支援范围内。

## L4 InputBatch 与模型运行时

`InputBatch.build()` 依排程工作项目建立本轮的 padded token ID tensor，并检查 request 数量、token length、position offset 和 block table 一一对应。Qwen forward 仍接收 per-request 长度／偏移／block tables，padding 位置不参与有效 token 计算。

InputBatch 暂时每轮动态分配 GPU tensor。后续若性能样本确认 CPU tensor 组装或分配是瓶颈，可将容量预配置、Pinned Memory、非同步 H2D 和 CUDA Graph 作为独立最佳化，需先保持相同的 request mapping／SSE／Cache acceptance。首版不加入这些未测机制。

## L5 Qwen 模型与生态相容

Qwen 模型拓扑及 Attention 等算子留在模型包；模型专属 `utils/` 提供上游 config／tokenizer／固定 revision 权重格式适配等辅助功能。Safetensors loader 在 Worker ready 前验证 metadata、revision、必要 tensor 和 shape。模型 manifest 登记 alias、设定路径、对外 API ID、Runtime 工厂和选用 API 扩充；共用 route 不以模型 alias 分支。

模型设定使用与架构相符的 Transformers config 类别：Qwen3 以 `Qwen3Config.from_pretrained()` 载入固定本机 revision，并将 `Qwen3Config` 传入原生模型计算图；Mage-VL 使用模型专属 `MageVLConfig`，其中固定 revision 的文字骨干由 `Qwen3Config` 表示。Tokenizer 与多模态 Processor 依固定 revision 的 `tokenizer_config.json`／`processor_config.json` 等 metadata，透过 `AutoTokenizer`／`AutoProcessor` 建立；这保留 Hugging Face 对不同模型 tokenizer 类别的解析能力，不代表模型 forward 由 `AutoModel` 执行。Qwen Runner 的 GQA、Mage 视觉 Attention、SigLIP pooling Attention 与 Mage 文字骨干 Attention 均在模型包以矩阵乘法、遮罩及 softmax 明确实作；Mage 文字骨干另有独立 Qwen3 Attention、RMSNorm、MLP 与 Decoder layer，使用分块 Query 计算控制暂存记忆体。所有载入仍使用本机 snapshot 与固定 revision，不可在服务启动时解析浮动版本或隐式下载。

## 新模型接入与共用装配

`api/application/service_app.py` 读取 Launcher 设定的模型 alias，建立共用 `EngineClient`、API request limits、输入处理器和选用 Live manager。`engine/bootstrap/model_bootstrap.py` 从 manifest 载入模型 Runner／Adapter 工厂，建立 `InferenceEngine`、`LocalProcessExecutor` 和 `InProcessModelWorker`。Runner／Adapter 工厂使用 `module:attribute` 字串；工厂仅在相应程序中载入，因此 API process 不会汇入模型 Runner 或 GPU Runtime。

模型包的 `configs/model.yml` 中 `api.model_id` 是推论 API 接受及回传的识别码；上游 model repo ID 留在 `inference.yml` 的 `model.id`。`api.capabilities` 明确列出聊天输入模态、文字串流、`text_delta`／`token_ids` 输出、影片上传及 Live 能力；API app 依此挂载路由并拒绝不支援的输入。`api.input_processor` 与 `api.live_session_manager` 是依能力需要登记的工厂。特殊前处理、Tokenizer、权重载入、模型拓扑和 cache 逻辑仍由模型包负责。

新模型最少实作 Runner／Adapter 并登记工厂和 YAML；不需写模型专属 `app.py` 或 `engine_core.py`。逐步生成 Adapter 维持 Engine 的 prepare／prefill／decode／finish／cancel 契约，Runner 持有模型 cache state 并依 request ID 回传批次结果。共用 KV Manager 的租约不替模型宣称支援相同的物理 cache 或 batch 语义。各文件职责、Runtime 工厂接口和完整接入步骤见[新模型接入指南](model-integration.md)。

Mage-VL 与 Qwen 均由共用入口装配；Mage 的 `model/mage_vl.py` 明确组合视觉塔、Qwen3 文本骨干与多模态输入，`vision_features.py`、`multimodal_inputs.py`、`streammind.py` 分别负责特征、placeholder 注入与 Gate；视觉层按 RoPE、patch 表示、注意力、Encoder 堆叠分置于 `layers/`。`conditional_generation.py` 维护 Transformers 生成契约，`components/` 仍负责逐步生成／替换注册等 Runtime 元件；`multimodal_model.py` 和 `vision_encoder.py` 只保留兼容导出。Qwen 拓扑与层亦各自分置于 `model/` 和 `layers/`。模型目录不保留未使用的 `app.py`／`engine_core.py` 包装档。

### 标准文字模型预设元件

`integration_profile: standard` 可省略模型专属 Runner、Adapter 和 `api.input_processor`。API Server 使用 `api/processors/default_chat_processor.py`；Worker 使用 `engine/execution/standard_text.py` 中的 `StandardTextModelRunner`／`StandardTextModelAdapter`。模型仍需提供薄的 `runtime.py` 工厂，以便共用 Worker 使用一致的 Runtime facade。

预设 Runner 以 `AutoModelForCausalLM` 载入固定 revision，仅支援单一计算装置，并优先使用 CUDA；CUDA 不可用时回退 CPU。模型类型仍限 decoder-only causal LM、可重用 `past_key_values`、文字 `chat_template` 和 token ID 输出。API 处理器只接收文字，不接受图片、影片或 Live。Scheduler 提交的 `execute_batch()` 仍按 request 逐项呼叫 forward，各 request 的 Cache／生成状态分开持有；这是可用的标准接入路径，不代表连续批次 forward、实体 block cache 或 PagedAttention。

预检以 YAML 与 AST 确认 capability／方法合同，不 import 工厂或权重。Worker 在模型载入前再次检查实例方法；如架构使用 encoder-decoder、`device_map` 多装置、不可重用 Cache，或需要自订采样／影音流程，应改登记 `custom` 并提供模型 Runner／Adapter。即使预检通过，将模型标成标准相容前仍需以固定 revision snapshot 实际验证 Cache 延续和 token 输出。

## 模型包结构与 manifest

模型专属程式、设定和依赖以 `models/{model_alias}/` 为唯一扩充边界。共用 `api/`、`engine/`、`media/` 不按模型 alias 分叉；Launcher 只负责发现、检查与启动，模型 Runtime 工厂在 Worker 子程序建立模型实作。

```text
models/{model_alias}/
├── runtime.py                 # create_runtime()，组合单一模型 Runtime facade
├── processor.py               # 可选：Tokenizer／Chat Template／模型输入处理
├── model/                     # 可选：模型拓扑
├── components/                # 可选：Runner、Adapter、批次与 Cache 支援元件
├── layers/                    # 可选：模型专属 Attention、Norm、Linear、RoPE
├── utils/                     # 模型 Config／Tokenizer／Processor、snapshot 与权重载入
│   └── loader.py
├── requirements.txt           # 可选：模型相依套件
└── configs/
    ├── model.yml             # alias、能力、Runtime 工厂及设定档名
    ├── inference.yml           # 模型与推论参数
    └── network.yml            # API 绑定位址
```

MediaMTX 配置位于项目根目录的独立部署目录：

```text
deploy/mediamtx/
├── mediamtx.yml               # 独立 MediaMTX sidecar 服务配置
└── README.md                  # 独立启停与端点同步方式
```

每个模型的 `configs/model.yml` 是该模型唯一的登记入口。Launcher 扫描 `models/*/configs/model.yml` 发现模型，并拒绝重复 alias；`list`／`validate` 可处理所有已发现模型，`start` 必须明确指定 `--model`。manifest 的设定路径是相对于 manifest 所在 `configs/` 目录解析；模型套件的 Runtime／API 工厂以 `module:attribute` 字串指定。

`runtime.py:create_runtime()` 回传由 `engine.execution.worker.ModelRuntime` 封装的 Runner／Adapter，或提供相同方法契约的单一 facade。模型若使用共用标准文字元件，Runtime 工厂可只组合共用预设元件；模型特有程式依责任放入 `model/`、`layers/`、`components/` 与 `utils/loader.py`，不要求建立所有目录。Mage-VL 的 Transformers 子元件及 Qwen 的 PyTorch 拓扑均继续留在各自模型包。

这种组织方式减少新增模型时需要编辑的共用档案数量，但不保证接入免写程式：新架构仍需实作其模型计算、权重载入、前处理与 Cache 适配。共用层只负责稳定的 API、Engine 与 Worker 契约。

## 模型接入 profile 与能力预检

`integration_profile` 分成 `standard` 与 `custom`。`standard` 使用共用的受限文字 Runner，适用于符合本设计中 decoder-only、单装置、文字 Chat Template 与可延续 `past_key_values` 契约的模型；它不代表任意 `AutoModel` 都相容。`custom` 代表模型包需拥有专属计算图、权重转换、影音前处理、Tokenizer 或 Cache 行为。Profile 是接入方式标记，不代表套件或模型名称保证相容。目前 Mage-VL 和 Qwen3-0.6B 都是 `custom`；专案已有标准文字预设 Runner，但尚无任意模型通用相容层。

`launcher.py validate` 为无权重预检：载入登录与 YAML，检查 profile、API capability、Runner／Adapter／选用工厂模组、`token_ids` 解码依赖及 Live manager；它不汇入工厂、不下载或载入 snapshot、不建立 CUDA context。MediaMTX 配置由部署文档独立维护，不属于模型预检。精简前曾以 AST 契约与跨服务 YAML 比对验证接入；该组测试已移除，现在由逐模型三项测试提供计算图、性能和 Gateway 请求验收，不宣称覆盖逐栏位静态设定一致性。

## 设定与营运

Qwen `configs/inference.yml` 保存 model repo／revision、dtype/device、`runtime.platform`、本机模型路径、Context、generation、Scheduler、KV block、Worker 和 Engine timeout。模型网路端点仍来自各自的 `configs/network.yml`。MediaMTX sidecar 配置独立放在 `deploy/mediamtx/mediamtx.yml`，不出现在模型 manifest，也不由推论 Launcher 读取或启停；部署者需单独执行 MediaMTX，并在推论 API 连接埠变更时同步更新授权回呼。API／Gateway 边界不变；MediaMTX 只供宣告 Live 能力的模型使用。健康资讯如实区分 API、Engine Core、Worker、逻辑 reservation、实体 KV page 和 GPU memory。

错误按所在边界传递：InputProcessor 的非法模型输入回 4xx；Engine 准入／Cache 容量错误沿既有 `InferenceError` 映射；Executor／Worker 启动和 RPC 超时标为服务错误并走清理。Worker 不得在初始化失败后回退到 Transformers `generate()`。

## 验收策略

- 模型元件对照：`tests/{model_alias}/test_components.py` 在 `meta` device 比较本地模型包与 Transformers 原生模型模组树，不载入权重、不建立 CUDA context。
- 推理性能对照：`tests/{model_alias}/test_performance.py` 以同一输入比较已启动的 Inference API 与本机 Transformers 直接生成，报告平均 latency 和 token/s；需明确指定 YAML、模型 snapshot、GPU 与 API。
- Gateway 端到端：`tests/{model_alias}/test_end2end.py` 验证公开 alias、SSE 和模型已宣告的媒体 API；服务／金钥／素材缺一时不构成实机通过证据。
- 历史真机报告按 alias 放在 `tests/{model_alias}/results/`；每个模型的分析、优化和验收工作流唯一记录在根目录 `analysis/{model_alias}/analysis.md` 与 `analysis.xlsx`。
- 三类测试不执行资料集精度评估，也不取代对 Engine／Scheduler／Cache／IPC 的专项单元测试；本次采用逐模型简化套件后，旧分层测试树不再是维护入口。
- API 兼容：对 Qwen 呼叫一般 Chat、SSE、并发 SSE、非法输入和 disconnect；Mage-VL 的旧文字增量路径保留回归检查。
- 固定 snapshot：确认 Qwen input tokenization、native model output、逐 token event 合并文字、SSE finish usage／`[DONE]` 和取消后 Worker page 回收。
- 性能：在推理 API 验收后记录 request／token throughput、TTFT、inter-token latency、request latency p50/p95、错误率和 GPU peak memory；与端到端 Gateway performance test 分开，不设定硬 SLA。
- 不执行资料集精度测试；vLLM/nano-vLLM 公开结果不作为本地性能证据。

## 标准工具调用 adapter

`api/contracts/tool_calling.py` 维护跨模型的 OpenAI-style tool definitions、assistant tool calls、role=tool 历史标准化及 JSON Schema 验证；此层不解析任何模型 marker。`api/routes/chat_completions.py` 依 app state capability 与 adapter，在普通 JSON 回应中加入 `tool_calls`，在 SSE 将 detokenized text 交给 adapter parser，再发送结构化 `delta.tool_calls`。

模型 manifest 透过 `api.capabilities.tool_calling` 与 `api.tool_call_adapter` 启用能力。Qwen3 adapter 消费 tokenizer 生成的原生 JSON 工具调用块，并把 ID/name/arguments 转成共用类型；future model 用同一 Protocol 实现自己的输出解析。Qwen processor 将 OpenAI 标准 `tools` 和 tool-history messages传给上游 Chat Template。Launcher 检查能力和 factory 符号，启动时检查 adapter 实例接口。

首版允许 `tool_choice=auto/none` 和单个调用；不做 constrained decoding、并行调用或函数指定型 tool_choice。流式参数 delta 无法在完整生成前判定全体 JSON Schema，故 Inference 在块结束前只提供 SDK 可累积的 partial arguments；块完整后执行 JSON/Schema 验证。客户端／Agent 应用必须等待完整 SSE、`finish_reason=tool_calls` 与有效参数后才执行，并使用自身的工具注册表、权限和资源限制；模型输出本身不构成执行授权。标准工具循环由 `tests/client.py` 示范，不另设项目专用 Agent HTTP API。

## Qwen3 算子分析入口

逐模型三项测试保持为唯一维护入口，并直接从脚本顶层执行，不用测试 helper 函数隐藏流程。Qwen `test_components.py` 首先比较 `meta` 模型树，再以子进程按路径顺序运行每个带数值基线的 Qwen3 模块 `__main__`；子进程隔离避免模块导入状态污染，并保留单文件独立调试方式。模块前向优先使用 CUDA，只有 CUDA 不可用才使用 CPU；测试不载入完整模型权重。

可选 `--profile-attention` 使用 inference YAML 的固定本机 snapshot，从 Safetensors 只读取第 0 层 Attention 投影和 Q/K norm 权重，构造与该模型 config 一致的 Attention layer 及 physical KV page pool。对合成 prefill hidden states 和一个 decode token 进行预热与重复量测；CUDA 可用时用 CUDA Event 计时，不可用时用 CPU wall time，再用 `torch.profiler` 记录可用的 CPU/CUDA 算子、shape 与 memory。报告同时列出设备、PyTorch/CUDA、dtype、工作负载、样本及限制。Profiler 结果含采集开销，只用于定位；CPU 回退结果不能作为 GPU 性能数据。

阅读算子表时，先检查 `PagedKVCache.read_batch` 的 gather、padding、零初始化与张量复制，再对照 attention score/softmax/value aggregation 与 Q/K/V/O projection。Profiler 排序提示热点所在，不足以单凭某个 `aten::` 项判断 memory-bound 或 compute-bound；需结合 shape、CUDA 时间、分配量，并在必要时用 Nsight Systems 看调度/传输/空档、用 Nsight Compute 看候选 kernel 的硬件 counters。优化每次只改变一个路径，先复跑模块数值基线，再比较相同 workload 和环境下的 profiler 外延迟与显存，最后跑完整模型 API/Gateway 检查。

每个模型在 `analysis/{model_alias}/` 维护一份 `analysis.md` 与一份 `analysis.xlsx`。Markdown 单表严格按“问题、解决方案、效果”组织，每行引用 Excel 中的证据 ID；Excel 保留该行的测量数据和验收状态。算子实现变更后在对应模型表中保留前后测量和验收结果。旧 `val.md/xlsx` 的记录已按模型 alias 迁入对应目录。
