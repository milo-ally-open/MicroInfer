# 技术设计：五层推理引擎

> 提案：[proposal.md](proposal.md)｜行为规格：[spec.md](spec.md)｜任务与验收：[tasks.md](tasks.md)

## 参考数据与本地证据

vLLM V1 官方架构将 API Server、Engine Core 和 GPU Worker 分成不同程序；API Server 执行输入处理并流式结果，Engine Core 执行 Scheduler、管理 KV Cache 并协调 Worker，Worker 加载模型并执行 forward。Model Runner 负责模型输入张量和模型执行细节。参考：[vLLM V1 Architecture Overview](https://docs.vllm.ai/en/v0.19.0/design/arch_overview/)。本设计采用这些责任边界，不复制 vLLM 的内部类别、IPC、Executor 或 Kernel，也不声称 vLLM 兼容。

[nano-vLLM README](https://github.com/GeeeekExplorer/nano-vllm) 将自身描述为精简、可读的 vLLM 实现，展示其以较小代码结构承载 Engine、批次和优化概念。本项目参考其「用可读的核心流程学习／组织 Engine」方向；不把 README 中宣称的功能或 benchmark 当成本项目能力，也不将 nano-vLLM 加入执行依赖。

本地现况包含固定 revision 的 Qwen3 config／tokenizer／Safetensors 加载、自有 PyTorch 模型、批次 prefill/decode、Worker 实体 KV pool 和文字 SSE。Qwen3 当前使用 PyTorch Attention，并在 Attention 计算前将历史 KV pages gather 为连续张量。2026-10-25 曾完成单模型 CUDA prototype 的数值、profile、API 与 Gateway SSE 验收；后续依使用者决定已撤除其源码、构建脚本和运行时接口，历史测量保存在 `analysis/qwen3_0_6b/` 与 `tests/qwen3_0_6b/results/`。当前工作树中已不存在原 kernel 目录或编译产物。Mage-VL 仍使用 Transformers 模型路径；项目没有 SGLang/vLLM backend、Ray 或第二个原生模型。

## 五层责任与依赖

| 层 | 责任 | 本地代码边界 | 不拥有 |
|---|---|---|---|
| L1 接口与门面 | HTTP、OpenAI schema、模型输入 Tokenize、增量 Detokenize、SSE 和 Engine IPC client | `network/`、`engine/engine_client.py` | GPU tensor、Scheduler、KV pool、模型 forward |
| L2 编排与调度 | Engine Core 请求状态、准入、Scheduler、逻辑 Cache reservation、Worker 结果事件化 | `engine/inference_engine.py`、`engine/scheduler.py`、`engine/block_manager.py` | HTTP/SSE schema、实体 GPU tensor、模型拓扑 |
| L3 执行与 Worker | Worker 管理本地子进程和 RPC；Platform 解析设备与资源观测；模型 Worker 加载 Runner | `engine/worker.py`、`engine/platform.py`、`engine/bootstrap.py` | request policy、模型输入格式、HTTP |
| L4 模型运行时 | Runner 组装模型输入批次、forward、Cache 访问和 Sampling | 各模型 `components/model_runner.py`、模型批次及 Cache 组件 | OpenAI request／response、模型别名路由 |
| L5 模型与算子 | 模型拓扑、层、Attention、权重映射和上游兼容 | 各模型 `model/`、`layers/`、`utils/` | 共用 HTTP API、Engine 队列和租户政策 |

模型生态兼容是跨层责任：API 端的 Chat Template／Tokenizer 在 L1 依模型创建输入；模型 config、Safetensors 权重映射和计算图在 L5。通用视频读取与 Live session 由 `processors/` 持有；Mage-VL 的视频采样和 Processor 留在 Mage 模型包。五层以代码职责划界，不代表五个新 package 或五个 OS process。

## 共用 Inference 模块边界

Network 将 HTTP 应用、协议、路由和处理器放在 `network/` 平面目录；Engine 采用扁平、可直读的关键模块，不为每个类再创建 package：

```text
network/
├── app.py                       # FastAPI app 与路由装配
├── service.py                   # 依模型配置创建 API service
├── schemas.py / serializers.py  # HTTP schema 与序列化
├── tool_calling.py              # 共用 tool protocol
├── *_processor.py               # API 输入与输出处理
└── chat_completions.py 等       # HTTP endpoints

engine/
├── sequence.py          # 跨进程请求、批次、结果与进度状态
├── block_manager.py     # Engine 侧 KV block lease
├── scheduler.py         # prefill/decode 规划与 Token Scheduler
├── inference_engine.py  # 请求生命周期、准入、执行、取消与事件分发
├── worker.py            # Model Runtime、Worker 子进程与本地 Executor
├── engine_client.py     # API Server 到 Engine Core 的 IPC client
├── config.py            # manifest 与 YAML 读取
├── bootstrap.py         # Engine Core 与 Model Worker 工厂
├── platform.py          # Worker 设备及显存观测
└── standard_text.py     # 受限的标准文字 Runner
```

`sequence.py`、`block_manager.py`、`scheduler.py` 和 `inference_engine.py` 对应 Engine 的主要执行概念；IPC／子进程相关代码仍单独留在 `engine_client.py` 与 `worker.py`，避免把生命周期与并行通信揉成难以阅读的单体文件。Scheduler 的 batch planner 合并在同一文件。项目不额外维护 Executor Protocol package、通用插件注册中心或重复的 Engine facade。

模型差异集中在 `models/{model_alias}/`：模型 Processor／Tokenizer、Runtime 工厂、Runner、Adapter、拓扑、Cache 计算、Sampling 与 kernels 均归模型包。Launcher 读取模型 manifest 选择工厂；共用 Engine 不按 alias 分支。API 与模型只依赖根目录的稳定 Engine 模块，不需要理解目录层级或组合入口。

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

仍为 API Server、Engine Core、单一 Model Worker 三个程序。`LocalProcessExecutor` 目前只管理一个 `spawn` 子进程；接口为未来更换执行拓朴留出位置，没有多 GPU、Ray 或 Worker pool 实现。Engine Core 不汇入模型包，模型包 factory 只在对应程序创建该模型所需组件。

## 请求与文本流式

1. Qwen app 将 `Qwen3InputProcessor` 注入共用 `create_app()`。API lifespan 先从 YAML 指定的本地 snapshot 加载 Tokenizer，再启动 EngineClient；此处只加载 CPU tokenizer，不加载权重或 GPU Runtime。
2. Chat route 验证 model alias，调用模型 `InputProcessor` 形成 `PreparedChatInput(model_alias, input_ids, sampling_params)`，再透过 EngineClient 传入 Engine Core。数据结构只包含可 pickle 的小型 Python 数据，不包含 tokenizer、GPU tensor 或模型 Cache。
3. Engine Core 将 PreparedChatInput 传给 Worker Adapter。Adapter 创建 request state，Engine 保留 token 容量 reservation，Scheduler 排入 prefill/decode 批次。
4. Worker 的 Qwen Model Runner 以 `InputBatch` 组装 token tensor、长度、position offset 和 block table，调用本地 Qwen forward 及 Sampler。生成步骤只返回 token ID；Mage-VL 仍返回文字增量。
5. Engine `EngineOutputProcessor` 依 request ID 将 token ID 或文字增量转成 Engine stream event。EngineClient 将事件送回 API process。
6. Qwen API 的 `IncrementalDetokenizer` 累积 token IDs 并对完整前缀解码；当前前缀若含不完整 Unicode replacement，先缓冲该位置之后的文字。可安全输出的 delta 立即序列化成 OpenAI SSE chunk；finish 时 flush 剩余内容，再送 finish usage 和 `[DONE]`。已送出的前缀若被 tokenizer 改写，输出处理器明确失败，避免静默重复／漏字。
7. HTTP SSE 的取消仍由 route 关闭 EngineClient stream，依 Engine cancellation path 停止 request 并释放 Scheduler／Cache／Worker 状态。一般非流式请求由 Worker 返回 `InferenceResult`；Qwen token IDs 在 API L1 解码成 `content` 后，再沿用现有 OpenAI response serializer。

InputProcessor／Tokenizer 位于 API process，因此 Qwen API 和 Worker 各保留一份小型 tokenizer instance；这是 Tokenize 与模型加载隔离的首版成本。固定 snapshot 验收需确认其 Chat Template 结果与 Worker 最终 decode 使用同一 revision。

## L2 调度与 KV 所有权

Engine Core 保持接收、准入、调度、Executor 执行、更新状态、发布事件和收尾流程。`EngineOutputProcessor` 只转换 Worker result；request terminal state、finish operation、取消和 Cache release 仍由 Engine Core 管理。

Token Scheduler 继续依 `max_num_seqs`、等待上限、`max_num_scheduled_tokens`、prefill chunk 和 batch wait 配置建批；批次按 request ID 返回结果。混合 prefill/decode 可同一次 Worker RPC 传输，Runner 仍按 phase 分批 forward。Sampler 保持在 L4 Worker，模型产生的 token 不交由 API 重新采样。

`BlockManager` 是 L2 逻辑资源管理器。Mage-VL 对应 Transformers Cache request state；Qwen Engine block ID 映射到 Worker 内的实体 page pool。Qwen PyTorch Attention gather paged K/V 后以分组矩阵乘法执行 GQA；选用式 CUDA decode prototype 则直接按 block table 读取 page，prefill 仍走 PyTorch 路径。CUDA extension 与 API 路径已完成本地功能验收，但当前慢于 PyTorch，YAML 默认保持 PyTorch backend。Engine 的 reservation／容量统计与 Worker 实际页面使用量分开回报。

## L3 Worker 与 Platform

`engine/worker.py` 直接包含 `LocalProcessExecutor`、子进程 Worker RPC 与 `ModelRuntime`／`InProcessModelWorker`；这是当前唯一的本地执行路径，不再单独维护 Executor Protocol package。`engine/bootstrap.py` 组合 Engine 和 Worker 工厂，Engine Core 通过 Worker 执行模型操作。

`Platform` 是 Worker Model Runner 使用的 device／memory contract。模型优先使用 CUDA，只有 CUDA 不可用时才回退到 CPU。Qwen YAML 的 `runtime.platform: torch_cuda` 由 `create_platform()` 创建 `TorchCudaPlatform`；设备选择、VRAM／peak memory 观测及 CUDA cache 清理经该接口执行。未知名称在 Runner 加载前明确失败；Ray、NPU 和多 GPU 不在支持范围内。

## L4 InputBatch 与模型运行时

`InputBatch.build()` 依调度工作项目创建本轮的 padded token ID tensor，并检查 request 数量、token length、position offset 和 block table 一一对应。Qwen forward 仍接收 per-request 长度／偏移／block tables，padding 位置不参与有效 token 计算。

InputBatch 暂时每轮动态分配 GPU tensor。后续若性能样本确认 CPU tensor 组装或分配是瓶颈，可将容量预配置、Pinned Memory、非同步 H2D 和 CUDA Graph 作为独立优化，需先保持相同的 request mapping／SSE／Cache acceptance。首版不加入这些未测机制。

## L5 Qwen 模型与生态兼容

Qwen 模型拓扑及 Attention 等算子留在模型包；模型专属 `utils/` 提供上游 config／tokenizer／固定 revision 权重格式适配等辅助功能。Safetensors loader 在 Worker ready 前验证 metadata、revision、必要 tensor 和 shape。模型 manifest 登记 alias、配置路径、对外 API ID、Runtime 工厂和选用 API 扩充；共用 route 不以模型 alias 分支。

模型配置使用与架构相符的 Transformers config 类别：Qwen3 以 `Qwen3Config.from_pretrained()` 加载固定本地 revision，并将 `Qwen3Config` 传入原生模型计算图；Mage-VL 使用模型专属 `MageVLConfig`，其中固定 revision 的文字骨干由 `Qwen3Config` 表示。Tokenizer 与多模态 Processor 依固定 revision 的 `tokenizer_config.json`／`processor_config.json` 等 metadata，透过 `AutoTokenizer`／`AutoProcessor` 创建；这保留 Hugging Face 对不同模型 tokenizer 类别的解析能力，不代表模型 forward 由 `AutoModel` 执行。Qwen Runner 的 GQA、Mage 视觉 Attention、SigLIP pooling Attention 与 Mage 文字骨干 Attention 均在模型包以矩阵乘法、遮罩及 softmax 明确实现；Mage 文字骨干另有独立 Qwen3 Attention、RMSNorm、MLP 与 Decoder layer，使用分块 Query 计算控制暂存内存。所有加载仍使用本地 snapshot 与固定 revision，不可在服务启动时解析浮动版本或隐式下载。

## 新模型接入与共用装配

`network/service.py` 读取 Launcher 配置的模型 alias，创建共用 `EngineClient`、API request limits、输入处理器和选用 Live manager。`engine/bootstrap.py` 从 manifest 加载模型 Runner／Adapter 工厂，创建 `InferenceEngine`、`LocalProcessExecutor` 和 `InProcessModelWorker`。Runner／Adapter 工厂使用 `module:attribute` 字符串；工厂仅在相应程序中加载，因此 API process 不会汇入模型 Runner 或 GPU Runtime。

模型包的 `configs/model.yml` 中 `api.model_id` 是推理 API 接受及返回的标识符；上游 model repo ID 留在 `inference.yml` 的 `model.id`。`api.capabilities` 明确列出聊天输入模态、文本流式、`text_delta`／`token_ids` 输出、视频上传及 Live 能力；API app 依此挂载路由并拒绝不支持的输入。`api.input_processor` 与 `api.live_session_manager` 是依能力需要登记的工厂。特殊前处理、Tokenizer、权重加载、模型拓扑和 cache 逻辑仍由模型包负责。

新模型最少实现 Runner／Adapter 并登记工厂和 YAML；不需写模型专属 `app.py` 或 `engine_core.py`。逐步生成 Adapter 维持 Engine 的 prepare／prefill／decode／finish／cancel 契约，Runner 持有模型 cache state 并依 request ID 返回批次结果。共用 KV Manager 的租约不替模型宣称支持相同的物理 cache 或 batch 语义。各文件职责、Runtime 工厂接口和完整接入步骤见[新模型接入指南](model-integration.md)。

Mage-VL 与 Qwen 均由共用入口装配；Mage 的 `model/mage_vl.py` 明确组合视觉塔、Qwen3 文本骨干与多模态输入，`vision_features.py`、`multimodal_inputs.py`、`streammind.py` 分别负责特征、placeholder 注入与 Gate；视觉层按 RoPE、patch 表示、注意力、Encoder 堆叠分置于 `layers/`。`conditional_generation.py` 维护 Transformers 生成契约，`components/` 仍负责逐步生成／替换注册等 Runtime 组件；`multimodal_model.py` 和 `vision_encoder.py` 只保留兼容导出。Qwen 拓扑与层亦各自分置于 `model/` 和 `layers/`。模型目录不保留未使用的 `app.py`／`engine_core.py` 包装文件。

### 标准文字模型默认组件

`integration_profile: standard` 可省略模型专属 Runner、Adapter 和 `api.input_processor`。Network 使用 `network/default_chat_processor.py`；Worker 使用 `engine/standard_text.py` 中的 `StandardTextModelRunner`／`StandardTextModelAdapter`。模型仍需提供薄的 `runtime.py` 工厂，以便共用 Worker 使用一致的 Runtime facade。

## 嵌入式 Engine 的 YAML 选择

模型插件的 `create_engine_client()` 可接收可选 `inference_config_path` 和 `network_config_path`。插件 manifest 仍决定模型 alias、Runtime 和 API 能力；传入路径只覆盖两份运行 YAML。未传入时沿用 manifest 相对 `configs/` 目录声明的路径。显式绝对路径直接使用，显式相对路径相对调用进程当前工作目录解析。工厂在返回 Engine client 前读取两份 YAML 并校验顶层 mapping，再把规范化路径和 manifest 写入 Worker 子进程将继承的环境。模型 YAML 中相对 checkpoint 路径仍由 `INFERENCE_PROJECT_ROOT` 解析。配置选择或 YAML 校验失败时，不启动 Engine Core、Worker 或模型加载。

默认 Runner 以 `AutoModelForCausalLM` 加载固定 revision，仅支持单一计算设备，并优先使用 CUDA；CUDA 不可用时回退 CPU。模型类型仍限 decoder-only causal LM、可重用 `past_key_values`、文字 `chat_template` 和 token ID 输出。API 处理器只接收文字，不接受图片、视频或 Live。Scheduler 提交的 `execute_batch()` 仍按 request 逐项调用 forward，各 request 的 Cache／生成状态分开持有；这是可用的标准接入路径，不代表连续批次 forward、实体 block cache 或 PagedAttention。

预检以 YAML 与 AST 确认 capability／方法合同，不 import 工厂或权重。Worker 在模型加载前再次检查实例方法；如架构使用 encoder-decoder、`device_map` 多设备、不可重用 Cache，或需要自定义采样／影音流程，应改登记 `custom` 并提供模型 Runner／Adapter。即使预检通过，将模型标成标准兼容前仍需以固定 revision snapshot 实际验证 Cache 延续和 token 输出。

## 模型包结构与 manifest

模型专属代码、配置和依赖以 `models/{model_alias}/` 为唯一扩充边界。共用 `network/`、`engine/`、`processors/` 不按模型 alias 分叉；`processors/` 可继续扩展视频等跨模型处理能力。Launcher 只负责发现、检查与启动，模型 Runtime 工厂在 Worker 子进程创建模型实现。

```text
models/{model_alias}/
├── runtime.py                 # create_runtime()，组合单一模型 Runtime facade
├── processor.py               # 可选：Tokenizer／Chat Template／模型输入处理
├── model/                     # 可选：模型拓扑
├── components/                # 可选：Runner、Adapter、批次与 Cache 支持组件
├── layers/                    # 可选：模型专属 Attention、Norm、Linear、RoPE
├── utils/                     # 模型 Config／Tokenizer／Processor、snapshot 与权重加载
│   └── loader.py
├── requirements.txt           # 可选：模型依赖包
└── configs/
    ├── model.yml             # alias、能力、Runtime 工厂及配置文件名
    ├── inference.yml           # 模型与推理参数
    └── network.yml            # API 绑定地址
```

MediaMTX 配置位于项目根目录的独立部署目录：

```text
deploy/mediamtx/
├── mediamtx.yml               # 独立 MediaMTX sidecar 服务配置
└── README.md                  # 独立启停与端点同步方式
```

每个模型的 `configs/model.yml` 是该模型唯一的登记入口。Launcher 扫描 `models/*/configs/model.yml` 发现模型，并拒绝重复 alias；`list`／`validate` 可处理所有已发现模型，`start` 必须明确指定 `--model`。manifest 的配置路径是相对于 manifest 所在 `configs/` 目录解析；模型包的 Runtime／API 工厂以 `module:attribute` 字符串指定。

`runtime.py:create_runtime()` 返回由 `engine.worker.ModelRuntime` 封装的 Runner／Adapter，或提供相同方法契约的单一 facade。模型若使用共用标准文字组件，Runtime 工厂可只组合共用默认组件；模型特有代码依责任放入 `model/`、`layers/`、`components/` 与 `utils/loader.py`，不要求创建所有目录。Mage-VL 的 Transformers 子组件及 Qwen 的 PyTorch 拓扑均继续留在各自模型包。

这种组织方式减少新增模型时需要编辑的共用文件数量，但不保证接入免写代码：新架构仍需实现其模型计算、权重加载、前处理与 Cache 适配。共用层只负责稳定的 API、Engine 与 Worker 契约。

## 模型接入 profile 与能力预检

`integration_profile` 分成 `standard` 与 `custom`。`standard` 使用共用的受限文字 Runner，适用于符合本设计中 decoder-only、单设备、文字 Chat Template 与可延续 `past_key_values` 契约的模型；它不代表任意 `AutoModel` 都兼容。`custom` 代表模型包需拥有专属计算图、权重转换、影音前处理、Tokenizer 或 Cache 行为。Profile 是接入方式标记，不代表包或模型名称保证兼容。目前 Mage-VL 和 Qwen3-0.6B 都是 `custom`；项目已有标准文字默认 Runner，但尚无任意模型通用兼容层。

`launcher.py validate` 为无权重预检：加载登记与 YAML，检查 profile、API capability、Runner／Adapter／选用工厂模块、`token_ids` 解码依赖及 Live manager；它不汇入工厂、不下载或加载 snapshot、不创建 CUDA context。MediaMTX 配置由部署文档独立维护，不属于模型预检。精简前曾以 AST 契约与跨服务 YAML 比对验证接入；该组测试已移除，现在由逐模型三项测试提供计算图、性能和 Gateway 请求验收，不宣称覆盖逐字段静态配置一致性。

## 配置与营运

Qwen `configs/inference.yml` 保存 model repo／revision、dtype/device、`runtime.platform`、本地模型路径、Context、generation、Scheduler、KV block、Worker 和 Engine timeout。模型网络端点仍来自各自的 `configs/network.yml`。MediaMTX sidecar 配置独立放在 `deploy/mediamtx/mediamtx.yml`，不出现在模型 manifest，也不由推理 Launcher 读取或启停；部署者需单独执行 MediaMTX，并在推理 API 端口变更时同步更新授权回调。API／Gateway 边界不变；MediaMTX 只供宣告 Live 能力的模型使用。健康信息如实区分 API、Engine Core、Worker、逻辑 reservation、实体 KV page 和 GPU memory。

错误按所在边界传递：InputProcessor 的非法模型输入回 4xx；Engine 准入／Cache 容量错误沿既有 `InferenceError` 映射；Executor／Worker 启动和 RPC 超时标为服务错误并走清理。Worker 不得在初始化失败后回退到 Transformers `generate()`。

## 验收策略

- 模型组件对照：`tests/{model_alias}/test_components.py` 在 `meta` device 比较本地模型包与 Transformers 原生模型模块树，不加载权重、不创建 CUDA context。
- 推理性能对照：`tests/{model_alias}/test_performance.py` 以同一输入比较已启动的 Inference API 与本地 Transformers 直接生成，报告平均 latency 和 token/s；需明确指定 YAML、模型 snapshot、GPU 与 API。
- Gateway 端到端：`tests/{model_alias}/test_end2end.py` 验证公开 alias、SSE 和模型已宣告的媒体 API；服务／密钥／素材缺一时不构成实机通过证据。
- 历史真机报告按 alias 放在 `tests/{model_alias}/results/`；每个模型的分析、优化和验收工作流唯一记录在根目录 `analysis/{model_alias}/analysis.md` 与 `analysis.xlsx`。
- 三类测试不执行数据集精度评估，也不取代对 Engine／Scheduler／Cache／IPC 的专项单元测试；本次采用逐模型简化包后，旧分层测试树不再是维护入口。
- API 兼容：对 Qwen 调用一般 Chat、SSE、并发 SSE、非法输入和 disconnect；Mage-VL 的旧文字增量路径保留回归检查。
- 固定 snapshot：确认 Qwen input tokenization、native model output、逐 token event 合并文字、SSE finish usage／`[DONE]` 和取消后 Worker page 回收。
- 性能：在推理 API 验收后记录 request／token throughput、TTFT、inter-token latency、request latency p50/p95、错误率和 GPU peak memory；与端到端 Gateway performance test 分开，不配置硬 SLA。
- 不执行数据集精度测试；vLLM/nano-vLLM 公开结果不作为本地性能证据。

## 标准工具调用 adapter

`network/tool_calling.py` 维护跨模型的 OpenAI-style tool definitions、assistant tool calls、role=tool 历史标准化及 JSON Schema 验证；此层不解析任何模型 marker。`network/chat_completions.py` 依 app state capability 与 adapter，在普通 JSON 响应中加入 `tool_calls`，在 SSE 将 detokenized text 交给 adapter parser，再发送结构化 `delta.tool_calls`。

模型 manifest 透过 `api.capabilities.tool_calling` 与 `api.tool_call_adapter` 启用能力。Qwen3 adapter 消费 tokenizer 生成的原生 JSON 工具调用块，并把 ID/name/arguments 转成共用类型；future model 用同一 Protocol 实现自己的输出解析。Qwen processor 将 OpenAI 标准 `tools` 和 tool-history messages传给上游 Chat Template。Launcher 检查能力和 factory 符号，启动时检查 adapter 实例接口。

首版允许 `tool_choice=auto/none` 和单个调用；不做 constrained decoding、并行调用或函数指定型 tool_choice。流式参数 delta 无法在完整生成前判定全体 JSON Schema，故 Inference 在块结束前只提供 SDK 可累积的 partial arguments；块完整后执行 JSON/Schema 验证。客户端／Agent 应用必须等待完整 SSE、`finish_reason=tool_calls` 与有效参数后才执行，并使用自身的工具注册表、权限和资源限制；模型输出本身不构成执行授权。标准工具循环由 `tests/client.py` 示范，不另设项目专用 Agent HTTP API。

## Qwen3 算子分析入口

逐模型三项测试保持为唯一维护入口，并直接从脚本顶层执行，不用测试 helper 函数隐藏流程。Qwen `test_components.py` 首先比较 `meta` 模型树，再以子进程按路径顺序运行每个带数值基线的 Qwen3 模块 `__main__`；子进程隔离避免模块导入状态污染，并保留单文件独立调试方式。模块前向优先使用 CUDA，只有 CUDA 不可用才使用 CPU；测试不加载完整模型权重。

可选 `--profile-attention` 使用 inference YAML 的固定本地 snapshot，从 Safetensors 只读取第 0 层 Attention 投影和 Q/K norm 权重，构造与该模型 config 一致的 Attention layer 及 physical KV page pool。对合成 prefill hidden states 和一个 decode token 进行预热与重复量测；CUDA 可用时用 CUDA Event 计时，不可用时用 CPU wall time，再用 `torch.profiler` 记录可用的 CPU/CUDA 算子、shape 与 memory。报告同时列出设备、PyTorch/CUDA、dtype、工作负载、样本及限制。Profiler 结果含采集开销，只用于定位；CPU 回退结果不能作为 GPU 性能数据。

阅读算子表时，先检查 `PagedKVCache.read_batch` 的 gather、padding、零初始化与张量复制，再对照 attention score/softmax/value aggregation 与 Q/K/V/O projection。Profiler 排序提示热点所在，不足以单凭某个 `aten::` 项判断 memory-bound 或 compute-bound；需结合 shape、CUDA 时间、分配量，并在必要时用 Nsight Systems 看调度/传输/空文件、用 Nsight Compute 看候选 kernel 的硬件 counters。优化每次只改变一个路径，先复跑模块数值基线，再比较相同 workload 和环境下的 profiler 外延迟与显存，最后跑完整模型 API/Gateway 检查。

每个模型在 `analysis/{model_alias}/` 维护一份 `analysis.md` 与一份 `analysis.xlsx`。Markdown 单表严格按“问题、解决方案、效果”组织，每行引用 Excel 中的证据 ID；Excel 保留该行的测量数据和验收状态。算子实现变更后在对应模型表中保留前后测量和验收结果。旧 `val.md/xlsx` 的记录已按模型 alias 迁入对应目录。

## Qwen3 Engine 组件、KV 管理与 Attention 路径（REQ-17、REQ-22）

本次以现有 Engine contracts 与三程序拓扑为边界，采用 nano-vLLM 的职责概念，不复制其单体 Engine 结构：

- `engine/sequence.py` 定义 Engine 可序列化的请求进度状态；Scheduler 不再维护另一份互不一致的 prompt/output/block 进度。它不承载 Worker Cache、GPU tensor、媒体内容或 API stream queue。
- `engine/block_manager.py` 管理 Engine 侧空闲物理 block、request block table、准入时的原子容量保留和幂等释放。实际 KV tensor 由 Qwen Worker 的 `PagedKVCache` 持有。每个 `ScheduledWorkItem` 将带本轮一致的 block table；首轮保留完整最大序列容量。
- 唯一 Scheduler 仍在 `engine/scheduler.py`，唯一请求编排器仍在 `engine/inference_engine.py`。后者等价承担模型服务的 Engine 编排职责，不另建会重复的 `model_engine.py`。模型专属 Runner 仍由 Qwen 模型包提供，Worker process/IPC 仍由 `engine/worker.py` 管理。
- Qwen3 当前只使用 PyTorch Attention 路径。Worker 的实体 KV pages 由 `PagedKVCache` 管理，Attention 读取历史上下文时通过 `read_batch()` gather 为连续 K/V 张量；不提供单模型专用手写 CUDA kernel、extension 构建入口或 backend selector。

请求进度和 block 所有权按如下生命周期流转：prepare 后创建 Sequence；BlockManager 为最大序列长度原子分配 pages 并登记 lease；Scheduler 输出 work items 与 block table；Qwen Runner 接收 token/position/block table 并执行；finish/cancel/error 清除 Sequence 和 Engine blocks，并由 Worker 收尾对应 request page state。跨 Engine/Worker IPC 只传 ID、长度和小型 metadata，不传 GPU tensor。

当前不做 Prefix Cache、按需扩容、抢占、KV offload、CUDA Graph、Tensor Parallel 或单模型专用 CUDA kernel。最大序列预留维持当前容量准入保证；未来若需按需扩容，须连同资源不足时的抢占、终止或可证明安全的等待策略设计，避免多个请求持有部分 KV 后互相等待。既有 CUDA prototype 的数值与性能测量作为历史记录保留，不代表当前运行时仍提供该实现。
