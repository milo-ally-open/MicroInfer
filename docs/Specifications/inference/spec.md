# 推论引擎五层行为规格

> 提案：[proposal.md](proposal.md)｜技术设计：[design.md](design.md)｜实施与验收：[tasks.md](tasks.md)
> 本规格描述 Qwen3-0.6B 实验改造及应保持的共用服务行为。已完成、进行中和待验收的状态以 `tasks.md` 为准。工具调用增补由 REQ-14 追踪。

### REQ-01：L1 接口与门面维持 API 契约

- **AC-01.1** `GET /health`、`GET /v1/models`、`POST /v1/chat/completions` 的路径、HTTP 方法、模型别名和 OpenAI response schema 维持现有行为；Qwen3 只登录纯文字输入，可另外宣告其标准工具调用能力。
- **AC-01.2** 共用 `network/` 不汇入模型包，不载入模型权重、GPU Runtime 或模型专属计算图。
- **AC-01.3** Qwen `InputProcessor` 在 API Server 程序读取模型 YAML 指定的本机 Tokenizer／Chat Template，将请求转为包含 alias、token IDs 和 sampling params 的 `PreparedChatInput`；该资料可经 IPC 传递，且 API 程序不执行模型 forward。
- **AC-01.4** Qwen Worker 回传 token ID 后，API 增量 Detokenize 并发出文字 SSE；完成时输出尚未送出的文字、finish reason、usage 和 `data: [DONE]`，串流合并文字须等于完整生成文字。
- **AC-01.5** Mage-VL Worker 既有 `text_delta` 路径仍转成相同 OpenAI SSE schema；不得要求 Mage-VL 改用 Qwen tokenizer 或 token ID。
- **AC-01.6** API 客户端断线时，原有取消 IPC 与 shielded cleanup 仍生效；取消后不得继续向该 stream 发送 token。
- **AC-01.7** Qwen 非串流聊天回应由 API Server 将 Worker 回传的 token ID 序列解码为 OpenAI `content`；Worker 不产生对外聊天字串。

### REQ-02：L2 Engine Core 统一管理请求执行闭环

- **AC-02.1** 每个可排程生成依序经过输入准备、准入与 Cache reservation、Scheduler 建批、Executor 执行 Worker、批次结果更新、逐 request 事件发布及完成收尾。
- **AC-02.2** 成功、拒绝、取消、超时、Worker 异常和关闭均回收 Scheduler 名额、Engine lease 及 Worker request state；释放流程幂等。
- **AC-02.3** `EngineOutputProcessor` 将 Worker token ID 或文字增量转成不依赖 FastAPI／SSE 的 Engine 输出事件；同一 Worker 结果不得同时包含 token ID 与文字增量。
- **AC-02.4** Engine 依 request ID 分派事件；一个请求的错误、取消、完成或文字输出不得串入另一个 request stream。

### REQ-03：Scheduler 与 KV Cache Manager 保留资源所有权

- **AC-03.1** Scheduler 依活动请求上限、等待伫列上限和每轮 token 预算建立 prefill／decode 批次；decode 每 request 每轮最多执行一个 token。
- **AC-03.2** 相容的同 phase Qwen request 由一次批次 Model Runner 呼叫处理；不同 phase 可由同一 Worker RPC 分组，但不得把不相容的 Cache 或 attention 状态混入同一次 forward。
- **AC-03.3** 共用 `KVCacheManager` 管理 Engine 逻辑 block reservation；Qwen Worker pool 管理与 block ID 对应的实体 GPU page，Mage-VL 仍使用 Transformers Cache。
- **AC-03.4** 请求成功、取消、Worker 失败或 Engine 关闭后，Engine lease 和 Qwen 实体 page 都可回收及供后续请求重用。
- **AC-03.5** Qwen Attention 对 KV page 的 gather 后 GQA 计算须如实描述；除非程式直接透过 block table 读写实体 page，否则不得称为 PagedAttention。

### REQ-04：L3 Executor、Worker 与 Platform 隔离执行细节

- **AC-04.1** Engine Core 只依赖 `Executor` 呼叫生命周期、RPC、批次执行、健康资讯和关闭；Engine 不自行建立 Pipe 或操作 multiprocessing process。
- **AC-04.2** `LocalProcessExecutor` 以 `spawn` 启动本机 Model Worker 子程序，完成 ready handshake 后才回报可执行，并保留既有 timeout、错误和 graceful close 行为。
- **AC-04.3** Worker 在自己的程序内载入模型与 Runtime；API Server 不载入模型权重，Engine Core 不持有 GPU tensor。
- **AC-04.4** `Platform` 定义装置解析、记忆体观测与资源释放契约；推理优先使用 CUDA，只有 CUDA 不可用时才回退 CPU。未知平台必须明确失败；未实作的 Ray、NPU 和多 GPU 不得出现在支援清单。
- **AC-04.5** 进程 IPC／执行层的 fake Worker 测试验证 API Server、Engine Core 与 Worker PID 隔离，并覆盖 startup、RPC、stream、cancel、timeout、错误和关闭。

### REQ-05：L4 Model Runner 以 InputBatch 执行批次模型呼叫

- **AC-05.1** Qwen `InputBatch` 将每轮 token tensor、每 request token 长度、position offset、request ID 和 block table 对齐组装；数量或长度不一致时在 forward 前拒绝。
- **AC-05.2** 变长 prefill 使用明确 padding 长度资料；每个 request 的 position、Cache block table、sampling state 和输出依 request ID 对应。
- **AC-05.3** Model Runner 负责模型输入 tensor、模型 forward、Sampling 和 Worker result；HTTP schema、SSE JSON 和 OpenAI response 不进入 Runner。
- **AC-05.4** 首版 `InputBatch` 每轮建立；不得宣称持久 GPU buffer、CUDA Graph capture/replay 或 fused Attention 已实作。

### REQ-06：L5 模型包持有模型拓扑与生态相容

- **AC-06.1** Qwen3-0.6B 模型拓扑、Attention、RoPE、RMSNorm、MLP、Sampling、Paged KV storage 及 Safetensors key mapping 均由 `models/qwen3_0_6b/` 持有。
- **AC-06.2** Qwen3 模型设定必须以 Transformers `Qwen3Config` 载入及传递至模型计算图；Mage-VL 的文字骨干设定使用 `Qwen3Config`，外层设定使用 Mage-VL 专属 `MageVLConfig`。Tokenizer／Processor 依固定 revision 的 Hugging Face metadata 使用 `AutoTokenizer`／`AutoProcessor`，不得以模型别名猜测其具体类别；Qwen forward 不使用 `AutoModelForCausalLM` 或 `generate()`。
- **AC-06.3** 权重载入验证固定 repo、revision、必要 key 和 tensor shape；未知或不相容权重在 Worker ready 前明确失败。
- **AC-06.4** `mage-vl-4b` 与 `qwen3-0.6b` 的上游模型识别码、专案 alias、执行后端及 API 能力各自登录；本规格不选定第三个模型。

### REQ-07：设定、健康状态和失败回报反映实际能力

- **AC-07.1** 模型、Runner、Scheduler、KV pool、Worker timeout、API 网路和模型使用的媒体端点保存在对应模型 YAML；MediaMTX sidecar 自身设定独立保存在 `deploy/mediamtx/mediamtx.yml`，不由模型 manifest 登记或推论 Launcher 管理；密钥不写入 YAML 或 Git。
- **AC-07.2** Qwen `/health` 可回报 backend、platform、API／Engine／Worker PID、Scheduler、Engine lease、Worker page 使用和 CUDA memory；GPU 栏位只在 Worker ready 后提供。
- **AC-07.3** Worker 启动、Executor timeout、Tokenizer 载入、平台不支援、Cache 容量不足和模型输入不合法均回报明确错误；不得静默切换到未排程的 `generate()`。
- **AC-07.4** Go Gateway、Inference API 和模型 Runner 仍保持程序／责任分离；不把 gateway API Key 或租户政策移至推理层。

### REQ-08：以精简的逐模型测试验证模型接入

- **AC-08.1** 每个已登录模型目录 `tests/{model_alias}/` 恰有三个测试入口：`test_components.py`、`test_performance.py`、`test_end2end.py`，及一份 `test.yml`；不再维护独立的 `tests/inference/`、`tests/gateway/`、`tests/end2end/` 测试树。
- **AC-08.2** `test_components.py` 以指定的本机固定 snapshot 和 `meta` device 比较模型包计算图与 Transformers 原生基线，依序输出 `==========engine===========` 和 `==========baseline==========`，不得载入权重或建立 CUDA context。
- **AC-08.3** `test_performance.py` 使用相同本机 snapshot、输入与生成上限，比较已启动的 Inference API 和 Transformers 原生直接生成；输出样本数、平均延迟、生成 token 数及平均 token/s。该测试需模型、GPU 与可用 Inference API，不宣称固定 SLA。
- **AC-08.4** `test_end2end.py` 以设定的模型 alias 从 Go Gateway 发送请求，验证 HTTP 成功、文字 SSE 内容及公开 alias；Mage-VL 同时验证图片 SSE 和影片上传，Qwen 验证纯文字 SSE。测试需 Gateway、相应 Inference API、API Key 和本机素材。
- **AC-08.5** 测试设定以 `--config` 明确传入；相对素材／snapshot 路径以该 YAML 所在目录解析。历史实测 JSON 保存在各模型目录的 `results/`；不新增资料集精度或 Video-MME 评测。

### REQ-09：新模型沿用共用服务装配

- **AC-09.1** 新模型在 `models/{model_alias}/configs/model.yml` 登记设定路径、Runtime 工厂及 API 识别码；Launcher 自动发现模型 manifest，`start` 必须明确指定 `--model`。Launcher 使用同一个 `network/service.py`，Engine 使用同一组 Engine／Worker bootstrap，不要求模型包提供 `app.py` 或 `engine_core.py`。
- **AC-09.2** GPU Worker 呼叫 Runner 的 `load()`、`gpu_status()`、`close()`；Adapter 依既有 Engine 操作契约完成准备、Cache 绑定、批次执行、finish 和 cancel，并按 request ID 回传独立结果。
- **AC-09.3** Tokenizer／Chat Template、视讯与 Live 整合以可选工厂及登录栏位加入；未登记的路由不出现在 API。`api.model_id` 与 YAML 中的上游 repo ID 分开设定。
- **AC-09.4** Mage-VL 与 Qwen3 经共用服务入口启动；Mage-VL 的影片／Live 路由与文字增量、Qwen 的文字 API 与逐 token SSE、finish usage 和 `[DONE]` 行为维持现有契约。
- **AC-09.5** 新模型接入操作指南列出必需程式介面、选用元件、Registry／YAML 范例及启动检查；新模型流程不需在共用 API route、Launcher 或 Engine 加入模型 alias 分支。

### REQ-10：模型接入宣告能力并提供无权重预检

- **AC-10.1** 每笔模型登录明确宣告 `integration_profile`（`standard`／`custom`）及 `api.capabilities`：聊天输入模态、串流输出、输出型态、影片上传与 Live 能力；目前两个已登录模型如实采用 `custom` profile。
- **AC-10.2** `launcher.py validate` 检查 profile、能力栏位、模型设定 YAML、factory 模组路径与方法、`token_ids` 输出所需 InputProcessor，以及 Live 所需 manager；此命令只读模型 YAML 和 Python AST，不得 import 模型工厂、载入权重、下载 snapshot 或初始化 GPU。MediaMTX sidecar 配置由独立部署说明管理，不纳入模型预检。
- **AC-10.3** 共用 Chat route 依已登记模态和输出能力拒绝不支援的输入／串流；影片与 Live API 只在相应 capability 为 true 时加入服务，Live manager 缺漏时启动失败。
- **AC-10.4** 新模型的结构比较由 `tests/{model_alias}/test_components.py` 输出模型包与 Transformers 原生模组树；Launcher `validate` 继续负责 manifest／工厂 AST 预检，不需要额外测试目录。
- **AC-10.5** 新模型指南区分标准相容路径与自订模型路径，列出标准预设实作的架构、Cache、Tokenizer、装置和采样限制；不相容模型必须登记 custom 并由模型包适配。
- **AC-10.6** Inference manifest 与 Gateway 设定的别名及路由由实际 Gateway 端到端测试验证；部署端点与能力差异依模型 `test.yml` 和 Gateway YAML 维护，不宣称此项测试静态比对全部设定栏位。

### REQ-11：符合契约的文字模型可使用标准预设 Runner

**需求**：符合标准文字生成契约的模型可省略模型专属 Runner、Adapter 和 InputProcessor 工厂，使用共用预设元件；标准路径不得把不相容架构或 Cache 行为伪装为通用支援。

**验收条件**

- **AC-11.1** Given `integration_profile: standard` 且模型未覆写元件工厂，When 共用 Network／Worker 启动，Then 分别使用 `network/default_chat_processor.py`、`engine/standard_text.py` 的预设元件；明确登记的工厂可逐项覆写预设。
- **AC-11.2** Given 标准文字聊天请求，When API 处理输入，Then 预设 InputProcessor 使用固定 revision Tokenizer 的 `chat_template` 产生 `PreparedChatInput`，只接受文字内容，并以同一 Tokenizer 提供非串流解码和增量文字 SSE。
- **AC-11.3** Given 标准 Runner 收到 Scheduler 的 prefill／decode work item，When 执行 Worker batch，Then 每个 request 使用自己的 `past_key_values`、Cache 长度及采样状态，依序执行 work items 并按 ID 回传 token 结果；Runner 不宣称此预设实作会合并模型 forward 或使用物理 PagedAttention。
- **AC-11.4** Given 模型宣告图片／影片／Live、encoder-decoder、`device_map` 多装置或无法重用 `past_key_values`，When 执行标准 profile 预检或 Worker 载入，Then 明确拒绝或要求切换 `custom`；不得退回完整 `generate()` 绕过 Token Scheduler。
- **AC-11.5** Given 本机固定 revision snapshot，When 使用标准 Runner 做实际模型 smoke，Then 至少验证 Tokenizer template、prefill、逐 token decode、Cache 延续、finish/cancel 和文字重建；合成 CPU 契约测试不得替代此模型 smoke。

### REQ-12：单一模型接入资料集中于模型目录

**需求**：每个模型以单一目录承载 Runtime 入口、模型专属程式、设定和依赖；新增模型主要透过新增模型包及其 manifest 完成，不需修改共用 API／Engine 装配或集中式模型清单中的多个位置。

**验收条件**

- **AC-12.1** 每个模型包以 `models/{model_alias}/runtime.py` 提供 `create_runtime()` 工厂；该工厂回传单一 Runtime facade，符合 Worker 的 load、health、generation、batch、finish、cancel、GPU status 及 close 契约。
- **AC-12.2** 每个模型的 `configs/model.yml` 登记 alias、display name、profile、backend、Runtime 工厂、推论／网路设定档名及 API 能力；MediaMTX 不在模型 manifest 中登记，其独立配置位于 `deploy/mediamtx/mediamtx.yml`。模型相对设定路径以 manifest 所在的 `configs/` 目录解析。
- **AC-12.3** Launcher 由 `models/*/configs/model.yml` 自动发现模型，拒绝重复 alias、缺失设定及非法工厂宣告；`start` 不依赖全域模型登录档，且必须指定 `--model`。
- **AC-12.4** 模型拓扑置于模型包 `model/`；模型层置于 `layers/`；Runtime 支援元件置于 `components/`；模型设定、Transformers Config／Tokenizer／Processor 选择、snapshot 与权重载入集中于 `utils/loader.py`。不再为相同职责拆分多个设定／HF loader 档案；模型专属推理参数仍由 YAML 管理。以上目录依模型实际需要建立，不建立空目录，也不把模型专属逻辑移入共用 `engine/`。
- **AC-12.5** Qwen3 与 Mage-VL 均由自身 manifest 和 Runtime 工厂装配，维持现有 API alias、网路设定、推理行为及文字 SSE 输出；Qwen token ID 仍在 API 边界增量 Detokenize，Mage-VL `text_delta` 路径不变。
- **AC-12.6** 每个模型的三项测试均只使用该目录的 `test.yml` 和模型 manifest；跨服务 alias／端点一致性由 `test_end2end.py` 的 Gateway 回应验证。
- **AC-12.7** 新模型接入指南以模型目录为主体，逐项说明必需／按需文件、各文件应承载的内容、接入顺序、standard/custom 边界、Gateway 与 Live 的条件及验证方法；指南不要求提供模型实现代码范例，也不要求维护全局模型登录清单或 Runner／Adapter 元件清单。

### REQ-13：模型包按计算责任分层并使用明确注意力路径

**需求**：复杂模型将拓扑、模型层与 Runtime 元件分置于各自模型包；模型注意力路径可直接检视、替换，不隐式分派至 fused SDPA。

**验收条件**

- **AC-13.1** Mage-VL 的模型拓扑／Config／输出型别位于 `models/mage_vl_4b/model/`，视觉 Encoder 与模型层位于 `layers/`，Runner／Adapter／生成生命周期元件位于 `components/`，载入流程位于 `utils/loader.py`；旧 `components/` 模型拓扑汇入路径不得残留。
- **AC-13.2** Qwen3 自有 GQA 与 Mage-VL 视觉注意力明确执行 QKᵀ、softmax 与权重乘 V；两个模型包的执行程式不得呼叫 `F.scaled_dot_product_attention()`，Mage-VL 文字骨干及其 Transformers 基线均固定选用模型包注册的分块 eager attention。
- **AC-13.3** 逐模型 `test_components.py`、`test_performance.py`、`test_end2end.py` 均以目前固定本机 snapshot／设定实际执行并通过；性能对照须两侧生成相同设定的 token 数，端到端须验证 Gateway SSE 与模型已宣告的多模态路由。
- **AC-13.4** Mage-VL 视觉 Attention 类别与模组分别使用 `MultiHeadAttention`、`multi_head_attention.py` 命名；模型模组中的前向对照程式直接置于 `if __name__ == "__main__":` 区块，CUDA 可用时默认使用 CUDA，仅在不可用时回退 CPU，不再透过测试函式包装呼叫。
- **AC-13.5** 每个含独立数值计算的模型层、视觉组件、KV/采样执行组件及模型前向文件，在文件内的 `if __name__ == "__main__":` 直接写小型基线对照；入口输出 `==========engine===========` 与 `==========baseline==========`，对比 Transformers 或等价 PyTorch 基线并断言数值／输出一致，不通过辅助测试函数包装。纯配置、输出数据结构和无前向计算的适配文件不强制添加入口；需要模型 YAML 环境的 Runner 入口须在开发文档说明启动环境。
- **AC-13.6** Mage-VL 由 `model/mage_vl.py` 明确组装视觉塔、Qwen3 文本骨干和多模态前向；视觉特征、placeholder 注入及 StreamMind Gate 位于独立模块。视觉 RoPE、Patch、Attention、Encoder stack 分文件维护；旧 `multimodal_model.py`／`vision_encoder.py` 导入路径保持兼容，重构不得改变 checkpoint state-dict key 路径。
- **AC-13.7** Mage-VL 新增计算模块的内联入口对照固定 revision 官方 `modeling_mage_vl.py` 的小型前向；Qwen3 模型、层和 Runner 对照 Transformers Qwen3。对照载入随机小型模型参数，不载入大型 checkpoint tensor；前向优先使用 CUDA，仅在 CUDA 不可用时回退 CPU；只验证模块前向与结构兼容，不将其作为真实权重／GPU 服务验收。

### REQ-14：以标准 Chat Completions 格式支援模型工具调用

**需求**：共用 Inference API 接受标准 function tool message/schema，模型包以 manifest adapter 将原生输出转换成 JSON 或 SSE 标准工具调用；Inference 不执行工具。

**验收条件**

- **AC-14.1** Given模型 manifest 登记 `tool_calling=true` 和 `api.tool_call_adapter`，When启动/validate，Then检查 adapter factory 路径及 `parse_output`、`create_stream_parser` 方法；无 adapter 的模型不得宣告该 capability，Mage-VL 保持 false。
- **AC-14.2** Given Qwen3 `tools` 与 `tool_choice="auto"`，When原生 JSON 工具调用块有效且 Schema 符合，Then回应标准 `assistant.tool_calls`、唯一 call ID、JSON 字串 arguments 和 `finish_reason="tool_calls"`；一般文字行为保持原契约。
- **AC-14.3** Given标准 SSE 生成跨 chunk Qwen tool-call marker/name/arguments，When解析输出，Then以 `delta.tool_calls` 分片输出且不把 Qwen marker 泄漏到文字；完整 marker 结束时需验证 JSON 与 tool Schema，失败时发 error event 且不产生有效 tool_calls finish。
- **AC-14.4** Given调用方送 `tool_choice="auto"` 或 `"none"`，When进入模板/生成，Then只向模型呈现 auto 模式的 tool definitions；函数指定型对象、`parallel_tool_calls=true`、`strict=true`、非法定义或参数必须明确拒绝，不静默放宽。
- **AC-14.5** Given历史包含标准 assistant function calls 与关联 `role="tool"` message，When Qwen InputProcessor 标记 token，Then保留 tool_call ID、名称、参数、结果与 Chat Template 顺序，并拒绝孤儿、重复 ID 或未闭合调用历史。
- **AC-14.6** Given模型输出函数名称不在该轮允许工具清单、JSON arguments 非 object 或不符 JSON Schema，When adapter解析，Then返回 invalid tool-call 错误；不得将结果呈现为可执行工具调用。

**验证方式**：Qwen adapter 对 JSON/SSE 的 chunk-split 与 Schema 案例、Inference API contract；真实 Qwen revision 的 JSON/SSE smoke。客户端工具执行范例透过既有 Gateway `/v1/chat/completions` 验收，不增加专用 Agent endpoint。实际状态见本规格与 Gateway `tasks.md`。

## 工具调用需求追踪

| 上游条款 | REQ | 模块 | 任务 ID | AC | 验收证据/状态 |
|---|---|---|---|---|---|
| 使用者要求第2、4点：标准非流式协议和 tool-call SSE delta | REQ-14 | `api/contracts/schemas.py`、`api/contracts/tool_calling.py`、`api/routes/chat_completions.py`、Qwen `processor.py`／`tool_calling.py` | T-26 | AC-14.2–AC-14.6 | Qwen 本机模型的非流式客户端工具循环和 SSE tool-call delta 已实测；完整 Gateway 请求受 Windows Application Control 阻挡 |
| 使用者要求为后续工具调用底座保持通用 | REQ-14 | Launcher、`api/application/service_app.py`、model manifest | T-26 | AC-14.1、AC-14.2 | Qwen adapter 已登记；Mage false；后续模型以自身 adapter 接入 |

### REQ-15：共用 Inference 程式按职责分包

**需求**：API 与 Engine 的共用程式依变化原因和职责组织为可独立搬移的 Python packages；模型专属计算仍由模型包持有，外部 API 与执行行为不因整理目录改变。

**验收条件**

- **AC-15.1** API 共用模块按 `application/`、`contracts/`、`processors/`、`routes/` 分组；应用装配、OpenAI schema／序列化／工具协议、输入输出转换与 HTTP route 分别归属对应 package。
- **AC-15.2** Engine 共用模块按 `core/`、`contracts/`、`scheduling/`、`cache/`、`execution/`、`transport/`、`bootstrap/`、`config/` 分组；Engine Core 依赖共享契约、Scheduler、KV 管理、`core/ports.py` 的 Executor Protocol 及现有 `media/` 文件辅助，不直接依赖 `execution/` 的本机执行器实现或模型包。
- **AC-15.3** （Engine 路径由 REQ-18 取代）历史目标为依赖方向单向，跨进程类型不依赖 API 框架或模型 runtime；当前由 `engine/sequence.py` 定义值对象，不改变 API 和模型对契约的依赖方向。
- **AC-15.4** Launcher、模型 manifest factory、所有 Python import、README 与当前设计文件使用新模块路径；`launcher.py list/validate`、模块 import smoke、双模型 `test_components.py` 与模块级 CPU 前向入口通过；不增加或改变 HTTP endpoint、请求／回应 schema、模型能力或运行配置。

**验证方式**：Python 语法解析、模块导入、Launcher `list`／`validate`、逐模型 `test_components.py`，以及适用的 Engine／API contract 测试；真实 GPU 服务性能不是目录重构的验收条件。

## 目录标准需求追踪

| 上游条款 | REQ | 模块 | 任务 ID | AC | 验收证据／状态 |
|---|---|---|---|---|---|
| 使用者要求按高内聚、低耦合整理共用 Inference 目录，保持统一且可整体搬移 | REQ-15 | `api/{application,contracts,processors,routes}/`、`engine/{core,contracts,scheduling,cache,execution,transport,bootstrap,config}/` | T-29 | AC-15.1–AC-15.4 | 以 import graph、Launcher 预检、逐模型组件测试和 API contract 检查验收 |

### REQ-18：以扁平 Engine 核心承载当前模型执行

**需求**：Engine 使用少量位于 `engine/` 根目录的关键模块承载请求状态、KV block、调度、Engine 编排、进程执行、IPC 和模型配置；模型专属前处理、Runner、Adapter、计算图及算子集中在各模型包。REQ-18 取代 REQ-15 对 Engine 子 package 结构的要求，并保持 Network/API 对外行为。

**验收条件**

- **AC-18.1** Engine 当前实现使用 `sequence.py`、`block_manager.py`、`scheduler.py`、`inference_engine.py`、`worker.py`、`engine_client.py`、`config.py`、`bootstrap.py` 等根目录模块；Engine 不再按 `core/`、`contracts/`、`scheduling/`、`cache/`、`execution/`、`transport/`、`bootstrap/`、`config/` 建立重复子 package。
- **AC-18.2** 模型 Tokenizer／Processor、模型 Runner／Adapter、模型拓扑、Cache 计算、Sampling 与 kernels 仍由 `models/{model_alias}/` 持有；共用 Engine 不按模型 alias 分支，也不增加通用插件注册框架。
- **AC-18.3** API Server、Engine Core、Model Worker 的进程边界，Scheduler 与 BlockManager 所有权、Qwen/Mage 已实现行为、HTTP schema、SSE、Launcher alias 和模型 YAML 均保持不变；Engine IPC 和 Worker factory 使用扁平模块路径。
- **AC-18.4** 双模型 Launcher `list`/`validate`、Engine/Worker 导入与启动配置解析、Qwen3 Engine 生命周期组件入口和 Mage-VL 共用组件回归通过；代码与 README/design/tasks 中的 Engine 路径一致。

**验证方式**：扁平模块导入和 Launcher 无权重预检、Qwen3 与 Mage-VL 组件入口、适用的 API schema/route contract 检查。此目录收敛不要求启动完整权重或 GPU 服务；若执行 GPU 验证，需避免同时启动多个服务或重复加载 checkpoint。

### REQ-19：以扁平 Network 模块承载推论 HTTP API

**需求**：共用 HTTP API 模块统一命名为 `network/`，将原 `api/application/`、`api/contracts/`、`api/processors/`、`api/routes/` 的职责文件直接放在该目录，保持现有 HTTP 行为和模型插件边界。

**验收条件**

- **AC-19.1** `network/` 根目录直接包含 `app.py`、`service.py`、schemas／serializers／tool calling、聊天／视频／Live 路由及 API 输入／输出处理器；不再存在 `api/` package 或上述四个嵌套子 package。
- **AC-19.2** `launcher.py` 载入 `inference.network.service`；模型 Processor 与 tool adapter 依赖 `inference.network` 稳定接口，不依赖旧 `inference.api` 导入路径。
- **AC-19.3** 健康检查、模型列表、聊天 JSON/SSE、工具调用协议、视频及 Live 路由的路径、schema、能力门控和 Engine IPC 行为保持不变；API 程序不载入模型 Runner 或 GPU Runtime。
- **AC-19.4** 双模型 Launcher `list`／`validate`、Network 模块导入与 FastAPI route/schema smoke、逐模型组件检查及 README／design／AGENTS 当前路径说明通过。

**验证方式**：Python 编译、Launcher 无权重预检、Network app／schema／route 导入与 FastAPI contract smoke、双模型组件入口及 Spec validator；不要求启动完整权重或推理服务。

### REQ-20：以共享 Processors package 承载跨模型处理能力

**需求**：将共用视频文件处理与 Live session 元件从 `media/` 迁移至 `processors/`，为未来跨模型处理能力提供统一扩展位置；维持现有调用语义，不将模型专属前处理移出模型包。

**验收条件**

- **AC-20.1** `src/inference/processors/` 提供共用 `video.py` 与 `live_sessions.py`；`src/inference/media/` 不再存在，源码、工厂路径和模块 imports 不引用旧 package。
- **AC-20.2** Network、Engine、Mage-VL 通过 `inference.processors` 使用共用视频与 Live 元件；上传视频临时文件处理、视频元数据／采样辅助和 Live session manager 行为保持不变。
- **AC-20.3** Processor package 的职责边界在 AGENTS、Inference README 与设计文件中说明为跨模型共用扩展点；模型专属 tokenizer、输入转换与视觉处理仍留在各自模型包。
- **AC-20.4** 双模型 Launcher `list`／`validate`、共享 processor 导入、Network app 导入、逐模型组件检查及 Spec validator 通过。

**验证方式**：Python 编译、旧导入路径搜索、双模型 Launcher 预检、共享 Processor 与 Network app 导入 smoke、逐模型组件检查及 Spec validator；不要求加载完整 checkpoint 或启动服务。

## Engine 简化需求追踪

| 上游条款 | REQ | 模块 | 任务 ID | AC | 验收证据／状态 |
|---|---|---|---|---|---|
| 使用者决定保留简单模型插件架构，Engine 仅保留 nano-vLLM 风格的关键组件文件 | REQ-18 | `engine/{sequence,block_manager,scheduler,inference_engine,worker,engine_client,config,bootstrap}.py`、`models/*/` | T-33 | AC-18.1–AC-18.4 | 实施与验收状态见 `tasks.md`；旧 REQ-15 的 Engine 子 package 结构由本需求取代 |

### REQ-16：逐模型脚本与 Qwen3 算子分析形成可重复工作流

**需求**：逐模型组件、性能与端到端入口以直读的单一 Python 脚本呈现；Qwen3 组件入口集成执行模块内的 `__main__` 基线，并提供可选的本机 Attention 算子分析；各模型的 Markdown 以表格严格说明“问题、解决方案、效果”，Excel 以编号数据作为对应证据。

**验收条件**

- **AC-16.1** Qwen3 `test_components.py` 不载入完整 checkpoint 权重，对照模型结构并依序执行 8 个 Qwen3 模块 `__main__`；模块前向默认使用 CUDA，仅在 CUDA 不可用时回退 CPU；失败时明确指出模块路径并以非零状态停止。
- **AC-16.2** Mage-VL 与 Qwen3 的 `test_components.py`、`test_performance.py`、`test_end2end.py` 均为顶层顺序脚本，不定义自有 `main()` 或测试辅助函数；命令列仍明确要求 `--config`。
- **AC-16.3** Qwen3 `--profile-attention` 从本机固定 revision 载入第 0 层 Attention checkpoint 权重，依配置记录执行设备、dtype、batch/token shape、KV block size、warmup、iteration 与可用峰值显存，分开报告 prefill/decode 延迟及 Profiler 可用的 CPU/CUDA op、shape、memory；CUDA 可用时默认使用 CUDA，合成 activation 与单层测量限制须同结果一起声明。
- **AC-16.4** 报告按模型 alias 放在根目录 `analysis/{model_alias}/`，每个模型目录仅维护一份 `analysis.md` 和一份 `analysis.xlsx`。Markdown 表格按“问题、解决方案、效果”组织，并引用对应 Excel 证据 ID；Excel 保留测量值、验收条件和状态。Qwen3 表可追加算子变更的前后数据，历史 Mage-VL Gateway 结果保存在 Mage-VL 目录。旧 `val.md/xlsx` 不再作为维护入口。

**验证方式**：双模型测试脚本 AST 检查没有自定义函数、Qwen3 组件模块入口集成检查、Qwen3 固定 snapshot 单层 CUDA Profiler、Inference Spec validator、工作簿导入/内容检查/渲染/导出。模型服务与 Gateway 只有在实际启动并通过请求后才记录端到端通过。

## 算子分析需求追踪

| 上游条款 | REQ | 模块 | 任务 ID | AC | 验收证据/状态 |
|---|---|---|---|---|---|
| 使用者要求集成检查 Qwen3 各模块 `__main__`、测试入口改成直接易读脚本、实做算子分析并形成长期工作流 | REQ-16 | `tests/{qwen3_0_6b,mage_vl_4b}/`、Qwen3 `layers/attention.py`、`analysis/qwen3_0_6b/analysis.md`、`analysis.xlsx` | T-30 | AC-16.1–AC-16.4 | 实作完成后以 WSL 本机环境、固定 Qwen snapshot、RTX 4060 和 Artifact Tool 实测；状态以本任务验收纪录为准 |

### REQ-17：Qwen3 Engine 组件明确分工并可替换执行后端

**需求**：以 Qwen3 为首个模型，将请求状态、KV block 分配、逐轮调度、Engine 编排、模型执行与 CUDA Attention 后端整理为具有明确所有权的可用组件；保留现有 API、Worker 子程序边界及 Mage-VL 共用路径。

**验收条件**

- **AC-17.1** Engine 侧 `SequenceState` 是 Scheduler 使用的单一请求进度对象，保存 request ID、prefill/decode 进度、输出上限与 block table；完成、取消及失败由 Engine Core 终止请求并移除该状态，不持有 GPU tensor、Transformers Cache、媒体 bytes 或 API response 对象。
- **AC-17.2** `BlockManager` 拥有 Engine 侧物理 block ID 空闲表、逐请求 block table、准入时的原子容量保留与幂等释放；Qwen Worker 持有对应实体 GPU page。每轮 Worker 工作项携带该轮有效 block table，失败、取消与完成后两侧均可释放，不发生跨请求 page 复用污染。首轮继续预留 prompt 与最大输出所需容量，不采用会造成 KV 等待死锁的按需分配。
- **AC-17.3** 现有 `TokenScheduler` 仍是唯一逐轮批次调度器；现有 Engine Core 仍是唯一请求编排器。模块命名不得复制第二套 Scheduler／Engine，Qwen Runner 仍由模型包持有，Mage-VL 共用契约和输出路径维持不变。
- **AC-17.4** Qwen Attention 后端以稳定调用契约接收 Q/K/V、实体 page pool、block tables、各序列长度和位置资料；CUDA 路径直接按 block table 读取 KV 并执行 decode attention，不先 gather 成连续完整 KV 张量。输入边界、dtype、装置及 block table 不合法时明确失败。
- **AC-17.5** 原生 CUDA op 为选用后端；PyTorch 参考路径保留为 CPU fallback、数值基线和 CUDA 扩展不可用时的明确 fallback。YAML/Runtime 能选择后端并如实回报实际后端，不可把 PyTorch gather 路径标示为 CUDA PagedAttention。
- **AC-17.6** 自定义算子支持 Qwen 当前部署的 CUDA、float16/bfloat16 数据路径，并覆盖多序列、不同 context length、非连续物理页表、block 边界和空/非法输入；kernel 不依赖模型别名或 HTTP 类型。
- **AC-17.7** Qwen 组件入口对照 CUDA 输出与 PyTorch 参考输出，固定 snapshot 单层 profile 可分开报告 decode latency 和使用路径；没有真实模型／GPU 验收时不得宣称服务性能提升或生产可用。

**验证方式**：Sequence/BlockManager 合成生命周期与容量边界检查、Scheduler/Engine/Worker 契约检查、Qwen CUDA extension build 与自定义 op contract、同输入 CUDA/PyTorch 数值比较、固定 checkpoint 单层 decode profile 与 CUDA backend 服务／Gateway SSE。当前实际执行结果、设备限制与性能结论见 `tasks.md` 和 `analysis/qwen3_0_6b/`。

## Qwen3 Engine 组件重构追踪

| 上游条款 | REQ | 模块 | 任务 ID | AC | 验收证据／状态 |
|---|---|---|---|---|---|
| 使用者要求先提交工作树，再以 Qwen3 为基线系统重构 nano-vLLM 风格的 Engine 组件，并以原生 CUDA 实作可复用的分页 Attention | REQ-17 | `engine/sequence.py`、`engine/block_manager.py`、`engine/scheduler.py`、`engine/inference_engine.py`、Qwen `components/model_runner.py`、`models/qwen3_0_6b/kernels/cuda/` | T-31、T-32 | AC-17.1–AC-17.7 | T-31/T-32 实施及实际验收状态见 `tasks.md`；模型与 GPU 性能声明仅依据固定环境的实测 |
