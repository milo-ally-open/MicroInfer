# 推论引擎五层行为规格

> 提案：[proposal.md](proposal.md)｜技术设计：[design.md](design.md)｜实施与验收：[tasks.md](tasks.md)
> 本规格描述 Qwen3-0.6B 实验改造及应保持的共用服务行为。已完成、进行中和待验收的状态以 `tasks.md` 为准。

### REQ-01：L1 接口与门面维持 API 契约

- **AC-01.1** `GET /health`、`GET /v1/models`、`POST /v1/chat/completions` 的路径、HTTP 方法、模型别名和 OpenAI response schema 维持现有行为；Qwen3 仍只登录文字聊天能力。
- **AC-01.2** 共用 `api/` 不汇入模型包，不载入模型权重、GPU Runtime 或模型专属计算图。
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
- **AC-04.4** `Platform` 定义装置解析、记忆体观测与资源释放契约；Qwen YAML 选用已实作的 `torch_cuda`。未知平台必须明确失败；未实作的 Ray、CPU、NPU 和多 GPU 不得出现在支援清单。
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

- **AC-08.1** 每个已登录模型目录 `tests/{model_alias}/` 恰有三个测试入口：`test_components.py`、`test_performance.py`、`test_end2end.py`，及一份 `test.yaml`；不再维护独立的 `tests/inference/`、`tests/gateway/`、`tests/end2end/` 测试树。
- **AC-08.2** `test_components.py` 以指定的本机固定 snapshot 和 `meta` device 比较模型包计算图与 Transformers 原生基线，依序输出 `==========engine===========` 和 `==========baseline==========`，不得载入权重或建立 CUDA context。
- **AC-08.3** `test_performance.py` 使用相同本机 snapshot、输入与生成上限，比较已启动的 Inference API 和 Transformers 原生直接生成；输出样本数、平均延迟、生成 token 数及平均 token/s。该测试需模型、GPU 与可用 Inference API，不宣称固定 SLA。
- **AC-08.4** `test_end2end.py` 以设定的模型 alias 从 Go Gateway 发送请求，验证 HTTP 成功、文字 SSE 内容及公开 alias；Mage-VL 同时验证图片 SSE 和影片上传，Qwen 验证纯文字 SSE。测试需 Gateway、相应 Inference API、API Key 和本机素材。
- **AC-08.5** 测试设定以 `--config` 明确传入；相对素材／snapshot 路径以该 YAML 所在目录解析。历史实测 JSON 保存在各模型目录的 `results/`；不新增资料集精度或 Video-MME 评测。

### REQ-09：新模型沿用共用服务装配

- **AC-09.1** 新模型在 `models/{model_alias}/configs/model.yaml` 登记设定路径、Runtime 工厂及 API 识别码；Launcher 自动发现模型 manifest，`start` 必须明确指定 `--model`。Launcher 使用同一个 `api/service_app.py`，Engine 使用同一组 Engine Core／Worker bootstrap，不要求模型包提供 `app.py` 或 `engine_core.py`。
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
- **AC-10.6** Inference manifest 与 Gateway 设定的别名及路由由实际 Gateway 端到端测试验证；部署端点与能力差异依模型 `test.yaml` 和 Gateway YAML 维护，不宣称此项测试静态比对全部设定栏位。

### REQ-11：符合契约的文字模型可使用标准预设 Runner

**需求**：符合标准文字生成契约的模型可省略模型专属 Runner、Adapter 和 InputProcessor 工厂，使用共用预设元件；标准路径不得把不相容架构或 Cache 行为伪装为通用支援。

**验收条件**

- **AC-11.1** Given `integration_profile: standard` 且模型未覆写元件工厂，When 共用 API／Worker 启动，Then 分别使用 `api/default_chat_processor.py`、`engine/standard_text.py` 的预设元件；明确登记的工厂可逐项覆写预设。
- **AC-11.2** Given 标准文字聊天请求，When API 处理输入，Then 预设 InputProcessor 使用固定 revision Tokenizer 的 `chat_template` 产生 `PreparedChatInput`，只接受文字内容，并以同一 Tokenizer 提供非串流解码和增量文字 SSE。
- **AC-11.3** Given 标准 Runner 收到 Scheduler 的 prefill／decode work item，When 执行 Worker batch，Then 每个 request 使用自己的 `past_key_values`、Cache 长度及采样状态，依序执行 work items 并按 ID 回传 token 结果；Runner 不宣称此预设实作会合并模型 forward 或使用物理 PagedAttention。
- **AC-11.4** Given 模型宣告图片／影片／Live、encoder-decoder、`device_map` 多装置或无法重用 `past_key_values`，When 执行标准 profile 预检或 Worker 载入，Then 明确拒绝或要求切换 `custom`；不得退回完整 `generate()` 绕过 Token Scheduler。
- **AC-11.5** Given 本机固定 revision snapshot，When 使用标准 Runner 做实际模型 smoke，Then 至少验证 Tokenizer template、prefill、逐 token decode、Cache 延续、finish/cancel 和文字重建；合成 CPU 契约测试不得替代此模型 smoke。

### REQ-12：单一模型接入资料集中于模型目录

**需求**：每个模型以单一目录承载 Runtime 入口、模型专属程式、设定和依赖；新增模型主要透过新增模型包及其 manifest 完成，不需修改共用 API／Engine 装配或集中式模型清单中的多个位置。

**验收条件**

- **AC-12.1** 每个模型包以 `models/{model_alias}/runtime.py` 提供 `create_runtime()` 工厂；该工厂回传单一 Runtime facade，符合 Worker 的 load、health、generation、batch、finish、cancel、GPU status 及 close 契约。
- **AC-12.2** 每个模型的 `configs/model.yaml` 登记 alias、display name、profile、backend、Runtime 工厂、推论／网路设定档名及 API 能力；MediaMTX 不在模型 manifest 中登记，其独立配置位于 `deploy/mediamtx/mediamtx.yml`。模型相对设定路径以 manifest 所在的 `configs/` 目录解析。
- **AC-12.3** Launcher 由 `models/*/configs/model.yaml` 自动发现模型，拒绝重复 alias、缺失设定及非法工厂宣告；`start` 不依赖全域模型登录档，且必须指定 `--model`。
- **AC-12.4** 模型拓扑置于模型包 `model/`；模型层置于 `layers/`；Runtime 支援元件置于 `components/`；模型设定、Transformers Config／Tokenizer／Processor 选择、snapshot 与权重载入集中于 `utils/loader.py`。不再为相同职责拆分多个设定／HF loader 档案；模型专属推理参数仍由 YAML 管理。以上目录依模型实际需要建立，不建立空目录，也不把模型专属逻辑移入共用 `engine/`。
- **AC-12.5** Qwen3 与 Mage-VL 均由自身 manifest 和 Runtime 工厂装配，维持现有 API alias、网路设定、推理行为及文字 SSE 输出；Qwen token ID 仍在 API 边界增量 Detokenize，Mage-VL `text_delta` 路径不变。
- **AC-12.6** 每个模型的三项测试均只使用该目录的 `test.yaml` 和模型 manifest；跨服务 alias／端点一致性由 `test_end2end.py` 的 Gateway 回应验证。
- **AC-12.7** 新模型指南以模型目录为唯一接入主体，包含最小 manifest、Runtime 工厂和设定范例；文件不得要求维护全域模型登录清单或 Runner／Adapter 元件清单。

### REQ-13：模型包按计算责任分层并使用明确注意力路径

**需求**：复杂模型将拓扑、模型层与 Runtime 元件分置于各自模型包；模型注意力路径可直接检视、替换，不隐式分派至 fused SDPA。

**验收条件**

- **AC-13.1** Mage-VL 的模型拓扑／Config／输出型别位于 `models/mage_vl_4b/model/`，视觉 Encoder 与模型层位于 `layers/`，Runner／Adapter／生成生命周期元件位于 `components/`，载入流程位于 `utils/loader.py`；旧 `components/` 模型拓扑汇入路径不得残留。
- **AC-13.2** Qwen3 自有 GQA 与 Mage-VL 视觉注意力明确执行 QKᵀ、softmax 与权重乘 V；两个模型包的执行程式不得呼叫 `F.scaled_dot_product_attention()`，Mage-VL 文字骨干及其 Transformers 基线均固定选用模型包注册的分块 eager attention。
- **AC-13.3** 逐模型 `test_components.py`、`test_performance.py`、`test_end2end.py` 均以目前固定本机 snapshot／设定实际执行并通过；性能对照须两侧生成相同设定的 token 数，端到端须验证 Gateway SSE 与模型已宣告的多模态路由。
- **AC-13.4** Mage-VL 视觉 Attention 类别与模组分别使用 `MultiHeadAttention`、`multi_head_attention.py` 命名；模型模组中的 CPU 对照程式直接置于 `if __name__ == "__main__":` 区块，不再透过测试函式包装呼叫。
- **AC-13.5** 每个含独立数值计算的模型层、视觉组件、KV/采样执行组件及模型前向文件，在文件内的 `if __name__ == "__main__":` 直接写小型基线对照；入口输出 `==========engine===========` 与 `==========baseline==========`，对比 Transformers 或等价 PyTorch 基线并断言数值／输出一致，不通过辅助测试函数包装。纯配置、输出数据结构和无前向计算的适配文件不强制添加入口；需要模型 YAML 环境的 Runner 入口须在开发文档说明启动环境。
