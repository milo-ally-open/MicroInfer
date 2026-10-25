# 提案：以 Qwen3-0.6B 验证五层推理引擎与模型包接入

> 行为规格：[spec.md](spec.md)｜技术设计：[design.md](design.md)｜实施与验收：[tasks.md](tasks.md)
> 状态：五层引擎已落地；本次改造将模型接入入口集中到单一模型目录。Qwen3-0.6B 与 Mage-VL-4B 是目前已登记模型；尚未完成的验收以 `tasks.md` 标示。

## 背景与已验证基线

推理服务目前包含共用 FastAPI API、Engine Core、Token Scheduler、KV Cache Manager 和子程序 Model Worker。Mage-VL-4B 维持 Transformers 推理；Qwen3-0.6B 已有本地 PyTorch 计算图、批次 prefill/decode、Safetensors 载入及 Worker 实体 KV page pool。两者共用对外 API 与 Engine 主链路，但模型输入、计算图、Cache 实作不同。

改造前，Qwen3 Worker 会在生成步骤将 token ID 解码成文字；Worker 程序管理和 Engine Core RPC 集中在 `engine/worker.py`；Qwen Runner 在 forward 前组装批次张量。这使 API 输入处理、Engine 输出转换、执行器、硬体平台和 Runner 批次组装的责任边界仍不够清晰。

既有 RTX 4060 验收及性能记录曾保存在 `checkpoints/val.md`、`checkpoints/val.xlsx`，现已按模型迁入根目录 `analysis/<model_alias>/`。这些是先前版本的基线证据；本次结构改造后的 SSE、Executor 与 InputBatch 行为仍需按 `tasks.md` 重新验收。

## 目标与决策

以 vLLM V1 的职责拆分为参考，将推理引擎明确划分为五层：

1. **L1 接口与门面层**：HTTP／OpenAI schema、模型输入处理、增量 Detokenize、SSE 序列化及 EngineClient。
2. **L2 编排与调度层**：Engine Core 请求生命周期、Token Scheduler、KV reservation 与 Worker 结果事件化。
3. **L3 执行抽象与 Worker 层**：Executor 管理程序及 IPC；Platform 封装装置和记忆体观测；Worker 在子程序载入模型。
4. **L4 模型运行时层**：Model Runner 组装 `InputBatch`、执行批次 forward、管理生成状态和呼叫 Sampler。
5. **L5 模型与算子层**：模型拓扑、模型元件、权重映射及模型生态相容程式。

模型相容能力会横跨 L1 输入适配与 L5 权重／模型适配；效能能力会横跨 L2 至 L5。五层用来界定程式责任，不要求每层独立成程序或目录。API Server、Engine Core、Model Worker 仍维持现有程序拓朴。

本次决策：

- 保持 Mage-VL 与 Qwen3 的模型包、载入方式及能力差异；不把任何一个模型的前处理或计算图移入 Go Gateway 或共用 Engine。
- Qwen3 API 程序使用模型专属 `InputProcessor` 载入本机 Tokenizer，产生可序列化的 `PreparedChatInput`；不在 API 程序载入模型权重或建立 GPU Runtime。
- Qwen3 Worker 回传 token ID；API 输出处理器按 tokenizer 规则逐步形成文字 chunk。Mage-VL 现有文字增量事件维持原路径。
- L2 `EngineOutputProcessor` 只把 Worker token／文字结果转成 Engine 事件，不负责 HTTP、JSON 或 SSE。
- L3 先提供 `Executor`、`Platform` 契约及目前实际使用的 `LocalProcessExecutor`、`TorchCudaPlatform`。Ray、多 GPU、其他硬体平台不宣称已实作。
- L4 `InputBatch` 明确承载本轮 token 张量、长度、位置偏移和每 request block table；目前每轮建立，不宣称已采用固定缓冲区或 CUDA Graph。
- L5 Qwen3 自有 PyTorch 模型计算图；Transformers 仅提供 config／tokenizer 等生态相容能力，Safetensors loader 负责权重映射。
- 新模型只需实作模型 Runner／Adapter 并登记工厂；API app、Engine Core、Executor 和 Worker 建立流程由共用入口负责，不再要求每个模型复制 `app.py`／`engine_core.py`。
- 模型对外 ID、输入处理器和选用的影片／Live 入口由 Registry 登记；上游 repo ID 保持在模型 YAML，两者分开维护。
- 每个模型目录透过 `configs/model.yml` 自我登记 alias、Runtime 工厂、API 能力与相对设定路径；Launcher 自动发现模型 manifest，启动时必须明确选择模型，不设全域预设模型档。
- 每个模型提供 `runtime.py:create_runtime()`，由共用 Worker 取得统一 Runtime facade；Runner／Adapter 等模型元件由该工厂在 Worker 子程序内组装。
- 模型目录采按需分层：`model/` 放拓扑，`layers/` 放模型层，`components/` 放 Runner／Adapter／批次与 Cache 元件，`utils/loader.py` 集中设定、Transformers 资产及权重载入。简单模型不需建立不使用的目录。

## 范围

- 重写本目录 `proposal.md`、`spec.md`、`design.md`、`tasks.md`，以五层责任和验收追踪取代旧的三层规格文字。
- 将 Qwen3 tokenization／增量 detokenization 放到 API 输入输出边界，保持 OpenAI Chat、逐步文字 SSE、finish usage、`[DONE]` 和取消收尾。
- 将 Engine Core 的 Worker RPC 改由 Executor 介面承接，并以 Platform 介面处理 Qwen CUDA 装置与记忆体观测。
- 将 Qwen 每轮批次张量组装抽成 `InputBatch`，保留目前 prefill/decode 批次和每 request cache 隔离。
- 建立共用模型工厂与设定载入流程，移除 Mage-VL／Qwen 重复的服务装配；提供逐步生成模型 Adapter／Runner 接入契约与操作指南。
- 将模型登记与启动配置由集中式多模型 Registry 迁移至模型包内 manifest，降低新模型接入时需编辑的跨目录档案数，并保留共用 API／Engine／Worker 边界。
- 为符合单装置 causal LM、Tokenizer chat template 与逐步 `past_key_values` 契约的文字模型提供预设 InputProcessor／Runner／Adapter；多模态或不同 Cache／计算图仍由模型包持有。
- 透过合成／契约测试及固定 Qwen3-0.6B 本机 snapshot 验证；不把旧版本 GPU 结果当作新架构的验收结果。

## 非目标

- 不改 Go Gateway、Mage-VL 模型元件或既有模型服务 API；不把 API Server 与 Engine Core 改成远端服务。
- 不引入 vLLM 或 nano-vLLM runtime 依赖，不复制其专用程式码，也不宣称 vLLM 相容或达到其效能。
- 不在本次实作 Ray／多 GPU／Tensor Parallel、自订 CUDA/Triton Attention、PagedAttention kernel、prefix cache、KV offload、量化、CUDA Graph 或静态 GPU InputBatch。
- 不选定第二个模型；Qwen3-0.6B 是目前实验模型，Mage-VL 只作为既有共用 API／Engine 相容路径。
- 不做模型资料集精度评测，不下载大型模型，不启动 GPU 或 MediaMTX 作为文件编辑流程的一部分。

## 风险、回复与提交基线

主要风险是 API／Worker token 契约变更造成 SSE 内容、分块时机或取消清理退化。所有 token event 必须保留 request ID；模型增量输出需与最终文字一致。若 tokenizer 在已送出的文字前缀上重新解码出不同内容，输出处理器必须报错，不得静默重复或丢字。

KV block 所有权仍分成 Engine 的逻辑 reservation 和 Qwen Worker 的 GPU 实体 page；本次不把这种映射描述成 Attention kernel 的分页直接读取。Executor 启动失败、逾时或正常关闭须维持 Worker 收尾行为。

改造前的工作树已建立本地回复提交 `0cb3c56`（`feat: add native Qwen3-0.6B inference engine`）。它只在本地 Git 历史中，没有推送。此次五层改造也只保存在本地 Git 历史，不推送远端；既有验收数据现由 `analysis/<model_alias>/analysis.md`、`analysis.xlsx` 接续维护。

## 完成条件

- 五层责任、实际模组位置、依赖方向及限制在本规格、Inference README、程式和测试一致。
- Qwen API 的 Tokenize、Engine domain event、Worker token ID、增量 Detokenize、OpenAI SSE 顺序形成可测链路；Mage-VL 文字事件路径仍通过原有契约。
- Engine 依赖 `Executor` 契约；线上实作由 `LocalProcessExecutor` 执行 Worker 子程序。Qwen Runner 透过 YAML 选用 `torch_cuda` 平台并使用 `InputBatch`。
- 所有新增／更新任务均以唯一 `REQ`、`AC`、`T` ID 互相追踪；只将实际执行的检查记为通过。
- 固定 snapshot 的 API／SSE／取消及性能验收完成前，不宣称本次改造已通过真实模型整合。

## 新模型接入工作流补强

为缩短后续模型接入并避免能力描述漂移，Registry 增加 `standard`／`custom` 整合 profile 和明确 API capability；共用 API 启动时验证其可提供的路由，Launcher 提供不载入权重或 GPU 的 `validate` 预检。逐模型测试统一放在 `tests/{model_alias}/`，仅保留计算图基线、实际推理性能比较和 Gateway 端到端三项入口；详细证据规则见 `spec.md`／`design.md`，任务进度见 `tasks.md`。

`standard` 使用共用文字预设元件，限定于单装置 causal LM、具有固定 revision 的 Tokenizer chat template、可逐步重用的 `past_key_values` 和 token ID 输出；每个 Scheduler work item 仍独立 forward，首版不宣称批次 forward。`custom` 用于需要模型包持有计算图、Cache、Tokenizer 或媒体流程的模型。Mage-VL 和 Qwen3-0.6B 保持目前的 `custom` 实作；Qwen snapshot 可用来实际验证标准预设执行路径，但不因此改变其 Registry profile 或原生 PyTorch 服务方式。预检与合成测试仍不能取代目标部署环境的实际模型／硬体验收。

## 本次实作补强

实际影片测试显示，Mage-VL 长视讯输入会令文字骨干 eager attention 建立过大的完整 `Q×K` 中间矩阵。本次按模型责任将 Mage-VL 拓扑／输出结构放入 `model/`、视觉／文字注意力放入 `layers/`，并在 Transformers `AttentionInterface` 登记模型专属分块 eager Attention；分块大小由 Mage `inference.yml` 管理。Qwen3 自有 GQA 亦改用分组矩阵乘法。两者均维持既有 Runtime、外部 API 与文字 SSE 契约，不引入 fused SDPA；这是长序列暂存显存控制，不代表 PagedAttention Kernel。

本次维护进一步统一 Mage-VL 视觉注意力类名为 `MultiHeadAttention`，源码文件命名为 `multi_head_attention.py`；三个模型组件 CPU 对照入口将测试逻辑直接放在文件的 `if __name__ == "__main__":` 区块中。程式、文档、注释与 YAML 中的中文统一使用简体。MediaMTX 配置移至项目根目录 `deploy/mediamtx/`，不再由模型 manifest 登记或由 Inference Launcher 读取／启动；部署者单独管理 sidecar 进程。模型 `network.yml` 仍保留模型所需的 RTSP／RTMP 连接端点。

## 新模型接入文档澄清

本次只细化模型接入说明，不改变模型实现、API 契约、运行配置或启动方式。接入指南逐项描述需要新增或修改的文件及其内容，并说明 profile、Gateway、MediaMTX 和三项测试的使用条件；不附模型实现代码示例。验收以文件清单完整、内容职责与当前代码一致、README 链接有效及 Inference Spec validator 通过为准。

## Qwen 标准工具调用补强

按新增 REQ-14，共用 Chat schema／消息历史／JSON Schema 校验提供跨模型协议，Qwen3 adapter 解析原生工具调用标记，输出标准 JSON/SSE；Inference 只解析模型意图，不执行工具。工具由 OpenAI 兼容客户端或上层 Agent 应用执行，再用标准 tool message 回传至同一 Chat Completions endpoint；不增加专用 Agent endpoint。其他模型只需登记自有 adapter 和能力，不需改变共用 Engine。

本次代码完成后仍需执行静态验证、Qwen API JSON/SSE 与安全边界验收；8 个合成样本评估工具选择与参数准确率，但在实际报告写出前不记录分数，也不把准确率当工具授权证明。

## 共用 Inference 模块目录标准化

本次依据 REQ-15 将 API 与 Engine 的共用程式从平铺文件改为职责 package：API 分为应用装配、协议契约、处理器与路由；Engine 分为核心编排、共享契约、调度、Cache、执行、传输、配置与启动组合。Engine Core 依赖独立 Executor Protocol，进程执行细节留在 execution package。模型目录、HTTP endpoint、运行配置及模型行为维持原样；Launcher、模型工厂路径、Python imports 与当前架构文档同步迁移。目的在于让同一职责的模块可以作为 package 单独理解、测试和搬移，同时避免模型包依赖 API route 或 Engine 组合入口。

## Qwen3 算子分析与优化工作流

本次依据使用者要求，把 Qwen3 各自带有 `__main__` 的计算模块串成可复跑的组件集成检查；逐模型三项测试改为从顶部执行的单一脚本，不使用自定义测试 helper。新增可选 Attention profile 模式，使用本机固定 snapshot 的第 0 层真实 Attention 权重、Qwen3 配置及 KV page 大小，量测 CUDA prefill/decode 并输出 PyTorch Profiler 明细。该模式只测单层算子，activation 为合成输入；不得把它描述成完整模型性能验收。Qwen3 算子的实测、分析、优化方向和验收状态记录在 `analysis/qwen3_0_6b/`；Mage-VL Gateway 历史验收事实放在 `analysis/mage_vl_4b/`。旧 `val.*` 文件名由各模型目录中的报告取代，并保留样本数与限制。
