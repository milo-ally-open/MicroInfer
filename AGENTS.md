# MicroInfer 项目协作规范

## 项目目标

本项目用于构建可直接对外提供服务的模型推理 API。共用 Python FastAPI Network 层负责 API Key 认证、tenant 限流与 OpenAI 兼容 API；各模型 Runtime 负责模型加载、多模态处理和 GPU 推理。当前仓库不包含 API Gateway 或控制面。

## 目前状态

- `src/inference/` 已有 Mage-VL-4B 与 Qwen3-VL-2B Transformers 推理服务；共用 API／Engine 分层、Token Scheduler 与子进程 Model Worker。
- Python API 使用模型 manifest 中的名称：`mage-vl-4b` 与 `qwen3-vl-2b`；Mage-VL 上游权重仓库 ID `microsoft/Mage-VL` 只用于模型加载，不作为 API `model` 值。
- SGLang 与 vLLM inference backend 尚未实现；Inference Engine 参考 vLLM／nano-vLLM 的 Engine、Scheduler、KV 与 Worker 职责，但不宣称其兼容性或效能。Mage-VL 与 Qwen3-VL 的 Engine KV 均为逻辑 reservation，模型 Cache 由对应 Transformers Runtime 管理。

## 重要目录与文件

```text
checkpoints/                         # 本地模型权重与 Hugging Face 缓存
docs/                                # 项目文件
scripts/download.py                    # 通用 Hugging Face 模型下载工具
tests/mage_vl_4b/                    # Mage-VL 专属模型级测试与分析
tests/qwen3_vl_2b/                   # Qwen3-VL-2B 专属模型级测试与分析
src/inference/network/               # 共用 FastAPI app、HTTP schemas、路由与处理器
src/inference/network/service.py     # 按模型 manifest 创建共用 API app
src/inference/engine/                # 扁平 Engine 核心模块：Scheduler、KV block、Engine、Worker、IPC 与配置
src/inference/engine/bootstrap.py    # 依模型 manifest 创建共用 Engine Core／Worker
src/inference/processors/            # 跨模型共用的视频处理与 Live session 组件
src/inference/models/<model>/        # 模型专属代码；YAML 与依赖分别归档在 profiles/、requirements/
src/inference/models/mage_vl_4b/     # Mage-VL Model Runner、前处理、配置及依赖
src/inference/launcher.py            # 推理启动器
tests/                               # 手动模型组件与 API 验收说明；scripts/client.py 是手动 API 客户端
```

Inference 参考 vLLM／nano-vLLM 的 API Server、Engine Core、Scheduler、KV Cache Manager、Worker／Model Runner 职责，Engine 内部保持少数平面模块，不复制完整 vLLM 插件体系。API Server、Engine Core 与 Model Worker 为三个程序；Engine Core 逐轮调度 prompt prefill 与单 token decode。Mage-VL 与 Qwen3-VL 的 Engine KV block 是逻辑 reservation；各自模型 Runner 负责调用 Transformers 模型前向。没有 vLLM backend 或 prefix cache。

`engine/scheduler.py` 直接包含线上批次规划；Mage-VL Runner 对兼容的同阶段请求执行多序列 forward，不兼容时逐请求执行。Qwen3-VL 使用官方多模态顶层编排与 DynamicCache；其 Vision Block、文字 Decoder Layer 和 Embedding、Attention、MLP、Norm、RoPE、Merger、LM Head 位于模型包中，并通过固定组件槽位装配。Attention、MLP、Norm、RoPE 和 Vision/Text 组合层已有本地前向实现，并通过组件级测试对齐 Transformers；Attention backend 仍由 Transformers 选择。两个模型的 Engine KV 都是逻辑 reservation。模型专属模型级测试与分析工具按 Python 模型目录名放在 `tests/<model_name>/`，不从 `src/` 汇入组件实现，且强制使用本地权重；目录名与 API `alias` 可能不同，例如 `qwen3_vl_2b` 对应 `qwen3-vl-2b`。

开始修改前先阅读根目录 `README.md` 和 `docs/README.md`，并依任务阅读 `TUTORIAL.md`、`src/inference/README.md` 及对应模型目录 README。以实际工作树内容为准；若文件和代码不一致，先查明差异，再只更新任务相关的内容。

## 架构边界

- 共用 FastAPI Network 层负责 Bearer API Key 验证、tenant 识别与限流、模型 API、请求大小限制、SSE、视频上传和 Live API。它直接作为模型服务入口。
- Engine 与模型 Runtime 负责模型权重生命周期、图像／视频理解、生成及 GPU 资源管理。API 认证属于 FastAPI Network 层，不应放入模型组件或 Engine Worker。
- MediaMTX 是实时推流时使用的 sidecar，不是模型服务或 API 网关。图片与上传视频文件流程不应要求 MediaMTX。
- 模型 API 监听地址与端口由各模型 `profiles/network.yml` 控制；通过 `inference serve --model <alias> --host 0.0.0.0` 对受控网络提供服务。
- Inference 的 `/v1/chat/completions` 支持文字／图片输入与 SSE 文字输出。视频文件理解及实时视频输入依各自 API 逐段／单次响应；「流式输出」指模型文字 token 的 SSE 输出，不代表视频输入必须逐 token 或逐帧返回。

## 配置与权重

- 推理与模型前向计算默认使用 CUDA；仅当 `torch.cuda.is_available()` 为 false 时回退到 CPU。不要将实际推理设备硬编码为 CPU；权重分片读取、Tokenizer 和请求输入暂存于 CPU 不代表模型前向运行在 CPU。

- 每个模型的 `profiles/model.yml` 登记 alias、API 能力、Runtime／Processor 工厂及配置文件名。Launcher 自动发现 `models/*/profiles/model.yml`，启动时必须透过 `--model` 指定 alias；同一 `profiles/` 中的配置文件名相对该目录解析。所有模型共用 `network/service.py` 和 `engine/bootstrap.py`；新模型不需创建专属 `app.py` 或 `engine_core.py`。
- 模型目录顶层只放 `example.py`、`loader.py`、`modeling.py` 和 `runtime.py`。运行配置统一放 `profiles/`，模型依赖放 `requirements/`，模型专属能力和架构辅助代码放 `architecture/`，运行时执行组件放 `components/`，计算层放 `layers/`。MageVL 和 Qwen3-VL 均保留 `architecture/__init__.py`；专属配置、输出类型、图像／视频输入适配和 Live 能力归入 `architecture/`。MediaMTX sidecar 配置集中在 `deploy/mediamtx/mediamtx.yml`，不得登记到模型配置中。
- 模型推理参数写在模型 `profiles/inference.yml`；API 绑定地址及模型连接的媒体端点写在 `profiles/network.yml`；MediaMTX 服务本身的监听、认证与协议配置放在独立的 `deploy/mediamtx/mediamtx.yml`。新增或调整可调参数时，优先加入合适的 YAML，不要把实验超参数散落在 Python 或 shell 脚本。
- 当前模型配置使用本地固定 revision 快照并启用离线加载；启动服务前先通过 `scripts/download.py` 或受控模型制品流程准备 `checkpoints/` 中的权重。通用下载工具要求明确指定 `--repo-id`、`--revision` 和 `--local-dir`。
- `checkpoints/` 内的大型权重及缓存由 `.gitignore` 整体忽略。分析报告按模型目录名放在根目录 `analysis/<model_name>/`，每个模型目录维护一份 `analysis.md` 和一份 `analysis.xlsx`；Markdown 按“问题、解决方案、效果”汇报，逐项编号并指向 Excel 证据；更新时保留历史记录与来源限制。
- 不要在 Git 中加入模型权重、缓存、私密密钥或 API token。认证数据由环境变量或安全配置提供。

## 修改与验证方式

- Python 不维护 pytest 测试套件。共用 API 认证模块通过自身 `__main__` 入口做 CPU 级自检；模型计算层和运行组件的模块级验证写在各自 `__main__` 入口。`scripts/client.py` 是手动 FastAPI 冒烟客户端；模型级 Test/Bench/Profile 工作负载及说明放在 `tests/<model_name>/`。模型实现说明放在 `src/inference/models/<model_name>/README.md`；具体执行方式见对应模型测试 README。

- 先检查相关代码、配置、文件和现有工作树修改，再进行最小范围的变更。保留使用者已有的未提交修改，不要用整文件覆写方式抹除它们。
- 不要自行下载大型模型、启动 GPU 推理服务、修改远端服务器或推送远端仓库；只有在使用者明确要求时才执行。
- 不要自行提交或推送 Git 变更，除非使用者明确要求。不要未经要求创建新分支。
- 除非使用者明确要求测试／验证，或任务明确要求交付验证结果，否则不要新增或执行测试。若未执行，清楚说明；不得宣称未做的检查已通过。
- 执行任何删除、搬移、清理或覆写前，先核对目标路径和影响范围；涉及模型权重或工作流数据时格外小心。
- 完成后摘要实际修改、相关文件、验证方式与尚未处理的事项。不要把计划描述成已实现。

## 语言与文件

- 与使用者沟通使用简体中文。
- 项目文件、代码注解和面向开发者的说明使用简体中文；代码识别字、命令、API 字段及上游专有名称维持原文。
- 文件中的启动顺序、端口、端点和配置路径必须与实际代码及 YAML 一致。
