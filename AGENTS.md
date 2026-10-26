# MicroInfer 专案协作规范

## 专案目标

本专案用于建置视讯理解模型的 API 服务与模型无关的 API 闸道。闸道和推论服务是不同元件，必须分开维护：闸道处理外部客户端接入，推论服务负责模型载入、GPU 推理及模型专属功能。

## 目前状态

- `src/inference/` 已有 Mage-VL-4B Transformers 推论服务，以及 Qwen3-0.6B 原生 PyTorch 推论引擎；共用 API／Engine 分层、Token Scheduler 与子程序 Model Worker。
- `src/gateway/` 已实作 Go API 闸道 v1；目前能力与限制见 `src/gateway/README.md` 及 `docs/Specifications/gateway/`。
- 目前登录模型为 `mage-vl-4b`（上游 `microsoft/Mage-VL`）及 `qwen3-0.6b`（上游 `Qwen/Qwen3-0.6B`）。不要把专案别名和 Hugging Face `repo_id` 混为一谈。
- SGLang 与 vLLM inference backend 尚未实作；Inference Engine 参考 vLLM／nano-vLLM 的 Engine、Scheduler、KV 与 Worker 职责，但不宣称其相容性或效能。Mage-VL 的 KV 是逻辑 reservation；Qwen Worker 将 Engine block ID 对应至实体 GPU page，PyTorch GQA Attention 会 gather KV pages。Go 闸道 v1 已实作，但仍须依实际验收证据描述其能力，不代表已部署或通过真实模型整合。

## 重要目录与文件

```text
checkpoints/                         # 本机模型权重与 Hugging Face 快取
docs/                                # 专案文件
scripts/download.py                  # 通用 Hugging Face 模型下载工具
scripts/mage_vl_4b/                      # Mage-VL 专属探索与诊断工具
scripts/qwen3_0_6b/                      # Qwen3-0.6B 专属探索与诊断工具
scripts/README.md                    # Scripts 工具使用说明
src/gateway/                         # Go API 闸道与 route/stream/upstream/videoproxy 模组
src/inference/network/               # 共用 FastAPI app、HTTP schemas、路由与处理器
src/inference/network/service.py     # 依模型登录建立共用 API app
src/inference/engine/                # 扁平 Engine 核心模块：Scheduler、KV block、Engine、Worker、IPC 与配置
src/inference/engine/bootstrap.py    # 依模型 manifest 建立共用 Engine Core／Worker
src/inference/processors/            # 跨模型共用的视频处理与 Live session 元件
src/inference/models/<model>/        # 模型专属程式、configs/ 与 requirements/
src/inference/models/mage_vl_4b/     # Mage-VL Model Runner、前处理、设定及依赖
src/inference/launcher.py            # 推论启动器
tests/                               # 逐模型测试：tests/<model_alias>/（三项测试与 test.yml）
```

Inference 参考 vLLM／nano-vLLM 的 API Server、Engine Core、Scheduler、KV Cache Manager、Worker／Model Runner 职责，Engine 内部保持少数平面模块，不复制完整 vLLM 插件体系。API Server、Engine Core 与 Model Worker 为三个程序；Engine Core 逐轮排程 prompt prefill 与单 token decode。Mage-VL 的 Engine KV block 是逻辑 reservation；Qwen3 Worker 以同一 block ID 管理 GPU 实体 KV pages，Qwen Runner 支援同 phase 多序列批次 forward。没有 vLLM backend 或 prefix cache。

`engine/scheduler.py` 直接包含线上批次规划；Mage-VL Runner 对相容的同阶段请求执行多序列 forward，不相容时逐请求执行。Mage-VL 视觉与文字骨干的注意力均由模型包实现；文字骨干亦自行组装 Qwen3 Attention、RMSNorm、MLP 与 Decoder layer，保留 Transformers 预训练权重及 Cache 介面。Mage-VL 的 Engine KV 是逻辑 reservation。Qwen3-0.6B 使用本地 PyTorch 计算图、批次 Runner 与 Worker 实体 KV pool；其配置／tokenizer／Safetensors 仍沿用 Hugging Face 生态接口。Qwen PyTorch GQA 会将 paged KV gather 为连续张量。模型专属探索工具依别名放在 `scripts/<model_alias>/`，不从 `src/` 汇入模组，且强制使用本机权重。

开始修改前先阅读根目录 `README.md`，并依任务阅读 `src/inference/README.md`、`src/gateway/README.md` 或 `scripts/README.md`。以实际工作树内容为准；若文件和程式码不一致，先查明差异，再只更新任务相关的内容。

## 架构边界

- 闸道层负责客户端认证与 API Key、租户识别、请求政策、限流、路由、可观测性，以及视讯请求和 SSE 回应的转送。
- 推论层负责模型权重与生命周期、影像／视讯理解、生成及 GPU 资源管理。不要把模型推论或 GPU 管理移入闸道。
- MediaMTX 是即时推流时使用的 sidecar，不是模型服务或 API 闸道。图片与上传视讯档案流程不应要求 MediaMTX。
- 推论 API 预设为 `127.0.0.1:8001`；Gateway 预设监听 `127.0.0.1:8000`。实际端点以目前设定档为准，不要在程式中另行硬编码。
- `/v1/chat/completions` 支援文字／图片输入与 SSE 文字输出。视讯档案理解及即时视讯输入依各自 API 逐段／单次回应；「流式输出」指模型文字 token 的 SSE 输出，不代表视讯输入必须逐 token 或逐影格回传。

## 设定与权重

- 推理与模型前向计算默认使用 CUDA；仅当 `torch.cuda.is_available()` 为 false 时回退到 CPU。不要将实际推理设备硬编码为 CPU；权重分片读取、Tokenizer 和请求输入暂存于 CPU 不代表模型前向运行在 CPU。

- 每个模型的 `configs/model.yml` 登记 alias、API 能力、Runtime／Processor 工厂及相对设定档名。Launcher 自动发现 `models/*/configs/model.yml`，启动时必须透过 `--model` 指定 alias，且相对设定路径以 manifest 的 `configs/` 目录为基准。所有模型共用 `network/service.py` 和 `engine/bootstrap.py`；新模型不需建立专属 `app.py` 或 `engine_core.py`。
- 模型程式、设定和依赖清单集中在 `src/inference/models/<model>/`。依需要建立 `runtime.py`、`processor.py`、`model/`、`components/`、`layers/`、`utils/`、`loader.py`；不需建立空目录。推论参数、Scheduler／KV 与请求上限放对应模型 `configs/inference.yml`；模型服务位址放 `configs/network.yml`。MediaMTX sidecar 设定集中在 `deploy/mediamtx/mediamtx.yml`，不得登记到单一模型 manifest 或模型设定目录。
- 模型推论参数放在模型包的 `configs/inference.yml`；API 绑定位址及模型连接的媒体端点放在 `configs/network.yml`；MediaMTX 服务本身的监听、认证与协定设定放在独立的 `deploy/mediamtx/mediamtx.yml`。新增或调整可调参数时，优先加入合适的 YAML，不要把实验超参数散落在 Python 或 shell 脚本。
- API 启动时会依固定 Hugging Face revision 载入模型；缺少权重时可自动下载至根目录 `checkpoints/`。通用下载工具则使用 `scripts/download.py`，并要求明确指定 `--repo-id`、`--revision` 和 `--local-dir`。
- `checkpoints/` 内的大型权重及快取由 `.gitignore` 整体忽略。分析报告按模型 alias 放在根目录 `analysis/<model_alias>/`，每个模型目录维护一份 `analysis.md` 和一份 `analysis.xlsx`；Markdown 按“问题、解决方案、效果”汇报，逐项编号并指向 Excel 证据；更新时保留历史记录与来源限制。
- 不要在 Git 中加入模型权重、快取、私密金钥或 API token。认证资料由环境变数或安全设定提供。

## 修改与验证方式

- 测试按模型放在 `tests/<model_alias>/`，每个模型维护 `test_components.py`、`test_performance.py`、`test_end2end.py` 和明确传入的 `test.yml`；结果放同目录 `results/`。执行方式见 `tests/README.md`。`test_performance.py` 需要足以同时执行服务与直接基线生成的装置记忆体；只执行元件对照时不载入权重或初始化 GPU。

- 先检查相关程式、设定、文件和现有工作树修改，再进行最小范围的变更。保留使用者已有的未提交修改，不要用整档覆写方式抹除它们。
- 不要自行下载大型模型、启动 GPU 推论服务、修改远端伺服器或推送远端仓库；只有在使用者明确要求时才执行。
- 不要自行提交或推送 Git 变更，除非使用者明确要求。不要未经要求建立新分支。
- 除非使用者明确要求测试／验证，或任务明确要求交付验证结果，否则不要新增或执行测试。若未执行，清楚说明；不得宣称未做的检查已通过。
- 执行任何删除、搬移、清理或覆写前，先核对目标路径和影响范围；涉及模型权重或工作流资料时格外小心。
- 完成后摘要实际修改、相关档案、验证方式与尚未处理的事项。不要把计划描述成已实作。

## 语言与文件

- 与使用者沟通使用简体中文。
- 专案文件、程式注解和面向开发者的说明使用简体中文；程式识别字、命令、API 栏位及上游专有名称维持原文。
- 文件中的启动顺序、连接埠、端点和设定路径必须与实际程式及 YAML 一致。
