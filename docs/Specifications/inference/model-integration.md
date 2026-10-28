# 新模型接入参考规范

> 从 Hugging Face 下载模型并按步骤完成接入，请先阅读仓库根目录的[模型接入实操教程](../../../TUTORIAL.md)。本文保留模型 Runtime、API 能力和配置边界的参考规范；实现状态与可运行行为以当前代码和模型 YAML 为准，历史验收记录见对应 `tasks.md`。

本文档说明接入一个新模型时要新增或修改哪些文件，以及每个文件应承载的内容。这里不提供模型实现代码；具体计算图、权重适配和预处理由模型包自行实现。

目前项目只登记 Mage-VL-4B 与 Qwen3-0.6B。后续模型尚未选定；本指南描述通用流程，不表示其他模型已经接入。

## 接入原则

- 模型专属代码、配置和依赖集中在一个模型包中；API、Engine Core、Scheduler、Executor 和 Worker 继续共用。
- Launcher 通过模型包内的 configs/model.yml 自动发现模型，不维护全局模型清单。
- 只要现有 API 能力和 Runtime 契约能够表达新模型，就不修改共用 API、Engine 或 Launcher。
- 不把模型计算图、Tokenizer、权重映射和模型 Cache 放入共用 engine/。
- Gateway 是独立服务。只有需要通过 Gateway 对客户端开放时，才修改 Gateway 配置。
- MediaMTX 是独立 sidecar。只有模型使用实时视频时才配置和部署，不属于模型包。

### 模型目录名、别名与上游 ID

这三个名称用途不同，不能互相替代：

| 名称 | 用途 | 示例 |
| --- | --- | --- |
| 模型包目录名 | Python 可导入的目录名，通常使用小写字母和下划线 | qwen3_0_6b |
| 模型 alias | 客户端通过 Gateway 选择模型的公开名称，通常使用连字符 | qwen3-0.6b |
| 上游模型 ID | 模型权重来源或模型服务标识符，例如 Hugging Face repo ID | Qwen/Qwen3-0.6B |

模型 alias 写在 manifest 的 alias 中；上游权重来源写在 inference.yml 的 model.id 中。manifest 的 api.model_id 是 Inference API 接受并返回的模型 ID。若经 Gateway 调用，Gateway 的 upstream_model 必须与该模型的 api.model_id 对齐；Gateway 对客户端仍公开模型 alias。

### 选择接入 profile

| profile | 适用条件 | 当前限制 |
| --- | --- | --- |
| standard | 纯文字、decoder-only 因果语言模型；能复用 Transformers Cache；使用 token ID 输出；单一计算设备 | 当前不接受图片、视频、Live、多设备 device_map 或模型专属 Cache 行为；Launcher 会按这些约束预检 |
| custom | 多模态模型、自有模型计算图、特殊 Cache、模型专属权重映射或文字增量输出 | 模型包负责适配 Runtime、Adapter、Processor 和模型生命周期；profile 名称本身不会自动提供后端实现 |

不确定是否满足 standard 时先按 custom 设计，并根据实际 Runner 行为声明能力。不要为了通过预检而虚报能力或把模型包装成不兼容的 standard 模型。

## 接入工作流

### 1. 确认模型事实与 API 契约

开始写模型代码前，先固定并记录：

- 上游模型名称、repo ID、不可变 revision，以及该 revision 的模型配置和权重格式。
- Tokenizer、Chat Template、图像或视频 Processor 的来源与加载方式。
- 模型接受的输入模态、返回方式、Cache 行为、生成停止条件及可否逐 token 输出。
- 运行所需依赖、设备类型、dtype、量化方式、显存需求和最大上下文。
- 计划公开的 API 能力：文字/图片聊天、文字 SSE、视频上传或 Live。

这些是接入决策和测试依据。模型卡、上游代码或模型 snapshot 尚未证实的行为，应标为待验证，不写成已支持功能。

### 2. 创建模型包与必需配置

在 src/inference/models/ 下创建新的模型包目录，并在该目录下创建 configs/。每个模型目前都需要 manifest、推理配置和网络配置：

| 文件 | 必需性 | 文件中应写入什么 |
| --- | --- | --- |
| src/inference/models/{package_name}/configs/model.yml | 必需 | 模型 alias、展示名称、integration_profile、backend 标识、Runtime 工厂路径、推理与网络配置文件名、Inference API 的 api.model_id，以及真实已实现的能力宣告。配置路径相对于此 manifest 的目录解析。 |
| src/inference/models/{package_name}/configs/inference.yml | 必需 | 上游 model.id 和固定 revision；dtype、device 或 device_map、量化和权重加载选项；runtime 的本地模型路径、离线行为、平台和超时；API 请求大小及上下文上限；generation 默认值；engine 的并发、等待队列、token 预算、prefill 分块和 KV 容量；确实使用的优化项。 |
| src/inference/models/{package_name}/configs/network.yml | 必需 | Inference API 监听 host 和 port。使用 Live 时，还记录推理服务连接 MediaMTX 的 RTSP/RTMP 地址及返回给发布端的公开主机名。顶层必须有 api 网络配置。 |

推理参数、Scheduler/KV 限制、视频大小和运行超时写在 inference.yml；API 监听及模型连接的媒体网络端点写在 network.yml。不要把可调运行参数写成 Python 常量，也不要在这些文件中保存 API Key、密码、签名密钥或其他秘密。

revision 应固定到可复现的提交版本。模型别名、显示名称和 Hugging Face repo ID 分别维护；repo ID 不是模型 alias。

### 3. 提供模型 Runtime 工厂

src/inference/models/{package_name}/runtime.py 是必需入口。它提供 create_runtime()，组合该模型的 Runner 与 Adapter，并返回共用 Worker 能使用的 ModelRuntime 门面。即使采用 standard，也必须满足 manifest 所要求的 Runtime 工厂入口。

Runtime/Adapter 应覆盖当前 Worker 契约中的这些职责：

- 模型加载、关闭、健康状态和 GPU 状态。
- 生成参数解析、请求输入准备、Cache 绑定、finish 和 cancel。
- execute_batch 接收 Scheduler 的 work items，并按 request ID 返回各请求的结果。
- 请求结果使用 token_ids 或 text_delta 中的一种；不要同时混用两种传输形式。

宣告 video_upload 时，还要提供视频生成准备与完成操作，并在推理配置中给出有效的视频请求上限。宣告 live_video 时，还要提供 Live 帧/视频生成及完成操作，并在 manifest 登记 Live session manager。取消、超时、Worker 异常和正常完成都必须结束模型请求状态并释放该模型实际持有的 Cache。

ModelRuntime 与共用 Engine 的契约是边界。不要为单一模型另建 app.py、engine_core.py，也不要复制 Scheduler 或 Worker。

### 4. 按模型职责新增可选文件

模型目录只创建确有职责的目录和文件，不需要为了目录整齐而创建空文件。

| 文件或目录 | 何时需要 | 应写入的内容 |
| --- | --- | --- |
| processor.py | 模型需要专属 Tokenizer、Chat Template、图片/视频转换、消息规范化或 sampling 参数适配时 | API 侧输入准备或模型专属预处理的规则。文字 token 输出模型还要负责解码与增量 detokenization 所需的处理器。 |
| utils/loader.py | 模型需要专属加载流程时 | 集中加载模型专属 Transformers Config、AutoTokenizer/AutoProcessor、固定 snapshot、Safetensors 权重及必要的 key/shape 映射。使用准确的 Transformers 配置类；不要再拆出冗长的 config.py、hf_config.py、hf_tokenizer.py 或 model_loader.py。 |
| model/ | 有自定义模型拓扑、模型输出结构或专属模型配置时 | 模型结构与拓扑、模型专属 Config/输出类型。不要把拓扑塞进 components/。 |
| layers/ | 有需要直接查看或替换的模型层或算子时 | Attention、Norm、Linear、RoPE、MLP、视觉 Encoder 等计算层及其数值行为。 |
| components/ | 有模型执行时协作组件时 | Model Runner、Adapter、InputBatch、Sampler、Cache manager、生成状态等 Runner 周边职责。 |
| requirements.txt | 当前环境缺少该模型所需依赖时 | 该模型专属的 Python 包及兼容版本。依赖安装到项目指定的现有环境；不为每个模型重复创建 Conda 环境。 |
| live.py | 模型宣告 Live 能力时 | Live session manager 工厂及模型专属的实时帧处理连接。MediaMTX 服务自身不放在这里。 |

utils/loader.py 是模型加载与生态适配的集中入口，不是另一套服务配置。Transformers Config、Tokenizer、Processor 和权重格式的选择放在这里；服务端可调参数仍由 YAML 提供。模型不需要自定义的职责就沿用现有生态或共用实现，不为文件数量而拆分。

### 5. 声明 API 输入、输出和能力

manifest 的 api 部分决定共用 API 会为该模型启用哪些行为。当前静态预检支持的 chat_inputs 是 text 和 image；streaming、output_mode、video_upload、live_video 均需按布尔值和真实实现填写。

output_mode 有两种当前契约：

- token_ids：Worker 返回生成 token ID；API 进程的 InputProcessor 使用同一 Tokenizer 解码普通响应和增量 SSE。需要在 manifest 提供 input_processor 工厂。该类应实现启动、请求准备和关闭；并提供 token 解码与 stream output processor。
- text_delta：Worker 逐次返回文本增量；API 将增量转换为 OpenAI 风格 SSE。该模式不能宣称为逐 token ID 输出。

如果 chat_inputs 包含 image，Processor 和 Runner 必须真正处理图片。视频上传和 Live 是独立能力，不是 chat_inputs 的值。Live 能力必须登记 Live manager，并提供必要的 Live 请求上限。只将代码实际完成且测试覆盖的功能设为 true。

### 6. 若要经 Gateway 对外开放，更新 Gateway 配置

只有需要客户端通过 Go Gateway 访问新模型时，才在 src/gateway/configs/gateway.yml 中增加模型映射。该项配置应包含：

- 对外模型 alias。
- 与模型 manifest 的 api.model_id 一致的 upstream_model。
- 与该模型 network.yml 监听地址相符的上游 base_url。
- text_chat、image_chat、streaming、video_upload、live_video 等 Gateway 能力；它们必须是推理端真实能力的子集。
- 只有要改变默认视频/Live 路由时，才更新对应 defaults。

Gateway 配置是模型路由与客户端政策，不是推理模型配置。API Key 和 HMAC 密钥通过环境变量指向 Git 外的安全文件或环境，不写进 YAML。

如果新模型需要当前 API 尚未定义的输入模态、路由或返回协议，单靠新增模型包和 Gateway 映射不够。此时要另行修改并验证共用 Inference/Gateway 契约，同时更新对应 Specifications；不要在本指南中把扩展 API 的工作伪装成普通模型登记。

### 7. 只有使用 Live 时才配置 MediaMTX

MediaMTX 配置集中在 deploy/mediamtx/mediamtx.yml，由部署者独立启动，不写入模型 manifest，也不由模型 Runtime 或 Inference Launcher 启停。使用 Live 的模型只需在自身 network.yml 提供推理侧媒体端点及公开主机名；部署文档负责 MediaMTX 的监听端口和授权回调。图片和上传视频不依赖 MediaMTX。

### 8. 为模型创建三项测试

在 tests/{package_name}/ 下维护该模型的三项测试及一份 test.yml：

| 文件 | 应验证的内容 | 依赖 |
| --- | --- | --- |
| test.yml | 模型 alias、本地 snapshot 路径、推理配置、Inference/Gateway URL、API Key 环境变量名称、prompt、素材路径、生成上限、warmup/iterations、timeout 及组件测试所需模块/dtype。只放环境变量名称，不放密钥值。 | 配置文件本身不启动服务。 |
| test_components.py | 显示模型包与 Transformers 原生基线的结构；对有独立计算的关键组件执行小型前向对照并断言数值或 token 一致。应避免加载权重和创建大型 GPU 测试负载。 | 需要配置文件指向本地 snapshot 的结构信息；不加载模型权重。 |
| test_performance.py | 在相同本地 revision、输入和生成上限下比较已启动的 Inference API 与 Transformers 原生基线；记录样本数、生成 token 数、平均延迟和 token/s。多模态模型使用同一输入素材。 | 本地权重、模型所需环境、可用 GPU 和已启动的 Inference API。 |
| test_end2end.py | 经 Gateway 使用公开 alias 验证成功响应、返回 alias、SSE 内容和每项已宣告能力的路由。上传/Live 模型要覆盖对应路径；未覆盖的能力不要宣称已端到端验证。 | Gateway、Inference API、环境变量提供的 API Key，以及该模型需要的本地素材或 MediaMTX。 |

测试脚本都从明确传入的 test.yml 读取路径和参数，不在脚本中推导项目根目录或默认配置。测试脚本将 JSON 摘要输出到标准输出，不会自动写入文件。需要归文件时，可由执行者将结果保存到对应 tests/{package_name}/results/，并记录模型 revision、硬件、测试次数和环境；不得将密钥、权重或媒体素材纳入报告。性能样本用于实际对比，不是数据集精度评测，也不自动构成 SLA。

## 验证顺序

1. 确认模型包目录与 manifest alias 唯一，并确认 manifest 引用的 inference.yml、network.yml 和 Runtime 工厂路径存在。
2. 运行 launcher.py list，确认新模型被自动发现；运行 launcher.py validate --model {alias}，先通过不读取权重、不初始化 GPU 的静态预检。
3. 检查推理与网络 YAML、revision、依赖版本、Tokenizer/Processor 和请求限制。
4. 运行该模型的 test_components.py，比较模型结构及关键小型前向行为。
5. 单独启动 Inference API，运行 test_performance.py；逐步确认输出、finish reason、文字 SSE、Cache 延续和请求清理。
6. 若公开给客户端，再登记 Gateway 模型映射，启动 Gateway，运行 test_end2end.py 验证 alias、认证、SSE 和已宣告路由。
7. 若声明 Live，再单独部署 MediaMTX 并测试会话创建、媒体发布、事件读取和停止。不要把 Live 单次测试结果描述为容量或稳定性结论。
8. 保存实际命令、环境、模型 revision、硬件、样本数、错误、延迟和限制；未执行的测试保持未验收状态。

Launcher validate 是静态预检，不代表权重能够加载或生成结果正确。结构/小型前向测试、真实 Inference 性能测试和 Gateway 端到端测试证明不同层次，不可互相替代。

## 常见边界错误

- 把公开 alias、package_name、api.model_id 和 model.id 当成同一名称。
- 新模型声明支持图片、视频、Live 或 streaming，但 Processor、Runner、API 工厂或测试并未实现。
- 将 Scheduler、Engine KV lease、模型实际 Cache 和物理 KV page 当成同一资源。
- 只因为 Runner 接受批次接口，就宣称模型 forward 已合批；必须确认模型实际执行行为。
- 把 KV reservation 或 gather 后的连续张量计算描述成 PagedAttention。
- 把 MediaMTX 配置放进模型目录，或让 Inference Launcher 管理 MediaMTX 生命周期。
- 为了新模型而新增全局 Registry、模型专属 app.py、engine_core.py，或在共用 API/Engine 按 alias 写分支。

当前目录和实现状态见 [Inference README](../../../src/inference/README.md)；行为要求见 [Inference Spec](spec.md)，实施状态见 [tasks.md](tasks.md)。
