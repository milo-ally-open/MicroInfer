# 推理服务

推理服务由 src/inference/launcher.py 启动。Launcher 自动发现各模型包中的 configs/model.yaml；启动时必须用 --model 选择一个模型。API Server、Engine Core 和 Model Worker 是独立进程；Go Gateway 与 Inference 分别配置和运行。

## 快速检查与启动

从仓库根目录运行：

- python src/inference/launcher.py list：列出已发现模型。
- python src/inference/launcher.py validate --model {alias}：检查 manifest、YAML、Runtime 工厂路径和 API 能力，不导入模型工厂、不载入权重、不下载 snapshot、不初始化 GPU。
- python src/inference/launcher.py start --model {alias}：启动指定模型。

目前登记的 alias 是 mage-vl-4b 和 qwen3-0.6b。模型 API 位址分别由各自的 configs/network.yml 决定。

## 新模型接入概要

详细步骤和文件内容要求见 [新模型接入指南](../../docs/Specifications/inference/model-integration.md)。模型包目录名是 Python 包名，alias 是客户端选模名称，两者可以不同；例如 qwen3_0_6b 对应 qwen3-0.6b。上游权重 ID 单独写在 inference.yml 的 model.id。

| 文件或目录 | 必需性 | 内容职责 |
| --- | --- | --- |
| models/{package_name}/configs/model.yaml | 必需 | alias、display name、standard/custom profile、backend、Runtime 工厂、配置路径、api.model_id 和已实现的能力。 |
| models/{package_name}/configs/inference.yml | 必需 | 固定模型 revision、dtype/device/量化、权重路径、服务限制、generation、Scheduler/KV、Worker 超时和优化参数。 |
| models/{package_name}/configs/network.yml | 必需 | Inference API 监听位址；使用 Live 时另列推理服务所需媒体端点。 |
| models/{package_name}/runtime.py | 必需 | create_runtime() 工厂及模型 Runner/Adapter 的组合。 |
| models/{package_name}/processor.py | 按模型需要 | Tokenizer、Chat Template、文本规范化、图片/视频预处理和模型专属采样参数适配。 |
| models/{package_name}/utils/loader.py | 按模型需要 | 集中选择 Transformers Config、Tokenizer/Processor、snapshot 和权重载入方式。 |
| model/、layers/、components/ | 按模型需要 | 分别承载拓扑、数值计算层、Runner/Adapter/Batch/Cache/Sampler 等执行元件。 |
| models/{package_name}/requirements.txt | 按模型需要 | 模型专属依赖及兼容版本；沿用项目指定环境，不为每个模型重复创建 Conda 环境。 |
| tests/{package_name}/ | 必需 | test.yaml 和 test_components.py、test_performance.py、test_end2end.py 三项测试。 |

runtime.py 是每个模型包都需要的工厂入口。符合当前 standard 条件的模型可以使用共用文字 Runner，减少模型专属 Runner/Adapter/Processor；standard 当前只适用于单装置、纯文字 decoder-only、token_ids 输出和可延续 past_key_values 的模型。多模态、特殊 Cache、自有计算图或权重映射使用 custom。profile 只表达接入路径，不会自动提供模型实现。

api.model_id 是 Inference API 接受并返回的模型 ID；model.id 是上游模型 repo ID；Gateway 公开 alias 由模型 alias 决定。通过 Gateway 开放新模型时，修改 src/gateway/configs/gateway.yml 的模型映射，使 upstream_model 与 api.model_id 一致、base_url 指向 network.yml 的 Inference API，并只宣告已实现的 Gateway 能力。

## 配置边界

所有模型服务和推理参数放在对应模型的 configs/inference.yml；API 监听与模型侧媒体连接端点放在 configs/network.yml。API Key、HMAC 密钥和密码只通过环境变量或 Git 外部的密钥文件提供，不写入 YAML 或 Git。

MediaMTX 是独立 sidecar，不属于模型包、manifest 或 Launcher。配置统一位于 deploy/mediamtx/mediamtx.yml，由部署者单独启动。模型 network.yml 只描述推理服务连接媒体端点所需的信息。独立部署方式见 [MediaMTX README](../../deploy/mediamtx/README.md)；图片和上传视频不依赖 MediaMTX。

## 共用架构边界

| 组件 | 位置 | 职责 |
| --- | --- | --- |
| API Server | api/ | OpenAI 风格请求、输入处理、增量 Detokenize、JSON/SSE 输出。 |
| Engine Core | engine/engine.py | 请求生命周期、调度循环和输出事件分发。 |
| Scheduler 与 KV 管理 | engine/scheduler.py、engine/kv_cache.py | 建立 prefill/decode 批次并管理 Engine 侧逻辑资源租约。 |
| Executor 与 Worker | engine/executor.py、engine/worker.py | 子进程、IPC、模型操作调用及资源清理。 |
| 模型 Runtime | models/{package_name}/ | 模型载入、拓扑、模型专属输入与 Cache 行为。 |

模型 Runner 批次接口不代表一定执行合并 forward。Qwen3 使用实体 KV page pool，但 Attention 前仍 gather KV pages；尚无自定义 PagedAttention kernel。Mage-VL 使用 Transformers Cache，Engine KV 管理仍是逻辑 reservation。项目借鉴 vLLM/nano-vLLM 的组件职责，不表示兼容 vLLM，也不代表支持尚未实现的后端或优化。

## 三项模型测试

每个模型目录只维护三项测试，设定和本机路径由 tests/{package_name}/test.yaml 提供，执行时明确传入配置：

1. test_components.py：检查模型结构和组件基线；meta 结构检查不载入权重，不做大型 GPU 测试，但需要 YAML 指向本机 snapshot 的配置资料。
2. test_performance.py：比较已启动的 Inference API 与同 revision Transformers 基线的生成表现；需要本机权重、GPU 和推理 API。
3. test_end2end.py：通过 Go Gateway 验证公开 alias、认证、SSE 和模型已声明的多模态路由；需要 Gateway、Inference API、API Key 及适用素材。

测试脚本将 JSON 摘要输出到标准输出，不会自动写入文件。需要归档时，可将结果保存到对应模型目录的 results/，并记录模型 revision、硬件和测试环境。三项测试不做资料集精度评估。性能测试是实测比较，不自动构成 SLA；未运行的测试应保持未验收。

## 模型文件内的小型基线入口

独立数值计算的模型层、视觉组件、KV/采样执行组件和模型前向文件，在自身 if __name__ == "__main__": 入口中直接建立小型输入、运行引擎实现与基线、打印 ==========engine=========== 和 ==========baseline========== 并断言输出。入口不通过另行封装的测试方法执行。纯配置、输出结构和无独立前向行为的文件不需要空测试。

在 WSL 的 magevl-transformers 环境中，从仓库根目录设置 PYTHONPATH=src/inference 后，用 python -m {module_path} 执行对应文件。Mage-VL 和 Qwen3 的模型层、视觉组件、模型拓扑与 KV/采样入口均采用此模式；Model Runner 入口还需要显式提供其模型 alias、inference.yml 和 network.yml 环境变量。模型入口使用小型合成配置与 CPU，不读取模型权重，也不启动服务。VideoMamba 的入口在可选依赖 mamba_ssm 缺失时会明确跳过对应分支。

真实模型性能和 Gateway 链路使用上面的逐模型三项测试。测试目录、配置字段及能力覆盖要求见 [新模型接入指南](../../docs/Specifications/inference/model-integration.md)，设计约束见 [Inference Specifications](../../docs/Specifications/inference/)。
