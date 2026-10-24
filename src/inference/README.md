# 推论服务

推论服务由 `launcher.py` 启动。启动器自动发现模型包内的 `configs/model.yaml`；执行 `start` 时必须以 `--model` 指定 Runtime 工厂、能力及设定路径。API Server、Engine Core 和 Model Worker 为独立程序；Go Gateway 不属于推论程序。

## 快速检查与启动

从专案根目录执行：

```bash
python src/inference/launcher.py list
python src/inference/launcher.py validate
python src/inference/launcher.py validate --model qwen3-0.6b
python src/inference/launcher.py start --model mage-vl-4b
python src/inference/launcher.py start --model qwen3-0.6b
```

`validate` 读取清单、YAML 和 Python AST，不汇入模型 Runtime、不读取权重、不下载 snapshot，也不初始化 GPU。启动模型前需安装该模型依赖；实际权重载入及装置选择依模型 `configs/inference.yml` 执行。

## 模型目录与设定

```text
src/inference/
├── api/                         # 共用 HTTP API、路由、schema 与 SSE 序列化
├── engine/                      # Engine Core、Scheduler、KV 管理、Executor／Worker
├── media/                       # 共用影片与 Live 工作阶段元件
├── models/
│   ├── mage_vl_4b/
│   │   ├── runtime.py           # Runtime 工厂
│   │   ├── processor.py         # Mage 图片／影片输入处理
│   │   ├── model/               # Mage-VL 拓扑、Transformers Config 与输出结构
│   │   ├── layers/              # 视觉 Encoder、Qwen3 Attention、多头注意力与 SigLIP MLP
│   │   ├── components/          # Runner、Adapter、逐步生成及元件替换
│   │   ├── utils/loader.py      # YAML、MageVLConfig、AutoProcessor 与权重载入
│   │   ├── requirements.txt
│   │   └── configs/             # model.yaml、inference.yml、network.yml
│   └── qwen3_0_6b/
│       ├── runtime.py           # Runtime 工厂
│       ├── processor.py         # Tokenizer／Chat Template 输入处理
│       ├── model/               # Qwen3 模型拓扑
│       ├── layers/              # Attention、Norm、RoPE 等模型层
│       ├── components/          # Runner、Adapter、InputBatch、KV pool、Sampler
│       ├── utils/loader.py       # YAML、Qwen3Config、AutoTokenizer 与权重载入
│       ├── requirements.txt     # 模型依赖
│       └── configs/             # model.yaml、inference.yml、network.yml
├── launcher.py
```

每个 `configs/model.yaml` 描述单一模型的 alias、能力、Runtime 工厂及相对设定档名。`inference.yml` 保存模型与推论参数、API 请求上限、Scheduler／KV、Worker timeout；`network.yml` 保存 API 绑定位址及模型连接的媒体端点。MediaMTX 是独立 sidecar，其服务设定位于 `deploy/mediamtx/mediamtx.yml`，由部署者单独启动，不属于模型 manifest 或推论引擎配置。若推论 API 连接埠改变，需同步更新 MediaMTX 的 `authHTTPAddress`。密钥不得写入 YAML 或 Git。

从专案根目录独立启动 MediaMTX：

```bash
mediamtx deploy/mediamtx/mediamtx.yml
```

模型的 `network.yml` 只负责推论服务如何连接 RTSP／RTMP 端点及向用户端公布的主机名称；MediaMTX 的监听协定、埠号和授权回呼由上述部署配置管理。

`launcher.py list` 和不带 `--model` 的 `validate` 会扫描所有模型 manifest；`start` 必须指定一个模型 alias。

模型目录可按模型需要建立 `model/`、`layers/`、`components/`、`utils/loader.py`、`processor.py` 及依赖清单；不需要的目录不必建立。`utils/loader.py` 集中处理模型设定、Transformers Config／Tokenizer／Processor、snapshot 和权重载入；服务推理参数仍留在 YAML。模型拓扑、权重格式、模型 Cache 和专属前处理不能放进共用 `engine/`。

## 引擎职责与模型差异

| 元件 | 位置 | 职责 |
| --- | --- | --- |
| API Server | `api/` | OpenAI 相容 API、Tokenizer 输入处理、增量 Detokenize、SSE／JSON 输出 |
| Engine Core | `engine/engine.py` | 请求生命周期、排程回圈及结果事件分派 |
| Scheduler／KV Manager | `engine/scheduler.py`、`engine/kv_cache.py` | 建立 prefill／decode 批次及管理 Engine cache 容量租约 |
| Executor／Worker | `engine/executor.py`、`engine/worker.py` | 启动 Worker 子程序、IPC、模型执行和资源清理 |
| 模型 Runtime | `models/<alias>/runtime.py` 与模型档案 | 组合该模型 Runner／Adapter，持有模型载入和生成行为 |

Mage-VL 使用 Transformers 的预训练模型载入、Processor 与 Cache 介面；模型包自行组装视觉 Encoder、SigLIP MLP、多头注意力，以及逐层拆分的 Qwen3 文字骨干。Qwen3 文字骨干使用模型包内的 GQA、RoPE、RMSNorm、MLP 与分块 eager 注意力，不再透过 `AutoModel` 建立。Qwen3-0.6B 使用模型包内 PyTorch 计算图、批次 prefill/decode 和实体 KV page pool；Attention 以分组矩阵乘法计算 GQA，先 gather pages，尚非 PagedAttention kernel。Mage-VL 的 Engine KV 仍是逻辑 reservation；两个模型均沿用共用 Engine 与 Worker 程序边界。

文字 token SSE 经 API 的增量 detokenizer 输出；Mage-VL 的文字增量路径保持不变。多模态路由依各模型 manifest 能力启用。Inference Engine 借鉴 vLLM／nano-vLLM 的元件职责，不代表 vLLM 相容，也不代表支援尚未实作的后端或最佳化。

## 测试

- `tests/qwen3_0_6b/`、`tests/mage_vl_4b/`：每个模型三项测试，分别比较计算图、比较真实 Inference API 与 Transformers 原生生成性能，以及从 Gateway 验证端到端请求。`test_components.py` 使用 `meta` device，不载入模型权重。
- 各测试的模型 snapshot、服务 URL、生成参数、API Key 环境变数名称及素材路径由同目录的 `test.yaml` 提供，执行时以 `--config` 明确指定；历史实测报告放在模型目录的 `results/`。

## 模型文件内的基线前向对照

值得单独验证的模型层、注意力、MLP、RoPE、视觉组件、KV／采样组件及完整模型文件，在文件内提供 `if __name__ == "__main__":` 小入口。测试逻辑直接写在入口中，不调用另行封装的测试方法；输出 `==========engine===========` 和 `==========baseline==========` 分隔实现与基线，并以断言确认结果。配置／输出结构等没有独立前向行为的文件不添加空测试。

在 WSL 激活 `magevl-transformers` 环境后，从仓库根目录运行无需模型权重的 CPU 对照：

```bash
export PYTHONPATH=src/inference
for module in \
  models.mage_vl_4b.layers.multi_head_attention \
  models.mage_vl_4b.layers.qwen3_text_attention \
  models.mage_vl_4b.layers.qwen3_text_layers \
  models.mage_vl_4b.layers.siglip_mlp \
  models.mage_vl_4b.layers.streammind_gate \
  models.mage_vl_4b.layers.vision_encoder \
  models.mage_vl_4b.model.text_backbone \
  models.mage_vl_4b.model.multimodal_model \
  models.mage_vl_4b.model.conditional_generation \
  models.mage_vl_4b.model.vision_tower \
  models.mage_vl_4b.components.stepwise_generation \
  models.qwen3_0_6b.layers.attention \
  models.qwen3_0_6b.layers.layers \
  models.qwen3_0_6b.layers.rotary_embedding \
  models.qwen3_0_6b.model.qwen3 \
  models.qwen3_0_6b.components.input_batch \
  models.qwen3_0_6b.components.paged_kv_cache \
  models.qwen3_0_6b.components.sampler
do
  python -m "$module"
done
```

两个 Model Runner 入口在模块导入时读取启动器选择的模型 YAML，因此需显式提供对应环境变量：

```bash
INFERENCE_MODEL_ALIAS=mage-vl-4b \
INFERENCE_CONFIG=src/inference/models/mage_vl_4b/configs/inference.yml \
INFERENCE_NETWORK_CONFIG=src/inference/models/mage_vl_4b/configs/network.yml \
python -m models.mage_vl_4b.components.model_runner

INFERENCE_MODEL_ALIAS=qwen3-0.6b \
INFERENCE_CONFIG=src/inference/models/qwen3_0_6b/configs/inference.yml \
INFERENCE_NETWORK_CONFIG=src/inference/models/qwen3_0_6b/configs/network.yml \
python -m models.qwen3_0_6b.components.model_runner
```

这些小入口使用随机小型配置和 CPU，不读取本地权重、不启动 API 或 GPU 服务。Mage-VL Runner 与 Stepwise 入口同时验证逐 token 文字增量；`VideoMamba` 的小对照会在可选依赖 `mamba_ssm` 未安装时明确输出跳过，其余投影层和分类头仍会运行。真实模型、性能和 Gateway 链路仍由上方逐模型三项测试负责。

端到端性能测试需待 Gateway 与指定模型都可运行时执行，并记录吞吐、错误率、延迟分位数、SSE 首 token 延迟及影片上传完成时间；目前不设定硬 SLA，也不做模型精度资料集评测。测试结果需区分 mock、CPU synthetic 和固定 revision 实机证据。

新模型接入步骤、manifest 范例和验收检查见 [`docs/Specifications/inference/model-integration.md`](../../docs/Specifications/inference/model-integration.md)；推论引擎规格见 [`docs/Specifications/inference/`](../../docs/Specifications/inference/)。
