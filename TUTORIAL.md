# 新 VLM 模型接入流程

本文说明如何从 Hugging Face 模型页面开始，完成一个视觉语言模型（VLM）的评估、组件拆解、性能分析，再接入本仓库 Inference Engine。示例使用仓库已有的 **Qwen3-VL-2B-Instruct**；接入其他模型时替换 repo、revision、模型类、输入格式和模型专属组件。

命令假设在仓库根目录的 WSL/Linux Bash 和 `inference` Conda 环境中运行。模型权重只放在被 Git 忽略的 `checkpoints/`，不提交到仓库。CUDA 可用时默认使用 CUDA；CUDA 不可用时才回退 CPU，显存不足应调整负载，不能把 OOM 当作回退条件。

## 接入顺序

```text
浏览并筛选 Hugging Face 模型
  → 固定 revision、检查许可证和文件
  → 下载到 checkpoints/，核对文件
  → Transformers 原生基线：Test → Bench → Profile
  → 按计算图拆解模块，逐层与官方实现对齐
  → 组装 modeling.py，接入 loader / runtime / components
  → 安装模型依赖，静态校验
  → 本地 Engine 和 Gateway 端到端验收
```

Test、Bench、Profile 是正式接入 Engine 前的基线工作。先测清官方模型的正确性、资源和时间开销，再决定哪些代码需要接入或优化；不能用 Engine 结果替代 Transformers 基线。

## 1. 从 Hugging Face 页面筛选模型

打开目标模型的 Hugging Face 仓库页面，依次检查 **Model card** 和 **Files and versions**。不要只根据模型名称或参数量判断能否接入。

| 页面位置 | 要核对的内容 | 接入时记录 |
| --- | --- | --- |
| Model card | 任务类型、推荐用法、支持的图片/视频输入、示例、限制和评测说明 | 实际要支持的模态与测试提示词 |
| License / 使用条款 | 权重许可证、额外限制、是否需申请访问 | 许可证名称和访问条件 |
| Files and versions | `config.json`、权重和索引、Tokenizer、Processor、模板文件是否齐全 | repo ID、目标 revision、必需文件清单 |
| `config.json` | `model_type`、`architectures`、文本与视觉配置、层数、维度、注意力头数 | Transformers 对应配置类和模型类 |
| 权重文件 | `.safetensors` 或分片索引是否存在，索引引用的分片是否齐全 | 文件格式、分片数量和总大小 |
| Processor / Tokenizer 配置 | `processor_config.json`、`preprocessor_config.json`、`tokenizer_config.json`、词表、特殊 token、chat template | 官方 Processor/Tokenizer 类和输入字段 |
| Python 文件 | 仓库是否带自定义 `modeling_*.py`、`processing_*.py` 等代码 | 是否必须加载远程代码；代码审查结论 |

在页面切换到准备接入的提交版本，记录 **完整 40 位 commit SHA**。不要将 `main` 或会变化的 tag 当成可复现版本。检查模型文件是否与该 commit 对应，确认模型大小、许可证、GPU 显存及本地磁盘足够。

安全要求：默认不设置 `trust_remote_code=True`。如果模型必须依赖仓库里的 Python 代码，先逐文件检查代码来源、执行行为和依赖，再决定是否使用；不要因为 `AutoModel.from_pretrained()` 报错就直接打开远程代码执行。

Qwen3-VL 示例信息：

| 字段 | 值 |
| --- | --- |
| Hugging Face repo | `Qwen/Qwen3-VL-2B-Instruct` |
| 固定 revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| 本地目录 | `checkpoints/Qwen3-VL-2B-Instruct-8964489` |
| Engine / Gateway alias | `qwen3-vl-2b` |
| 当前实现模态 | 文字、图片；流式文字输出 |

## 2. 配置环境并下载固定版本

先进入项目环境，确认 Python、PyTorch、Transformers 与 GPU 状态。安装 PyTorch 时按机器的 NVIDIA 驱动和 CUDA 兼容矩阵选对应版本；模型依赖后续通过模型 extra 安装。

```bash
cd /mnt/e/workspace/MicroInfer
conda activate inference
python -c 'import torch, transformers; print("torch", torch.__version__); print("transformers", transformers.__version__); print("cuda", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
```

### 国内网络

Hugging Face Hub 客户端会读取 `HF_ENDPOINT` 作为 Hub 地址。若运行环境访问官方站点不稳定，可在 **启动下载命令前** 设置镜像。下面的 `hf-mirror.com` 是第三方镜像，不是 Hugging Face 官方服务；按组织网络和安全要求选择可用镜像。镜像访问失败时，去掉该变量或改用组织批准的 Hub 代理。

```bash
# 正确写法：等号两侧不能有空格；设置在 import huggingface_hub / transformers 之前。
export HF_ENDPOINT=https://hf-mirror.com
echo "$HF_ENDPOINT"

# 使用官方 Hub 时：
# unset HF_ENDPOINT
```

Hugging Face 访问令牌只用于 gated/private 仓库。不要把令牌写进脚本、YAML、命令历史或 Git；在终端安全地导出 `HF_TOKEN`，或使用本机 `hf auth login`。

然后使用仓库下载脚本。把 `MODEL_ID`、`REVISION` 和 `LOCAL_DIR` 替换成刚才在网页核对的信息：

```bash
export MODEL_ID=Qwen/Qwen3-VL-2B-Instruct
export REVISION=89644892e4d85e24eaac8bacfd4f463576704203
export LOCAL_DIR=checkpoints/Qwen3-VL-2B-Instruct-8964489

python scripts/download.py \
  --repo-id "$MODEL_ID" \
  --revision "$REVISION" \
  --local-dir "$LOCAL_DIR"
```

`scripts/download.py` 使用 `huggingface_hub.snapshot_download()` 下载固定 revision 的完整快照。检查下载结果：

```bash
test -s "$LOCAL_DIR/config.json"
find "$LOCAL_DIR" -maxdepth 1 -type f -printf '%f\n' | sort
find "$LOCAL_DIR" -maxdepth 1 -name '*.safetensors' -o -name 'model.safetensors.index.json'
```

对分片权重，还要确认 `model.safetensors.index.json` 中 `weight_map` 引用的每个文件都存在。Processor 文件名因模型而异；依照模型页和官方 Transformers 文档核对，不要假设所有 VLM 都使用完全相同的一组文件。下载完成后，推理时设置本地路径和 `local_files_only: true`，让服务启动不再访问 Hub。

可以用下面的小检查确认权重分片没有缺失：

```bash
python - "$LOCAL_DIR" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
index = model_dir / "model.safetensors.index.json"
if index.is_file():
    weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    missing = sorted(name for name in set(weight_map.values()) if not (model_dir / name).is_file())
    if missing:
        raise SystemExit(f"Missing weight shards: {missing}")
elif not list(model_dir.glob("*.safetensors")):
    raise SystemExit("No safetensors weights or safetensors index found")
print("weight files: complete")
PY
```

## 3. 先建立 Transformers 原生基线

在写 Engine 接入代码之前，先用 Hugging Face 官方模型类和 Processor 从本地目录加载模型。确定性测试使用固定输入、固定生成长度和关闭采样的配置；同一组输入之后也用于组件对齐、Bench 和 Profile。

### Test：验证能力与正确性

模型负责人在 `scripts/<model_alias>/` 维护上游基线的 Test、Bench 和 Profile 入口及其配置副本；`scripts/` 用于模型分析，`tests/` 保留 Gateway/API 请求验收，不混用。至少覆盖模型页宣称、且计划对外提供的每种输入模态。测试脚本应使用固定 revision、本地权重和固定素材；模型不同，输入格式与测试集不同，因此由接入工程师按该模型官方调用方式实现，不复制另一个 VLM 的假输入或 Cache 逻辑。

```python
# 上游基线的核心模式。用目标模型官方文档中的模型类、Processor 和消息格式替换占位内容。
processor = OfficialProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model = OfficialModel.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    dtype=DTYPE,
    device_map=0 if torch.cuda.is_available() else "cpu",
    attn_implementation="sdpa",
).eval()

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
)
input_device = model.get_input_embeddings().weight.device
model_inputs = {}
for name, value in inputs.items():
    if isinstance(value, torch.Tensor):
        is_visual_float = name in {"pixel_values", "pixel_values_videos"} and value.is_floating_point()
        model_inputs[name] = value.to(
            device=input_device,
            dtype=model.dtype if is_visual_float else value.dtype,
        )
    else:
        model_inputs[name] = value
model_inputs.pop("token_type_ids", None)
with torch.inference_mode():
    output_ids = model.generate(**model_inputs, do_sample=False, max_new_tokens=32)
prompt_length = model_inputs["input_ids"].shape[-1]
print(processor.batch_decode(output_ids[:, prompt_length:], skip_special_tokens=True)[0])
```

这是结构示意：不同模型的类名、视频帧组织方式、模板参数、输入 tensor 和输出解码方式可能不同，必须以该模型的官方示例为准。Test 至少记录：

- 文本单轮和多轮请求能否成功生成，EOS/停止 token 是否正确。
- 图片数量、分辨率和图文混排输入能否按模型 Processor 正确编码；检查 `input_ids`、像素张量、grid/size 等模型所需字段。
- 若需要视频能力，验证帧抽取、时间戳、视频 Processor 和长度限制；仅在模型与服务端完整支持后才宣告。
- 输入输出 shape、dtype、device、生成 token 数、错误输入的行为，以及一次真实输出样例。
- Cache 是否创建和随生成增长。以官方 `generate()` 为正确性基线，不要手写续 Cache 来代替模型自己的生成准备逻辑。

固定模型 revision、依赖版本、提示词、图片/视频素材、采样参数和随机种子。将输出保存到 `artifacts/` 或分析记录中，避免只凭肉眼判断“跑通”。

### Bench：测 Transformers 基线

Bench 脚本与 Test 使用同一个本地 snapshot、输入、dtype、attention backend 和 `max_new_tokens`。包括预热和多次正式样本；CUDA 计时前后调用 `torch.cuda.synchronize()`，否则异步执行会让延迟结果偏小。分别报告：

| 指标 | 记录方式 |
| --- | --- |
| Prefill latency | 输入固定 prompt（图像/视频也固定），记录首次前向时间 |
| Decode latency / token | 按新增 token 数统计 Decode 时间，不把 Prefill 混入 |
| End-to-end latency | 从编码开始到完整输出结束的时间 |
| Throughput | 输出 token/s；注明是否包含预处理和首 token 时间 |
| Peak GPU memory | 预热后重置峰值统计，报告峰值 allocated/reserved |
| Workload | prompt token、输出 token、图片数量/尺寸或视频帧数、batch/concurrency |

报告中同时写明 GPU、驱动、CUDA、PyTorch、Transformers、dtype、attention backend、模型 SHA 和测量次数。不要把不同输入长度或不同精度的结果直接比较，也不要把单请求 Bench 宣称为服务容量或并发吞吐。

### Profile：分析前向与算子

`scripts/<model_alias>/forward.py` 由负责该模型的工程师按实际结构编写。它应在相同模型快照和工作负载下分段分析 Prefill 与单 token Decode，并保存完整的 profiler trace/JSON 到 `artifacts/`；脚本所需模型配置副本放在 `scripts/<model_alias>/profiles/`，不要从 `src/` 动态读取配置。

接入 Engine 前，Profile 先测 Transformers 官方实现；Engine 接入并通过 Test 后，再把同一输入扩展到本项目实现。两段输出分别标为 `==================== transformers ====================` 与 `==================== inference ====================`，同时保留相同模型 revision、输入和生成配置，便于直接比较。不要在 Transformers 基线中提前混入 Engine 的结果。

Profile 至少记录 operator/kernel 名称、输入 shape、CUDA 时间、显存/临时分配和调用次数。先用 PyTorch Profiler 定位；需要看 CPU 调度、CUDA API 和 Kernel 时间线时用 Nsight Systems；只有定位到值得优化的具体 Kernel 后，再用 Nsight Compute 看硬件计数器。Profile 用于解释开销分布，不代替 Bench。

VLM 的重点检查项：视觉预处理和视觉塔开销、视觉 token 数对 Prefill 的影响、文本 Attention/MLP、Cache 分配与读写、Decode 是否每步只输入新增 token。Qwen3-VL 有 MRoPE、`rope_deltas` 和视觉状态；Decode 输入准备应沿用 Transformers 官方 `prepare_inputs_for_generation()` / `generate()` 路径，并核实 query length 为 1、Cache 长度每次只增加 1。其他 VLM 也必须按其官方生成实现检查，不能将 Qwen 的假设照搬。

每个优化候选用一行记录：**问题、证据、拟改模块、正确性门槛、Bench baseline、Profile 证据、结果**。没有 Profile 证据时先不要改写算子；正确性不一致时先解决一致性，再谈速度。

## 4. 拆解模型并逐个组件对齐

根据 `config.json`、官方 modeling 源码和权重 key，画出该模型的前向结构。按计算职责拆解，不按文件数量拆解：例如 `Patch Embedding → Vision Blocks → Merger`，以及 `Token Embedding → Decoder Layers → Norm → LM Head`。如果模型有额外的 Resampler、Projector、跨模态融合、视频时序模块或额外视觉特征，就按真实拓扑添加；纯图片模型不要为了目录完整伪造视频模块。

| 模块 | 应确认的官方行为 | 本仓库常见归属 |
| --- | --- | --- |
| Embedding / Patch Embedding | 输入布局、patch 顺序、位置编码、输出 shape | `layers/` |
| Attention / RoPE / Norm / MLP | QKV 布局、mask、位置编码、激活和数值精度 | 各自的 `layers/*.py` |
| Decoder Layer / Vision Block | 残差顺序、Norm 位置、子模块连接 | `layers/` 中的组合层 |
| Projector / Merger / Resampler | 视觉到文本维度映射、token 合并策略 | `layers/` 或 `architecture/` |
| 完整视觉塔、文本塔、跨模态主干、生成模型 | checkpoint 模块路径和整体前向 | `modeling.py` 直接组装；模型专属类型放 `architecture/` |
| Tokenizer / Processor / 图像或视频适配 | Chat Template、预处理、视觉字段、校验 | `components/` 与 `architecture/` 按通用运行职责/模型特性划分 |
| Prefill / Decode / Cache 状态 | 新 token 准备、缓存续接、取消与结束清理 | `components/stepwise_generation.py` 等 |
| Runner / Adapter | Engine 生命周期和请求结果映射 | `components/` |

每个自研计算层都提供独立的小型 `__main__` 前向入口：在 CUDA 可用时默认用 CUDA，否则 CPU；构造相同配置的官方参考层和本地层，将参考权重复制/映射到本地层，用同一输入做 `eval()` 与 `torch.inference_mode()` 对比，断言输出 shape 和 `torch.testing.assert_close()`。测试随机参数只能验证接口形状；有 checkpoint 时还要做真实权重 key/数值对齐。

**组件入口必须以包模块方式运行**，不要直接执行 `python layers/vision_attention.py`。模型层使用包内相对导入，直接运行文件会丢失包上下文并报 `attempted relative import with no known parent package`。在仓库根目录、激活对应模型依赖环境后运行：

```bash
python -m inference.models.mage_vl_4b.layers.vision_attention
python -m inference.models.qwen3_vl_2b.layers.text_attention
```

按模型结构编写，不是固定清单：模型不需要的文件不要创建。出现权重 key 无法匹配、Processor 字段缺失或官方前向不支持时，先定位架构差异，不用 `strict=False` 掩盖未加载权重。

## 5. 按仓库规范实现模型包

模型目录统一为下面的形状。根目录只放四个 Python 入口；YAML 放 `profiles/`，依赖放 `requirements/`；`architecture/__init__.py` 保留为空。目录内部文件依据模型能力增减：

```text
src/inference/models/<model_alias>/
├── example.py                 # 最小实际调用示例；VLM 展示图片输入和流式输出
├── loader.py                  # 本模型配置读取、路径解析和 Processor/权重加载所需设置
├── modeling.py                # 直接导入本地层并显式组装模型拓扑
├── runtime.py                 # 简单组合 ModelRunner、Adapter，导出 Engine Client
├── architecture/
│   ├── __init__.py            # 空包标记，保持模型间目录一致
│   └── ...                    # Config、输出类型和模型专属模态能力（按需）
├── layers/                    # 细粒度计算层；每个独立层可运行 main 对齐测试（按需）
├── components/                # tokenizer、encoder、generation、runner、adapter（按需）
├── profiles/
│   ├── model.yml              # alias、能力、runtime_factory、配置文件名
│   ├── inference.yml          # 模型版本/路径及 Engine 运行参数
│   └── network.yml            # Inference 地址和媒体端点
└── requirements/
    └── requirements.txt       # 本模型专属 Python 依赖
```

### 四个入口各写什么

| 文件 | 实现内容 | 约束 |
| --- | --- | --- |
| `modeling.py` | 导入已实现的 `layers/`，显式组装 Vision、Text、Fusion、Conditional Generation 模块 | 让官方 checkpoint key 按标准模块路径加载；不要用动态工厂、注册表或黑盒整体替换 |
| `loader.py` | 读取启动器已选中的配置，解析本地权重目录，提供模型类/Processor 初始化所需常量 | 配置里的权重路径使用本地 `checkpoints/`；生产请求不临时下载 |
| `runtime.py` | 创建 Runner 与 Adapter，封装成共用 `ModelRuntime`，导出 `create_engine_client()` | 复用共用 Engine/Worker，不复制一套服务和调度器 |
| `example.py` | 创建 Engine Client、传入文本和图片、打印流式结果 | 所有 import 放顶部；保留直接可读的调用流程，使用本模型 `profiles/inference.yml` |

`modeling.py` 是完整架构组装处。不要把 assembled model 放到 `model/` 子目录，也不要新增 `party.py` 或模型级 `utils/` 作为杂项聚集点。模型 YAML 只存模型版本、路径、dtype、设备、Attention backend、上下文/缓存容量和请求限制等运行值，不用 YAML 选择或拼装 Python 层组件。

### 配置和依赖

`profiles/model.yml` 是模型发现和能力声明入口。下面仅是 Qwen3-VL 案例；新模型 alias、输入模态和能力必须依据代码及实测结果填写：

```yaml
alias: qwen3-vl-2b
display_name: Qwen3-VL-2B-Instruct
integration_profile: custom
backend: transformers
runtime_factory: inference.models.qwen3_vl_2b.runtime:create_runtime
inference_config: inference.yml
network_config: network.yml
api:
  model_id: qwen3-vl-2b
  capabilities:
    chat_inputs: [text, image]
    streaming: true
    tool_calling: false
    output_mode: text_delta
    video_upload: false
    live_video: false
```

`profiles/inference.yml` 保存运行参数；至少固定 Hub repo ID、完整 revision、dtype、attention backend、本地权重路径和 `local_files_only: true`。图片最大尺寸、图片数、上下文、生成长度和 KV reservation 按实际 GPU 与 API 限制制定。`profiles/network.yml` 配置监听地址。`scripts/<model_alias>/profiles/` 保存供本地分析脚本使用的配置副本。

将缺少的模型专属 Python 包写入 `requirements/requirements.txt`，并在 `src/inference/pyproject.toml` 的 optional dependencies 增加单独 extra：

```toml
<model-extra> = { file = ["models/<model_alias>/requirements/requirements.txt"] }
```

PyTorch/CUDA 应按目标环境单独安装；其他模型依赖按需安装，不要让 `pip install -e .` 拉入所有模型依赖。Qwen3-VL 的实际命令如下，新模型将 extra 名称替换为自己的名字：

```bash
python -m pip install -e "src/inference[qwen3-vl-2b]"
python -m pip check
```

## 6. 接入 Engine 并逐级验收

所有 Transformers 基线与组件对齐通过后，再接 Engine。按顺序实现并检查：

1. `components/tokenizer.py`：调用模型官方 tokenizer/processor 和 chat template；保留多模态 tensor 与 grid/时间戳等必要 metadata。
2. `architecture/` 的输入能力实现及 `components/encoder.py`：验证 OpenAI 消息内容、图片/帧大小、请求上限和生成参数；将 Processor 输入放到正确 device/dtype。
3. `components/stepwise_generation.py`：把 Prefill、每步 Decode、Cache、位置编码状态、EOS、取消和资源清理适配到 Engine 工作项。先让官方 `generate()` 作为正确性来源，再接入逐步生成。
4. `modeling.py` 与 `components/model_runner.py`：显式加载本地模块和权重，验证完整输入至 logits/生成文本。
5. `components/adapter.py`、`runtime.py`：把请求和结果映射到共用 worker 契约；不要让模型提出 tool call 就直接执行外部工具，工具授权与执行在 Agent 层。
6. 更新 `model.yml` 的能力、`pyproject.toml` extra 和必要的 Gateway 映射。只声明已经实现并测试的能力。

启动器预检不加载权重：

```bash
inference list
inference validate --model <model-alias>
```

然后启动模型并运行模型目录中的示例：

```bash
inference start --model <model-alias>
# 另一个终端，仍在 inference 环境和仓库根目录：
python -m inference.models.<model_package>.example
```

端到端验收使用已有 OpenAI 兼容 `/v1/chat/completions`，不新建模型专属接口。若要经 Go Gateway 访问，再配置 Gateway alias/upstream，并用仓库根目录的 `client.py` 验证路由、鉴权、请求/响应和 SSE。图像、视频、工具调用等能力分别准备真实请求验收；没有测过的能力必须标为 `false`。

## 7. 结果与复现记录

每个模型至少保留以下结果：

| 结果 | 内容 |
| --- | --- |
| Transformers Test | 成功/失败、模型 SHA、输入素材标识、输出与 tensor 检查 |
| Transformers Bench | 输入长度/视觉规模、生成 token 数、延迟、吞吐、峰值显存和环境 |
| Transformers Profile | Prefill/Decode 分段 trace、算子 shape/时间/内存、分析结论 |
| 层级对齐 | 官方参考模块、本地模块、加载权重情况、误差阈值和断言结果 |
| Engine 验收 | 静态 validate、模型启动、图文流式结果、Gateway 路由（若启用） |

Profile 脚本由熟悉该模型的工程师维护在 `scripts/<model_alias>/forward.py`；它需要按模型官方生成逻辑准备输入和 Cache，不能照抄其他模型的内部状态假设。机器可读数值交给脚本生成 `analysis/<model_alias>/analysis.xlsx`；问题、证据、解决方案、效果及未验证项写在同目录 `analysis.md`。权重、个人密钥和大体积 trace 不进 Git。

## 常见问题

| 现象 | 先检查 |
| --- | --- |
| 下载命令仍访问 huggingface.co | `HF_ENDPOINT` 是否在 Python 进程启动前导出；Python 是否已在 shell 中提前 import Hub 包；镜像当前是否可访问 |
| 下载成功但模型加载找不到文件 | 固定 revision 是否一致；分片 index 是否指向缺失文件；Processor/Tokenizer 文件是否完整；运行 YAML 是否指向正确目录 |
| `AutoModel` 无法识别架构 | Transformers 版本是否与模型要求匹配；是否要更新到官方支持该架构的版本；远程代码需审阅后再决定，不要直接启用 |
| 视觉输出与官方不同 | chat template、图片缩放/裁剪、grid/视频帧 metadata、dtype、attention backend 和 special token 是否一致 |
| Decode query 长度不是 1 | 使用模型官方 `prepare_inputs_for_generation()` 检查新 token 输入；避免把整个 prompt 再送入 Decode |
| CUDA OOM | 记录图片分辨率/patch 数、prompt、输出长度、dtype、并发和显存；降低受控负载后重新做同配置 Bench/Profile |

## 官方资料

- [Hugging Face Hub：环境变量](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)
- [Hugging Face Hub：snapshot_download](https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download#huggingface_hubsnapshot_download)
- [Hugging Face Hub：模型仓库文件说明](https://huggingface.co/docs/hub/transformers)
- [Transformers：Processor](https://huggingface.co/docs/transformers/processors)
- [Transformers：Qwen3-VL](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)
- [Qwen3-VL-2B-Instruct：固定 revision 文件树](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/89644892e4d85e24eaac8bacfd4f463576704203)
- [Hugging Face Hub 当前源码：`HF_ENDPOINT` 的读取位置](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/constants.py)
- [hf-mirror 项目使用说明](https://github.com/Mirrors-Project/hf-mirror/blob/main/docs/en/quickstart.md)（第三方项目）
