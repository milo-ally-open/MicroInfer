# 新模型接入流程

按本顺序操作：先用 Hugging Face 原生实现完成 **Test、Bench、Profile**，再接入 Inference，最后验收服务。本文用仓库现有的 **Qwen3-0.6B** 说明实际目录、配置和命令；VLM、视频模型按能力增加组件与测试。

## 1. 判断接入方式

| 模型 | 接入方式 | 必须验证 |
| --- | --- | --- |
| 纯文本 | `standard`（模型符合标准 Causal LM/Cache 行为）；否则 `custom` | Tokenizer、生成、Prefill/Decode、Cache |
| VLM / 视频 | `custom` | 文本及对应图片、视频输入；声明支持的每项能力 |

`model.yml` 中每个能力都必须有实现和测试；不支持的能力设为 `false`。

## 2. 固定模型并下载权重

以下命令在仓库根目录的 WSL/Linux Bash 中执行。推理默认使用 CUDA，仅在 CUDA 不可用时回退 CPU；CUDA OOM 应调整配置，不能当作回退条件。

```bash
conda activate inference
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

在模型卡确认许可证，并记录 Hub 仓库和完整 commit revision。仓库中的 Qwen3-0.6B 定版信息如下：

```text
repo_id: Qwen/Qwen3-0.6B
revision: c1899de289a04d12100db370d81485cdf75e47ca
package: qwen3_0_6b
alias: qwen3-0.6b
```

将权重保存到被 Git 忽略的 `checkpoints/`：

```bash
python scripts/download.py \
  --repo-id Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir checkpoints/Qwen3-0.6B-c1899de \
  --ignore-pattern 'onnx/**'
```

确认目录中有 `config.json`、模型权重和 Tokenizer 文件。私有模型用 `HF_TOKEN` 环境变量认证，不要把令牌写入文件。

## 3. 接入 Engine 前完成原生基线

此阶段只用固定 revision 的 Hugging Face 模型和 Processor，不导入项目 `inference`、不改模型实现。按 **Test → Bench → Profile** 顺序执行，并保存脚本、配置和原始结果。

| 顺序 | 文件 | 要写的代码 | 通过条件 |
| --- | --- | --- | --- |
| 1. Test | `scripts/<package>/forward.py` | 只用 Transformers 加载本地快照；按官方 Processor/Chat Template 编码固定输入；执行一次生成；断言输出非空并打印输入 token 数、输出 token 数和设备 | 固定 revision 离线加载成功，模型确实生成内容 |
| 2. Bench | `tests/<package>/test_performance.py` | 加载与 Test 相同的模型/输入；先预热，再同步 GPU 并多轮计时；分别记录 Prefill、Decode 或 API 总延迟、生成 token 数、吞吐和峰值显存 | 输出机器可读的数值报告；固定输入和环境可重复运行 |
| 3. Profile | `scripts/<package>/forward.py` | 对 Prefill 和单步 Decode 分别包 PyTorch Profiler/NVTX；开启 shape、memory 采集；导出 trace 和算子表 | trace 可打开，记录包含阶段、shape、调用次数和耗时 |

Bench 与 Profile 分开运行；Profiler 有采集开销，不能用 Profile 耗时作性能基准。不同模态和输入规模分别测量。

### 3.1 先写 Hugging Face 原生 Test

新模型的 `scripts/<package>/forward.py` 先只导入 Transformers，不导入 `inference`。以下代码可作为 Qwen3 的起步版：固定本地 snapshot、revision、prompt 和 dtype，用官方 Chat Template 编码，再确认输出非空。

```python
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "checkpoints/Qwen3-0.6B-c1899de"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DTYPE = torch.bfloat16
DEVICE_MAP = 0 if torch.cuda.is_available() else "cpu"


def forward_transformer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, revision=REVISION, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        revision=REVISION,
        local_files_only=True,
        use_safetensors=True,
        dtype=DTYPE,
        device_map=DEVICE_MAP,
    ).eval()
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "用一句话说明大型语言模型推理服务。"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(model.get_input_embeddings().weight.device)
    with torch.inference_mode():
        output_ids = model.generate(input_ids, max_new_tokens=8, do_sample=False)
    answer = tokenizer.decode(output_ids[0, input_ids.shape[-1]:], skip_special_tokens=True)
    assert answer.strip(), "模型没有生成文本"
    print(answer)
    return model, tokenizer, input_ids


if __name__ == "__main__":
    model, tokenizer, input_ids = forward_transformer()
```

运行前确认本地有完整权重。Test 通过条件：模型离线加载成功、Chat Template 能编码、生成文本非空。先固定 prompt、dtype、revision 与 GPU 型号，这些值后续必须用于同输入对照。

### 3.2 Bench：在 `test_performance.py` 单独测速度

`test_performance.py` 是独立测试程序，不从 `scripts/` 导入 Python 代码；它从自己的 `test.yml` 读取模型快照、revision、prompt 和测量轮数。下面代码是纯文本 Transformers 基线的可运行测量主体。多模态模型还要通过对应 Processor 加载固定图片/视频，并把相同模态输入用于所有轮次。集成后在同一文件中增加 HTTP API 测量，不能把 HF 模型和 Engine 模型在同一进程中的显存峰值误当成同一条推理路径。

测速期间不要启用 Profiler。CUDA 计时前后同步；实际生成 token 数从输出形状计算，不用请求上限代替：

```python
from pathlib import Path
import json
import statistics
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

TEST_CONFIG = Path(__file__).resolve().with_name("test.yml")
settings = yaml.safe_load(TEST_CONFIG.read_text(encoding="utf-8"))
snapshot = (TEST_CONFIG.parent / settings["snapshot"]).resolve()
revision = str(settings["revision"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = getattr(torch, settings["component"]["dtype"])
tokenizer = AutoTokenizer.from_pretrained(snapshot, revision=revision, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    snapshot,
    revision=revision,
    local_files_only=True,
    use_safetensors=True,
    torch_dtype=dtype,
).to(device).eval()
messages = [{"role": "user", "content": settings["prompt"]}]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(device)
max_new_tokens = int(settings["max_new_tokens"])
warmups = int(settings["warmup_requests"])
iterations = int(settings["iterations"])
for _ in range(warmups):
    with torch.inference_mode():
        model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
if device.type == "cuda":
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
latencies, token_rates, token_counts = [], [], []

for _ in range(iterations):
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    generated_tokens = int(output_ids.shape[-1] - input_ids.shape[-1])
    latencies.append(elapsed)
    token_counts.append(generated_tokens)
    token_rates.append(generated_tokens / elapsed)

assert token_counts and min(token_counts) > 0, "模型没有生成 token"
report = {
    "model_alias": settings["model_alias"],
    "revision": revision,
    "device": str(device),
    "warmup_requests": warmups,
    "iterations": iterations,
    "input_tokens": int(input_ids.shape[-1]),
    "completion_tokens": token_counts,
    "latency_seconds_median": statistics.median(latencies),
    "tokens_per_second_median": statistics.median(token_rates),
    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
}
print(json.dumps(report, ensure_ascii=False, indent=2))
```

测试 YAML 需包含 `revision` 字段，例如 `c1899de289a04d12100db370d81485cdf75e47ca`。代码输出包含模型、输入输出 token 数、median 延迟、tokens/s 和峰值显存；接入后的完整配置解析、HF/API 双路径和汇总实现见 [Qwen3 test_performance.py](tests/qwen3_0_6b/test_performance.py)。新模型按自己的模型类、Processor 和输出协议改写，不能保留 Qwen3 类型假设。

集成之后，在同一 `test_performance.py` 中增加下面代码请求启动中的 Qwen3 服务；测量 HTTP 的完整请求延迟，并记录返回的生成 token 数。新模型接入前不要执行这段：

```python
import json
import time
from urllib.request import Request, urlopen

body = {
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "用一句话说明大型语言模型推理服务。"}],
    "max_tokens": 32,
    "stream": False,
}
request = Request(
    "http://127.0.0.1:8011/v1/chat/completions",
    data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
started = time.perf_counter()
with urlopen(request, timeout=300) as response:
    result = json.loads(response.read())
elapsed = time.perf_counter() - started
assert result["choices"][0]["message"]["content"]
generated_tokens = int(result["usage"]["completion_tokens"])
print("engine_seconds:", elapsed)
print("engine_tokens_per_second:", generated_tokens / elapsed)
```

固定 warmup 次数、测量次数、输入长度、输出上限和并发；同时记录 Prefill/首 Token、Decode 速度及峰值显存。短 prompt 与长 prompt、不同 batch 分开测，不把不同配置的结果放在同一组比较。

### 3.3 Profile：在 `forward.py` 分开采 Prefill 和 Decode

在 `scripts/<package>/forward.py` 的 `forward_transformer()` 中，独立采集 Transformers Prefill 和 Decode；接入后在 `forward_inference()` 对 Inference 执行同样输入。先对完整 prompt 做 Prefill 并保留 Transformers Cache，再用刚产生的一个 token 做 Decode。以下代码用于说明采集边界；完整版本还要在文件顶部导入 `torch`、定义 `TRACE_DIR` 并先创建目录。仓库可直接运行的实现见 [Qwen3 forward.py](scripts/qwen3_0_6b/forward.py)：

```python
TRACE_DIR = ROOT / "artifacts/qwen3_0_6b/traces"
TRACE_DIR.mkdir(parents=True, exist_ok=True)
activities = [torch.profiler.ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(torch.profiler.ProfilerActivity.CUDA)

with torch.profiler.profile(
    activities=activities, record_shapes=True, profile_memory=True
) as prefill_profile:
    with torch.inference_mode():
        prefill = model(input_ids=input_ids, use_cache=True, return_dict=True)
next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
prefill_profile.export_chrome_trace(str(TRACE_DIR / "transformers-prefill.json"))

decode_ids = torch.cat((input_ids, next_token), dim=-1)
decode_inputs = model.prepare_inputs_for_generation(
    decode_ids,
    past_key_values=prefill.past_key_values,
    attention_mask=torch.ones_like(decode_ids),
    use_cache=True,
)
with torch.profiler.profile(
    activities=activities, record_shapes=True, profile_memory=True
) as decode_profile:
    with torch.inference_mode():
        decode = model(**decode_inputs, return_dict=True)
decode_profile.export_chrome_trace(str(TRACE_DIR / "transformers-decode.json"))
for phase, profile in (("prefill", prefill_profile), ("decode", decode_profile)):
    sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    print(phase, profile.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=20))
```

Bench 和 Profile 必须分开；Profiler 耗时不能当作基准。模型负责人还要决定是否用 Nsight Systems 查 CPU/GPU 时间线、用 Nsight Compute 分析热点 Kernel，并保留采集命令、输入 shape 和原始报告。

写入 `analysis.py` 的原始 JSON 应保留可自动生成 XLSX 的数值字段，至少有运行环境、实现名称、阶段和每个算子的 shape/调用次数/耗时/显存。例如：

```json
{
  "run": {
    "model_alias": "qwen3-0.6b",
    "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "gpu_name": "GPU name",
    "dtype": "torch.bfloat16",
    "batch_size": 1,
    "query_tokens": 128,
    "kv_tokens": 129
  },
  "operator_records": [
    {
      "implementation": "transformers",
      "phase": "prefill",
      "operator": "aten::linear",
      "input_shapes": [[1, 128, 1024], [1024, 1024]],
      "call_count": 1,
      "self_cuda_time_total_us": 123.4,
      "self_cuda_memory_usage_bytes": 4096
    }
  ]
}
```

示例数值只说明字段类型，实际值必须由测量脚本写出。`analysis.py` 从这些记录映射到 XLSX 列；问题、方案、效果只写在 Markdown，不写进机器数据表。

### 3.4 接入后增加 Inference 路径

模型接入并通过组件测试后，才在同一个 `forward.py` 加入 `forward_inference()`。两条路径必须接收同一份 `input_ids`；Prefill 比较最后位置 logits，Decode 用相同的首 token 和 Cache 长度比较 logits。项目配置必须先指向 `scripts/<package>/` 下的本地副本，再导入读取配置的模型模块：

```python
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "scripts/qwen3_0_6b/inference.yml"
NETWORK_CONFIG_PATH = ROOT / "scripts/qwen3_0_6b/network.yml"
MANIFEST_PATH = ROOT / "scripts/qwen3_0_6b/model.yml"
os.environ["INFERENCE_PROJECT_ROOT"] = str(ROOT)
os.environ["INFERENCE_MODEL_ALIAS"] = "qwen3-0.6b"
os.environ["INFERENCE_MODEL_MANIFEST"] = str(MANIFEST_PATH)
os.environ["INFERENCE_CONFIG"] = str(CONFIG_PATH)
os.environ["INFERENCE_NETWORK_CONFIG"] = str(NETWORK_CONFIG_PATH)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 配置环境后，在模块顶部导入模型 Runtime。
from inference.models.qwen3_0_6b.runtime import create_runtime
```

Qwen3 的 Inference 核心调用如下，完整配置和指标收集见 [`forward.py`](scripts/qwen3_0_6b/forward.py)：

```python
def forward_inference():
    runtime = create_runtime()
    try:
        runtime.load()
        runner = runtime.model_runner
        token_count = int(INPUT_IDS_CPU.shape[-1])
        block_size = runner.cache.block_size_tokens
        block_count = (token_count + 1 + block_size - 1) // block_size
        if block_count > runner.cache.num_blocks:
            raise ValueError("prompt plus one decode token exceeds KV page capacity")
        request_id = "qwen3-forward-analysis"
        block_table = tuple(range(block_count))
        runner.cache.bind(request_id, block_table)
        input_ids = INPUT_IDS_CPU.to(runner.input_device)

        with torch.inference_mode():
            prefill_logits = runner.model.forward_batch(
                input_ids,
                token_lengths=(token_count,),
                position_offsets=(0,),
                request_ids=(request_id,),
                block_tables=(block_table,),
                cache=runner.cache,
            )
            next_token = TRANSFORMERS_NEXT_TOKEN.to(runner.input_device)
            decode_logits = runner.model.forward_batch(
                next_token,
                token_lengths=(1,),
                position_offsets=(token_count,),
                request_ids=(request_id,),
                block_tables=(block_table,),
                cache=runner.cache,
            )
        runner.cache.release(request_id)
        return prefill_logits, decode_logits
    finally:
        runtime.close()
```

启动前把 `INFERENCE_CONFIG`、`INFERENCE_NETWORK_CONFIG`、`INFERENCE_MODEL_MANIFEST` 指向 `scripts/<package>/` 下的副本；不能让分析脚本依赖别的模型目录。实际脚本还要分别对两个阶段采集 Profiler/NVTX，并校验 logits 与 Transformers 基线的差值。

**Test、Bench、Profile 脚本由模型接入负责人编写和维护；项目不会自动生成。** `forward.py` 在接入前测 Hugging Face 基线，接入后再增加上面的 Inference 路径；负责人固定工作负载、预热、计时和采集范围，并选择配置 PyTorch Profiler、Nsight Systems 或 Nsight Compute。`analysis.py` 从原始数值 JSON/Profiler CSV 生成 XLSX；结论记录在 `analysis/<package>/analysis.md`，简要说明问题、方案和效果。XLSX 数据必须来自脚本生成的测量值，不手工填入结论文字。

脚本与配置放在独立目录，不从 `src/` 读取配置：

```text
scripts/<package>/
├── forward.py
├── analysis.py
├── model.yml
├── inference.yml
└── network.yml
```

新模型接入前，`forward.py` 只运行 Hugging Face 路径。接入后再增加 Inference 路径，与基线使用相同输入对照。仓库中已集成的 Qwen3 示例可这样运行：

```bash
python scripts/qwen3_0_6b/forward.py
python scripts/qwen3_0_6b/analysis.py --input artifacts/qwen3-0.6b-forward.json
```

`analysis.py` 默认将 XLSX 写入 `analysis/qwen3_0_6b/analysis.xlsx`。

## 4. 建立模型包

模型代码放在 `src/inference/models/<package>/`。新建模型包时，先从仓库根目录创建必需目录和入口文件：

```bash
PACKAGE=my_model
mkdir -p "src/inference/models/$PACKAGE/configs" "scripts/$PACKAGE" "tests/$PACKAGE" "analysis/$PACKAGE"
touch "src/inference/models/$PACKAGE/__init__.py" \
      "src/inference/models/$PACKAGE/runtime.py" \
      "src/inference/models/$PACKAGE/example.py" \
      "src/inference/models/$PACKAGE/requirements.txt" \
      "scripts/$PACKAGE/forward.py" \
      "scripts/$PACKAGE/analysis.py" \
      "tests/$PACKAGE/test.yml" \
      "tests/$PACKAGE/test_components.py" \
      "tests/$PACKAGE/test_performance.py" \
      "tests/$PACKAGE/test_end2end.py" \
      "analysis/$PACKAGE/analysis.md"
cp scripts/qwen3_0_6b/model.yml scripts/qwen3_0_6b/inference.yml scripts/qwen3_0_6b/network.yml "scripts/$PACKAGE/"
cp src/inference/models/qwen3_0_6b/configs/model.yml \
   src/inference/models/qwen3_0_6b/configs/inference.yml \
   src/inference/models/qwen3_0_6b/configs/network.yml \
   "src/inference/models/$PACKAGE/configs/"
cp tests/qwen3_0_6b/test.yml "tests/$PACKAGE/"
```

这些是 Qwen3 配置模板；必须改模型 ID、revision、alias、模块工厂、权重路径、网络端口和测试输入。不要把 Qwen3 专属模型结构复制成另一个架构的实现。只在确实要新增的子目录中写入组件文件。

下面是仓库现有 Qwen3-0.6B 的完整结构；其中模型拓扑、计算层、执行组件各自独立：

```text
src/inference/models/qwen3_0_6b/
├── __init__.py
├── runtime.py
├── example.py
├── requirements.txt
├── processor.py
├── tool_calling.py
├── configs/
│   ├── model.yml
│   ├── inference.yml
│   └── network.yml
├── model/
│   └── qwen3.py
├── layers/
│   ├── attention.py
│   ├── layers.py
│   └── rope.py
├── components/
│   ├── adapter.py
│   ├── decoder.py
│   ├── detokenizer.py
│   ├── encoder.py
│   ├── input_batch.py
│   ├── model_runner.py
│   ├── paged_kv_cache.py
│   ├── sampler.py
│   └── tokenizer.py
└── utils/
    └── loader.py
```

新模型目录名换成自己的 `<package>`。`tool_calling.py`、VLM 的图像/视频 Encoder、`live.py` 都按能力添加；没有实现的能力不要声明为 `true`。不要为了凑目录创建空文件。

Qwen3 的实际拆分是：`model/qwen3.py` 组装网络；`layers/` 实现 Attention、RoPE 等计算；`components/` 分别放 Tokenizer、Encoder、Decoder/Detokenizer、Sampler、KV Cache、Runner 和 Adapter；`processor.py` 适配 API 输入输出，`tool_calling.py` 解析工具调用。新模型按需要选取组件，不要复制不支持的能力。

各文件不是空壳；需要实现的职责和对应代码入口如下：

| 文件 | 要写的代码 | Qwen3 实现 / 验收重点 |
| --- | --- | --- |
| `utils/loader.py` | 从固定 snapshot 读取官方 Config、Tokenizer 和 Safetensors，并校验权重 key/shape | [loader.py](src/inference/models/qwen3_0_6b/utils/loader.py)：不能静默漏载权重 |
| `model/qwen3.py` | 按官方层数和维度组装 Embedding、Decoder Layer、Norm、LM Head；实现带 Cache 的 forward | [qwen3.py](src/inference/models/qwen3_0_6b/model/qwen3.py)：输出 logits 与 HF 对照 |
| `layers/attention.py` | GQA 的 Q/K/V 投影、Q/K Norm、RoPE、因果注意力和 KV 读写 | [attention.py](src/inference/models/qwen3_0_6b/layers/attention.py)：覆盖 Prefill、单 token Decode、不同 KV 长度 |
| `layers/rope.py`、`layers/layers.py` | RoPE、RMSNorm、MLP 等数学算子 | 对照 Transformers 对应层；检查 dtype、shape、误差 |
| `components/paged_kv_cache.py` | 物理 KV pages 的分配、写入、读取、释放和 block table 映射 | 多请求隔离、边界、变长 Prefill 和 Decode |
| `components/model_runner.py` | 加载权重；实现请求状态、Prefill、Decode、采样、完成/取消和资源清理 | [model_runner.py](src/inference/models/qwen3_0_6b/components/model_runner.py)：遵守 Engine 传入的阶段和 block table |
| `components/adapter.py` | 把 Engine 的请求/调度项转换成模型 Runner 操作，再把结果转换回共用类型 | [adapter.py](src/inference/models/qwen3_0_6b/components/adapter.py)：校验 alias、token 数和 finish reason |
| `components/tokenizer.py`、`encoder.py` | 应用 Chat Template、整理消息、工具 schema、采样参数和上下文长度 | tokenizer 输出 IDs；encoder 输出 `PreparedChatInput` |
| `components/decoder.py`、`detokenizer.py` | 完整解码及增量解码，支持逐 token SSE | [decoder.py](src/inference/models/qwen3_0_6b/components/decoder.py)：分片拼接后与完整 decode 一致 |
| `processor.py`、`tool_calling.py` | 组合输入/输出组件；把模型工具格式解析为标准 Tool Call | 实现 Processor 契约；只解析和校验，不执行工具 |

### 4.1 按依赖顺序逐个实现组件

不要先写一个大 `forward()` 再把代码拆散。按下面顺序实现：低层张量运算先通过数值对照，模型前向再复用这些层，最后才接调度和 API。每个代码文件要有明确输入、输出和失败条件；不能只建空类或用 `pass` 占位。

| 顺序 | 文件 | 必须实现的代码 | 先做的局部验收 |
| --- | --- | --- | --- |
| 1 | `layers/rope.py`、`layers/layers.py` | RoPE 的频率缓存和旋转；RMSNorm 的归一化；MLP 的 gate/up/down 投影及激活；构造参数从模型 Config 读取 | 输入输出 shape 不变、dtype/device 不变、值有限；与 Transformers 对应层共享权重后 `assert_close` |
| 2 | `layers/attention.py` | Q/K/V 投影和 head reshape；Q/K norm（若模型有）；RoPE；GQA/MQA head 映射；causal mask；Prefill/Decode 分支；按层读写 Cache | Prefill 的每个 token 只读可见历史；Decode query 长度为 1；Cache 的逻辑长度、物理 page 写入位置正确；对比 HF attention 输出 |
| 3 | `components/paged_kv_cache.py` | 分配固定形状的 K/V page；维护 request 到 block table 映射；支持 bind/extend/write/read/release；检查层号、偏移和 page 容量 | 写入后读回相同数据；跨 page 边界正确；两个 request 互不污染；释放后可重新分配 |
| 4 | `model/<model>.py` | 从 Config 构造 embedding、N 层 decoder、final norm 和 LM head；实现单批次 `forward_batch`；将 token 长度、位置偏移、request ID、block table 和 Cache 传至每一层 | HF 与自有实现加载同一权重；分别对比 Prefill logits、单 token Decode logits；检查 batch 元数据错误会明确报错 |
| 5 | `utils/loader.py` | 读取本模型配置和本地 snapshot；创建模型结构；逐个加载 Safetensors；校验 missing/unexpected key、shape、dtype；加载 Tokenizer/Processor | 加载权重报告没有未解释的缺失或多余 key；参数总量、设备和 dtype 符合配置 |
| 6 | `components/sampler.py`、`model_runner.py` | logits 按 sampling 参数处理并选 token；Runner 管每个 request 的输入、生成 token、Prefill 进度、Cache 长度、停止条件、取消和资源释放 | greedy 输出可复现；EOS/max tokens 正确结束；取消请求后 Cache 被释放；Prefill chunk 与连续 Prefill 等价 |
| 7 | `components/adapter.py`、`processor.py` | Adapter 将 Engine 的生命周期/调度操作转成 Runner 调用；Processor 在 API 边界解析请求并构造 `PreparedChatInput`，负责完整与增量解码 | alias/参数错误返回明确的 4xx；流式 delta 拼接等于完整 decode；完成、取消都清理 request 状态 |
| 8 | `runtime.py`、`__init__.py`、`example.py` | Runtime 组合 Runner 和 Adapter；包入口提供 Engine Client 工厂；example 展示可复制的 Python 用法和流式输出 | `inference validate --model <alias>` 通过；example 能实际生成并在结束时关闭 Engine |

#### 模型计算层必须遵守的前向接口

以 Qwen3 为例，模型层接收 token 和 Engine 已经分配好的 KV 资源；它不自行调度请求，也不自行执行 HTTP。变长 batch 的每个样本用相同索引位置的 metadata 描述：`token_lengths[i]` 是本次 query token 数，`position_offsets[i]` 是 query 在上下文中的起点，`request_ids[i]` 和 `block_tables[i]` 指向该请求的 KV。

```python
def forward_batch(
    self,
    input_ids: torch.Tensor,                  # [batch, padded_query_tokens]
    *,
    token_lengths: Sequence[int],             # 每条实际 query 长度
    position_offsets: Sequence[int],          # Prefill chunk 起点或 Decode 位置
    request_ids: Sequence[str],
    block_tables: Sequence[Sequence[int]],
    cache: PagedKVCache,
) -> torch.Tensor:                            # [batch, vocab_size]，每条取最后有效 token logits
    ...
```

入口至少校验：`input_ids` 是二维且非空；batch 大小与 metadata 数量一致；request ID 不重复；每个有效长度在 `[1, query_width]`；position offset 非负；模型、输入和 Cache 的设备及 dtype 一致。最后只返回每条序列最后一个有效位置的 logits，避免把 padding token 当作预测结果。Qwen3 的真实实现见 [`model/qwen3.py`](src/inference/models/qwen3_0_6b/model/qwen3.py)。

Attention 的一次前向按固定顺序实现，后续优化 Kernel 时也保持这个数学契约：

```text
hidden_states
  → q_proj / k_proj / v_proj
  → reshape 为 [batch, heads, query, head_dim]
  → Q/K norm（模型定义有时才做）
  → RoPE(position_ids)
  → 将本次 K/V 写入 paged cache
  → query 对可见的历史 K/V 做 causal attention
  → merge heads → o_proj
```

Prefill 处理一段 query，query 长度可以大于 1；Decode 每步只处理一个新 token，query 长度必须为 1，但 K/V 读取范围是完整历史。把 “写入新 K/V” 和 “attention 读取范围” 分开核对：Prefill 第 `j` 个 query 不能看见比自身位置更晚的 K/V；Decode 要能读取之前所有 token 加本步 token。使用 FlashAttention 或其他算子时，先保持输入布局、mask、RoPE 位置和 Cache 更新语义一致，再测速度。

#### Runner 与 Adapter 的 Engine 调用边界

Engine 通过 Adapter 调用模型专属生命周期，再把 `ScheduledWorkItem` 批次交给 `execute_batch()`。Runner 拥有模型、Cache 和逐请求状态；Adapter 转换 Engine 的类型与模型输入，不直接实现矩阵运算。Qwen3 的 Adapter/Runner 已给出可运行参考：[`adapter.py`](src/inference/models/qwen3_0_6b/components/adapter.py)、[`model_runner.py`](src/inference/models/qwen3_0_6b/components/model_runner.py)。

```python
class ModelAdapter:
    def prepare_chat_generation(self, request_id, request_body, prepared_input=None):
        # 校验 alias/输入；生成 PreparedChatInput；通知 Runner 建立 request 状态。
        ...

    def bind_generation_cache(self, request_id, block_ids):
        # 把 Engine 分配的 KV block 绑定给 Runner。
        ...

    def execute_batch(self, requests):
        # 只委托给 Runner；结果必须按传入 work item 顺序返回。
        return self.model_runner.execute_batch(requests)

    def finish_chat_generation(self, request_id):
        # 回收 Runner 状态并转换为 InferenceResult。
        ...

    def cancel_generation(self, request_id):
        # 释放 Cache 和 request 状态；允许重复取消按契约处理。
        ...
```

`ModelRuntime(model_runner, model_adapter)` 负责把 Adapter 操作暴露给 Worker，把 `load()`、`close()`、GPU 状态交给 Runner。新模型的 `runtime.py` 只做组装，不要把加载权重、tokenize、采样或 API 逻辑塞进工厂函数。

#### Tokenize、Encoder、Decoder、Detokenize 分工

这四个名称表示不同边界，不要重新合并成一个笼统的 `InputProcessor`：

1. **Tokenizer**：封装 HF tokenizer/processor 的底层调用。输入是字符串或按模型协议整理好的消息，输出 token IDs（多模态时还包括 image/video tensors 与对应 metadata）。这里不决定生成参数，也不产生 HTTP 响应。
2. **Encoder**：校验 API message 和模态输入；应用模型 Chat Template；调用 Tokenizer；校验上下文长度；组装 `PreparedChatInput`（模型 alias、token IDs、sampling params 和模态张量）。同一消息只在此处转换一次。
3. **Decoder**：把完整 token ID 序列解成完整文本，用于非流式响应、日志或离线比较。需要保留模型的 special-token 和 skip-special-token 规则。
4. **Detokenizer**：保留增量解码状态；每收到 token ID 返回当前可安全输出的文本增量；`finish()` 刷出缓存中的最后部分。必须验证所有增量拼起来与 Decoder 对同一 token 序列的完整解码一致。

统一 API 契约见 [`network/input_processor.py`](src/inference/network/input_processor.py)：Processor 实现 `start()`、`prepare_chat(request_body)`、`close()`；流式能力实现 `create_stream_output_processor()`，非流式 token 输出实现 `decode_tokens(token_ids)`。多模态项目应把图片预处理/视频帧采样独立放在对应 Encoder 子组件中，并把模态张量随 `PreparedChatInput` 传给 Runner；不能只把图片路径转成文本假装图像理解。

#### 在接入前做模块数值对照

每个数学模块先用合成张量对齐 Transformers，再用真实权重做整模型对照。Qwen3 的典型本地入口是 `python -m inference.models.qwen3_0_6b.layers.rope` 或直接执行对应模块文件（以模块自身说明为准）。入口至少写出以下逻辑：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(17)
module = MyLayer(...).to(device).eval()
inputs = torch.randn(..., device=device, dtype=module_dtype)
with torch.inference_mode():
    actual = module(inputs)
assert actual.shape == expected_shape
assert actual.dtype == inputs.dtype and actual.device == device
assert torch.isfinite(actual).all()
torch.testing.assert_close(actual, reference, atol=..., rtol=...)
```

当官方层输入输出格式不同，先显式转换张量布局；不要为了让测试通过随意放宽容差。无法做逐层官方对照的项目，要至少写明参考实现、转换方式和采用的误差阈值。跑完模块入口后，再执行 `test_components.py` 验证真实 checkpoint 的权重映射和整模型 Prefill/Decode。

Qwen3 的模型主前向边界是 `Qwen3ForCausalLM.forward_batch(input_ids, token_lengths, position_offsets, request_ids, block_tables, cache)`。Runner 管请求和 KV 状态；Attention/Model 只做张量计算。把这条边界保持清楚，后续才能单独替换算子或 Cache。

### 把设计落实成代码：Qwen3-0.6B 逐文件顺序

下面是实际写代码时的顺序。每一步都先写当前文件，再运行对应的局部入口；不要一次创建全部空文件后再集中调试。Qwen3 的类名和实现可直接在链接文件中查看；新模型应保留 Engine 契约，只替换模型结构、权重映射和模态处理。

| 步骤 | 先写的文件 | 必须落地的代码 | 写完立即运行 |
| --- | --- | --- | --- |
| 1 | `layers/rope.py` | `RoPE.__init__` 校验维度并建立 cos/sin cache；`forward(pos, query, key)` 按位置旋转 Q/K，保留非旋转维度；文件尾的 `__main__` 用官方 RoPE 共享输入对照 | `python -m inference.models.qwen3_0_6b.layers.rope` |
| 2 | `layers/layers.py` | `Qwen3RMSNorm.forward(x)` 做 FP32 方差计算、归一化后乘可学习权重；`Qwen3MLP.forward(x)` 实现 `down_proj(act(gate_proj(x)) * up_proj(x))` | `python -m inference.models.qwen3_0_6b.layers.layers` |
| 3 | `components/paged_kv_cache.py` | 初始化 `[layer, block, K/V, kv_head, block_offset, head_dim]` 的设备张量；`bind/extend` 管 block table；`write` 按 `position_offset` 跨页写入；`read_batch` 按有效长度拼出历史 K/V；`release` 解除请求与物理页关联 | `python -m inference.models.qwen3_0_6b.components.paged_kv_cache` |
| 4 | `layers/attention.py` | 从 hidden state 计算 Q/K/V；Q/K 做 RMSNorm 和 RoPE；把 K/V 写入 Cache；读回历史 K/V；应用 causal mask 并计算 GQA；合并 heads 后过 `o_proj` | `python -m inference.models.qwen3_0_6b.layers.attention` |
| 5 | `model/qwen3.py` | 按配置构建 embedding、N 个 DecoderLayer、final norm 和 LM head；每层做 `x += attention(norm(x))`、`x += mlp(norm(x))`；最后取每条序列最后一个有效 token 的 logits | `python -m inference.models.qwen3_0_6b.model.qwen3` |
| 6 | `utils/loader.py`、`components/model_runner.py` | 校验本地 snapshot/revision；从 Safetensors 按 key 和 shape 装载权重；Runner 初始化模型、Tokenizer 和 Cache，并实现请求状态、Prefill、Decode、采样、EOS/长度结束与释放 | `python tests/qwen3_0_6b/test_components.py --config tests/qwen3_0_6b/test.yml` |
| 7 | `components/adapter.py`、`runtime.py` | Adapter 将 Engine work item 转给 Runner，保持返回顺序；Runtime 只负责组装 `ModelRuntime(runner, adapter)` | `inference validate --model qwen3-0.6b` |
| 8 | `components/tokenizer.py`、`encoder.py`、`decoder.py`、`detokenizer.py` | 依次完成 Chat Template→IDs、请求校验→`PreparedChatInput`、完整 decode、增量 decode；流式增量拼接必须等于完整 decode | 运行 `src/inference/models/qwen3_0_6b/example.py`，再跑 `test_end2end.py` |

以下三段是实现时要保持的关键代码结构。变量和类名与 Qwen3 对齐；如果新模型的接口不同，应转换到同一组 metadata 契约，而不是删去校验。

**Decoder Layer：先归一化，再加 Attention 残差，再归一化并加 MLP 残差。**

```python
def forward_batch(self, hidden_states, *, layer_index, request_ids,
                  token_lengths, position_offsets, block_tables, cache):
    attention_output = self.self_attn.forward_batch(
        self.input_layernorm(hidden_states),
        layer_index=layer_index,
        request_ids=request_ids,
        token_lengths=token_lengths,
        position_offsets=position_offsets,
        block_tables=block_tables,
        cache=cache,
    )
    hidden_states = hidden_states + attention_output
    return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
```

**Attention：写入本轮 K/V 后，读取到当前有效上下文长度，再做因果注意力。**

```python
query, key = self.rope(position_ids, query, key)
key_lengths = []
for index, (request_id, offset, length, blocks) in enumerate(
    zip(request_ids, position_offsets, token_lengths, block_tables, strict=True)
):
    key_lengths.append(cache.write(
        request_id=request_id,
        layer_index=layer_index,
        block_table=blocks,
        position_offset=offset,
        key=key[index, :, :length, :],
        value=value[index, :, :length, :],
    ))
key_states, value_states = cache.read_batch(
    layer_index=layer_index,
    request_ids=request_ids,
    block_tables=block_tables,
    lengths=key_lengths,
)
# attention_mask 需同时屏蔽 padding query、未写入的 KV 和未来 token。
# 后续再按模型的 GQA/MQA 头映射计算 softmax(QKᵀ / sqrt(head_dim))V。
```

**模型输出：batch metadata 必须与 input_ids 的 batch 对齐，返回最后有效位置 logits。**

```python
hidden_states = self.model.embed_tokens(input_ids)
for layer_index, layer in enumerate(self.model.layers):
    hidden_states = layer.forward_batch(
        hidden_states,
        layer_index=layer_index,
        request_ids=request_ids,
        token_lengths=token_lengths,
        position_offsets=position_offsets,
        block_tables=block_tables,
        cache=cache,
    )
hidden_states = self.model.norm(hidden_states)
batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
last_indices = torch.as_tensor(token_lengths, device=input_ids.device) - 1
return self.lm_head(hidden_states[batch_indices, last_indices])
```

每段结构在仓库中的完整实现分别见 [`Qwen3DecoderLayer`](src/inference/models/qwen3_0_6b/model/qwen3.py)、[`Qwen3Attention`](src/inference/models/qwen3_0_6b/layers/attention.py) 和 [`PagedKVCache`](src/inference/models/qwen3_0_6b/components/paged_kv_cache.py)。上面的片段解释控制流，不代替完整实现：完整代码还必须包含输入检查、GQA 形状处理、mask 构造、Cache 越界检查、设备/dtype 校验和异常清理。

**权重加载要显式校验，不要用 `strict=False` 掩盖结构不匹配。** 读取 Safetensors 前先对照模型 `state_dict()` 检查 checkpoint key、重复 key 和每个 tensor shape；所有必需 key 匹配后，再逐 tensor 复制到模型设备。只有经过解释并记录的 tied weights 或模型头差异才可以做映射。Qwen3 的参考实现是 [`utils/loader.py`](src/inference/models/qwen3_0_6b/utils/loader.py) 中的 `load_safetensors()`；新模型需根据自己的 checkpoint 命名写 key 映射，并在组件测试中断言缺失/多余 key 为空。

**Runner 和 Adapter 不写模型数学。** Runner 的实现顺序是：`load()` 加载模型 → `begin_generation()` 创建 request 状态 → `bind_generation_cache()` 接受 Engine 分配的页 → `execute_batch()` 按 `ExecutionPhase.PREFILL/DECODE` 执行并产出 `WorkerResult` → `finish_generation()` 返回 token IDs 和结束原因；取消或异常时统一释放 Cache 和 request 状态。Adapter 则实现 Worker 所需入口：`prepare_chat_generation()` 校验 alias 并构造 prompt，`bind_generation_cache()` / `execute_batch()` 委托 Runner，`finish_chat_generation()` 转为 `InferenceResult`，`cancel_generation()` 清理状态。可运行实现见 [`model_runner.py`](src/inference/models/qwen3_0_6b/components/model_runner.py) 与 [`adapter.py`](src/inference/models/qwen3_0_6b/components/adapter.py)。

逐层数值调试时，先用小随机配置验证公式，再用正式 checkpoint 验证权重映射。至少保存以下对照点，某一级失败就停在该层修复：

| 对照点 | 输入/输出 | 要断言什么 |
| --- | --- | --- |
| Tokenizer | 固定 messages → token IDs | 与 Hugging Face `apply_chat_template` IDs 完全一致 |
| Embedding/RoPE/Norm/MLP | 同一 tensor 和同一份权重 → hidden states | shape、dtype、device 一致，`assert_close` 在声明的容差内 |
| Attention | 固定 Q/K/V、位置与 mask → attention output | Prefill 和单 token Decode 分别与官方实现比较 |
| KV Cache | 写入位置和 block table → 读回 K/V | 跨 block 边界数据一致；不同 request 隔离；释放可复用 |
| 整层/整模型 | 同 checkpoint、同 token IDs → hidden/logits | Prefill 最后位置 logits 与 Decode logits 分别通过 `assert_close` |
| 生成协议 | greedy 参数与固定 prompt → token IDs | 首个不同 token 定位；EOS/max token stop reason 正确 |

不要只比较最终自然语言文本：相同文本可能来自不同 token 序列，数值或 Cache 错误也可能被掩盖。局部检查通过后，才进入完整服务集成。

`api.input_processor` 指向的类必须实现 `start()`、`prepare_chat()`、`close()`；当 `output_mode: token_ids` 时还必须实现 `decode_tokens()` 和 `create_stream_output_processor()`。启用 `tool_calling` 时，tool adapter 必须提供 `parse_output()` 和 `create_stream_parser()`。Launcher 的 `inference validate` 会检查这些方法是否存在。

每个可独立优化的 `layers/*.py` 和轻量 `components/*.py` 都要有模块级前向入口：随机种子固定、CUDA 优先、执行一次前向、检查输出 shape/dtype/finite；能对齐 Transformers 官方模块时，把权重复制到两边后用 `torch.testing.assert_close` 验证数值。例子：

```python
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(43)
    rope = RoPE(
        head_size=16, rotary_dim=16, max_pos_emb=64, base=10_000.0,
        dtype=torch.float32, device=device,
    )
    positions = torch.arange(5, device=device)
    query = torch.randn(1, 4, 5, 16, device=device)
    key = torch.randn(1, 2, 5, 16, device=device)
    with torch.inference_mode():
        rotated_query, rotated_key = rope(positions, query, key)
    assert rotated_query.shape == query.shape and rotated_key.shape == key.shape
    assert torch.isfinite(rotated_query).all() and torch.isfinite(rotated_key).all()
    print(f"RoPE forward OK: device={device}, shape={tuple(rotated_query.shape)}")
```

Qwen3 的可运行对照入口包括 [RoPE](src/inference/models/qwen3_0_6b/layers/rope.py)、[RMSNorm/MLP](src/inference/models/qwen3_0_6b/layers/layers.py)、[Attention](src/inference/models/qwen3_0_6b/layers/attention.py) 和 [PagedKVCache](src/inference/models/qwen3_0_6b/components/paged_kv_cache.py)。它们直接对照 Transformers 或检查 Cache 不变量；不是只打印一个 tensor 就算通过。完整模型与多组件仍由 `tests/<package>/test_components.py` 验收。

请求在系统中的执行顺序是：

```text
OpenAI Chat Request
  → Qwen3InputProcessor
  → Encoder (message / sampling validation)
  → Tokenizer (Qwen3 Chat Template → token IDs)
  → Engine Scheduler (Prefill / Decode work items)
  → Qwen3ModelAdapter → Qwen3ModelRunner
  → Qwen3ForCausalLM → Attention / RoPE / MLP + PagedKVCache
  → logits → Sampler → token ID
  → Decoder / Detokenizer → text delta or tool-call delta
```

Qwen3 使用 `custom` Runtime。它在 `runtime.py` 中把模型专属 Runner 和 Adapter 交给共用 Worker：

```python
from inference.engine.worker import ModelRuntime

from .components.adapter import Qwen3ModelAdapter
from .components.model_runner import Qwen3ModelRunner

def create_runtime() -> ModelRuntime:
    runner = Qwen3ModelRunner()
    return ModelRuntime(runner, Qwen3ModelAdapter(runner))
```

只有符合标准 Causal LM/Cache 行为的纯文本模型才适合复用共用 `StandardTextModelRunner`；其他模型实现自己的 Runner/Adapter，并遵守 `inference.engine.worker.ModelRuntime` 契约。

Qwen3 包入口 `__init__.py` 对外提供 Engine Client 工厂。新模型可照此实现，让使用者不用了解 Launcher 内部：

```python
from importlib import import_module
from pathlib import Path

from inference.engine.engine_client import EngineClient, create_engine_client_for_package


def create_engine_client(
    *,
    inference_config_path: str | Path | None = None,
    network_config_path: str | Path | None = None,
) -> EngineClient:
    return create_engine_client_for_package(
        import_module(__name__),
        inference_config_path=inference_config_path,
        network_config_path=network_config_path,
    )
```

`example.py` 验证库的 Python 调用路径。Qwen3 的最小流式流程如下；保留顶部导入、显式组件和默认流式打印：

```python
from pathlib import Path

from inference.models.qwen3_0_6b import create_engine_client
from inference.models.qwen3_0_6b.components.decoder import Qwen3Decoder
from inference.models.qwen3_0_6b.components.encoder import Qwen3Encoder
from inference.models.qwen3_0_6b.components.tokenizer import Qwen3Tokenizer

INFERENCE_CONFIG = Path(__file__).resolve().parent / "configs" / "inference.yml"
engine = create_engine_client(inference_config_path=INFERENCE_CONFIG)
tokenizer = Qwen3Tokenizer()
encoder = Qwen3Encoder()
decoder = Qwen3Decoder(tokenizer)
request = {
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "介绍一下 Paged KV Cache。"}],
    "max_tokens": 64,
}

engine.start()
tokenizer.start()
plan = encoder.prepare(request)
token_ids = tokenizer.tokenize_chat(
    plan.messages, tools=plan.template_tools, enable_thinking=plan.enable_thinking
)
prepared_input = encoder.build(plan, token_ids)
detokenizer = decoder.create_detokenizer()
stream = engine.stream_chat(request, prepared_input)
for event in stream:
    if event.kind == "token" and event.token_id is not None:
        print(detokenizer.push(event.token_id), end="", flush=True)
    elif event.kind == "finish":
        print(detokenizer.finish(), end="", flush=True)
print()
stream.close()
tokenizer.close()
engine.close()
```

新模型的 `example.py` 需要按自己实际的 Tokenizer/Encoder/Decoder 名称调整；不要为了示例把它们合并成一个大 `InputProcessor`。

## 5. 配置、安装并预检

`configs/model.yml` 声明模型别名、运行工厂及真实能力；`configs/inference.yml` 固定模型 ID、revision、dtype、设备和本地权重路径；`configs/network.yml` 配置 API 地址。可复制最相近模型的配置后修改，确保三份配置与权重 revision 一致。

Qwen3 的 `configs/model.yml` 声明自定义 Runtime、Tokenizer 处理器和工具调用能力：

```yaml
alias: qwen3-0.6b
display_name: Qwen3-0.6B
integration_profile: custom
backend: qwen3_native_torch
runtime_factory: inference.models.qwen3_0_6b.runtime:create_runtime
inference_config: inference.yml
network_config: network.yml
api:
  model_id: qwen3-0.6b
  input_processor: inference.models.qwen3_0_6b.processor:Qwen3InputProcessor
  capabilities:
    chat_inputs: [text]
    streaming: true
    tool_calling: true
    output_mode: token_ids
    video_upload: false
    live_video: false
  tool_call_adapter: inference.models.qwen3_0_6b.tool_calling:Qwen3ToolCallAdapter
```

Qwen3 的 `configs/inference.yml` 固定 revision、离线权重路径和运行参数。容量值是该模型配置，不应原样套用到其他模型：

```yaml
model:
  id: Qwen/Qwen3-0.6B
  revision: c1899de289a04d12100db370d81485cdf75e47ca
  dtype: bfloat16
  device: cuda:0
runtime:
  platform: torch_cuda
  local_model_path: checkpoints/Qwen3-0.6B-c1899de
  local_files_only: true
  worker_startup_timeout_seconds: 900
  worker_request_timeout_seconds: 300
  engine_core_startup_timeout_seconds: 930
  engine_core_request_timeout_seconds: 330
api:
  max_chat_request_bytes: 2097152
  max_context_tokens: 8192
engine:
  max_num_seqs: 4
  max_waiting_requests: 16
  max_num_scheduled_tokens: 8192
  batch_wait_seconds: 0.01
  prefill_chunk_size_tokens: 2048
  kv_cache:
    block_size_tokens: 16
    num_blocks: 1024
generation:
  max_new_tokens: 256
  temperature: 0.0
  top_p: 0.8
  top_k: 20
  min_p: null
  repetition_penalty: 1.0
  no_repeat_ngram_size: 0
  num_beams: 1
  enable_thinking: false
```

`network.yml` 示例：

```yaml
api:
  host: 127.0.0.1
  port: 8011
```

依赖只装该模型使用的包。Qwen3 当前的 `requirements.txt` 是：

```text
transformers==5.19.0
accelerate==1.15.0
safetensors==0.8.0
jsonschema==4.26.0
```

Qwen3 的配置副本位于 `scripts/qwen3_0_6b/`，其服务配置位于 `src/inference/models/qwen3_0_6b/configs/`。新模型要把自己的配置放在对应目录，不能在脚本中读取另一个模型的配置。

在 `src/inference/pyproject.toml` 注册模型专属依赖，PyTorch/CUDA 由环境单独安装：

Qwen3 的 `qwen3` extra 已在该表注册。新模型在同一个表中新增一行，例如：

```toml
my_model = { file = ["models/my_model/requirements.txt"] }
```

然后安装并预检：

```bash
python -m pip install -e "src/inference[qwen3]"
python -m pip check
inference list
inference validate --model qwen3-0.6b
```

`validate` 通过后，再编写或适配 `tests/<package>/test.yml`、`test_components.py`、`test_performance.py` 和 `test_end2end.py`。测试脚本也由模型负责人按能力维护，不会自动生成。Qwen3 测试配置如下；相对路径以 `test.yml` 所在目录为基准：

```yaml
model_alias: qwen3-0.6b
revision: c1899de289a04d12100db370d81485cdf75e47ca
snapshot: ../../checkpoints/Qwen3-0.6B-c1899de
inference_config: ../../src/inference/models/qwen3_0_6b/configs/inference.yml
inference_url: http://127.0.0.1:8011
gateway_url: http://127.0.0.1:8000
api_key_env: VIDEO_API_KEY
prompt: "请用一句话说明大型语言模型推理服务。"
max_new_tokens: 32
warmup_requests: 1
iterations: 3
timeout_seconds: 300
component:
  engine_module: ../../src/inference
  dtype: bfloat16
operator_analysis:
  batch_size: 1
  prompt_tokens: 512
  warmup_iterations: 2
  iterations: 5
  top_operators: 20
```

### 每个测试文件要写什么

| 文件 | 必须写入的代码 | 明确的通过断言 | 可逐段对照的仓库实现 |
| --- | --- | --- | --- |
| `test_components.py` | 解析 `test.yml`；读取固定快照 Config；构造 HF 基线和模型自有实现；逐 key/shape 检查并加载权重；按层检查 Tokenizer、RoPE、Attention、Cache、Sampler；对相同输入前向 | 加载权重无缺失/多余项；输出 shape/dtype 正确且 finite；Cache 写读/释放后状态正确；HF 与自有 logits 在定义的 `atol/rtol` 内 | [Qwen3 test_components.py](tests/qwen3_0_6b/test_components.py) |
| `test_performance.py` | 读同一个 `test.yml`；独立加载 HF 和运行中的 Inference API；分别做 warmup 与正式计时；GPU 计时同步；保存每轮延迟/token 数并输出 JSON | 两条路径都返回非空结果和正 token 数；报告 median 延迟与 tokens/s；运行配置和输入一致 | [Qwen3 test_performance.py](tests/qwen3_0_6b/test_performance.py) |
| `test_end2end.py` | 读取 Gateway URL/alias/key；发真实 OpenAI Chat Completions SSE 请求；逐行解析 `data:` JSON；收集增量文本和结束事件 | HTTP 200；返回 alias 正确；至少一个文本 delta；无错误事件；观察到 `finish_reason` 和 `[DONE]` | [Qwen3 test_end2end.py](tests/qwen3_0_6b/test_end2end.py) |

按这个顺序把代码写进测试文件：先让 `test_components.py` 在不启动服务时通过，再让 `test_performance.py` 对运行中的 Inference 完成 HF/API 对照，最后启动 Gateway 运行 SSE 验收。不要把测试写成只检查进程存活或只打印结果；失败条件必须通过 Python `assert` 或抛出异常让进程以非零状态退出。Qwen3 的三个测试文件是可直接运行的成品参考，应复制测试逻辑后替换模型专属 Config、类名、输入格式和预期数值范围。

端到端 SSE 循环必须明确记录终止条件。收到文本 delta、结束原因和 `[DONE]` 是三个独立事实，不能只因为连接关闭就判定成功：

```python
saw_text_delta = False
saw_finish_reason = False
saw_done = False
for line in response:
    if not line.startswith(b"data:"):
        continue
    data = line[5:].strip()
    if data == b"[DONE]":
        saw_done = True
        break
    event = json.loads(data)
    if event.get("error"):
        raise RuntimeError(f"Gateway SSE error: {event['error']}")
    for choice in event.get("choices", []):
        delta = choice.get("delta", {})
        saw_text_delta |= bool(delta.get("content"))
        saw_finish_reason |= choice.get("finish_reason") is not None

assert saw_text_delta, "SSE 没有返回文本增量"
assert saw_finish_reason, "SSE 缺少 finish_reason 事件"
assert saw_done, "SSE 流没有以 [DONE] 结束"
```

仓库当前 [Qwen3 SSE 示例](tests/qwen3_0_6b/test_end2end.py)检查 HTTP、文本 delta、alias 和错误事件，并在读到 `[DONE]` 时停止；把它适配到新模型时，还应按上面代码显式断言终止标记及 `finish_reason`，避免连接提前关闭被误判为通过。

新模型测试要替换 Qwen3 专属 import 和配置，不要只改目录名。组件测试发现数值不一致时先定位层级：Tokenizer IDs、Embedding、单层 Attention、整层 logits、完整模型 logits；不要直接用 API 最终文本相似度代替算子正确性。

`tests/<package>/test_components.py`、`test_performance.py`、`test_end2end.py` 采用可直接运行的脚本：常量在顶部，测试步骤和断言直接写在文件主体，不再包一层测试框架或辅助方法。共用输入参数放在 `test.yml`，避免相同 prompt/revision 在三个脚本中各自漂移。

Qwen3 的模块级算子 Profile 可单独运行；`--operator-report` 写入机器可读 JSON：

```bash
python tests/qwen3_0_6b/test_components.py \
  --config tests/qwen3_0_6b/test.yml \
  --profile-attention \
  --operator-report artifacts/qwen3-0.6b-attention.json
```

该报告与整模型 `scripts/qwen3_0_6b/forward.py` 的 Profile 目标不同：前者对照指定 Attention 层和合成输入，后者对照 Prefill/Decode 的整体执行路径。两种结果不要混成一项指标。

## 6. 集成验收

先运行组件测试：

```bash
python tests/qwen3_0_6b/test_components.py --config tests/qwen3_0_6b/test.yml
```

再在终端 A 启动 Inference（该命令持续运行）：

```bash
inference start --model qwen3-0.6b
```

终端 B 运行服务性能测试：

```bash
python tests/qwen3_0_6b/test_performance.py --config tests/qwen3_0_6b/test.yml
```

如需 Gateway 端到端验收，按 `src/gateway/README.md` 配置并启动 Gateway，在终端 C 设置 API Key 后运行：

```bash
export VIDEO_API_KEY='<Gateway API Key>'
python tests/qwen3_0_6b/test_end2end.py --config tests/qwen3_0_6b/test.yml
python tests/client.py
```

Qwen3 声明了 `tool_calling: true`，因此还要运行 `tests/client.py` 验证工具调用；该示例由客户端执行工具。

性能结果与接入前基线使用相同硬件、dtype、输入和测量方式对照。API 至少验证非流式和流式请求；多模态模型还要为声明的每种输入准备真实素材并验收。Health 检查不能替代生成请求。未实际运行的项标为未验证。

## 7. 常见问题

| 现象 | 检查 |
| --- | --- |
| 下载 401/403 | Hub 权限、模型许可和 `HF_TOKEN` |
| `inference validate` 失败 | alias、配置路径、Runtime 工厂和能力声明 |
| 找不到权重 | 本地 snapshot 路径、revision 和文件是否完整 |
| CUDA OOM | dtype、上下文长度、并发、KV 容量及其他 GPU 进程 |
| Health 正常但生成失败 | Worker 日志、Tokenizer/Processor、模型 forward 和 Cache |

## 官方资料

- Hugging Face Hub [固定 revision 下载](https://huggingface.co/docs/huggingface_hub/en/guides/download)
- Transformers [Auto Classes](https://huggingface.co/docs/transformers/model_doc/auto)、[Processors](https://huggingface.co/docs/transformers/processors)、[Benchmarks](https://huggingface.co/docs/transformers/main/benchmarks)
- [Qwen3-0.6B 固定 revision](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca)
