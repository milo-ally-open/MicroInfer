# 新模型接入教程

本教程默认使用仓库中的 **Qwen3-VL-2B-Instruct**，按 **Test → Bench → Profile → Engine 集成 → 服务验收** 讲解新模型接入。命令从仓库根目录的 WSL/Linux Bash、inference Conda 环境执行。

本模型包支持文字、图片、流式文字输出；工具调用、视频上传和实时视频未实现，能力必须为 false。仓库的 [Qwen3-0.6B](src/inference/models/qwen3_0_6b/) 保留为纯文字、自研 PyTorch 对照，不再是默认案例。

## 1. Qwen3-VL 的实现边界

模型执行复用 Transformers 官方 Qwen3VLForConditionalGeneration、AutoProcessor 和 DynamicCache。项目代码实现消息/图片校验、Processor 调用、逐步生成、Engine Runner/Adapter 和流式输出；不复制官方视觉塔或语言模型层，也不宣称自研 Kernel。

<pre><code>
Chat 请求
  → Encoder：验证消息、图片和生成参数
  → AutoProcessor：Chat Template、tokenize、视觉预处理
  → Scheduler：完整图文 Prefill，之后每步一个 Decode token
  → Runner：官方模型前向、DynamicCache、MRoPE 状态
  → Decoder / Detokenizer：完整文本或增量文本
  → Chat Completions / SSE
</code></pre>

当前 max_num_seqs 为 1，隔离模型级 MRoPE 状态。Engine 预留逻辑 KV blocks，实际 K/V 由 Transformers DynamicCache 管理，不是实体 paged KV 内存池。

## 2. 固定模型版本、硬件和权重

| 项目 | 当前案例 |
| --- | --- |
| Hub repo | Qwen/Qwen3-VL-2B-Instruct |
| 完整 revision | 89644892e4d85e24eaac8bacfd4f463576704203 |
| Python 包 / API alias | qwen3_vl_2b / qwen3-vl-2b |
| 模型依赖 | Transformers 5.19.0、Accelerate 1.15.0、Safetensors 0.8.0 |
| 已验收环境 | PyTorch 2.14.1+cu126、BF16、RTX 4060 Laptop GPU |

固定 revision 的模型卡声明 Apache-2.0；部署方仍需确认许可证适用性。[固定 revision 模型卡](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/89644892e4d85e24eaac8bacfd4f463576704203/README.md)

CUDA 可用时默认用 CUDA，只有 CUDA 不可用时才 fallback CPU。OOM 应通过资源参数处理，不能静默回退 CPU。

<pre><code>
conda activate inference
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python scripts/download.py --repo-id Qwen/Qwen3-VL-2B-Instruct --revision 89644892e4d85e24eaac8bacfd4f463576704203 --local-dir checkpoints/Qwen3-VL-2B-Instruct-8964489
</code></pre>

权重放在 .gitignore 忽略的 checkpoints/，不可提交。下载后确认 config、Safetensors、Tokenizer、Processor 和 Chat Template 文件完整；服务配置启用 local_files_only。

## 3. Engine 集成前完成上游 Test、Bench、Profile

固定同一 revision、图片、prompt、dtype 和 attention 实现。先验证 Transformers 本身和工作负载，再开始 Engine 接入。

| 门禁 | 文件 | 需要写的代码 | 运行命令 |
| --- | --- | --- | --- |
| Test | tests/qwen3_vl_2b/test_components.py | 官方 Processor 图文编码；检查 input_ids、pixel_values、image_grid_thw；Prefill logits/Cache；生成非空；stepwise token 与 HF generate 对齐；纯文本输入 | python tests/qwen3_vl_2b/test_components.py |
| Bench | tests/qwen3_vl_2b/test_performance.py | warmup、多轮图文生成、CUDA 同步、真实输出 token、median latency、吞吐、峰值显存和 JSON | python tests/qwen3_vl_2b/test_performance.py |
| Profile | scripts/qwen3_vl_2b/forward.py | Prefill 和单 token Decode 分别采集；记录 shape/memory；导出 Chrome trace 与算子记录 | python scripts/qwen3_vl_2b/forward.py |

### Profile 要验证单 token Decode

Prefill 用全部图文 prompt，保存 past_key_values。Decode 每步只送一个新 token。Qwen3-VL 有 MRoPE 和 rope_deltas 状态，手动续算必须用官方 generation 输入准备逻辑：

<pre><code>
prefill = model(**model_inputs, use_cache=True, return_dict=True)
next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
decode_ids = torch.cat((model_inputs["input_ids"], next_token), dim=-1)
generation_kwargs = {
    **vision_inputs,
    "attention_mask": torch.ones_like(decode_ids),
    "past_key_values": prefill.past_key_values,
}
position_ids = model._prepare_position_ids_for_generation(decode_ids, generation_kwargs)
decode_inputs = model.prepare_inputs_for_generation(
    input_ids=decode_ids,
    next_sequence_length=1,
    past_key_values=prefill.past_key_values,
    attention_mask=generation_kwargs["attention_mask"],
    position_ids=position_ids,
    use_cache=True,
    is_first_iteration=False,
    **vision_inputs,
)
assert decode_inputs["input_ids"].shape[-1] == 1
decode = model(**decode_inputs, return_dict=True)
assert decode.past_key_values.get_seq_length() == prefill.past_key_values.get_seq_length() + 1
</code></pre>

Profile 的通过条件是 query 长度 1、Cache 长度加 1、logits shape 合法且 finite。shape 错误时先记录实际 inputs/position IDs/rope_deltas/vision kwargs/Cache 长度和 traceback，再对照 Transformers generate 路径；不能只看脚本退出码。

仓库 test.jpg 是 960×540，Processor 输出 528 prompt tokens。在 RTX 4060 Laptop GPU、BF16 上，三轮基线 median 1.638 秒、19.530 tokens/s；Decode query=1、Cache 528→529。样本量有限，不是 SLA；Profiler 时间不可作为 Bench latency。原始 JSON 在 artifacts/，analysis.xlsx 由 scripts/qwen3_vl_2b/analysis.py 生成。

## 4. 模型目录与职责

<pre><code>
src/inference/models/qwen3_vl_2b/
├── __init__.py                  # Engine Client 工厂
├── runtime.py                   # 组合 Runner 和 Adapter
├── example.py                   # 图片输入的流式示例
├── requirements.txt
├── configs/{model,inference,network}.yml
├── utils/loader.py              # 加载所选配置和本地 snapshot
└── components/
    ├── image_encoder.py         # 图片解码、格式/大小限制
    ├── tokenizer.py             # AutoProcessor Chat Template
    ├── encoder.py               # 消息、图片、context、sampling 校验
    ├── stepwise_generation.py   # Prefill、Decode、Cache、MRoPE
    ├── model_runner.py          # 官方模型及 Engine 执行入口
    ├── adapter.py               # Engine 请求和结果适配
    ├── decoder.py               # 完整解码
    └── detokenizer.py           # 增量解码
scripts/qwen3_vl_2b/              # forward.py、analysis.py、配置副本
tests/qwen3_vl_2b/                # test.yml、三类测试脚本
analysis/qwen3_vl_2b/             # analysis.md、脚本生成的 analysis.xlsx
</code></pre>

| 文件 | 编写内容 | 验收点 |
| --- | --- | --- |
| image_encoder.py | 解码图片 URL/data URL；限制格式、字节、像素、边长；转 RGB | 非法或超限输入明确拒绝 |
| tokenizer.py | 调官方 apply_chat_template，保留视觉张量和 grid metadata | input_ids、pixel_values、image_grid_thw 齐全 |
| encoder.py | 规范消息、校验 context/sampling、转换 tensor device/dtype | 文本和图像输入均可编码 |
| stepwise_generation.py | 每请求状态；完整 Prefill；单 token Decode；Cache、MRoPE、EOS/max token、取消清理 | 确定性输出与 HF generate 对齐 |
| model_runner.py | 固定 snapshot 加载官方模型/Processor；按 Scheduler 阶段执行；返回 WorkerResult | 当前单 active sequence，生命周期正确 |
| adapter.py | alias/参数校验；请求/结果转换；错误、结束、取消清理 | 共用 Engine 契约通过 |
| decoder.py/detokenizer.py | 完整解码、增量解码、尾部 flush | 增量拼接等于完整文本 |
| runtime.py/__init__.py | ModelRuntime 组合、Engine Client 工厂 | validate 和 Python 调用通过 |

复用官方 Transformers 网络时不要创建空 model/、layers/ 目录。只有自行实现或优化视觉/语言算子时才拆这些组件，并用同权重、同输入逐层对照。

runtime.py 核心组合如下：

<pre><code>
from inference.engine.worker import ModelRuntime
from .components.adapter import Qwen3VLModelAdapter
from .components.model_runner import Qwen3VLModelRunner

def create_runtime() -> ModelRuntime:
    runner = Qwen3VLModelRunner()
    return ModelRuntime(runner, Qwen3VLModelAdapter(runner))
</code></pre>

包入口通过 create_engine_client_for_package() 导出 Engine Client，沿用现有 /v1/chat/completions，不增加模型专属 endpoint。

## 5. 配置和模型依赖

包配置放 src/inference/models/qwen3_vl_2b/configs/；scripts/qwen3_vl_2b/ 内存放独立的 model.yml、inference.yml 和 network.yml 副本，不从 src/ 读取配置。

model.yml 核心配置：

<pre><code>
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
</code></pre>

inference.yml 固定 revision、dtype、设备、权重路径和服务容量。仓库当前关键配置如下，容量必须按部署 GPU 重测：

<pre><code>
model:
  id: Qwen/Qwen3-VL-2B-Instruct
  revision: 89644892e4d85e24eaac8bacfd4f463576704203
  dtype: bfloat16
  device_map: 0
  attn_implementation: sdpa
  use_safetensors: true
  weights_only: true
runtime:
  local_model_path: checkpoints/Qwen3-VL-2B-Instruct-8964489
  local_files_only: true
api:
  max_context_tokens: 8192
  max_new_tokens: 256
  max_images_per_request: 4
  max_image_bytes: 10485760
  max_image_pixels: 16777216
  max_image_side: 1344
engine:
  max_num_seqs: 1
  max_waiting_requests: 4
  prefill_chunk_size_tokens: 8192
  kv_cache:
    block_size_tokens: 16
    capacity_tokens: 8192
generation:
  max_tokens_default: 128
  temperature: 0.0
  top_p: 0.8
  top_k: 20
  num_beams: 1
  use_cache: true
  advanced_kwargs: {}
</code></pre>

Engine capacity_tokens 是调度预留值，不是物理 paged KV；max_num_seqs=1 是 MRoPE 隔离限制。network.yml 监听 127.0.0.1:8012。

requirements.txt：

<pre><code>
transformers==5.19.0
accelerate==1.15.0
safetensors==0.8.0
</code></pre>

在 src/inference/pyproject.toml 加模型 extra：

<pre><code>
qwen3-vl-2b = { file = ["models/qwen3_vl_2b/requirements.txt"] }
</code></pre>

PyTorch/CUDA 单独安装后执行：

<pre><code>
python -m pip install -e "src/inference[qwen3-vl-2b]"
python -m pip check
inference list
inference validate --model qwen3-vl-2b
</code></pre>

validate 只检查静态配置和接口，不代替权重加载与真实生成。

## 6. 测试和服务验收

tests/qwen3_vl_2b/test.yml 是测试参数唯一来源，包含模型 alias、revision、snapshot、图片、Inference URL、prompt、轮数、dtype、attention。

按顺序运行：

<pre><code>
python tests/qwen3_vl_2b/test_components.py
python tests/qwen3_vl_2b/test_performance.py
python scripts/qwen3_vl_2b/forward.py
python scripts/qwen3_vl_2b/analysis.py
inference validate --model qwen3-vl-2b
</code></pre>

终端 A 启动，终端 B 验收：

<pre><code>
inference start --model qwen3-vl-2b
python tests/qwen3_vl_2b/test_end2end.py
</code></pre>

端到端脚本验证文本非流式及图片 SSE：HTTP 200、alias、非空内容、usage、text delta、finish_reason、[DONE]。health 成功不能替代生成请求。Gateway 发布时另外验证认证与路由。本模型没有 tool/video/live 能力，不应宣称这些功能通过。

模型 Python 示例位于 src/inference/models/qwen3_vl_2b/example.py，使用 test.jpg 并流式打印。所有 imports 在顶部，启动代码放进 if __name__ == "__main__":，避免 multiprocessing spawn 重新导入时重复启动；不需封装 main()。

## 7. 问题复现和通用检查表

| 问题 | 先记录 | 排查 |
| --- | --- | --- |
| Decode query 不为 1 | 实际 decode shape、prompt token、next_sequence_length 实参 | 是否重新输入完整 prefix；官方 hook 是否收到 1 |
| 手动续 Cache shape 错 | position IDs、rope_deltas、vision kwargs、Cache 长度、traceback | 是否调用官方 MRoPE 与 prepare_inputs_for_generation |
| Cache 长度不符 | Prefill/Decode 前后的 Cache length | 是否复用 Cache，Decode 是否只计算新增 token |
| 图像 shape 错 | 原图尺寸、Processor 字段和 tensor shape/dtype/device | Chat Template、grid metadata、视觉 tensor 是否完整 |
| CUDA OOM | GPU、dtype、输入 token、并发、峰值显存 | 缩小图片/上下文/输出/并发后复测，不静默回退 CPU |

发现代码问题时使用 [problem-representing-workflow Skill](.codex/skills/problem-representing-workflow/SKILL.md)：记录复现、预期/实际和环境；修改前提交基线；每个假设独立改动、验证、留提交；最后以 YAML 关联尝试、命令、结果和前后 commit。失败尝试也保留，不先改后补造基线。

新模型统一门禁：固定版本和依赖 → 上游 Test → Bench → Profile → Engine 组件 → 配置/extra 注册 → HTTP/SSE 真实请求。Test/Bench/Profile 脚本由模型负责人维护，tests 和 scripts 分开。analysis.xlsx 由脚本从数值数据生成；问题、方案、效果、限制写进 analysis.md。未执行项标记未验证。

## 官方资料

- [Qwen3-VL-2B-Instruct 固定 revision 文件树](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/89644892e4d85e24eaac8bacfd4f463576704203)
- [固定 revision 模型卡](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/89644892e4d85e24eaac8bacfd4f463576704203/README.md)
- [Transformers Qwen3-VL 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)
- [Transformers Cache 文档](https://huggingface.co/docs/transformers/en/kv_cache)
- [Hugging Face Hub revision 下载文档](https://huggingface.co/docs/huggingface_hub/en/guides/download)
