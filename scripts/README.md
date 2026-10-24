# 模型下载与探索工具

共用 Hugging Face 下载器保留在 `scripts/download.py`。每个模型的探索、诊断及能力撷取工具放在 `scripts/<model_alias>/`，避免不同模型的设定、输入格式和报告逻辑混在一起。目前工具位于 `scripts/mage_vl_4b/` 及 `scripts/qwen3_0_6b/`。

## Hugging Face 模型下载工具

`download.py` 会从 Hugging Face Hub 下载指定模型 revision 的档案，并放入明确指定的本机目录。预设下载该 revision 的完整快照，包含权重、设定、Tokenizer 与模型程式码等档案；下载中断后再次执行会沿用 Hub 快取及目标目录中的既有档案。

## 安装依赖

在专案根目录执行：

```bash
python -m pip install "huggingface_hub==1.33.0"
```

此版本与 Mage-VL Transformers 环境目前锁定的 `huggingface_hub` 版本一致。

## 下载完整模型

每次执行都必须明确提供模型 `repo_id`、revision 与下载目录。建议 revision 使用 commit SHA，确保后续可重现：

```bash
python scripts/download.py \
  --repo-id microsoft/Mage-VL \
  --revision d88b153285f1633a61b2f693c59c8576693af185 \
  --local-dir checkpoints/microsoft-Mage-VL-d88b153
```

`--local-dir` 是模型档案的实际目标目录。专案根目录的 `.gitignore` 会忽略 `checkpoints/` 下的模型权重与快取，同时保留 `checkpoints/val.md` 及 `checkpoints/val.xlsx`。

## 私有模型与存取权杖

公开模型不需要权杖。下载私有或受限模型时，先将 Hugging Face 权杖放入环境变数，不要直接写进命令列或文件：

```bash
export HF_TOKEN=hf_your_token
python scripts/download.py \
  --repo-id organization/private-model \
  --revision 0123456789abcdef0123456789abcdef01234567 \
  --local-dir checkpoints/private-model
```

若使用的环境变数名称不是 `HF_TOKEN`，以 `--token-env NAME` 指定名称。

## 选择档案与下载参数

可重复提供 `--allow-pattern`，让 Hub 只下载符合模式的档案；也可用 `--ignore-pattern` 排除档案。若没有指定模式，则下载完整快照。部分下载可能缺少模型执行所需的设定或程式码，使用前请确认模型格式与推论服务的需求。

```bash
python scripts/download.py \
  --repo-id microsoft/Mage-VL \
  --revision d88b153285f1633a61b2f693c59c8576693af185 \
  --local-dir checkpoints/mage-vl-config-only \
  --allow-pattern '*.json' \
  --allow-pattern '*.py' \
  --ignore-pattern '*safetensors*'
```

`--max-workers N` 可调整并行下载数，预设为 `8`。需要重新抓取本机已有档案时，加入 `--force-download`。执行 `python scripts/download.py --help` 可查看完整参数。

Qwen3-0.6B 固定使用官方 `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`。从专案根目录下载完整快照：

```powershell
python scripts/download.py `
  --repo-id Qwen/Qwen3-0.6B `
  --revision c1899de289a04d12100db370d81485cdf75e47ca `
  --local-dir checkpoints/Qwen3-0.6B-c1899de
```

下载目录被 `.gitignore` 排除，不会提交模型权重。

## 撷取 Mage-VL 能力资讯

`scripts/mage_vl_4b/test.py` 是 Mage-VL 专属的本机探索工具，可记录固定 revision 的模型类别与结构、处理后输入 tensor、逐步 KV cache、GPU 记忆体及 prefill／decode 计时。每次执行都必须明确提供 `--config`、`--model-path`、`--image` 或 `--video`、`--question` 及 `--output`；脚本不从自身位置推导专案目录，也不预设同目录 YAML。所有相对路径均以启动命令时的目前工作目录解析。YAML 沿用正式服务的 `model`、`processor`、`runtime`、`api`、`generation` 和 `optimization` 区段，独立调整探索时使用的模型与输入限制。输出 token 数预设取自明确指定 YAML 的 `generation.max_tokens_default`，`--max-new-tokens` 可覆写。工具不从 `src/` 汇入程式，推理流程直接使用已安装的 PyTorch、Transformers 与明确指定的本机模型快照。

脚本只接受本机图片或影片，不呼叫推论 HTTP API。它会强制设定 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，并以 `local_files_only=True` 从 `--model-path` 指定的本机快照载入 Processor 与 Model。模型目录无效时会报错退出，不会搜寻快取或下载模型。执行环境需具备推理环境依赖、CUDA 和模型快照；影片探测另需 OpenCV。

从专案根目录执行：

```powershell
python scripts/mage_vl_4b/test.py `
  --config scripts/mage_vl_4b/test.yaml `
  --model-path checkpoints/microsoft-Mage-VL-d88b153 `
  --image path/to/frame.jpg `
  --question "描述画面中的人物正在做什么。" `
  --max-new-tokens 8 `
  --output artifacts/mage-vl-image-probe.json

python scripts/mage_vl_4b/test.py `
  --config scripts/mage_vl_4b/test.yaml `
  --model-path checkpoints/microsoft-Mage-VL-d88b153 `
  --video path/to/clip.mp4 `
  --question "总结影片中发生的事件。" `
  --max-new-tokens 8 `
  --output artifacts/mage-vl-video-probe.json
```

`scripts/mage_vl_4b/test.yaml` 预设依目前 RTX 4060 8 GiB 显卡采用 int8 CPU offload、GPU 5 GiB／CPU 6 GiB 放置上限、448 px 输入与最多 8 个输出 token；这是探索工具的低负载设定，不等于正式服务预设。`--config` 必须指向相同结构且使用 `video_backend: frames` 的 Mage-VL YAML，`--model-path` 必须指向已存在的本机快照目录。若从其他目录启动，请传入相对于该目前工作目录的路径或绝对路径。输出 JSON 不包含媒体 bytes 或模型生成文字，但包含模型路径、tensor metadata、计时和错误 traceback；请将报告留在本机 `artifacts/`，不要纳入 Git。生成使用模型原生 `generate()`，并以 forward hook 记录 prefill／decode 的耗时及 cache 长度。已在 RTX 4060、固定 Mage-VL revision 和本机图片上完成一次低负载 GPU 探测；这项探测用于检查模型执行介面和 cache 行为，不是模型精度评估或正式性能基线。

## 撷取 Qwen3-0.6B 能力资讯

`scripts/qwen3_0_6b/test.py` 只接受明确指定的本机权重、YAML、prompt 与 JSON 报告路径，不从 `src/` 汇入程式，也不会下载模型。它使用 Qwen chat template 执行文字生成，记录模型结构、权重载入时间、prompt／completion token 数、生成速度及 GPU 峰值记忆体。

```powershell
python scripts/qwen3_0_6b/test.py `
  --config scripts/qwen3_0_6b/test.yaml `
  --model-path checkpoints/Qwen3-0.6B-c1899de `
  --prompt "用两句话说明什么是推理引擎。" `
  --output artifacts/qwen3-0.6b-probe.json
```

输出 JSON 包含生成文字，应保留于本机 `artifacts/`，不要纳入 Git。此探测是单请求 GPU smoke test，不是精度评估或并发性能基线。

## Qwen3 API 性能采样

`scripts/qwen3_0_6b/benchmark_api.py` 只透过 OpenAI Chat Completions HTTP/SSE 呼叫已启动的 API，不汇入 `src/` 或载入第二份模型。每次执行须明确传入 API 根网址、模型别名、prompt、请求数、并发数、输出上限及 JSON 报告路径；报告记录 TTFT、inter-token latency、request/token throughput、request latency、硬体、套件版本与 `/health` 暴露的 KV／显存状态。

```powershell
python scripts/qwen3_0_6b/benchmark_api.py `
  --base-url http://127.0.0.1:8011 `
  --model qwen3-0.6b `
  --prompt "用两句话说明推理引擎如何处理多个请求。" `
  --requests 4 `
  --concurrency 2 `
  --max-tokens 32 `
  --output artifacts/qwen3-0.6b-api-baseline.json
```

这是服务端到端的短期基线，不是模型精度评测或 SLA。每个 SSE 可见文字 chunk 代表一次输出事件；Tokenizer special token 不产生可见文字时不计入 TTFT／inter-token 样本。将报告留在本机 `artifacts/`，不要纳入 Git。
