# 模型下载与离线分析

`scripts/` 提供通用模型下载器，以及按模型组织的本地前向和算子分析工具。逐模型组件、性能与端到端测试属于 `tests/`，两类脚本分开维护。

## 模型下载

下载器要求显式指定上游 Hugging Face `repo_id`、固定 revision 和本地目录。项目别名不能替代上游 `repo_id`。

```bash
python scripts/download.py \
  --repo-id Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir checkpoints/Qwen3-0.6B-c1899de

python scripts/download.py \
  --repo-id microsoft/Mage-VL \
  --revision d88b153285f1633a61b2f693c59c8576693af185 \
  --local-dir checkpoints/microsoft-Mage-VL-d88b153
```

运行 `python scripts/download.py --help` 查看全部参数。公开模型通常不需要 token；下载受限模型时，通过环境变量提供 Hugging Face token。权重和缓存保存在根目录 `checkpoints/`，该目录不会纳入 Git。

## 逐模型目录

```text
scripts/
├── download.py
├── mage_vl_4b/
│   ├── forward.py
│   ├── analysis.py
│   └── model.yml / inference.yml / network.yml
└── qwen3_0_6b/
    ├── forward.py
    ├── analysis.py
    └── model.yml / inference.yml / network.yml
```

每个模型目录保存一份独立配置副本；模型插件实际配置仍位于 `src/inference/models/<model>/configs/`。修改分析配置时，请确认脚本副本和模型配置意图一致。模型专属脚本只使用本地权重，不从 `src/` 导入脚本模块。

## 前向与 Profile

`forward.py` 使用本地权重分别执行 Transformers 和项目 Inference 的 Prefill、单步 Decode。每段都会写 PyTorch Profiler 算子记录，并导出带调用栈的 Chrome Trace；NVTX 范围让 Nsight 可按模型加载、输入准备、Prefill、Decode 定位。脚本不接受命令行参数，配置仍来自该模型目录的 YAML 副本。

| 工具 | 适合回答 | 输出 |
| --- | --- | --- |
| PyTorch Profiler | 哪些 PyTorch 算子、输入 shape、CPU/CUDA 时间和内存变化明显 | 原始 JSON 与 `artifacts/<model>/traces/*.json` |
| Nsight Systems (`nsys`) | CPU 调度、CUDA API、Kernel、拷贝和时间线之间的关系 | `.nsys-rep`，并可导出 CSV |
| Nsight Compute (`ncu`) | 单个 GPU Kernel 的 SM、带宽、占用率等硬件指标 | `.ncu-rep`，并可导出 CSV |

### Qwen3 采集

以下命令在仓库根目录的 Linux/WSL shell 运行。`forward.py` 自己会写 PyTorch Trace；Nsight 命令会额外采集同一次前向。Mage-VL 对应脚本和报告路径见下方说明。

```bash
python scripts/qwen3_0_6b/forward.py

mkdir -p artifacts/qwen3_0_6b
nsys profile --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  --output artifacts/qwen3_0_6b/qwen3-forward \
  env INFERENCE_PROFILE_MODE=external python scripts/qwen3_0_6b/forward.py

nsys stats --report nvtx_sum --format csv artifacts/qwen3_0_6b/qwen3-forward.nsys-rep \
  > artifacts/qwen3_0_6b/nsys-nvtx.csv
nsys stats --report cuda_gpu_kern_sum --format csv artifacts/qwen3_0_6b/qwen3-forward.nsys-rep \
  > artifacts/qwen3_0_6b/nsys-kernels.csv
nsys stats --report cuda_gpu_mem_time_sum --format csv artifacts/qwen3_0_6b/qwen3-forward.nsys-rep \
  > artifacts/qwen3_0_6b/nsys-memory.csv

# 先由 Nsight Systems 找到热点，再对所选范围中的一个 Kernel 采集详细硬件计数器。
ncu --nvtx --nvtx-include "inference.prefill/" --launch-count 1 --set full \
  --force-overwrite \
  --output artifacts/qwen3_0_6b/inference-prefill-ncu \
  env INFERENCE_PROFILE_MODE=external python scripts/qwen3_0_6b/forward.py
ncu --import artifacts/qwen3_0_6b/inference-prefill-ncu.ncu-rep --page raw --csv \
  > artifacts/qwen3_0_6b/ncu-inference-prefill.csv

python scripts/qwen3_0_6b/analysis.py \
  --input artifacts/qwen3-0.6b-forward.json \
  --output analysis/qwen3_0_6b/analysis.xlsx \
  --nsys-csv artifacts/qwen3_0_6b/nsys-nvtx.csv \
  --nsys-csv artifacts/qwen3_0_6b/nsys-kernels.csv \
  --nsys-csv artifacts/qwen3_0_6b/nsys-memory.csv \
  --ncu-csv artifacts/qwen3_0_6b/ncu-inference-prefill.csv
```

`--launch-count 1` 限定 NCU 只详细采集匹配范围中的一个 Kernel；先用 Nsys 找到值得分析的阶段和 Kernel，再针对 Transformer 或 Inference 的具体范围调整 `--nvtx-include`。移除此限制会分析范围内更多 Kernel，耗时会明显增加。

### Mage-VL 采集

确认仓库根目录 `test.jpg` 存在，或修改 `scripts/mage_vl_4b/forward.py` 的 `INPUT_IMAGE_PATH`。把上面命令中的脚本、目录和输入报告替换为 Mage-VL：

```bash
python scripts/mage_vl_4b/forward.py
mkdir -p artifacts/mage_vl_4b
nsys profile --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  --output artifacts/mage_vl_4b/mage-vl-forward \
  env INFERENCE_PROFILE_MODE=external python scripts/mage_vl_4b/forward.py
nsys stats --report nvtx_sum --format csv artifacts/mage_vl_4b/mage-vl-forward.nsys-rep \
  > artifacts/mage_vl_4b/nsys-nvtx.csv
nsys stats --report cuda_gpu_kern_sum --format csv artifacts/mage_vl_4b/mage-vl-forward.nsys-rep \
  > artifacts/mage_vl_4b/nsys-kernels.csv
nsys stats --report cuda_gpu_mem_time_sum --format csv artifacts/mage_vl_4b/mage-vl-forward.nsys-rep \
  > artifacts/mage_vl_4b/nsys-memory.csv

ncu --nvtx --nvtx-include "inference.prefill/" --launch-count 1 --set full \
  --force-overwrite \
  --output artifacts/mage_vl_4b/inference-prefill-ncu \
  env INFERENCE_PROFILE_MODE=external python scripts/mage_vl_4b/forward.py
ncu --import artifacts/mage_vl_4b/inference-prefill-ncu.ncu-rep --page raw --csv \
  > artifacts/mage_vl_4b/ncu-inference-prefill.csv

python scripts/mage_vl_4b/analysis.py \
  --input artifacts/mage-vl-image-forward.json \
  --output analysis/mage_vl_4b/analysis.xlsx \
  --nsys-csv artifacts/mage_vl_4b/nsys-nvtx.csv \
  --nsys-csv artifacts/mage_vl_4b/nsys-kernels.csv \
  --nsys-csv artifacts/mage_vl_4b/nsys-memory.csv \
  --ncu-csv artifacts/mage_vl_4b/ncu-inference-prefill.csv
```

PyTorch Trace 可导入 Perfetto 或 Chrome Trace Viewer 检查调用栈、算子和 shape；`analysis.py` 将 PyTorch、Nsys 和 Ncu 的数值记录写进同一个 XLSX 数据表，并保留外部 CSV 原始行。`--nsys-csv` 和 `--ncu-csv` 可重复传入；不传时仍只生成 PyTorch Profiler 行。`--input` 必须指向未配置环境变量时生成的 `forward.py` JSON，`--output` 指向 `analysis/<model_alias>/analysis.xlsx`。

`INFERENCE_PROFILE_MODE` 默认是 `torch`，启用 CPU/CUDA PyTorch Profiler；运行 Nsight 时设为 `external`，脚本仅记录 PyTorch CPU 事件并保留 NVTX，不启动 PyTorch CUDA Profiler，避免外部工具和 CUPTI 活动采集相互干扰。外部模式的 JSON 与 Trace 使用 `-external` 后缀，因此不会覆盖 PyTorch 基线报告。

CUDA 可用时工具使用 CUDA；仅在 CUDA 不可用时回退 CPU。Profiler 和 NCU replay 会增加开销：这些数据用于定位算子与 Kernel 热点，不作为延迟基准。前向脚本不包含 HTTP、Gateway 或 Engine Core 调度延迟；性能对比应使用 `tests/<model_alias>/test_performance.py`。

## 结果归文件与测试

可重复的算子优化证据维护在 `analysis/<model_alias>/analysis.md` 和 `analysis.xlsx`。原始 JSON 可以归文件至 `artifacts/`，并记录模型 revision、设备、dtype、输入 shape、warmup 和迭代次数。

组件、服务性能与 Gateway 端到端测试位于 `tests/<model_alias>/`，使用该目录的 `test.yml`。测试命令和运行条件见 [Tests README](../tests/README.md)。
