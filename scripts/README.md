# 模型下载与离线分析

`scripts/` 提供通用模型下载器，以及按模型组织的本机前向和算子分析工具。逐模型组件、性能与端到端测试属于 `tests/`，两类脚本分开维护。

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

每个模型目录保存一份独立配置副本；模型插件实际配置仍位于 `src/inference/models/<model>/configs/`。修改分析配置时，请确认脚本副本和模型配置意图一致。模型专属脚本只使用本机权重，不从 `src/` 导入脚本模块。

## 前向与算子报告

`forward.py` 使用本机模型权重分别运行 Transformers 基线和项目模型实现，以 PyTorch Profiler 汇总算子、输入 shape、CPU/CUDA 时间及内存变化，并将结果写入 `artifacts/`。这些脚本不接受命令行参数；输入图片、提示词和输出文件路径由脚本顶部及目录内 YAML 配置。

Qwen3 示例：

```bash
python scripts/qwen3_0_6b/forward.py
mkdir -p artifacts
python scripts/qwen3_0_6b/analysis.py \
  --input artifacts/qwen3-0.6b-forward.json \
  --output artifacts/qwen3-0.6b-operator-report.xlsx
```

Mage-VL 默认读取仓库根目录的 `test.jpg`。运行前请确认图片存在，或修改 `scripts/mage_vl_4b/forward.py` 中的 `INPUT_IMAGE_PATH`：

```bash
python scripts/mage_vl_4b/forward.py
mkdir -p artifacts
python scripts/mage_vl_4b/analysis.py \
  --input artifacts/mage-vl-image-forward.json \
  --output artifacts/mage-vl-operator-report.xlsx
```

`analysis.py --input` 必须指向已有的 JSON 报告。使用 `--output` 指定新工作簿路径，避免覆盖模型分析目录中维护的历史 `analysis.xlsx`。Profiler 只分析模型前向或 Runner 工作项，不包括 HTTP、Gateway 或 Engine Core 排程延迟。

CUDA 可用时工具使用 CUDA；仅在 CUDA 不可用时回退 CPU。CPU 结果会记录设备，不能作为 GPU 性能结果解读。Profiler 有采集开销；分析结果用于定位热点，不代表服务性能提升或 SLA。

## 结果归档与测试

可重复的算子优化证据维护在 `analysis/<model_alias>/analysis.md` 和 `analysis.xlsx`。原始 JSON 可以归档至 `artifacts/`，并记录模型 revision、设备、dtype、输入 shape、warmup 和迭代次数。

组件、服务性能与 Gateway 端到端测试位于 `tests/<model_alias>/`，使用该目录的 `test.yml`。测试命令和运行条件见 [Tests README](../tests/README.md)。
