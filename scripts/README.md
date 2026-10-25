# 模型下载与逐算子分析

`scripts/` 保存模型下载和离线模型分析工具；`tests/` 保存组件、性能与端到端测试。两者分开维护，脚本不会调用或导入 `tests/`。

模型权重和 Hub 快取放在根目录 `checkpoints/`，该目录由 `.gitignore` 忽略。逐算子分析结果的原始 JSON 建议放在 `artifacts/`，工作簿位于 `analysis/<model_alias>/analysis.xlsx`。

## 下载本机模型

通用下载器要求明确指定上游模型 `repo_id`、固定 revision 和本机目录。不要把专案 alias 当作 Hugging Face `repo_id`。

```powershell
python scripts/download.py `
  --repo-id Qwen/Qwen3-0.6B `
  --revision c1899de289a04d12100db370d81485cdf75e47ca `
  --local-dir checkpoints/Qwen3-0.6B-c1899de

python scripts/download.py `
  --repo-id microsoft/Mage-VL `
  --revision d88b153285f1633a61b2f693c59c8576693af185 `
  --local-dir checkpoints/microsoft-Mage-VL-d88b153
```

`scripts/download.py --help` 会列出完整参数。公开模型不需 token；下载私有或受限模型时将 Hugging Face token 放入环境变量，不要写入脚本或提交到 Git。

## 逐算子分析流程

每个模型目录采用相同流程：`forward.py` 从本机权重执行模型 forward，以 PyTorch Profiler 按算子名称和输入 shape 汇总 CPU/CUDA 时间、调用次数及内存变化，写出原始 JSON；`analysis.py` 只读该 JSON 并将原始字段写入单工作表 XLSX。两个模型的 `forward.py` 都分段运行 Transformers 基线与项目 `inference` Runtime，并在输出及 XLSX 中以 `transformers`、`inference` 区分。脚本不收命令列参数；权重和推理设定从各自目录下的 `model.yml`、`inference.yml` 与 `network.yml` 读取，不引用 `src/inference/models/` 内的设定档。提示词、输入图片和输出 JSON 路径在各自脚本顶部调整。Profiler 包住模型前向或 ModelRunner 工作项，不包含 HTTP、Gateway 或 Engine Core 排程时间。XLSX 转换不载入模型、不初始化 GPU，也不运行测试。

CUDA 可用时，前向分析和性能基线使用 CUDA；只有 CUDA 不可用时才回退 CPU。CPU 回退结果记录执行设备，不能作为 GPU 性能结果解读。

这些设定档是模型包配置的本地副本；需要更新模型配置时，请同步更新对应的脚本副本。`network.yml` 也保留在脚本目录，因为 Mage-VL Runtime 的模型模块会在导入时读取网络设定。

Qwen3-0.6B 采集文字 prefill 和单 token decode：

```powershell
python scripts/qwen3_0_6b/forward.py

python scripts/qwen3_0_6b/analysis.py `
  --input artifacts/qwen3-0.6b-forward.json
```

Mage-VL-4B 使用其脚本目录内的模型设定副本和本机图片。默认图片为仓库根目录 `test.jpg`；更换输入时编辑 `scripts/mage_vl_4b/forward.py` 顶部的 `INPUT_IMAGE_PATH`。如需调整本机权重路径或推理参数，编辑 `scripts/mage_vl_4b/inference.yml`；载入时强制离线。

```powershell
python scripts/mage_vl_4b/forward.py

python scripts/mage_vl_4b/analysis.py `
  --input artifacts/mage-vl-image-forward.json
```

两份 `forward.py` 都会在终端显示 Transformers、Inference 两段 profiler 表格；完整算子清单写入 JSON，随后交给对应 `analysis.py` 生成 XLSX。

## 测试目录

组件、性能和端到端验收脚本统一位于 `tests/<model_alias>/`，独立使用该目录的 `test.yml`。请参考 [tests/README.md](../tests/README.md)。`scripts/<model>/` 只放分析入口和该模型独立的配置副本，不放置测试入口或测试配置。
