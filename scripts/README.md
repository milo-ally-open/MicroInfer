# 模型下载与算子分析

`scripts/` 提供固定 revision 权重下载和逐模型前向分析。模型功能测试独立维护在 `tests/`。

## 下载模型

从仓库根目录运行，并明确指定 Hugging Face repo、revision 和本地目录：

```bash
python scripts/download.py --repo-id Qwen/Qwen3-VL-2B-Instruct --revision 89644892e4d85e24eaac8bacfd4f463576704203 --local-dir checkpoints/Qwen3-VL-2B-Instruct-8964489
python scripts/download.py --repo-id microsoft/Mage-VL --revision d88b153285f1633a61b2f693c59c8576693af185 --local-dir checkpoints/microsoft-Mage-VL-d88b153
```

权重保存在被 `.gitignore` 忽略的 `checkpoints/`，不会进入 Git。

## 逐模型分析

每个模型的 `scripts/<model_alias>/` 保存独立配置副本和分析脚本；`forward.py` 对比 Transformers 基线与 Inference 前向，`analysis.py` 更新 `analysis/<model_alias>/analysis.xlsx`。

Qwen3-VL-2B：

```bash
python scripts/qwen3_vl_2b/forward.py
python scripts/qwen3_vl_2b/analysis.py
```

Mage-VL-4B：

```bash
python scripts/mage_vl_4b/forward.py
python scripts/mage_vl_4b/analysis.py
```

Profile 输出保存在忽略目录 `artifacts/`。PyTorch Profiler 可检查算子、shape、CPU/CUDA 时间及内存；Nsight Systems 适合分析 CPU 调度、CUDA API 和 Kernel 时间线；Nsight Compute 用于检查单个 GPU Kernel 的硬件计数器。采集脚本按模型和输入特点实现，不能把单层或单请求 profile 当作服务容量结论。
