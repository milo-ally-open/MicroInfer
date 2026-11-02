# Qwen3-VL-2B 基线分析

本记录对应固定 Hugging Face revision `89644892e4d85e24eaac8bacfd4f463576704203`，在 RTX 4060 Laptop GPU 上使用 BF16、Transformers 5.19.0、PyTorch 2.14.1+cu126 和 SDPA 完成。测试输入为仓库中的 `test.jpg`（960×540）及固定中文问题；Processor 产生 528 个 prompt token。Test、Bench、Profile 均只针对 Transformers 官方模型实现，尚未代表 Inference Engine 的性能。

## 问题、解决方案、效果

| 问题 | 解决方案 | 效果 |
|---|---|---|
| 尚无 Qwen3-VL 的图文部署基线，无法判断服务集成是否改变输入处理或生成结果。 | 固定模型 revision；用官方 `AutoProcessor` 与 `Qwen3VLForConditionalGeneration` 分别验收图文和纯文本生成；把相同图文输入用于 Bench 与 Profile。 | Test 在 CUDA/BF16 通过。图文 Prefill 返回 logits 和 Cache，图片问题生成非空回答；纯文本路径也生成 3 个 token。完整原始记录见 `artifacts/qwen3-vl-2b-transformers-benchmark.json` 与 Profile JSON。 |
| 当前完整图文请求的服务级耗时与显存缺少基准值。 | 预热 1 次、正式测量 3 次，每次生成上限 32 token；GPU 计时前后同步，报告中位延迟、吞吐和峰值显存。 | 本次复测中位延迟 1.341 s，中位吞吐 23.865 token/s；每轮生成 32 token；峰值分配 4,383,617,024 bytes，峰值保留 4,435,476,480 bytes。此值只对应当前图片、输入长度和硬件；3 轮样本较少，不作为服务 SLA。 |
| Prefill 主要包含视觉编码、全 prompt 投影与注意力；需要区分视觉侧和语言侧算子，后续优化不能只看总时延。 | 用 PyTorch Profiler 分别包住 Prefill 和单 token Decode，导出 Chrome trace、算子形状、调用数、设备耗时和分配活动。Decode 显式使用 Qwen3-VL 官方 MRoPE 位置准备逻辑，并断言 query 长度为 1、Cache 长度从 528 增至 529。 | 本次 Profile 的 Prefill CUDA event 为 1,338.700 ms，单步 Decode 为 111.594 ms。Decode query 长度为 1，Cache 从 528 增至 529；算子详情由 `analysis.xlsx` 和 Chrome Trace 记录。Profiler 有采集开销，该时延不与 Bench 数字直接比较。 |
| Profile 指出候选热点，但目前没有算子替换或优化后复测，不能宣称已提升性能。 | 下一轮先以相同 revision、图片、dtype、GPU 和生成长度建立 Engine 对照；根据 Nsight 时间线判断 CPU 调度、视觉编码、GEMM 或 Cache 是否主导，再逐项替换实现并重复 Test/Bench/Profile。 | 本轮只建立可复现基线，没有修改推理算子；当前优化收益为“未测”。算子细节和数值由脚本直接生成在 [`analysis.xlsx`](analysis.xlsx)，不手工填写。 |

## 复现

在仓库根目录的 WSL/Linux Bash 中激活 `inference` 环境：

```bash
conda activate inference
python tests/qwen3_vl_2b/test_components.py
python tests/qwen3_vl_2b/test_performance.py
python scripts/qwen3_vl_2b/forward.py
python scripts/qwen3_vl_2b/analysis.py
```

Profiler trace 保存在 `artifacts/qwen3_vl_2b/traces/`；原始 Profile 与 Bench JSON 被脚本写入 `artifacts/`。XLSX 仅保存机器可读测量记录。GPU Profiler 的算子耗时是定位证据，受采集开销影响，不应与未开启 Profiler 的 Bench 延迟直接相加或混为同一指标。
