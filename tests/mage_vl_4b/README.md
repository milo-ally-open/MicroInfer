# Mage-VL-4B 模型级测试

本目录维护 Mage-VL-4B 的模型级 Test、Bench 和 Prefill Profile。模型实现及组件级入口见 [`src/inference/models/mage_vl_4b/`](../../src/inference/models/mage_vl_4b/) 和该目录的 [README](../../src/inference/models/mage_vl_4b/README.md)；新模型接入步骤见根目录 [TUTORIAL.md](../../TUTORIAL.md)。

## 固定工作负载

| 项目 | 配置 |
| --- | --- |
| Hugging Face 模型 / revision | `microsoft/Mage-VL` / `d88b153285f1633a61b2f693c59c8576693af185` |
| API alias | `mage-vl-4b` |
| 输入 | 仓库根目录 `test.jpg` 和固定中文图像问题 |
| 生成长度 | 2 token，greedy decoding |
| 推理设备 | CUDA 可用时使用 CUDA；只有 CUDA 不可用时回退 CPU |
| Profile 模式 | PyTorch Profiler；可设 `INFERENCE_PROFILE_MODE=external` 准备 Nsight 外部采集 |

先按各自 `profiles/inference.yml` 准备固定 revision 的本地权重。脚本启用离线加载，不负责下载权重。视频上传与 Live 视频不属于本目录的图片 workload。

## 目录与执行顺序

```text
tests/mage_vl_4b/
├── README.md
├── analysis.py
├── baseline/
│   ├── test.py
│   ├── bench.py
│   ├── prefill.py
│   └── profiles/inference.yml
└── enhanced/
    ├── test.py
    ├── bench.py
    ├── prefill.py
    └── profiles/inference.yml
```

在仓库根目录激活已安装 `mage-vl` extra 的环境后，依次执行：

```bash
conda activate inference
python tests/mage_vl_4b/baseline/test.py
python tests/mage_vl_4b/enhanced/test.py
python tests/mage_vl_4b/baseline/bench.py
python tests/mage_vl_4b/enhanced/bench.py
python tests/mage_vl_4b/baseline/prefill.py
python tests/mage_vl_4b/enhanced/prefill.py
python tests/mage_vl_4b/analysis.py
```

先运行 baseline Test，为 enhanced Test 和 Bench 准备对照 token。两条路径分别使用目录内的 `profiles/inference.yml`；enhanced 路径从模型源码 `profiles/` 读取注册用的 manifest 与 network 配置，不在测试目录另存副本。

## 输出数据与验收

六个入口各写一个结构一致的 JSON，记录 `schema_version`、模型与 revision、实现类型、任务名、硬件/软件环境、工作负载、结果、指标和算子记录。JSON 输出到 `artifacts/mage_vl_4b/{baseline,enhanced}/`，Prefill trace 输出到 `artifacts/mage_vl_4b/traces/`。

| 入口 | 主要检查或记录 |
| --- | --- |
| `test.py` | token ID 范围与生成结果；enhanced token 与 baseline 对齐 |
| `bench.py` | 预热与测量轮次、每轮延迟、中位延迟、tokens/s、峰值显存 |
| `prefill.py` | Prefill query 长度、Cache 长度、logits shape、算子 shape/次数/时间/显存和 trace |

`analysis.py` 汇总六份 JSON，生成 `analysis/mage_vl_4b/analysis.csv` 和 `analysis.xlsx`。测试脚本生成的数据是分析输入，不应手工修改。组件级入口和 FastAPI 服务验收仍需独立运行；权重、缓存和 profiler trace 不提交 Git。
