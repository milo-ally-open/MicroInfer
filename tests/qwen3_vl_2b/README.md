# Qwen3-VL-2B 模型级测试

本目录维护 Qwen3-VL-2B 的模型级 Test、Bench 和 Prefill Profile。模型实现与组件级入口见 [`src/inference/models/qwen3_vl_2b/`](../../src/inference/models/qwen3_vl_2b/) 及其 [README](../../src/inference/models/qwen3_vl_2b/README.md)；新模型接入步骤见根目录 [TUTORIAL.md](../../TUTORIAL.md)。

## 固定工作负载

| 项目 | 配置 |
| --- | --- |
| Hugging Face 模型 / revision | `Qwen/Qwen3-VL-2B-Instruct` / `89644892e4d85e24eaac8bacfd4f463576704203` |
| API alias | `qwen3-vl-2b` |
| 输入 | 仓库根目录 `test.jpg` 和固定中文图像问题 |
| 生成长度 | 2 token，greedy decoding |
| 推理设备 | CUDA 可用时使用 CUDA；只有 CUDA 不可用时回退 CPU |

先按对应 `profiles/inference.yml` 准备固定 revision 的本地权重。脚本启用离线加载，不负责下载权重。

## 目录与执行顺序

```text
tests/qwen3_vl_2b/
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

在仓库根目录激活已安装 `qwen3-vl-2b` extra 的环境后，依次执行：

```bash
conda activate inference
python tests/qwen3_vl_2b/baseline/test.py
python tests/qwen3_vl_2b/enhanced/test.py
python tests/qwen3_vl_2b/baseline/bench.py
python tests/qwen3_vl_2b/enhanced/bench.py
python tests/qwen3_vl_2b/baseline/prefill.py
python tests/qwen3_vl_2b/enhanced/prefill.py
python tests/qwen3_vl_2b/analysis.py
```

先运行 baseline Test，为 enhanced Test 与 Bench 生成对照 token。每个实现使用各自 `profiles/inference.yml`；增强路径从模型源码的 `profiles/model.yml` 和 `profiles/network.yml` 读取服务注册信息，不在测试目录重复保存这两份文件。

## 输出数据与验收

每个入口写一个结构相同的 JSON，统一包含 `schema_version`、模型与 revision、实现类型、任务名、环境、工作负载、结果、指标和算子记录。JSON 保存到 `artifacts/qwen3_vl_2b/{baseline,enhanced}/`；Prefill trace 保存到 `artifacts/qwen3_vl_2b/traces/`。

| 入口 | 主要检查或记录 |
| --- | --- |
| `test.py` | token ID 合法性；enhanced 生成结果与 baseline token 对齐 |
| `bench.py` | 预热次数、每轮延迟、中位延迟、tokens/s、峰值显存 |
| `prefill.py` | Prefill 输入长度、KV Cache 长度、logits shape、算子 shape/次数/时间/显存和 trace |

`analysis.py` 读取上述六份 JSON，生成 `analysis/qwen3_vl_2b/analysis.csv` 与 `analysis.xlsx`。Prefill JSON 中的字段是表格数据来源；不要手工修改 JSON 后再生成报告。模型组件入口和 FastAPI 服务级验收不由这六个脚本替代。权重、缓存和 profiler trace 不提交 Git。
