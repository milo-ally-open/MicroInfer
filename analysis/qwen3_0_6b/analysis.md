# Qwen3-0.6B 算子分析

本目录保存 Qwen3 的问题分析、解决方案和验收效果；每一行都用编号对应 [analysis.xlsx](analysis.xlsx) 中的原始测量证据。

## 测量范围

2026-10-25，固定 revision `c1899de289a04d12100db370d81485cdf75e47ca`。只载入本机 checkpoint 第 0 层 `self_attn` 权重，bfloat16，RTX 4060 Laptop 8,188 MiB，WSL Ubuntu 24.04.5，Python 3.12.15，PyTorch 2.14.1+cu126，Driver 566.24。batch=1；prefill 512 tokens；decode 读取 512-token KV 后增加 1 token；KV block size=16。Activation 为合成张量，预热 2 次、测量 5 次。

## 问题、解决方案、效果

| 问题 | 解决方案（待验证） | 效果 | XLSX 证据 |
|---|---|---|---|
| Prefill `aten::copy_`：shape `[1,8,2,512,512]`，109.092 μs，Profiler allocation 0 bytes，3 次；这是 score 尺寸张量 copy。 | 先追踪 copy 来源；只有确认它影响耗时或显存后，再评估减少中间 copy。 | 尚未修改；本次只建立基线，没有优化效果数据。 | QW-01 |
| Prefill `aten::mm`：`[512,1024] × [1024,1024]`，87.076 μs，2,097,152 bytes，2 次。 | 保留现状；扩大序列长度和 batch 测量后，再判断是否值得优化 projection。 | 尚未修改；本次只建立基线，没有优化效果数据。 | QW-02 |
| Prefill `aten::mm`：`[512,1024] × [1024,2048]`，83.586 μs，2,097,152 bytes，1 次。 | 保留现状；扩大序列长度和 batch 测量后，再判断是否值得优化 projection。 | 尚未修改；本次只建立基线，没有优化效果数据。 | QW-03 |
| Prefill `aten::_softmax`：shape `[1,8,2,512,512]`，76.834 μs，16,777,216 bytes，1 次。 | 扫描更长 query/key；确认临时显存影响后，再评估分块实现。 | 尚未修改；本次只建立基线，没有优化效果数据。 | QW-04 |
| Prefill `aten::mm`：`[512,2048] × [2048,1024]`，75.490 μs，1,048,576 bytes，1 次。 | 保留现状；扩大序列长度和 batch 测量后，再判断是否值得优化 projection。 | 尚未修改；本次只建立基线，没有优化效果数据。 | QW-05 |
| Prefill `aten::bmm`：`[8,1024,128] × [8,128,512]`，42.945 μs，8,388,608 bytes，1 次；用于 GQA score 计算。 | 扫描更长 query/key 和 batch，再决定是否需要调整 GQA score 计算。 | 尚未修改；本次只建立基线，没有优化效果数据。 | QW-06 |
| Decode `aten::mm`：`[1,1024] × [1024,1024]`，22.530 μs，4,096 bytes，2 次。 | 先作为单 token projection 基线；不根据单次采样直接改算子。 | 尚未修改；本次只建立基线，没有优化效果数据。 | QW-07 |
| Decode `aten::mm`：`[1,1024] × [1024,2048]`，20.866 μs，4,096 bytes，1 次。 | 先作为单 token projection 基线；不根据单次采样直接改算子。 | 尚未修改；本次只建立基线，没有优化效果数据。 | QW-08 |
| Decode `aten::mm`：`[1,2048] × [2048,1024]`，19.426 μs，2,048 bytes，1 次。 | 先作为单 token projection 基线；不根据单次采样直接改算子。 | 尚未修改；本次只建立基线，没有优化效果数据。 | QW-09 |
| Decode `aten::index`：KV page shape `[33,16,8,128]`，11.522 μs，2,162,688 bytes，2 次；trace 显示 KV page gather。 | 对 KV 长度 128/512/1024/2048、batch 1/4 量化 gather；再决定是否减少连续物化。 | Gather 基线已测；优化和优化后验收尚未执行。 | QW-10 |
| Decode `aten::copy_`：K/V shape `[8,513,128]`，11.075 μs，Profiler allocation 0 bytes，2 次；将 gather 后 K/V 写入连续输出。 | 与 KV gather 一起测量，评估能否减少复制；保留 Cache 生命周期检查。 | 尚未修改；本次只建立基线，没有优化效果数据。 | QW-11 |
| Decode `aten::bmm`：`[8,2,513] × [8,513,128]`，9.569 μs，4,096 bytes，1 次；用于 GQA Attention。 | 扫描更长 KV 后再判断算子占比和优化价值。 | 尚未修改；本次只建立基线，没有优化效果数据。 | QW-12 |

## 验收结论

Qwen3 的 8 个模块 `__main__` 对照测试通过：模型、Attention、RMSNorm/MLP、RoPE、ModelRunner、PagedKVCache、Sampler、InputBatch。本次没有修改算子，因此没有优化前后效果；未启动 Inference API，也未执行 API 端到端测试。CUDA Event 中位数为 prefill 2.784 ms、decode 1.459 ms，各 5 个样本且波动较大，只用于本机单层定位。Profiler allocation 是算子分组的内存活动，不是峰值显存；Profiler 耗时不能按行相加。
