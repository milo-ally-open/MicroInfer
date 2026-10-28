# Qwen3-0.6B 算子分析

本目录保存 Qwen3 的问题分析、解决方案和验收效果；每一行都用编号对应 [analysis.xlsx](analysis.xlsx) 中的原始测量证据。

> 状态更新（2026-10-25）：本报告中的 CUDA Paged Decode Attention 是已撤除的历史 prototype。按使用者决定，当前 Qwen3 代码不再包含手写 CUDA kernel 源码、构建脚本或 backend 选择接口；当前工作树中原 kernel 目录及编译产物均不存在。以下数值仅作为历史实测记录，不能用于配置或调用当前服务。

## 测量范围

2026-10-25，固定 revision `c1899de289a04d12100db370d81485cdf75e47ca`。只加载本地 checkpoint 第 0 层 `self_attn` 权重，bfloat16，RTX 4060 Laptop 8,188 MiB，WSL Ubuntu 24.04.5，Python 3.12.15，PyTorch 2.14.1+cu126，Driver 566.24。batch=1；prefill 512 tokens；decode 读取 512-token KV 后增加 1 token；KV block size=16。Activation 为合成张量，预热 2 次、测量 5 次。

## 问题、解决方案、效果

| 问题 | 解决方案/实施方式 | 效果 | XLSX 证据 |
|---|---|---|---|
| Prefill `aten::copy_`：shape `[1,8,2,512,512]`，109.092 μs，Profiler allocation 0 bytes，3 次；这是 score 尺寸张量 copy。 | 先追踪 copy 来源；只有确认它影响耗时或显存后，再评估减少中间 copy。 | 尚未修改；本次只创建基线，没有优化效果数据。 | QW-01 |
| Prefill `aten::mm`：`[512,1024] × [1024,1024]`，87.076 μs，2,097,152 bytes，2 次。 | 保留现状；扩大序列长度和 batch 测量后，再判断是否值得优化 projection。 | 尚未修改；本次只创建基线，没有优化效果数据。 | QW-02 |
| Prefill `aten::mm`：`[512,1024] × [1024,2048]`，83.586 μs，2,097,152 bytes，1 次。 | 保留现状；扩大序列长度和 batch 测量后，再判断是否值得优化 projection。 | 尚未修改；本次只创建基线，没有优化效果数据。 | QW-03 |
| Prefill `aten::_softmax`：shape `[1,8,2,512,512]`，76.834 μs，16,777,216 bytes，1 次。 | 扫描更长 query/key；确认临时显存影响后，再评估分块实现。 | 尚未修改；本次只创建基线，没有优化效果数据。 | QW-04 |
| Prefill `aten::mm`：`[512,2048] × [2048,1024]`，75.490 μs，1,048,576 bytes，1 次。 | 保留现状；扩大序列长度和 batch 测量后，再判断是否值得优化 projection。 | 尚未修改；本次只创建基线，没有优化效果数据。 | QW-05 |
| Prefill `aten::bmm`：`[8,1024,128] × [8,128,512]`，42.945 μs，8,388,608 bytes，1 次；用于 GQA score 计算。 | 扫描更长 query/key 和 batch，再决定是否需要调整 GQA score 计算。 | 尚未修改；本次只创建基线，没有优化效果数据。 | QW-06 |
| Decode `aten::mm`：`[1,1024] × [1024,1024]`，22.530 μs，4,096 bytes，2 次。 | 先作为单 token projection 基线；不根据单次采样直接改算子。 | 尚未修改；本次只创建基线，没有优化效果数据。 | QW-07 |
| Decode `aten::mm`：`[1,1024] × [1024,2048]`，20.866 μs，4,096 bytes，1 次。 | 先作为单 token projection 基线；不根据单次采样直接改算子。 | 尚未修改；本次只创建基线，没有优化效果数据。 | QW-08 |
| Decode `aten::mm`：`[1,2048] × [2048,1024]`，19.426 μs，2,048 bytes，1 次。 | 先作为单 token projection 基线；不根据单次采样直接改算子。 | 尚未修改；本次只创建基线，没有优化效果数据。 | QW-09 |
| Decode `aten::index`：KV page shape `[33,16,8,128]`，11.522 μs，2,162,688 bytes，2 次；trace 显示 KV page gather。 | 对 KV 长度 128/512/1024/2048、batch 1/4 量化 gather；再决定是否减少连续物化。 | Gather 基线已测；优化和优化后验收尚未执行。 | QW-10 |
| Decode `aten::copy_`：K/V shape `[8,513,128]`，11.075 μs，Profiler allocation 0 bytes，2 次；将 gather 后 K/V 写入连续输出。 | 与 KV gather 一起测量，评估能否减少复制；保留 Cache 生命周期检查。 | 尚未修改；本次只创建基线，没有优化效果数据。 | QW-11 |
| Decode `aten::bmm`：`[8,2,513] × [8,513,128]`，9.569 μs，4,096 bytes，1 次；用于 GQA Attention。 | 扫描更长 KV 后再判断算子占比和优化价值。 | 尚未修改；本次只创建基线，没有优化效果数据。 | QW-12 |
| Decode 从分页 KV gather 到连续 K/V；旧路径没有直接依 block table 读取 page 的 kernel。 | 历史实验曾加入 Qwen3 PyTorch C++/CUDA decode op，并以多序列、异构上下文和重排物理页对照 PyTorch 稠密参考。 | RTX 4060 Laptop 上 float16/bfloat16、context=[1,4,9] 的数值对照通过，最大误差为 0；同机同 shape checkpoint profile 确认执行 `qwen3_cuda::paged_decode`，kernel 614.267 μs。同 revision、GPU、shape、dtype 的完整 Attention decode event 中位数：CUDA 10.813 ms，PyTorch 1.877 ms；PyTorch `aten::index` gather 为 14.654 μs。该 prototype 已依后续决定撤除；性能测量慢于 PyTorch 路径。 | QW-13、QW-14 |

## 验收结论

T-32 prototype 验收在 2026-10-25 完成，随后按使用者决定由 T-37 撤除。历史记录：当时 Qwen3 组件 9 个模块入口和 Engine Sequence/BlockManager/Scheduler 生命周期检查通过；新增 native CUDA 入口在本地 RTX 4060 Laptop 上通过 float16/bfloat16、多序列、异构上下文、物理页重排和 block 边界检查；PyTorch C++/CUDA extension 使用 `MAX_JOBS=1` 构建成功；真实 checkpoint 第 0 层 CUDA profile 和 CUDA backend 推理 API／Gateway SSE 均通过。性能脚本的 3 次测量中，CUDA Engine API 平均 6.118 秒、5.26 tokens/s，Transformers baseline 平均 1.187 秒、27.17 tokens/s。以上只描述已撤除实现的历史结果，不代表当前服务能力。

QW-01–QW-12 保留原先 PyTorch gather 路径的 profiler 历史数据；QW-13/QW-14 是本轮 CUDA 与 PyTorch 同机、同 revision、同 shape、同 dtype 对照；完整 Attention decode CUDA event 中位数约为 PyTorch 的 5.76 倍。单层输入为合成 activation，模型 revision 与权重来自本地固定 snapshot；结果不代表完整服务吞吐规律。Profiler allocation 是算子分组的内存活动，不是峰值显存；Profiler 耗时不能按行相加。详细测试输出保存在 `tests/qwen3_0_6b/results/`，短测结果不构成 SLA。
