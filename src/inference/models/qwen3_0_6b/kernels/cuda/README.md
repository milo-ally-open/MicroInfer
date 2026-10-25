# Qwen3 CUDA Decode Attention

该模型包提供一个可选的 native CUDA 单 token decode attention prototype。Kernel 按 Qwen3 Worker 提供的 block table 直接读取实体 KV page，不先将完整历史 KV gather 成连续张量。Prefill 和多 token query 仍使用 PyTorch reference。

此实现只属于 Qwen3 模型包。其他模型如需自定义 CUDA 算子，应在自己的模型目录中维护代码和构建入口。

## 构建

需要 NVIDIA GPU、CUDA 版 PyTorch、兼容的 CUDA Toolkit/`nvcc` 和 C++ 编译器。请在目标运行环境中构建扩展；API 启动时不会即时编译。

从仓库根目录单进程构建：

```bash
MAX_JOBS=1 python src/inference/models/qwen3_0_6b/kernels/cuda/setup.py build_ext --inplace
```

## 运行

模型配置 `configs/inference.yml` 中的 `engine.attention_backend` 默认为 `pytorch`。只有在目标环境已成功构建扩展并完成验证后，才设置为 `cuda`。配置选择 CUDA 后，扩展不可载入、设备不符或输入不受支持时会明确报错，不会静默回退。

Qwen3 组件检查包含 CUDA 数值对照入口：

```bash
python tests/qwen3_0_6b/test_components.py \
  --config tests/qwen3_0_6b/test.yml
```

该入口覆盖 float16/bfloat16、多序列、异构上下文长度、非连续物理页、KV block 边界和非法元数据。

## 支持范围与性能

- 只替换单 token decode attention；prefill 和非单 token query 走 PyTorch 路径。
- 当前要求连续 CUDA tensor，head dimension 为不大于 128 的 2 次幂。
- 自定义 Kernel 的正确性已在本机 RTX 4060 Laptop 上与 PyTorch reference 对照。
- 当前单层 profile 和 API 测量均慢于 PyTorch 路径，因此默认配置保持 `pytorch`。

数值正确性不代表性能提升或生产环境普遍适用。实际表现取决于 GPU、PyTorch/CUDA ABI、模型配置和请求形状。
