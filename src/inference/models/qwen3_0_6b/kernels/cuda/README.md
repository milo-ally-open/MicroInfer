# Native CUDA 推理算子

本目录属于 Qwen3 模型包，提供 Qwen3 单 token decode 的 CUDA Paged Attention prototype。它读取 Worker 持有的实体 KV page pool 和请求 block table，不先 gather 为连续 K/V 张量。Prefill 仍走 Qwen PyTorch reference；其他模型应在各自模型包维护其算子与构建入口。

在已安装目标 Qwen/PyTorch 环境、且 `nvcc` 与 PyTorch CUDA ABI 可用的机器上，从专案根目录执行：

```bash
MAX_JOBS=1 python src/inference/models/qwen3_0_6b/kernels/cuda/setup.py build_ext --inplace
```

模型设定 `engine.attention_backend` 默认是 `pytorch`。本地已在 RTX 4060 Laptop 完成扩展构建、float16/bfloat16 多序列与异构长度数值对照、非连续物理页表和 block 边界测试，以及固定 checkpoint 单层 profile。设为 `cuda` 后，单 token decode 会调用 native op；prefill 与非单 token query 仍使用 PyTorch 参考路径。扩展无法载入、装置不符或输入 shape/dtype 不支持时明确失败。Kernel 要求连续 CUDA tensor 和不大于 128 的 2 次幂 head dimension；当前 profile 与 API 性能低于 PyTorch 路径，因此不建议将它设为默认后端。

此扩展随实际 PyTorch、CUDA Toolkit、编译器与 GPU compute capability 构建，不在 API 服务启动期间即时编译。部署 wheel 应在目标运行环境构建并验证。
