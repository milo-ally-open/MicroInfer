# Qwen3-VL-2B 模型实现

本文说明 Qwen3-VL-2B 在 MicroInfer 中的模型实现边界。通用接入步骤见仓库根目录 [TUTORIAL.md](../../../../TUTORIAL.md)；模型级手动测试见 [tests/qwen3_vl_2b/README.md](../../../../tests/qwen3_vl_2b/README.md)。

## 模型身份与能力

| 项目 | 当前值 |
| --- | --- |
| Hugging Face 仓库 | `Qwen/Qwen3-VL-2B-Instruct` |
| 固定 revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| Inference/API alias | `qwen3-vl-2b` |
| 已声明输入 | 文本、图片 |
| 已声明输出 | 流式文本增量 |
| 未声明能力 | 视频上传、Live 视频、Tool Calling |

能力声明的运行时来源是 [`profiles/model.yml`](profiles/model.yml)。Model Card 上的能力不等于本仓库已经实现的 API 能力；修改 manifest 前先完成对应验收。

## 前向结构与代码映射

本模型沿用 Qwen3-VL 的多模态前向拓扑：图片经视觉 Patch Embedding、Vision Transformer Blocks 和 Patch Merger 得到视觉特征；文本 Token Embedding 与视觉特征进入多模态主干，再通过文本 Decoder、Norm 和 LM Head 生成 logits。位置编码包含视觉 RoPE 与文本 MRoPE；逐 token 生成还要保留 Cache 和 Qwen3-VL 的位置状态。

| 路径 | 职责 |
| --- | --- |
| `modeling.py` | 组装完整 Qwen3-VL 视觉塔、文本塔和条件生成模型；明确复用上游 Transformers 类及本地覆盖层的边界。 |
| `layers/` | Embedding、视觉/文本 Attention、RoPE、Norm、MLP、Vision Block、Merger 和 LM Head 等计算层；有 `__main__` 的模块提供独立前向检查。 |
| `components/tokenizer.py`、`encoder.py` | 应用官方 Processor/Chat Template，检查文本与图片输入，并准备模型张量。 |
| `components/model_runner.py`、`stepwise_generation.py` | 将模型前向、Prefill、单 token Decode 和 Cache 状态接入共用 Engine。 |
| `components/decoder.py`、`detokenizer.py`、`adapter.py` | 处理生成结果解码、增量文本和 Engine 请求/响应适配。 |
| `architecture/image_encoder.py` | 图片输入的模型专属编码与图像预处理逻辑。 |
| `loader.py`、`runtime.py` | 读取模型配置、加载模型，并将 Runner/Adapter 组合为共用 Runtime 或 Engine Client。 |
| `example.py` | 最小图片输入和流式文本输出示例。 |

新增或改动层、组件时，更新本表并在对应源文件的 `__main__` 中维护该模块自己的前向检查；模块入口用 `python -m` 运行，避免相对导入失败。

## 配置、权重与依赖

- `profiles/model.yml`：服务发现所用 alias、Runtime 工厂和 API 能力。
- `profiles/inference.yml`：上游 repo/revision、本地 `checkpoints/` 路径、dtype、Attention backend 和请求/Engine 参数。
- `profiles/network.yml`：API 监听地址。
- `requirements/requirements.txt`：Qwen3-VL 专属 Python 依赖；安装 extra 为 `qwen3-vl-2b`。PyTorch/CUDA 按目标机器单独安装。
- 权重只从固定 revision 的本地快照加载，不提交模型权重、缓存、密钥或 Profile 大文件。

## 运行入口

```bash
# 预检不加载模型权重、不初始化 GPU
inference validate --model qwen3-vl-2b

# 启动 FastAPI 模型服务
inference serve --model qwen3-vl-2b

# 最小 Engine Client 图片流式示例
python -m inference.models.qwen3_vl_2b.example
```

显卡可用时模型前向默认走 CUDA；只有 CUDA 不可用时才回退 CPU。显存不足应降低输入规模或调整配置，不视为自动回退条件。
