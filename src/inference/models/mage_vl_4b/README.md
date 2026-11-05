# Mage-VL-4B 模型实现

本文说明 Mage-VL-4B 在 MicroInfer 中的模型实现边界。通用接入步骤见仓库根目录 [TUTORIAL.md](../../../../TUTORIAL.md)；模型级手动测试见 [tests/mage_vl_4b/README.md](../../../../tests/mage_vl_4b/README.md)。

## 模型身份与能力

| 项目 | 当前值 |
| --- | --- |
| Hugging Face 仓库 | `microsoft/Mage-VL` |
| 固定 revision | `d88b153285f1633a61b2f693c59c8576693af185` |
| Inference/API alias | `mage-vl-4b` |
| 已声明输入 | 图片；视频文件与 Live 视频由相应 API 路径处理 |
| 已声明输出 | 流式文本增量 |
| 已声明能力 | 图片聊天、视频上传、Live 视频 |
| 未声明能力 | Tool Calling |

能力声明以 [`profiles/model.yml`](profiles/model.yml) 为准。`microsoft/Mage-VL` 是 Hugging Face 仓库 ID；对外 API 的 `model` 值使用 `mage-vl-4b`。

## 前向结构与代码映射

Mage-VL 将图像/视频视觉特征与文字主干结合，再由条件生成模型输出文本。图像和视频输入的处理、视觉特征、StreamMind 视频能力及文字生成分别位于模型架构辅助模块中；实际前向顺序以当前 `modeling.py`、配置和上游 checkpoint 为准。

| 路径 | 职责 |
| --- | --- |
| `modeling.py` | 显式装配 Mage-VL 多模态模型、视觉特征和条件生成结构。 |
| `architecture/` | 模型配置/输出类型、图像/视频编码、多模态输入与融合、文字主干、StreamMind 和 Live session 能力。 |
| `layers/` | Vision Embedding/Attention/RoPE/Stack、SigLIP MLP、Qwen3 文字 Attention/Norm/MLP 和 StreamMind Gate 等计算层。 |
| `components/tokenizer.py`、`encoder.py` | Chat Template、图片/视频输入规范化、请求参数及上下文检查。 |
| `components/model_runner.py`、`stepwise_generation.py`、`cache_manager.py` | 模型前向、Prefill/Decode、请求级 Cache 生命周期与 Engine 适配。 |
| `components/decoder.py`、`detokenizer.py`、`adapter.py` | 输出解码、流式增量文本和 Engine 请求/响应转换。 |
| `loader.py`、`runtime.py` | 从本地快照加载权重并将模型 Runner/Adapter 交给共用 Runtime。 |
| `example.py` | 最小图片输入和流式文本输出示例。 |

新增或改动层、组件时，更新本表并在对应源文件的 `__main__` 中维护模块级检查；模块入口必须通过 `python -m` 运行。

## 配置、权重与依赖

- `profiles/model.yml`：注册 alias、Runtime 工厂和已验收能力。
- `profiles/inference.yml`：上游 repo/revision、本地权重目录、dtype、量化和推理参数。
- `profiles/network.yml`：FastAPI 监听地址及视频媒体端点。
- `requirements/requirements.txt`：Mage-VL 专属 Python 依赖；安装 extra 为 `mage-vl`。PyTorch/CUDA 按目标机器单独安装。
- 权重只从固定 revision 的本地快照加载。不要将权重、缓存、私密密钥或大体积 Profile 产物加入 Git。

## 运行入口

```bash
# 预检不加载权重或初始化 GPU
inference validate --model mage-vl-4b

# 启动 FastAPI 模型服务
inference serve --model mage-vl-4b

# 最小 Engine Client 图片流式示例
python -m inference.models.mage_vl_4b.example
```

显卡可用时模型前向默认使用 CUDA；仅 CUDA 不可用时回退 CPU。视频处理、量化和 Live 推理对显存与系统依赖有额外要求，按模型配置和测试 README 准备环境。
