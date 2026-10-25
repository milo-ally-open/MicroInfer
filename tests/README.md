# 模型测试与验收

每个模型目录包含三项直接执行的测试：组件检查、服务性能比较和 Gateway 端到端调用。测试配置位于同目录的 `test.yml`；运行时必须显式传入 `--config`。所有相对路径以该 YAML 所在目录为基准。

## 环境要求

在对应模型的 Python 环境安装依赖：

```bash
python -m pip install -e "src/inference[qwen3]"
# 或
python -m pip install -e "src/inference[mage-vl]"
```

PyTorch/CUDA wheel 需按目标设备单独安装。各模型的 `test.yml` 应指向固定 revision 的本机 snapshot；测试不会替用户下载大型模型。

## Qwen3-0.6B

### 组件检查

该脚本不载入完整模型权重或启动 Inference 服务。它在 meta device 对照模型结构，并运行 Qwen3 模块的数值入口，包括模型、Attention、RMSNorm/MLP、RoPE、Runner、Paged KV Cache、Sampler、InputBatch 和 CUDA paged decode。

```bash
python tests/qwen3_0_6b/test_components.py \
  --config tests/qwen3_0_6b/test.yml
```

CUDA paged decode 检查要求已构建本地扩展并可用的 CUDA 装置。它覆盖多序列、不同上下文长度、非连续物理页、block 边界和非法元数据；其他模块优先使用 CUDA，CUDA 不可用时回退 CPU。

### Attention 算子分析

`--profile-attention` 从本机固定 snapshot 读取第 0 层 Attention 权重，以合成 hidden states 比较 prefill 和单 token decode，并记录 Profiler 可见的算子、shape 和内存信息：

```bash
python tests/qwen3_0_6b/test_components.py \
  --config tests/qwen3_0_6b/test.yml \
  --profile-attention
```

该结果只代表指定层和合成输入，不代表真实 activation 分布、整模型吞吐或 API 延迟。Profiler 有采集开销；CPU 结果不能作为 GPU 性能数据。

### 服务性能

启动 Qwen3 Inference API，并确认 GPU 显存足以同时运行服务模型和 Transformers 基线。脚本比较生成耗时、token 数和 token/s：

```bash
python tests/qwen3_0_6b/test_performance.py \
  --config tests/qwen3_0_6b/test.yml
```

### Gateway 端到端

先启动 Qwen3 Inference 和 Go Gateway，并通过 `VIDEO_API_KEY` 提供 Gateway API Key：

```bash
export VIDEO_API_KEY='<Gateway API Key>'
python tests/qwen3_0_6b/test_end2end.py \
  --config tests/qwen3_0_6b/test.yml
```

该测试验证现有 `/v1/chat/completions` SSE 路径。

## Mage-VL-4B

### 组件检查

在 meta device 对照模型结构和 checkpoint key，并运行 Attention、文本 prefill/decode 数值对照；不载入完整权重：

```bash
python tests/mage_vl_4b/test_components.py \
  --config tests/mage_vl_4b/test.yml
```

### 服务性能

启动 Mage-VL Inference API，并确认 GPU 显存可同时容纳服务模型和 Transformers 基线：

```bash
python tests/mage_vl_4b/test_performance.py \
  --config tests/mage_vl_4b/test.yml
```

### Gateway 端到端

启动 Mage-VL Inference 和 Go Gateway，设置 API Key，并准备 `test.yml` 指定的图片和视频素材：

```bash
export VIDEO_API_KEY='<Gateway API Key>'
python tests/mage_vl_4b/test_end2end.py \
  --config tests/mage_vl_4b/test.yml
```

该测试覆盖模型已声明的图片聊天和视频上传路径。Live 会话需要另行部署 MediaMTX；其配置见 [MediaMTX README](../deploy/mediamtx/README.md)。

## 结果和限制

测试摘要输出到标准输出，不会自动创建历史结果文件。需要归档时，将结果保存到 `tests/<model_alias>/results/`，并记录模型 revision、GPU、驱动、CUDA、PyTorch、dtype 和配置。

性能测试结果是特定机器和配置下的比较，不构成 SLA。组件检查不评估数据集精度；服务或 Gateway 未实际运行时，不应将静态或组件检查描述为端到端验收。

## 客户端协议示例

[`client.py`](client.py) 演示 OpenAI SDK 聊天、Qwen3 工具调用循环，以及可选的视频理解请求。运行工具循环前应启动相应服务并设置 API Key：

```bash
export VIDEO_API_KEY='<Gateway API Key>'
python tests/client.py

# 可选：通过 /v1/videos/understand 发送视频
export VIDEO_FILE='/path/to/video.mp4'
python tests/client.py
```

示例中的工具由客户端执行。真实应用必须使用自身的工具注册表，并实施参数校验、权限、超时、结果大小和最大循环次数限制。
