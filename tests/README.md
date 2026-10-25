# 逐模型测试

每个模型目录维护三个直接执行的 Python 脚本。脚本从顶部开始读取 YAML、执行检查并输出 JSON；不再把主体包进 main() 或自定义辅助函数，便于按执行顺序调试。相对路径均以该模型目录的 test.yml 为基准。

先在当前模型的 Python 环境中执行 `python -m pip install -e "src/inference[mage-vl]"` 或 `python -m pip install -e "src/inference[qwen3]"`，只安装当前模型的依赖 extra。PyTorch/CUDA wheel 需按目标 GPU 环境预先安装。

## Qwen3-0.6B

组件测试不载入完整模型权重、不启动推理服务。它在 meta device 对照模型结构，然后依序执行 Qwen3 模块的 `__main__` 前向入口，覆盖模型、Attention、RMSNorm/MLP、RoPE、ModelRunner、Paged KV Cache、Sampler、InputBatch 和原生 CUDA paged decode。最后一项要求扩展已构建且 CUDA 可用，并对照多序列、异构上下文长度、非连续物理页、block 边界及非法元数据；其余模块前向优先使用 CUDA，CUDA 不可用时回退 CPU：

~~~powershell
python tests/qwen3_0_6b/test_components.py --config tests/qwen3_0_6b/test.yml
~~~

要对实际 Qwen3 Attention 算子做一次分析，使用本机 snapshot 的第 0 层 Q/K/V/O projection 与 Q/K norm 权重，执行 prefill 和单 token decode；CUDA 可用时记录 CUDA Event，CUDA 不可用时记录 CPU wall time，并由 PyTorch Profiler 采集可用的 CPU/CUDA 算子、shape 和 memory：

~~~powershell
python tests/qwen3_0_6b/test_components.py --config tests/qwen3_0_6b/test.yml --profile-attention
~~~

该模式读取 test.yml 的 operator_analysis 参数。输入 hidden states 是合成张量；模型层维度、dtype、KV page size 与 checkpoint 权重来自本机固定 revision。结果代表单层 Attention 算子分析，不代表整模型吞吐、API 延迟或真实 activation 分布。Profiler 会增加测量开销，用于定位算子，不将 profiler 采样当作发布性能数字；CPU 回退数据不能作为 GPU 性能结果。

test_performance.py 需要自行启动 Qwen3 Inference API，并载入 Transformers baseline。推理优先使用 CUDA，仅在 CUDA 不可用时使用 CPU；它比较 API 与 Transformers 原生生成的延迟、输出 token 数和 token/s：

~~~powershell
python tests/qwen3_0_6b/test_performance.py --config tests/qwen3_0_6b/test.yml
~~~

test_end2end.py 需要先启动 Go Gateway 与 Qwen3 Inference API，并通过环境变量提供 Gateway API Key。它走现有 /v1/chat/completions SSE 路由：

~~~powershell
$env:VIDEO_API_KEY = '<Gateway API Key>'
python tests/qwen3_0_6b/test_end2end.py --config tests/qwen3_0_6b/test.yml
~~~

## Mage-VL-4B

组件测试以 meta device 对照模型结构和 checkpoint keys，并直接执行注意力、文字 prefill/decode 数值对照，不载入完整权重：

~~~powershell
python tests/mage_vl_4b/test_components.py --config tests/mage_vl_4b/test.yml
~~~

性能对照需要启动 Mage-VL Inference API，并确认 GPU 显存足以同时装载服务模型和 Transformers baseline：

~~~powershell
python tests/mage_vl_4b/test_performance.py --config tests/mage_vl_4b/test.yml
~~~

端到端脚本需要启动 Gateway 与 Mage-VL Inference API、提供 API Key，并准备 YAML 指定的图片和视频素材：

~~~powershell
$env:VIDEO_API_KEY = '<Gateway API Key>'
python tests/mage_vl_4b/test_end2end.py --config tests/mage_vl_4b/test.yml
~~~

## 可重复的算子分析与优化记录

一次完整循环按以下顺序执行：

1. 跑目标模型 test_components.py，确认模块对照及各自前向入口通过；Qwen3 额外启用 --profile-attention。
2. 记录 commit、模型 alias/revision、GPU、driver、CUDA、PyTorch、dtype、batch、token shape、warmup、iteration、时间和峰值显存。
3. 用 Profiler 表和 CUDA Event 识别热点，分清 CPU launch/metadata、KV gather 与 padding、矩阵计算/softmax、显存分配等部分。
4. 在根目录 `analysis/<model_alias>/` 的 `analysis.md` 与 `analysis.xlsx` 中，以一张表严格按“问题、解决方案、效果”说明；Markdown 每行引用 Excel 中的证据 ID。算子实现变更后，在对应模型的表中补充前后测量和验收结果。
5. 每次只改一个算子或数据路径，重复组件对照、同 shape profile，再跑适用的模型性能及 Gateway 端到端验收。

单层 profile 的单位、shape 与限制必须和完整模型/API 数据分开。Profiler 负责诊断；正式前后性能应在不启用 Profiler 的情况下用 CUDA Event 或端到端脚本重复测量。脚本只打印结果，不自动覆盖分析记录；历史结果按时间追加，不把不同硬件、shape、dtype 或配置的数字直接比较。

tests/client.py 是面向调用方的最小示例：它使用 OpenAI SDK 调用普通聊天，并通过同一个 /v1/chat/completions 接口完成 tools → 客户端执行函数 → role: tool 回传的循环。视频理解示例通过环境变量 VIDEO_FILE 启用，使用现有 /v1/videos/understand 路由和 mage-vl-4b 模型。

~~~powershell
$env:VIDEO_API_KEY = '<Gateway API Key>'
python tests/client.py
$env:VIDEO_FILE = '<本机 MP4 文件路径>'
python tests/client.py
~~~

工具执行示例只用于演示协议循环，不代表模型输出即获得执行授权。真实应用必须通过自身已注册工具清单执行，并实施参数校验、权限、超时、结果大小和最大循环轮数限制。所有测试均不执行数据集精度评估；服务、密钥或素材缺失时，只能报告静态/组件检查，不能宣称端到端验收通过。
