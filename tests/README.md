# 测试与验收

自动化测试只保留 Go Gateway 集成测试。Python 模型代码不使用 pytest：新增或拆解模型组件时，在组件文件的 `if __name__ == "__main__"` 入口中写该组件自己的前向测试、必要的对齐检查和小规模基准。

## 目录约定

| 位置 | 内容 | 运行条件 |
| --- | --- | --- |
| `tests/integration/` | Go Gateway 对外 Handler 集成测试；通过公开 API 验证路由鉴权与健康检查 | Go；本地构造 Handler，不需要启动模型服务 |
| `src/inference/models/<model>/layers/` | 算子及计算层的模块级前向验证入口 | 对应模型依赖；由工程师在已分配的 GPU 时段手动运行，不进入任何 CI Runner |
| `src/inference/models/<model>/components/` | Tokenize、Encode、Runner 等运行组件的模块级验证入口 | 由工程师按组件前置条件手动运行；不进入任何 CI Runner |
| 根目录 `client.py` | 手动调用真实 Gateway 的 OpenAI SDK 冒烟示例，可选视频请求 | Gateway、Inference、API Key；模型请求需本地权重 |

组件入口应直接测试当前文件负责的行为，并对照 Transformers 官方组件或实现检查输入输出形状、dtype、数值误差和边界条件。完整模型的 Test、Bench、Profile 脚本仍放在 `scripts/<model_alias>/`；它们不替代模块级测试或真实服务验收。

## 手动运行 Python 模型测试

Python 组件测试以及模型级 Test、Bench、Profile **均由工程师手动运行，不加入 GitHub Actions、GitLab CI 或其他 Runner**。这些工作需要本地模型权重和 CUDA，且会占用 GPU/显存；让 Runner 自动并发调度会与推理服务及其他实验争抢资源。Gateway 的 Go CI 不负责这些模型测试。

### 运行前确认

1. 确认自己已获得 GPU 测试时段，并与正在使用该 GPU 的推理服务或实验协调；不要在已有负载上直接启动 Profile 或 Benchmark。
2. 激活目标模型对应的 Python 环境，确认 PyTorch、CUDA 和模型依赖已安装。
3. 在仓库根目录运行 `nvidia-smi`，记录 GPU 型号、显存占用和正在运行的进程；确认 checkpoint 和脚本所需图片已在配置路径中。
4. Profile 会增加执行开销并产生 `artifacts/` 输出；保证有足够磁盘空间。不要把权重、Profiler trace 或运行时产物提交到 Git。

### 组件级 Test、Bench、Profile

每个组件文件的 `__main__` 入口针对最小输入完成该模块的测试、基准或 PyTorch Profiler。根据本次要检查的组件选择对应入口，例如：

```bash
python src/inference/models/qwen3_vl_2b/layers/text_attention.py
python src/inference/models/mage_vl_4b/layers/vision_attention.py
```

如果目标组件的入口需要 checkpoint、额外素材或不同 Python 环境，先按该文件注释准备；不要通过 Runner 代跑。

### 模型级前向与算子分析

从仓库根目录按模型单独运行。`forward.py` 会加载本地模型配置/权重并产生对照、Bench 和 Profile 结果；随后运行 `analysis.py` 汇总脚本产物：

```bash
# Qwen3-VL-2B：生成 Transformers / Inference 前向与分析产物
python scripts/qwen3_vl_2b/forward.py
python scripts/qwen3_vl_2b/analysis.py

# Mage-VL-4B：生成前向与分析产物
python scripts/mage_vl_4b/forward.py
python scripts/mage_vl_4b/analysis.py --input artifacts/mage-vl-image-forward.json
```

每次记录 commit、模型 revision、GPU/驱动、CUDA、PyTorch、dtype、输入形状、执行时段和 GPU 并发情况。将分析结论写入 `analysis/<model_alias>/analysis.md`，表格数据由对应脚本生成；权重与临时 trace 只保留在本机。

模型服务端到端测试同样由工程师在已分配资源后手动执行，例如启动本地 Inference/Gateway 后运行根目录 `client.py`。不要将 Python 服务、权重下载或 GPU 模型测试追加进 Gateway Runner 工作流。

## 运行 Go Gateway 集成测试

在仓库根目录执行：

```bash
cd tests/integration
go test -race -count=1 ./...
go vet ./...
```

`tests/integration/go.mod` 将 Gateway 作为本地模块依赖，不把 Go 测试文件放回源码目录。测试构造 Gateway Handler 并经公开 HTTP 接口验证行为，不会启动 Inference 或加载模型。

## 运行组件入口

在仓库根目录用对应模型环境执行需要检查的组件文件，例如：

```bash
python src/inference/models/qwen3_vl_2b/layers/text_attention.py
```

每个模块的 main 入口负责构造最小输入并验证输出；按其注释准备模型依赖或 checkpoint。实际文件名和所需资源以目标组件为准。

## 运行真实服务冒烟

启动目标 Inference 与 Gateway，设置 `VIDEO_API_KEY` 和可选的 `GATEWAY_URL`，然后在仓库根目录运行 `python client.py`。需要验证视频上传时再设置 `VIDEO_FILE`。该调用涉及真实模型和服务，结果不属于 Go Gateway 单元／集成测试的替代品。
