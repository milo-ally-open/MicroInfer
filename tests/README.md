# 逐模型测试

每个模型只维护三项可执行测试，测试参数、服务地址、素材及本机 snapshot 路径集中在该模型目录的 `test.yaml`：

1. `test_components.py`：以 `meta` device 比较模型包与 Transformers 原生模型的模组树，不载入权重、不使用 GPU。
2. `test_performance.py`：以同一输入比较 Inference API 与 Transformers 原生生成的延迟、输出 token 数及 token/s。需要自行启动 Inference API 并提供本机模型及 GPU；两份模型同时占用装置记忆体，执行前须确认显存足够。
3. `test_end2end.py`：由 Go Gateway 发送请求，验证 SSE 和公开模型别名；Mage-VL 另验证图片 SSE 与影片上传。需要自行启动 Gateway 和相应 Inference API，并透过环境变数提供 API Key。

所有脚本都要求明确传入设定档，例如：

```powershell
python tests/qwen3_0_6b/test_components.py --config tests/qwen3_0_6b/test.yaml
python tests/qwen3_0_6b/test_performance.py --config tests/qwen3_0_6b/test.yaml
$env:VIDEO_API_KEY = '<Gateway API Key>'
python tests/qwen3_0_6b/test_end2end.py --config tests/qwen3_0_6b/test.yaml
```

Mage-VL 将目录名替换为 `mage_vl_4b`。脚本把 JSON 摘要打印到标准输出，不会自动写入结果文件。需要归档时，可手动保存到同模型目录的 `results/`，并记录模型 revision、硬件和执行环境；不要把密钥、权重或媒体素材放入报告。这三项测试不执行资料集精度评估。`checkpoints/val.md` 和 `checkpoints/val.xlsx` 是受保护的验收工作流文件。
