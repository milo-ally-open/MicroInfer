# 模型下載與探索工具

共用 Hugging Face 下載器保留在 `scripts/download.py`。每個模型的探索、診斷及能力擷取工具放在 `scripts/<model_alias>/`，避免不同模型的設定、輸入格式和報告邏輯混在一起。目前工具位於 `scripts/mage_vl_4b/` 及 `scripts/qwen3_0_6b/`。

## Hugging Face 模型下載工具

`download.py` 會從 Hugging Face Hub 下載指定模型 revision 的檔案，並放入明確指定的本機目錄。預設下載該 revision 的完整快照，包含權重、設定、Tokenizer 與模型程式碼等檔案；下載中斷後再次執行會沿用 Hub 快取及目標目錄中的既有檔案。

## 安裝依賴

在專案根目錄執行：

```bash
python -m pip install "huggingface_hub==1.33.0"
```

此版本與 Mage-VL Transformers 環境目前鎖定的 `huggingface_hub` 版本一致。

## 下載完整模型

每次執行都必須明確提供模型 `repo_id`、revision 與下載目錄。建議 revision 使用 commit SHA，確保後續可重現：

```bash
python scripts/download.py \
  --repo-id microsoft/Mage-VL \
  --revision d88b153285f1633a61b2f693c59c8576693af185 \
  --local-dir checkpoints/microsoft-Mage-VL-d88b153
```

`--local-dir` 是模型檔案的實際目標目錄。專案根目錄的 `.gitignore` 會忽略 `checkpoints/` 下的模型權重與快取，同時保留 `checkpoints/val.md` 及 `checkpoints/val.xlsx`。

## 私有模型與存取權杖

公開模型不需要權杖。下載私有或受限模型時，先將 Hugging Face 權杖放入環境變數，不要直接寫進命令列或文件：

```bash
export HF_TOKEN=hf_your_token
python scripts/download.py \
  --repo-id organization/private-model \
  --revision 0123456789abcdef0123456789abcdef01234567 \
  --local-dir checkpoints/private-model
```

若使用的環境變數名稱不是 `HF_TOKEN`，以 `--token-env NAME` 指定名稱。

## 選擇檔案與下載參數

可重複提供 `--allow-pattern`，讓 Hub 只下載符合模式的檔案；也可用 `--ignore-pattern` 排除檔案。若沒有指定模式，則下載完整快照。部分下載可能缺少模型執行所需的設定或程式碼，使用前請確認模型格式與推論服務的需求。

```bash
python scripts/download.py \
  --repo-id microsoft/Mage-VL \
  --revision d88b153285f1633a61b2f693c59c8576693af185 \
  --local-dir checkpoints/mage-vl-config-only \
  --allow-pattern '*.json' \
  --allow-pattern '*.py' \
  --ignore-pattern '*safetensors*'
```

`--max-workers N` 可調整並行下載數，預設為 `8`。需要重新抓取本機已有檔案時，加入 `--force-download`。執行 `python scripts/download.py --help` 可查看完整參數。

Qwen3-0.6B 固定使用官方 `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`。從專案根目錄下載完整快照：

```powershell
python scripts/download.py `
  --repo-id Qwen/Qwen3-0.6B `
  --revision c1899de289a04d12100db370d81485cdf75e47ca `
  --local-dir checkpoints/Qwen3-0.6B-c1899de
```

下載目錄被 `.gitignore` 排除，不會提交模型權重。

## 擷取 Mage-VL 能力資訊

`scripts/mage_vl_4b/test.py` 是 Mage-VL 專屬的本機探索工具，可記錄固定 revision 的模型類別與結構、處理後輸入 tensor、逐步 KV cache、GPU 記憶體及 prefill／decode 計時。每次執行都必須明確提供 `--config`、`--model-path`、`--image` 或 `--video`、`--question` 及 `--output`；腳本不從自身位置推導專案目錄，也不預設同目錄 YAML。所有相對路徑均以啟動命令時的目前工作目錄解析。YAML 沿用正式服務的 `model`、`processor`、`runtime`、`api`、`generation` 和 `optimization` 區段，獨立調整探索時使用的模型與輸入限制。輸出 token 數預設取自明確指定 YAML 的 `generation.max_tokens_default`，`--max-new-tokens` 可覆寫。工具不從 `src/` 匯入程式，推理流程直接使用已安裝的 PyTorch、Transformers 與明確指定的本機模型快照。

腳本只接受本機圖片或影片，不呼叫推論 HTTP API。它會強制設定 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，並以 `local_files_only=True` 從 `--model-path` 指定的本機快照載入 Processor 與 Model。模型目錄無效時會報錯退出，不會搜尋快取或下載模型。執行環境需具備推理環境依賴、CUDA 和模型快照；影片探測另需 OpenCV。

從專案根目錄執行：

```powershell
python scripts/mage_vl_4b/test.py `
  --config scripts/mage_vl_4b/test.yaml `
  --model-path checkpoints/microsoft-Mage-VL-d88b153 `
  --image path/to/frame.jpg `
  --question "描述畫面中的人物正在做什麼。" `
  --max-new-tokens 8 `
  --output artifacts/mage-vl-image-probe.json

python scripts/mage_vl_4b/test.py `
  --config scripts/mage_vl_4b/test.yaml `
  --model-path checkpoints/microsoft-Mage-VL-d88b153 `
  --video path/to/clip.mp4 `
  --question "總結影片中發生的事件。" `
  --max-new-tokens 8 `
  --output artifacts/mage-vl-video-probe.json
```

`scripts/mage_vl_4b/test.yaml` 預設依目前 RTX 4060 8 GiB 顯卡採用 int8 CPU offload、GPU 5 GiB／CPU 6 GiB 放置上限、448 px 輸入與最多 8 個輸出 token；這是探索工具的低負載設定，不等於正式服務預設。`--config` 必須指向相同結構且使用 `video_backend: frames` 的 Mage-VL YAML，`--model-path` 必須指向已存在的本機快照目錄。若從其他目錄啟動，請傳入相對於該目前工作目錄的路徑或絕對路徑。輸出 JSON 不包含媒體 bytes 或模型生成文字，但包含模型路徑、tensor metadata、計時和錯誤 traceback；請將報告留在本機 `artifacts/`，不要納入 Git。生成使用模型原生 `generate()`，並以 forward hook 記錄 prefill／decode 的耗時及 cache 長度。已在 RTX 4060、固定 Mage-VL revision 和本機圖片上完成一次低負載 GPU 探測；這項探測用於檢查模型執行介面和 cache 行為，不是模型精度評估或正式性能基線。

## 擷取 Qwen3-0.6B 能力資訊

`scripts/qwen3_0_6b/test.py` 只接受明確指定的本機權重、YAML、prompt 與 JSON 報告路徑，不從 `src/` 匯入程式，也不會下載模型。它使用 Qwen chat template 執行文字生成，記錄模型結構、權重載入時間、prompt／completion token 數、生成速度及 GPU 峰值記憶體。

```powershell
python scripts/qwen3_0_6b/test.py `
  --config scripts/qwen3_0_6b/test.yaml `
  --model-path checkpoints/Qwen3-0.6B-c1899de `
  --prompt "用兩句話說明什麼是推理引擎。" `
  --output artifacts/qwen3-0.6b-probe.json
```

輸出 JSON 包含生成文字，應保留於本機 `artifacts/`，不要納入 Git。此探測是單請求 GPU smoke test，不是精度評估或併發性能基線。

## Qwen3 API 性能採樣

`scripts/qwen3_0_6b/benchmark_api.py` 只透過 OpenAI Chat Completions HTTP/SSE 呼叫已啟動的 API，不匯入 `src/` 或載入第二份模型。每次執行須明確傳入 API 根網址、模型別名、prompt、請求數、併發數、輸出上限及 JSON 報告路徑；報告記錄 TTFT、inter-token latency、request/token throughput、request latency、硬體、套件版本與 `/health` 暴露的 KV／顯存狀態。

```powershell
python scripts/qwen3_0_6b/benchmark_api.py `
  --base-url http://127.0.0.1:8011 `
  --model qwen3-0.6b `
  --prompt "用兩句話說明推理引擎如何處理多個請求。" `
  --requests 4 `
  --concurrency 2 `
  --max-tokens 32 `
  --output artifacts/qwen3-0.6b-api-baseline.json
```

這是服務端到端的短期基線，不是模型精度評測或 SLA。每個 SSE 可見文字 chunk 代表一次輸出事件；Tokenizer special token 不產生可見文字時不計入 TTFT／inter-token 樣本。將報告留在本機 `artifacts/`，不要納入 Git。
