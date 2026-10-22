# VideoStreamAPI

## 元件職責

- `src/gateway/`：Go API 閘道 v1，負責客戶端 API Key、租戶限流、模型別名、請求轉送、SSE／視訊代理與可觀測性；實際能力與啟動方式見 [`src/gateway/README.md`](src/gateway/README.md)。
- `src/inference/launcher.py`：列出模型與推論後端、選取設定並啟動推論 API。
- `src/inference/model_registry.yaml`：維護模型別名、模型／網路／媒體設定檔與服務入口之間的對應關係。
- `src/inference/models/mage_vl_4b/configs/`：集中存放模型推論、網路與 MediaMTX YAML。
- `src/inference/models/mage_vl_4b/requirements/`：保存目前 Mage-VL-4B 的 Python 依賴清單與鎖定檔。
- `src/inference/models/qwen3_0_6b/`：Qwen3-0.6B 純文字聊天服務、Hugging Face 相容載入層、本地模型計算圖、批次 Runner 及實體 KV page pool。
- `src/inference/api/`、`engine/`、`media/`、`models/<model>/`：推論引擎依接口門面、編排調度、執行抽象、模型運行時、模型與算子五層職責組織；五層是程式邊界，不代表五個程序。當前 API Server、Engine Core 和 Model Worker 為三個程序。
- 新模型放入 `src/inference/models/<model>/`，並在 `model_registry.yaml` 登錄其 YAML 路徑及 `app.py`；共用 API、Runtime 和啟動器不需為模型新增分支。
- Gateway 預設監聽 `127.0.0.1:8000`；推論服務預設監聽 `127.0.0.1:8001`，兩者仍是分離程序。

Inference Engine 借鑑 vLLM／nano-vLLM 的元件職責，沒有 vLLM backend、自訂 PagedAttention kernel 或 prefix cache。五層模組歸屬、輸入與文字流輸出契約及支援範圍見 [`src/inference/README.md`](src/inference/README.md) 與 [Inference Spec](docs/Specifications/inference/)。Mage-VL 使用 Transformers Runtime；Qwen3-0.6B 實驗服務使用本地 PyTorch Runner。兩條模型路徑共用 API／Engine 職責，不混用模型專屬計算圖或把未實作後端標成已支援。

## 啟動指定模型設定

啟動順序為：啟用 Conda 環境、啟動 API（若缺少指定 revision，服務會自動下載模型權重至專案 checkpoints/）、確認健康狀態；使用即時推流時，再啟動 MediaMTX。完整步驟、命令與兩個程序的先後關係請參閱 [`src/inference/README.md`](src/inference/README.md)。以下為 API 啟動的簡例。在 WSL 中啟用對應的 Conda 環境，並從專案根目錄執行：

```bash
# 列出已登錄的模型與後端
python src/inference/launcher.py list

# 指定模型別名與後端啟動
python src/inference/launcher.py start --model mage-vl-4b
python src/inference/launcher.py start --model qwen3-0.6b
```

預設設定由 `model_registry.yaml` 登錄。若要讓實驗命令明確記錄全部設定檔，可在啟動時分別傳入 `--inference-config`、`--network-config` 及 `--media-config`；各檔案用途與完整命令範例請參閱 [`src/inference/README.md`](src/inference/README.md)。

`src/inference/models/mage_vl_4b/requirements/requirements.txt` 宣告直接相依套件；同目錄中的 `requirements-lock.txt` 記錄固定版本。模型快取使用專案內的 `checkpoints/`，Hugging Face 權重與快取檔案會被 `.gitignore` 排除；`val.md` 和 `val.xlsx` 則保留為可追蹤的工作流文件。啟動 API 時若固定 revision 尚未快取，服務會自動下載至該目錄。

## 目前提供的 API

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`：相容 OpenAI 格式的文字／圖片請求並支援 SSE；Mage-VL 要求圖片，Qwen3-0.6B 僅接受文字。
- `POST /v1/videos/understand`：上傳視訊檔案並取得理解結果。
- `/v1/live/*`：即時視訊工作階段 API；RTSP／RTMP 推流另需啟動 MediaMTX sidecar。

Qwen3-0.6B 是已登錄的純文字模型，上游權重位於 `checkpoints/Qwen3-0.6B-c1899de/`；其直接推論 API 預設使用 `127.0.0.1:8011`，Mage-VL 仍使用 `127.0.0.1:8001`。Qwen 使用本地 PyTorch 計算圖、批次 prefill/decode 與 Worker 實體 KV page pool；Mage-VL 維持原有 Transformers Runner 與邏輯 KV reservation。Inference Engine 借鑑 vLLM／nano-vLLM 的元件職責，沒有 vLLM backend、自訂 PagedAttention kernel 或 prefix cache。模型下載、服務啟動及 Qwen SSE 性能採樣命令見 [`scripts/README.md`](scripts/README.md)。根目錄 `example.py` 仍示範透過 Gateway 呼叫 Mage-VL 圖片聊天；Gateway 與推論服務的分層測試及性能基線方式見 [`src/gateway/README.md`](src/gateway/README.md)。
