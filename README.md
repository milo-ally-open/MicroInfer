# VideoStreamAPI

## 元件職責

- `src/gateway/`：Go API 閘道 v1，負責客戶端 API Key、租戶限流、模型別名、請求轉送、SSE／視訊代理與可觀測性；實際能力與啟動方式見 [`src/gateway/README.md`](src/gateway/README.md)。
- `src/inference/launcher.py`：列出模型與推論後端、選取設定並啟動推論 API。
- `src/inference/model_registry.yaml`：維護模型別名、模型／網路／媒體設定檔與服務入口之間的對應關係。
- `src/inference/mage_vl_4b/configs/`：集中存放 `inference.yml`、`network.yml` 與 `mediamtx.yml`。
- `src/inference/mage_vl_4b/requirements/`：存放 Transformers 服務的依賴清單與鎖定檔。
- `src/inference/mage_vl_4b/`：Mage-VL-4B 的 Python/FastAPI 推論服務。
- Gateway 預設監聽 `127.0.0.1:8000`；推論服務預設監聽 `127.0.0.1:8001`，兩者仍是分離程序。

## 啟動指定模型設定

啟動順序為：啟用 Conda 環境、啟動 API（若缺少指定 revision，服務會自動下載模型權重至專案 checkpoints/）、確認健康狀態；使用即時推流時，再啟動 MediaMTX。完整步驟、命令與兩個程序的先後關係請參閱 [`src/inference/README.md`](src/inference/README.md)。以下為 API 啟動的簡例。在 WSL 中啟用對應的 Conda 環境，並從專案根目錄執行：

```bash
# 列出已登錄的模型與後端
python src/inference/launcher.py list

# 指定模型別名與後端啟動
python src/inference/launcher.py start --model mage-vl-4b
```

預設設定由 `model_registry.yaml` 登錄。若要讓實驗命令明確記錄全部設定檔，可在啟動時分別傳入 `--inference-config`、`--network-config` 及 `--media-config`；各檔案用途與完整命令範例請參閱 [`src/inference/README.md`](src/inference/README.md)。

`src/inference/mage_vl_4b/requirements/requirements.txt` 宣告直接相依套件；同目錄中的 `requirements-lock.txt` 記錄固定版本。模型快取使用專案內的 `checkpoints/`，Hugging Face 權重與快取檔案會被 `.gitignore` 排除；`val.md` 和 `val.xlsx` 則保留為可追蹤的工作流文件。啟動 API 時若固定 revision 尚未快取，服務會自動下載至該目錄。

## 目前提供的 API

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`：相容 OpenAI 格式的圖片／圖文請求，支援以 SSE 串流輸出文字；目前實作要求至少一個 `image_url`，純文字請求會回 HTTP 400。
- `POST /v1/videos/understand`：上傳視訊檔案並取得理解結果。
- `/v1/live/*`：即時視訊工作階段 API；RTSP／RTMP 推流另需啟動 MediaMTX sidecar。

根目錄 `example.py` 使用 OpenAI Python SDK，示範透過 Gateway 呼叫一次圖片聊天。Go Gateway v1 已實作；SGLang、vLLM 與其他模型仍未實作。Gateway 與推論服務的分層測試及性能基線方式見 [`src/gateway/README.md`](src/gateway/README.md)。
