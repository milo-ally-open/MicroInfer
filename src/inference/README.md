# 推論服務設定與啟動

推論服務由 `launcher.py` 啟動。`model_registry.yaml` 將模型別名連結到推論設定、網路設定、媒體伺服器設定及 API 服務入口。一般啟動時，啟動器依登錄表載入這些路徑；進行實驗或切換設定時，也可以在命令列逐一指定設定檔。

## 設定檔分工

| 設定檔 | 用途 | 主要內容 |
| --- | --- | --- |
| `inference.yml` | 模型與推論行為 | 模型識別碼與 revision、權重載入、精度與量化、生成參數、請求限制、視訊處理及推論最佳化 |
| `network.yml` | API 與串流網路端點 | API 綁定位址與連接埠、MediaMTX 的 RTSP／RTMP 位址，以及回傳給推流端的公開主機名稱 |
| `mediamtx.yml` | MediaMTX sidecar | 推流協定、監聽連接埠及 MediaMTX 回呼 API 設定 |

三個設定檔目前位於 `src/inference/mage_vl_4b/configs/`；Python 依賴清單與鎖定檔位於 `src/inference/mage_vl_4b/requirements/`。`model_registry.yaml` 保存預設設定路徑及 `main.py` 服務入口；明確指定命令列設定時，該次啟動會以命令列路徑覆蓋相應的登錄表路徑。

## 完整啟動順序

啟動依賴順序如下：

**啟用推論環境 → 啟動 Mage-VL API（缺少權重時自動下載）→ 確認 API 就緒 →（即時推流時）啟動 MediaMTX → 客戶端呼叫 API**

### 啟用環境並確認 GPU

在 WSL 專案根目錄執行。以下流程假設 `magevl-transformers` Conda 環境與鎖定依賴已建置完成：

```bash
conda activate magevl-transformers
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
python src/inference/launcher.py list
```

確認 PyTorch 顯示 CUDA 可用，且登錄表中的模型與設定檔狀態正常。若是首次建置環境，請先依 `requirements/requirements-lock.txt` 和目標主機的 CUDA 環境完成套件安裝，再繼續以下步驟。

### 啟動 Mage-VL API

確認網路可連線至 Hugging Face 後，在終端機 A 從專案根目錄啟動服務。此命令明確指定模型及三個設定檔；啟動器會使用 `model_registry.yaml` 中登錄的 `service_entrypoint`：

```bash
python src/inference/launcher.py start \
  --model mage-vl-4b \
  --inference-config src/inference/mage_vl_4b/configs/inference.yml \
  --network-config src/inference/mage_vl_4b/configs/network.yml \
  --media-config src/inference/mage_vl_4b/configs/mediamtx.yml
```

啟動器會先載入設定，再由 Hugging Face Hub 準備 `inference.yml` 指定的固定 revision。模型快照會存放在專案的 `checkpoints/`；快取已有完整快照時會直接重用，缺少快照時會自動下載。下載及模型載入期間 API 尚未就緒，因此首次啟動需要等待較久。此命令會持續佔用終端機 A，請保持程序運行。若只需使用登錄表的預設設定，可簡寫為 `python src/inference/launcher.py start --model mage-vl-4b`。

### 確認 API 就緒

開啟終端機 B，等終端機 A 完成模型載入後檢查健康狀態：

```bash
curl --fail http://127.0.0.1:8001/health
curl --fail http://127.0.0.1:8001/v1/models
```

`/health` 回報 `status: ok` 後，再從客戶端呼叫圖片理解或視訊上傳 API。API 預設監聽 `127.0.0.1:8001`，綁定位址與連接埠由 `network.yml` 控制。

### 即時推流時再啟動 MediaMTX

只有使用 `/v1/live/*` 即時推流 API 才需要 MediaMTX；圖片理解及上傳視訊檔案不需要它。必須先讓 API 就緒，再於終端機 B 執行 MediaMTX，因為 MediaMTX 會呼叫 API 的授權回呼：

```bash
mediamtx src/inference/mage_vl_4b/configs/mediamtx.yml
```

MediaMTX 啟動後，客戶端才建立即時工作階段、依回應中的 RTSP／RTMP 位址推流，並透過 API 取得理解結果。若調整 `network.yml` 中的 API 或串流端點，請同步確認 `mediamtx.yml` 的授權回呼及監聽連接埠。

### 停止服務

若有使用即時推流，先在終端機 B 按 `Ctrl+C` 停止 MediaMTX，再在終端機 A 按 `Ctrl+C` 停止 API 並釋放 GPU 資源。若只使用圖片或視訊上傳 API，停止終端機 A 的 API 即可。

## 模型與 API

模型快取位於專案的 `checkpoints/`，Hugging Face 權重及快取檔案會由 `.gitignore` 排除；`checkpoints/val.md` 與 `checkpoints/val.xlsx` 例外保留為可追蹤的工作流文件。`runtime.local_files_only: false` 允許服務在指定 revision 尚未快取時自動下載。

目前登錄並實作的推論引擎為 Transformers，由 `default_backend` 及 `inference.yml` 中的 `backend.name` 標示。API 路由如下：

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`：文字／圖片請求，支援 SSE 串流文字輸出。
- `POST /v1/videos/understand`：上傳視訊檔案並取得理解結果。
- `/v1/live/*`：即時視訊工作階段 API；需另行啟動 MediaMTX。

`requirements/requirements.txt` 列出直接依賴，`requirements/requirements-lock.txt` 固定環境套件版本。瀏覽器展示頁與 Python 客戶端範例位於 `examples/`。
