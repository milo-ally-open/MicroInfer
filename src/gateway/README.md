# Go API 閘道

Go Gateway v1 已實作，負責外部客戶端 API Key、租戶限流、模型別名、HTTP 請求政策、上游錯誤轉換、SSE／視訊代理與請求觀測。推論服務仍獨立管理模型權重、GPU、生成及視訊前處理；RTSP／RTMP 媒體封包由客戶端直接傳至 MediaMTX。

## 目前能力

- Gateway 目前登錄 `mage-vl-4b`（上游 `microsoft/Mage-VL`）及 `qwen3-0.6b`（上游 `qwen3-0.6b`），分別連接各自的推論服務。
- 能力依模型路由設定：Mage-VL 接受圖片聊天、文字 SSE、影片上傳和 Live；Qwen3-0.6B 接受純文字聊天與文字 SSE。Gateway 會在上游轉送前拒絕不支援的輸入／路由。
- `GET /healthz` 僅檢查 Gateway 存活；`GET /readyz` 另檢查推論服務 `GET /health` 的 `status=ok`。
- 需要 Bearer API Key 的客戶端 API：`GET /v1/models`、`POST /v1/chat/completions`、`POST /v1/videos/understand`、`POST /v1/live/sessions`、live events SSE 與 `DELETE` 停止路由。
- 聊天 SSE 逐事件轉送，並將分塊中的上游 model 欄位改寫為專案別名。視訊 multipart 本文以有界串流送往推論服務，不整段載入 Gateway 記憶體。
- 聊天 JSON 的 `model` 使用公開別名；影片上傳可用 `?model=<alias>` 選擇已登記模型，預設由 `defaults.video_model` 指定。Live 建立本文可帶 `model`，預設使用 `defaults.live_model`；簽名 session handle 綁定建立時的模型路由。
- Live 工作階段控制 ID 是綁定租戶、建立時模型別名、上游 session ID 和有效期的 HMAC 句柄。`publish.rtsp`／`publish.rtmp` 維持直連 MediaMTX；Gateway 不代理媒體封包。
- OpenAI 風格錯誤、每租戶程序內 token bucket、JSON 結構化日誌，以及不含 API Key／租戶標籤的 Prometheus 格式 `/metrics`。

## 目錄

```text
src/gateway/
├── cmd/gateway/main.go            # 設定、密鑰載入與 HTTP server lifecycle
├── configs/gateway.yml            # 非敏感 Gateway 設定
├── gateway.go                     # 應用組裝、認證政策、API 業務流程及錯誤回應
└── internal/
    ├── auth/                      # API Key、HMAC live 句柄
    ├── config/                    # 嚴格 YAML 載入與設定驗證
    ├── middleware/                # 單實例租戶 token bucket
    ├── observability/             # 請求與延遲 metrics
    ├── route/                     # API 路徑與 HTTP method 分派
    ├── stream/                    # 聊天 SSE 改寫、限界解析與即時轉送
    ├── upstream/                  # 推論上游 HTTP transport 與請求建構
    └── videoproxy/                # multipart 請求長度檢查及串流代理
```

`gateway.go` 建立並組合以上元件；路由表、上游 transport、SSE 轉送及影片 multipart 代理分別由對應套件負責。API Key／租戶政策、live 句柄簽章與業務流程仍由 Gateway 應用層負責。

## 設定與密鑰

所有服務位址、公開 URL、模型映射／能力、本文限制、逾時、連線池／串流 buffer、限流、指標和 listener 設定都在 [`configs/gateway.yml`](configs/gateway.yml)。預設 Gateway 綁定 `127.0.0.1:8000`，Mage-VL 上游預設 `http://127.0.0.1:8001`，Qwen3-0.6B 上游預設 `http://127.0.0.1:8011`。Gateway 限制不得高於推論端自己的 YAML 上限；推論端的 `Content-Length` 規則仍由推論服務負責。

API Key 以 JSON array 放在 Git 工作樹以外的受保護檔案，透過 `GATEWAY_API_KEYS_FILE` 指向該檔。格式如下，請替換佔位值並使用隨機密鑰：

```json
[
  {"id": "local-client", "tenant_id": "demo", "key": "<至少 16 字元的隨機 API Key>"}
]
```

`GATEWAY_SESSION_HMAC_KEYS` 接受以逗號分隔的 base64url 密鑰，解碼後每把至少 32 bytes。第一把用於新句柄簽章，後續密鑰只用於驗證舊句柄。輪替時，保留舊密鑰至少至所有舊句柄過期；`handle_ttl_seconds` 預設 3660 秒，需與推論端 live session 最長期限保持相容。不得將 API Key 或 HMAC 密鑰寫進追蹤中的 YAML／JSON。

## 啟動

先依 [`src/inference/README.md`](../inference/README.md) 啟動推論 API。Gateway 可以在推論服務尚未就緒時啟動，但 `/readyz` 會回 `503`。

PowerShell 範例：

```powershell
$env:GATEWAY_API_KEYS_FILE = 'C:\secrets\gateway-api-keys.json'
$env:GATEWAY_SESSION_HMAC_KEYS = '<base64url-secret>[,<previous-base64url-secret>]'
Set-Location src/gateway
go run ./cmd/gateway --config configs/gateway.yml
```

從專案根目錄呼叫：

```powershell
curl.exe http://127.0.0.1:8000/healthz
curl.exe http://127.0.0.1:8000/readyz
```

`/readyz` 會檢查每個不同的已配置推論上游；只有全部回報 ready 才回成功。`/metrics` 預設啟用，提供請求量、錯誤類別、延遲 histogram、SSE 首文字 token／串流時間及視訊本文讀取完成時間。Prometheus 可用 histogram buckets 計算 p50、p95、p99；metrics 不使用原始 API Key 或租戶 ID 作為標籤。限流狀態保存在單一程序記憶體，重啟後重設；首版不支援多實例共享限流。

## 測試與端到端範例

從專案根目錄執行：

```powershell
Push-Location src/gateway
go test ./...
Pop-Location

Push-Location tests/gateway
go test ./...
Pop-Location

python -m unittest discover -s tests/inference -v
python -m unittest discover -s tests/end2end -v
```

`tests/gateway/` 使用 `httptest` 模擬上游；`tests/end2end/` 保留 mock／stub 測試，另有連接真實 Gateway 與推論服務的性能工具。2026-10-22 已以低顯存設定連接 Qwen3-0.6B 執行文字 JSON／SSE、雙請求並發與客戶端斷流回收 smoke；同日也透過真實 Mage-VL 完成圖片聊天 JSON／SSE、視訊上傳與 Live／MediaMTX 合成 RTSP smoke。Mage-VL 設定、結果和限制見 [`checkpoints/val.md`](../../checkpoints/val.md)、[`checkpoints/val.xlsx`](../../checkpoints/val.xlsx) 及 [`tests/end2end/results/`](../../tests/end2end/results/)。Live 僅有單次功能驗證；正式容量、高併發和媒體授權邊界仍未測。

根目錄的 `example.py` 只示範一次 OpenAI SDK 圖片聊天呼叫。此範例使用 `test.jpg`，因目前推論端的聊天 API 要求圖片輸入。若執行環境尚未安裝 OpenAI Python SDK，先安裝套件；Gateway 與推論服務就緒後，在專案根目錄執行：

```powershell
python -m pip install openai
$env:VIDEO_API_KEY = '<client API Key>'
$env:GATEWAY_URL = 'http://127.0.0.1:8000'
python example.py
```

端到端性能程式獨立位於 `tests/end2end/performance_baseline.py`，不由 `example.py` 提供客戶端功能。Qwen 純文字 SSE 不需媒體素材：

```powershell
python tests/end2end/performance_baseline.py --mode text-sse --model qwen3-0.6b --requests 3 --concurrency 1 --output tests/end2end/results/qwen3-0.6b-gateway-smoke-2026-10-22.json
```

Mage-VL 圖片／影片測試需已啟動相應服務並提供 API Key、`test.jpg` 和 `test.mp4`：

```powershell
python tests/end2end/performance_baseline.py --mode mixed --requests 10 --concurrency 1 --output results/gateway-baseline.json
```

報告記錄模型別名、吞吐、錯誤率、p50／p95／p99 延遲、SSE 首文字 token 延遲及視訊本文送完時間；v1 不設硬性 SLA，也不做模型精度評測。Qwen smoke 條件與結果見 [`tests/end2end/README.md`](../../tests/end2end/README.md)；Mage-VL 圖片／影片各有一次真實 Gateway smoke 記錄於 `checkpoints/val.*`，Mage JSON 與 Live／MediaMTX 合成 RTSP 報告置於 `tests/end2end/results/`。各項樣本不足以作為穩定性能基線。

## 部署與回復

初始部署保持推論 API 綁定 loopback，只讓受控前置入口連接 Gateway。TLS 可由前置 Proxy 終止，並將 `public_base_url` 設為客戶端使用的 HTTPS URL。回復時先停止外部入口並停止作用中的 live session，再停止 Gateway 或回復 `gateway.yml`；推論服務和 MediaMTX 不需隨 Gateway 一起回滾。
