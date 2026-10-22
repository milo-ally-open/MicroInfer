# 技術設計：Go API 閘道

> 關聯規格：[spec.md](spec.md)  
> 關聯提案：[proposal.md](proposal.md)  
> 任務追蹤：[tasks.md](tasks.md)  
> 狀態：Go Gateway v1 已依本設計實作；雙模型能力路由依 REQ-13 增補，驗收證據與尚未執行項目見 tasks.md。

## 當前實作與目標

- 當前程式位置／行為：src/gateway/ 已有 Go module、HTTP handler、設定 YAML、認證／設定／限流／觀測套件及 server lifecycle；實際檔案見 src/gateway/README.md。
- 現有推論服務：src/inference/api/service_app.py 依模型登錄組裝 /health、/v1/models、/v1/chat/completions，以及模型已啟用的影片／Live 路由。
- 現有模型路由：`mage-vl-4b` → `microsoft/Mage-VL`（port 8001），`qwen3-0.6b` → `qwen3-0.6b`（port 8011）；各模型能力在 Inference Registry 與 Gateway YAML 對應登錄。
- 目標邊界：Gateway 處理外部 HTTP/API Key/租戶政策/限流/路由/轉送/可觀測性；推論服務處理模型與推論；MediaMTX 處理 RTSP／RTMP 媒體流。

## Gateway 程式模組

| 模組 | 責任 |
|---|---|
| `gateway.go` | 組裝應用元件、執行認證與限流政策、實作聊天／即時工作階段業務流程及 OpenAI 風格錯誤 |
| `internal/route/` | 將公開 API 路徑和 HTTP method 分派至應用 handler，提供穩定的 route label |
| `internal/upstream/` | 建立共用 HTTP transport，轉送請求本文、`Accept`、`Last-Event-ID` 和 request ID，回傳上游延遲 |
| `internal/stream/` | 有界讀取 SSE 事件、改寫聊天模型別名、逐事件 flush、追蹤首 token／閒置與取消 |
| `internal/videoproxy/` | 驗證 multipart `Content-Length`，以串流方式轉送 request body，觀測本文讀取完成時間 |

上述套件只處理 Gateway 的 HTTP 邊界，不載入模型或代理 RTSP／RTMP 媒體封包。

## 元件與資料流

1. 客戶端對 Gateway 發送 HTTP API 請求並提供 Bearer API Key。
2. Gateway 驗證 API Key，取得伺服器端 tenant_id，再執行模型、請求大小、限流及逾時政策。
3. Gateway 依 src/gateway/configs/gateway.yml 將 API 請求送至推論服務；推論位址不可散落為程式常數。
4. 一般 JSON 回應由 Gateway 回傳；聊天 SSE 逐事件轉送並刷新，視訊 multipart 以有界串流轉送。
5. 即時工作階段控制 API 經 Gateway 轉送。Gateway 把推論服務的內部 session_id 包裝為租戶綁定的簽名句柄，並將事件／停止 URL 改寫為 Gateway URL。
6. 客戶端使用推論服務回傳的 RTSP／RTMP publish URL 直接連接 MediaMTX。MediaMTX 的授權回呼仍直接連接推論服務；Gateway 不承載媒體封包，也不更改回呼拓撲。

## 對外 HTTP 契約

### 系統端點

| 方法與路徑 | 認證 | 行為 |
|---|---|---|
| GET /healthz | 不需要 | Gateway 程序存活；不檢查模型是否就緒 |
| GET /readyz | 不需要 | 每個不同的已配置推論 URL 之 GET /health 都回報 `status=ok` 才成功；否則回 503 |

### 客戶端 API

| 方法與路徑 | 認證 | 上游路徑 | 設計要點 |
|---|---|---|---|
| GET /v1/models | Bearer API Key | GET /v1/models | 對外只列出設定的專案別名 |
| POST /v1/chat/completions | Bearer API Key | POST /v1/chat/completions | 以公開 model alias 路由；按 text/image/stream capability 驗證後改寫 upstream model ID |
| POST /v1/videos/understand?model=`{model_alias}` | Bearer API Key | POST /v1/videos/understand | 選用 query 選模型，省略則使用 YAML 預設；保留 multipart 原文串流 |
| POST /v1/live/sessions | Bearer API Key | POST /v1/live/sessions | JSON 選用 `model`；改寫工作階段控制識別碼及 URL；publish URL 仍指向 MediaMTX |
| GET /v1/live/sessions/{session_id}/events | Bearer API Key | GET /v1/live/sessions/{upstream_session_id}/events | 驗證租戶句柄；保留 Last-Event-ID 與 SSE 語義 |
| DELETE /v1/live/sessions/{session_id} | Bearer API Key | DELETE /v1/live/sessions/{upstream_session_id} | 驗證租戶句柄後停止上游工作階段 |

/v1/live/mediamtx/auth 是推論服務提供給 MediaMTX 的內部回呼，不作為 Gateway 客戶端路由。

## 設定設計

Gateway 非敏感設定位於 src/gateway/configs/gateway.yml。設定至少涵蓋：

| 類別 | 設定責任 |
|---|---|
| listener | host、port；預設 127.0.0.1:8000 |
| upstream | 共用推論服務 fallback URL；本機預設值 http://127.0.0.1:8001 |
| defaults | 視訊及 Live 的預設模型別名 |
| model routes | 每模型公開別名、上游模型識別碼、上游 URL 和 text/image/stream/video/live 能力；目前登錄 Mage-VL 與 Qwen3-0.6B |
| request limits | 聊天、視訊 multipart、即時工作階段本文上限 |
| timeouts | 上游連線、一般回應、SSE 閒置及整體逾時 |
| tenant policies | 每租戶請求速率、突發容量及必要的並行限制 |
| observability | 日誌等級、request ID、指標輸出設定 |
| secret references | API Key、簽名密鑰的環境變數名稱或掛載檔案路徑；不可包含秘密值 |
| transport／buffers | 上游連線池、keepalive、Expect-Continue 及請求標頭／串流 buffer 上限 |

所有可調整政策由 YAML 提供；必要欄位缺失或範圍無效時，Gateway 啟動失敗。Gateway 的視訊本文上限不可高於推論端 inference.yml 的 max_video_request_bytes；聊天及即時工作階段亦依其上游本文限制設定。推論設定仍由推論服務自己的 YAML 管理，不搬入 Gateway。

## 認證與租戶

- Authorization: Bearer 提供 API Key。Gateway 從安全注入的來源載入有效密鑰及其 tenant_id 對應。
- 客戶端提交的租戶標頭不具權限來源；Gateway 不採信其覆蓋已驗證身分。
- 比較密鑰時採用適當的常數時間比較；日誌只記錄安全的租戶代號或 Key ID，不記錄原始 API Key。
- /healthz 與 /readyz 不要求客戶端 API Key；所有 /v1/* 客戶端 API 都要求有效密鑰。
- 首版只支援單實例；租戶限流器位於程序記憶體，重啟後計數重置。多實例共享狀態不在首版範圍。

## 模型路由與回應改寫

- Gateway 啟動時載入每模型路由。聊天請求以 `model` 選公開別名，未設定或未知 alias 在呼叫上游前以 HTTP 400 拒絕；依 capability 拒絕純文字 Mage 請求及含圖片的 Qwen 請求，兩者都支援文字 SSE。
- 聊天上游依路由使用 `microsoft/Mage-VL` 或 `qwen3-0.6b`，回應 JSON／SSE 的 model 欄位改回該請求的專案 alias。
- GET /v1/models 僅回傳公開別名，不暴露上游 repo_id。
- SSE parser 以事件為單位處理 data 欄位，只改寫合法 JSON 的 model 欄位；註解、keep-alive、[DONE] 及非 JSON 控制資料保持不變。每個事件即時刷新，不聚合整條串流。

影片路由使用可選 query `model`，以便不讀取或重建 multipart 本文也能選擇上游；預設模型由 `defaults.video_model` 設定。Live 建立 JSON 使用可選 `model`（預設 `defaults.live_model`）。Gateway 在轉送前檢查 `video_upload`／`live_video`，不支援時拒絕。Inference route capability 不在 Gateway 內推導；`tests/end2end/` 比對兩份 YAML，避免部署配置漂移。

## 視訊本文與 Content-Length

- 目前推論 API 在讀取聊天 JSON、視訊 multipart、即時工作階段 POST 本文前要求 Content-Length；缺少時回 HTTP 411。
- Gateway 在轉送前檢查 Content-Length 格式及 YAML 上限；長度無效回 400，本文過大回 413。
- Gateway 以有界串流方式轉送 multipart，不把完整視訊寫入記憶體，也不改寫 multipart boundary。
- Gateway 對串流位元組數另設上限，避免只依賴宣告長度；到達上限時中止上游請求並回報請求過大。
- 推論端當前上限是配置基線而非 Gateway 常數：聊天本文 15,728,640 bytes、視訊請求本文 210,763,776 bytes、視訊檔 209,715,200 bytes、即時建立本文 65,536 bytes。部署時應讓 Gateway 上限不高於相應推論上限；推論服務仍負責檢查視訊檔本身大小。

## SSE、逾時與錯誤

- SSE 回應設定適當的 text/event-stream、Cache-Control: no-cache，關閉代理緩衝並逐事件 flush。
- Gateway 保留上游事件順序及必要串流標頭；客戶端取消時取消上游 HTTP 請求。遵守設定的閒置與整體逾時。
- 在 SSE 標頭送出前發生連線錯誤或逾時，分別回 HTTP 502 或 504 的 OpenAI 風格 JSON 錯誤；串流開始後發生錯誤則記錄 request ID 並結束串流。
- Gateway 自行產生的錯誤至少使用 error.message、error.type、error.code；HTTP 狀態依認證、請求、限流及上游失敗種類設定。上游錯誤保留有意義的 HTTP 狀態並正規化錯誤本文，不傳回內部堆疊或內部網路位址。
- 限流回 HTTP 429 並提供 Retry-After；本文過大回 413；缺少 Content-Length 回 411。

## 即時工作階段句柄

句柄採 URL 安全編碼，簽章涵蓋版本、tenant_id、上游 session_id、建立時的模型 alias 與到期時間。Gateway 不需要資料庫或程序內的租戶工作階段表即可驗證句柄並還原上游路由；舊版未含 alias 的 v1 句柄使用 `defaults.live_model` 相容處理。

- 建立工作階段後，Gateway 將對外 id 改寫為簽名句柄；events_url 與 stop_url 使用 Gateway 主機及公開句柄。
- Gateway 驗證 Bearer Key、句柄簽章、到期時間及 tenant_id 相符後，才將句柄中的上游 session_id 轉為推論服務路徑。
- 無效、過期或其他租戶的句柄一致回 HTTP 404，避免暴露工作階段存在性。
- RTSP／RTMP publish URL 保留推論服務生成的 MediaMTX 路徑。該 URL 是能力憑證，取得者可在工作階段有效期間使用 MediaMTX 對應路徑；工作階段停止或自然結束後失效。
- 密鑰輪替時保留舊驗證密鑰，至少涵蓋舊句柄最長有效期。目前 inference.yml 的即時工作階段最長時間為 3600 秒；部署值改變時須同步簽名句柄 TTL。

## 可觀測性、容量與敏感資料

- 每個請求帶有或產生 request ID，日誌記錄路由、HTTP 狀態、租戶安全代號、上游耗時、總耗時及可用的本文位元組數。
- 指標可觀測吞吐、錯誤率、延遲分位數、SSE 首 token 延遲與串流時間、視訊請求完成時間及上游連線／逾時錯誤。
- 不記錄 API Key、簽名密鑰、聊天／圖片／視訊本文、生成內容或完整 RTSP／RTMP 能力 URL。
- 指標不得以原始 API Key 為維度；租戶標籤應受控以避免高基數及敏感資訊暴露。

## 相容性、回復與測試

- 推論 API 保持可獨立啟動與測試；Gateway 只使用設定的 HTTP 上游，不載入推論套件或模型。
- 目前 MediaMTX 授權回呼維持直接連接推論 API，避免把內部回呼錯誤暴露為客戶端路由。
- 回復只需停止 Gateway listener、撤下外部入口並還原 Gateway YAML；推論服務與 MediaMTX 不需反向修改。切換前應停止作用中的即時工作階段。若程序意外停止，推論服務依自身 max_duration_seconds 結束工作階段。
- 測試分別置於 tests/gateway/、tests/inference/、tests/end2end/。Gateway mock 測試不下載模型；目前 inference 測試是靜態契約檢查；端到端性能基線工具已建立，必須在 Gateway、推論 API 與所需 MediaMTX 可用時另行實測。
- 首版性能基線記錄吞吐量、錯誤率、p50/p95/p99、SSE 首 token 延遲／串流時間及視訊上傳完成時間，不設硬性 SLA，不做模型精度評測。

## 取捨

| 方案 | 優點 | 放棄原因／代價 |
|---|---|---|
| 單實例、程序內限流、租戶簽名句柄 | 無需外部狀態服務；句柄驗證可跨 Gateway 重啟 | 多實例需新增共享限流方案 |
| API Key 對應租戶並以 HMAC 句柄授權控制 API | SSE／停止操作具租戶隔離且不需資料庫 | 媒體 URL 本身仍是能力憑證，須由客戶端保密 |
| Gateway 代理 RTSP／RTMP 媒體 | 集中網路入口 | 增加長連線與頻寬壓力，且違反 MediaMTX sidecar 邊界，故不採用 |
| 透過推論 HTTP API 轉送 SSE 與視訊 | 保持推論與閘道服務分離 | 需正確處理刷新、取消、Content-Length 及串流上限 |

## 未決項

無待確認的產品範圍項目。部署者需提供實際限流值、本文限制、逾時、上游 URL、密鑰注入來源及 TLS 前置代理設定。

## 變更記錄

| 日期 | 變更 |
|---|---|
| 2026-10-20 | 建立 Gateway 技術設計草稿 |

