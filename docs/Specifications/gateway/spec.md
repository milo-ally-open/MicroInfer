# 行為規格：Go API 閘道

> 關聯提案：[proposal.md](proposal.md)  
> 技術設計：[design.md](design.md)  
> 實施追蹤：[tasks.md](tasks.md)  
> 狀態：Go Gateway v1 基本路由已實作；本次以 REQ-13 補充雙模型能力路由與 Gateway／Inference 設定一致性，驗收證據見 tasks.md。
> 來源：使用者確認的 Go 閘道目標、根目錄 AGENTS.md、README.md、src/gateway/README.md 與 src/inference/README.md。

## 範圍和術語

本規格涵蓋外部客戶端至 Go Gateway、Gateway 至推論 API 的 HTTP 請求，以及即時工作階段對 MediaMTX 的媒體資料流。Gateway 與推論服務是分離元件。媒體資料流由客戶端直接連接 MediaMTX。

- 專案別名：`mage-vl-4b` 和 `qwen3-0.6b`，供客戶端使用。
- 上游模型識別碼：`microsoft/Mage-VL` 與 `qwen3-0.6b`，分別用於各模型推論服務路由。
- 租戶：由已驗證 API Key 對應的客戶或呼叫方。
- 工作階段句柄：Gateway 簽發、綁定租戶與上游工作階段的 URL 安全識別碼。
- OpenAI 風格錯誤：JSON 本文包含 error 物件，至少含 message、type、code；param 可為 null。

## 需求與驗收條件

### REQ-01：公開模型別名與上游路由

**需求**：Gateway 應以 YAML 登錄公開模型別名及其上游服務映射，只公開已登錄的模型別名。

**驗收條件**

- **AC-01.1** Given API Key 有效，When 客戶端呼叫 GET /v1/models，Then 回應清單包含目前登錄的 mage-vl-4b 和 qwen3-0.6b，且不暴露上游識別碼。
- **AC-01.2** Given 客戶端請求指定 model 為 mage-vl-4b，When Gateway 轉送聊天請求，Then 上游請求使用 microsoft/Mage-VL，非串流回應及每個 SSE JSON 分塊的 model 欄位均使用 mage-vl-4b。
- **AC-01.3** Given 請求未指定 model 或指定未登錄的名稱，When Gateway 驗證聊天請求，Then 回應 HTTP 400 與 OpenAI 風格錯誤，且不呼叫上游。

**驗證方式**：Gateway API 整合測試；驗證請求與回應的模型欄位。

### REQ-02：API Key 驗證與租戶識別

**需求**：除健康及就緒端點外，所有公開 /v1/* API 都應以 Bearer API Key 驗證呼叫者，並以伺服器端對應結果決定租戶。

**驗收條件**

- **AC-02.1** Given /v1/* 請求帶有有效 Bearer API Key，When Gateway 驗證請求，Then 使用該密鑰設定所對應的租戶執行政策判定。
- **AC-02.2** Given API Key 缺漏或無效，When 呼叫任一公開 /v1/* API，Then 回應 HTTP 401 與 OpenAI 風格錯誤，且不呼叫上游。
- **AC-02.3** Given 請求另帶客戶端自訂租戶標頭，When Gateway 建立呼叫者身分，Then 不以客戶端標頭覆蓋伺服器端租戶對應。

**驗證方式**：認證中介層測試及 API 整合測試。

### REQ-03：一般聊天請求

**需求**：Gateway 應依選定模型已登記的輸入能力，將文字或圖片／圖文聊天請求轉送至正確的推論服務，並保留相容的聊天欄位。Mage-VL 要求圖片輸入；Qwen3-0.6B 接受純文字，兩者都不接受對方未宣告的輸入模態。

**驗收條件**

- **AC-03.1** Given 聊天請求通過 API Key、模型及本文限制檢查，When stream 為 false 或未啟用，Then Gateway 轉送請求並回傳推論服務的 JSON 完成結果，僅改寫公開 model 欄位及 Gateway 規定的錯誤資訊。
- **AC-03.2** Given 上游回應包含非 hop-by-hop 的內容型別及必要標頭，When Gateway 回傳結果，Then 保留與客戶端契約相容的標頭及狀態碼，且不洩漏上游內部位址或伺服器堆疊資訊。

**驗證方式**：Gateway API 整合測試，涵蓋文字及圖片請求；不執行模型精度評估。

### REQ-04：聊天 SSE 串流

**需求**：Gateway 應轉送聊天 SSE token 串流，保持事件邊界、順序、即時送出、取消及完成標記語義。

**驗收條件**

- **AC-04.1** Given 聊天請求指定 stream 為 true，When 上游產生 SSE，Then Gateway 逐事件轉送，刷新回應而不等待完整生成，並保留 data: [DONE] 結尾。
- **AC-04.2** Given 上游 SSE JSON 分塊包含 model 欄位，When Gateway 對外轉送該分塊，Then 將欄位改寫為請求選定的公開別名；SSE 註解、keep-alive、非 JSON 控制資料及 [DONE] 不被誤解析為完成分塊。
- **AC-04.3** Given 客戶端在串流中斷線或取消，When Gateway 收到取消訊號，Then 取消上游請求並釋放相關連線與工作資源。

**驗證方式**：串流測試，檢查首事件延遲、事件順序、反向壓力及客戶端取消。

### REQ-05：視訊上傳轉送

**需求**：Gateway 應將視訊理解 multipart 請求轉送至推論服務，不在 Gateway 將完整檔案讀入記憶體。

**驗收條件**

- **AC-05.1** Given multipart 請求包含 video、question 及目前推論 API 支援的選用欄位，When Gateway 轉送請求，Then 保留 multipart 欄位與邊界語義並以串流方式送至 POST /v1/videos/understand。
- **AC-05.2** Given 請求缺少 Content-Length、長度格式錯誤或超過 Gateway 設定上限，When Gateway 在讀取大型本文前檢查請求，Then 分別回應 HTTP 411、HTTP 400 或 HTTP 413，使用 OpenAI 風格錯誤且不呼叫上游。
- **AC-05.3** Given Gateway 與推論服務的視訊本文上限已設定，When 部署設定被驗證，Then Gateway 對應上限不得高於推論端 inference.yml 的限制；Gateway 另以有界串流計數避免錯報長度造成無界轉送。

**驗證方式**：multipart API 整合測試及大本文串流測試；不下載模型權重。

### REQ-06：即時工作階段與租戶隔離

**需求**：Gateway 應代理即時工作階段控制 API，並確保只有建立工作階段的租戶可讀取事件或停止該工作階段。

**驗收條件**

- **AC-06.1** Given 有效租戶建立工作階段，When Gateway 呼叫推論服務 POST /v1/live/sessions，Then 對外回應使用該租戶綁定的 URL 安全簽名句柄，events_url 與 stop_url 指向 Gateway，publish RTSP／RTMP URL 仍直連 MediaMTX。
- **AC-06.2** Given 同租戶使用有效 API Key 與簽名句柄呼叫事件或停止路由，When Gateway 驗證句柄簽章、租戶及有效期，Then 以句柄內的模型別名和上游識別碼代理 GET /v1/live/sessions/{session_id}/events 或 DELETE /v1/live/sessions/{session_id}。
- **AC-06.3** Given 句柄無效、過期或屬於其他租戶，When 客戶端呼叫事件或停止路由，Then 回應一致的 HTTP 404 OpenAI 風格錯誤，不揭露工作階段是否存在。
- **AC-06.4** Given 客戶端透過 publish URL 連接 MediaMTX，When MediaMTX 授權回呼檢查路徑，Then 僅作用中的工作階段可發布或讀取；媒體 URL 視同能力憑證，客戶端應保密，且工作階段終止後失效。

**驗證方式**：租戶隔離測試、簽章與有效期測試、SSE Last-Event-ID 續接測試，以及受控 MediaMTX 整合測試。

### REQ-07：請求大小、長度與租戶限流

**需求**：Gateway 應在轉送前套用 YAML 設定的本文限制及每租戶限流，單實例首版的限流狀態由程序內維護。

**驗收條件**

- **AC-07.1** Given POST 聊天、視訊上傳或工作階段建立請求未帶 Content-Length，When Gateway 驗證請求，Then 回應 HTTP 411；格式錯誤長度回應 HTTP 400；超出設定上限回應 HTTP 413。
- **AC-07.2** Given 租戶超出 YAML 設定的請求速率或突發容量，When 該租戶再發送請求，Then Gateway 回應 HTTP 429 與 Retry-After，且其他租戶不受該租戶計數器直接影響。
- **AC-07.3** Given Gateway 程序重新啟動，When 新程序開始接受流量，Then 程序內限流狀態重新初始化；此行為須在運維文件中明確記載。

**驗證方式**：本文邊界、Content-Length、租戶隔離及限流測試。

### REQ-08：錯誤格式與上游失敗

**需求**：Gateway 自身及正規化後的上游錯誤應使用一致的 OpenAI 風格 JSON，且不得洩漏密鑰、內部位址或堆疊。

**驗收條件**

- **AC-08.1** Given Gateway 拒絕認證、模型、本文、限流或工作階段請求，When 回應錯誤，Then JSON error 物件包含 message、type、code，並搭配適當 HTTP 狀態碼。
- **AC-08.2** Given 上游回傳 HTTP 錯誤，When Gateway 回應客戶端，Then 保留有意義的 HTTP 狀態並將錯誤本文正規化；連線失敗回 HTTP 502，逾時回 HTTP 504。
- **AC-08.3** Given SSE 標頭已送出後發生上游錯誤，When Gateway 無法再更改 HTTP 狀態，Then 終止串流並記錄含 request ID 的錯誤，不把一般 JSON 錯誤插入 SSE token 資料。

**驗證方式**：錯誤契約測試、上游模擬器測試及串流中斷測試。

### REQ-09：健康與就緒

**需求**：Gateway 應分別提供程序存活檢查與上游就緒檢查。

**驗收條件**

- **AC-09.1** Given Gateway 程序運行，When 呼叫 GET /healthz，Then 回應成功且不要求 API Key；此端點只表示 Gateway 程序仍存活。
- **AC-09.2** Given Gateway 設定有效且每個不同推論 URL 的 GET /health 均回報模型狀態為 ok，When 呼叫 GET /readyz，Then 回應成功且不要求 API Key。
- **AC-09.3** Given 設定未載入或推論上游不可用／未就緒，When 呼叫 GET /readyz，Then 回應 HTTP 503，不宣稱模型 API 可服務。

**驗證方式**：健康端點測試及上游健康狀態模擬測試。

### REQ-10：可觀測性與敏感資料保護

**需求**：Gateway 應提供可依請求追蹤的結構化日誌與服務指標，且不記錄敏感本文或憑證。

**驗收條件**

- **AC-10.1** Given 每個 API 請求進入 Gateway，When 請求完成或失敗，Then 日誌可關聯 request ID、路由、租戶識別碼或安全的租戶代號、HTTP 狀態、上游延遲與總延遲。
- **AC-10.2** Given 任何認證、聊天、圖片、視訊、即時工作階段或媒體 URL，When Gateway 記錄請求，Then 不輸出 API Key、簽名密鑰、請求本文、圖片／視訊內容或完整媒體能力 URL。
- **AC-10.3** Given Gateway 提供性能觀測資料，When 管理者檢視基線，Then 可取得請求吞吐量、錯誤率、p50/p95/p99 延遲、SSE 首 token 延遲／串流時間及視訊上傳完成時間，且不以原始 API Key 作為維度。

**驗證方式**：日誌敏感欄位測試、指標檢查及端到端基線報告。

### REQ-11：YAML 設定與密鑰注入

**需求**：Gateway 的可調整服務參數應由 src/gateway/configs/gateway.yml 載入；秘密值只由環境變數或受保護的掛載檔案注入。

**驗收條件**

- **AC-11.1** Given Gateway 啟動，When 載入設定，Then 綁定位址、連接埠、推論上游、模型路由、本文限制、逾時、租戶限流與日誌／指標參數均從 YAML 取得；預設綁定 127.0.0.1:8000，上游預留預設值為 http://127.0.0.1:8001。
- **AC-11.2** Given YAML 缺少必要欄位、格式錯誤或存在無效上限，When Gateway 啟動，Then 明確失敗且不以未記錄的程式常數靜默替代必要設定。
- **AC-11.3** Given API Key、工作階段 HMAC 密鑰或其他憑證，When Gateway 載入秘密，Then 從環境變數或安全掛載檔案讀取，且不要求在追蹤中的 YAML 填入明文秘密。
- **AC-11.4** Given工作階段簽名密鑰輪替，When 新密鑰開始簽發句柄，Then 舊驗證密鑰至少保留至其簽發句柄最長有效期結束；目前推論端工作階段最長期間設定為 3600 秒，Gateway 必須依實際部署設定保持一致。

**驗證方式**：設定載入、無效設定拒絕、密鑰來源及輪替相容測試。

### REQ-12：相容、回復與測試分層

**需求**：Gateway 的實作與驗收應維持既有元件邊界、明確的測試分層及可回復部署方式。

**驗收條件**

- **AC-12.1** Given 新增 Gateway，When 專案檢視元件責任，Then 推論服務仍負責模型生命週期、視訊理解與 GPU；MediaMTX 仍負責 RTSP／RTMP 媒體流；Gateway 不載入模型、不管理 GPU，也不代理媒體封包。
- **AC-12.2** Given 後續實作與測試，When 新增測試檔案，Then Gateway 測試位於 tests/gateway/，推論測試位於 tests/inference/，端到端測試位於 tests/end2end/；推論測試可直接呼叫推論服務，與 Gateway 解耦。
- **AC-12.3** Given Gateway、推論服務及所需 MediaMTX 可用，When 執行端到端性能基線，Then 報告吞吐量、錯誤率、延遲分位數、SSE 首 token 延遲及視訊上傳完成時間；首版不設定硬性 SLA、不執行資料集精度測試。
- **AC-12.4** Given Gateway 發布需回復，When 部署者撤下 Gateway 入口並還原其 YAML，Then 推論服務與 MediaMTX 不需回滾變更；切換前應停止作用中的即時工作階段。

**驗證方式**：套件／責任邊界檢視、分層測試、回復演練及性能基線報告。

### REQ-13：依模型能力選擇推論上游並拒絕不支援功能

**需求**：Gateway 必須依公開模型別名、每模型上游 URL 及能力設定路由請求；Gateway 設定需與 Inference Registry 的已登錄 alias、上游識別碼和能力保持一致。

**驗收條件**

- **AC-13.1** Given `mage-vl-4b` 或 `qwen3-0.6b` 聊天請求，When Gateway 解析 JSON `model`，Then 轉送至該模型 YAML 上游 URL，改寫成相應上游模型識別碼，並將 JSON／SSE 回應模型欄位改回該專案別名。
- **AC-13.2** Given 聊天本文含有不受所選模型支援的輸入模態或串流模式，When Gateway 執行 capability policy，Then 回應 HTTP 400 `unsupported_model_capability`，且不呼叫上游。
- **AC-13.3** Given影片 multipart 請求以選用 query `model` 指定別名，When Gateway 驗證 `video_upload`，Then 以所選模型上游轉送；省略時使用 YAML `defaults.video_model`；unsupported capability 在讀取或轉送本文前拒絕；轉送保留原 multipart body、boundary 及 `Content-Length`。
- **AC-13.4** Given Live 建立 JSON 本文含選用 `model`，When Gateway 驗證 `live_video`，Then 使用該模型上游並在句柄簽章中綁定其 alias；省略時使用 `defaults.live_model`；events SSE 與停止請求必須使用句柄綁定的同一上游。句柄仍須驗證簽章、有效期及 API Key tenant。
- **AC-13.5** Given Gateway 配置多個不同推論 URL，When 客戶端呼叫 `/readyz`，Then 每個不同上游都回報 ready 才返回成功。
- **AC-13.6** Given Gateway 與 Inference 各自保存路由設定，When 執行 `tests/end2end/test_config_compatibility.py`，Then 驗證 alias 集合、`upstream_model`／`api.model_id`、文字／圖片／stream／video／live 能力及 network URL 一致。

**驗證方式**：Gateway `httptest` 覆蓋兩個模型文字／圖片／SSE、上傳與 Live capability 正負案例、模型上游隔離及 readiness；跨設定檔檢查由 `tests/end2end/` 執行。

## 不變項與相容要求

- 專案別名 mage-vl-4b 與上游 repo_id microsoft/Mage-VL 是不同識別碼，不可混用。
- 推論服務預設為 127.0.0.1:8001；Gateway 預設預留 127.0.0.1:8000；實際位址由各自 YAML 控制。
- 推論層目前提供聊天文字／圖片與 SSE、視訊上傳單次回應及即時工作階段控制；SSE 表示文字輸出串流，不表示影格逐幀回傳。
- 推論 API 目前要求相關 POST 請求提供 Content-Length；Gateway 必須兼容且限制記憶體使用。
- 圖片與上傳視訊不依賴 MediaMTX；即時媒體流才需 MediaMTX。

## 資料處理和保留

- Gateway 不持久化聊天、圖片、視訊本文或推論輸出。
- Gateway 不需資料庫；首版租戶限流狀態為程序內狀態。工作階段控制權由簽名句柄承載，不在 Gateway 建立持久工作階段資料表。
- 日誌只保留診斷所需的安全中繼資料；實際保留期限由部署環境的日誌政策設定。
- API Key 與 HMAC 密鑰不記錄、不放入 Git，也不放入普通 YAML。

## 明確不做

- 不提供模型精度或資料集評測功能。
- 不把 SGLang、vLLM、第二個模型、多實例共享狀態或 Redis 列為首版已完成能力。
- 不經 Gateway 轉送 RTSP／RTMP 媒體封包。
- 不在 Gateway 實作模型推論、權重下載、GPU 資源排程或 MediaMTX 管理。

## 需求追蹤

| 上游條款／決定 | REQ | 模組／服務 | 任務 ID | AC | 驗收證據／狀態 |
|---|---|---|---|---|---|
| 已登錄模型別名與上游路由 | REQ-01 | internal/route/、internal/upstream/、gateway.go | T-03、T-12、T-13 | AC-01.1、AC-01.2、AC-01.3、AC-13.1 | Mage／Qwen mock 路由已測；真實 Gateway smoke 已覆蓋 Qwen 文字 JSON／SSE、Mage 圖片 JSON／SSE／視訊上傳及 Live／MediaMTX 合成 RTSP |
| API Key 與租戶識別 | REQ-02 | internal/auth/、gateway.go | T-02 | AC-02.1、AC-02.2、AC-02.3 | 已實作；mock 測試 |
| 既有聊天 JSON API | REQ-03 | gateway.go、internal/upstream/ | T-03、T-12 | AC-03.1、AC-03.2 | 已實作；mock 測試 |
| 既有聊天 SSE | REQ-04 | internal/stream/、internal/upstream/、gateway.go | T-04、T-12 | AC-04.1、AC-04.2、AC-04.3 | 已實作；取消專項測試待補 |
| 視訊 multipart 與 Content-Length 限制 | REQ-05 | internal/videoproxy/、internal/upstream/、YAML | T-05、T-12 | AC-05.1、AC-05.2、AC-05.3 | 已實作；mock／靜態相容測試 |
| 即時 API、媒體直連與租戶綁定句柄 | REQ-06 | gateway.go、internal/auth/ | T-06 | AC-06.1、AC-06.2、AC-06.3、AC-06.4 | 控制 API mock 測試通過；已完成一次 Gateway／MediaMTX 合成 RTSP 端到端 smoke，過期／未授權媒體存取仍待驗證 |
| 單實例、每租戶限流及本文上限 | REQ-07 | internal/middleware/、internal/config/ | T-07 | AC-07.1、AC-07.2、AC-07.3 | 已實作；mock 測試 |
| OpenAI 風格錯誤及上游逾時 | REQ-08 | gateway.go、internal/upstream/、internal/stream/ | T-08、T-12 | AC-08.1、AC-08.2、AC-08.3 | 已實作；502 已測，504／中斷 case 未完整測 |
| 健康與就緒語義 | REQ-09 | gateway.go、internal/upstream/ | T-09、T-12 | AC-09.1、AC-09.2、AC-09.3 | 已實作；mock 上游測試 |
| 日誌、指標與敏感資料保護 | REQ-10 | internal/observability/、gateway.go | T-10 | AC-10.1、AC-10.2、AC-10.3 | 已實作；mock 指標／遮蔽測試，真實性能未執行 |
| YAML 設定及安全密鑰注入 | REQ-11 | internal/config/、internal/auth/ | T-01、T-02 | AC-11.1、AC-11.2、AC-11.3、AC-11.4 | 已實作；checked-in YAML 與 mock 密鑰測試 |
| 元件邊界、測試分層、性能基線與回復 | REQ-12 | Gateway 套件、tests/ | T-11、T-12 | AC-12.1、AC-12.2、AC-12.3、AC-12.4 | 套件責任與測試分層已檢視；實際性能與回復演練未執行 |
| 模型能力政策、雙模型路由與設定一致性 | REQ-13 | internal/config/、gateway.go、internal/auth/、tests/gateway/、tests/end2end/ | T-13 | AC-13.1、AC-13.2、AC-13.3、AC-13.4、AC-13.5、AC-13.6 | Gateway mock、Inference／End2End 設定一致性測試通過；真實 Gateway smoke 已覆蓋 Qwen 文字 JSON／SSE、Mage 圖片 JSON／SSE／視訊上傳及 Live／MediaMTX 合成 RTSP |

## 假設與未決問題

- [假設] TLS 由外部入口或反向代理終止；首版預設只在 127.0.0.1 綁定。部署直接對外暴露前需確認 TLS 保護。
- [假設] Gateway 和推論服務由同一部署管理者配置，能維持 Gateway 與推論端本文限制、即時工作階段期限一致。
- 沒有未解決的產品範圍決策；限流值、請求限制、逾時與觀測後端是部署設定，不是固定產品 SLA。

## 變更記錄

| 日期 | 變更 |
|---|---|
| 2026-10-20 | 建立可追蹤的 Gateway 行為規格與驗收條件 |

