# 提案：Go API 閘道

> 狀態：首版設計決策已記錄；Go Gateway v1 已實作，驗收進度見 tasks.md。  
> 路徑：跨邊界高風險  
> 負責人：專案維護者  
> 上游依據：[專案規範](../../../AGENTS.md)、[專案 README](../../../README.md)、[閘道目錄說明](../../../src/gateway/README.md)、[推論服務說明](../../../src/inference/README.md)

## 背景與問題

建立本提案時，專案已有 Mage-VL-4B Transformers 推論 API，但尚無 Go 閘道服務。外部 API 身分驗證、租戶識別、請求政策、限流、模型路由與 API 可觀測性需要由獨立閘道承擔，避免把客戶端接入職責和模型載入、GPU 推論、模型專屬處理混在一起。後續使用者明確要求依此規格實作 Gateway v1；目前實作及測試狀態由 tasks.md 記錄。

目前推論 API 預設監聽 127.0.0.1:8001；Go Gateway v1 預設監聽 127.0.0.1:8000。推論路由包含健康檢查、模型清單、聊天完成、視訊上傳及即時工作階段 API。MediaMTX 是即時媒體流的 sidecar，不是閘道或模型服務。

## 目標與成功觀察

- 提供以 Go 實作、可由 YAML 設定的外部 API 閘道，統一處理客戶端驗證、租戶政策、模型路由、請求限制及可觀測性。
- 在不承擔模型生命週期或 GPU 工作的前提下，轉送文字／圖片聊天、SSE 文字輸出、視訊上傳及即時工作階段控制 API。
- 讓對外模型名稱使用專案別名 mage-vl-4b，並將它映射到推論服務使用的上游識別碼 microsoft/Mage-VL。
- 建立 REQ、AC、T 任務及驗收證據之間可追蹤的規格流程。

## 範圍

### 本次包含

- Gateway 的 HTTP API、Bearer API Key 與租戶識別、單實例限流、錯誤契約、健康／就緒檢查、上游路由、日誌與指標需求。
- 聊天一般回應與 SSE、視訊檔案上傳、即時工作階段建立／SSE 事件／停止操作的轉送規格。
- MediaMTX 即時媒體流與推論服務之間的責任界線、安全要求、設定要求、失敗行為、回復方式及後續測試分層。
- 在本規格組內維護提案、行為規格、技術設計與實施任務。

### 本次不包含／保持不變

- 原始規格階段只建立文件，不實作 Go 服務；後續實作任務依本規格建立 Go Gateway，並只為將 live 本文限制移入 YAML 而調整推論設定／middleware。此提案不代表 SGLang、vLLM 或第二個模型已實作。
- 不將 SGLang、vLLM 或其他模型描述為已實作功能；目前沒有選定第二個模型。
- 不把模型權重、模型載入、生成、視訊解析、GPU 管理或 MediaMTX 媒體封包轉送移入 Gateway。
- 不進行模型精度或資料集基準測試。端到端性能測試待 Gateway 完成後進行。
- 不修改 checkpoints/val.md、checkpoints/val.xlsx，也不移動或清理既有工作樹內容。

## 現狀依據與已確認決策

### 程式與文件現況

- src/gateway/README.md 描述已實作的 Go Gateway v1 及各套件職責；真實推論服務與 MediaMTX 整合仍未驗證。
- src/inference/model_registry.yaml 將專案別名 mage-vl-4b 登錄到 src/inference/models/mage_vl_4b/ 模型包。
- 推論設定 src/inference/models/mage_vl_4b/configs/inference.yml 中的模型識別碼是 microsoft/Mage-VL；這是 Hugging Face 上游 repo_id，不是專案別名。
- 推論服務目前提供 GET /health、GET /v1/models、POST /v1/chat/completions、POST /v1/videos/understand，以及 /v1/live/* 工作階段 API。
- POST 聊天、視訊上傳及即時工作階段建立目前要求 Content-Length。當前推論設定的聊天本文上限為 15,728,640 bytes、視訊請求本文上限為 210,763,776 bytes、單一視訊檔上限為 209,715,200 bytes；即時工作階段建立上限目前為 65,536 bytes。
- 即時工作階段回應包含 RTSP／RTMP 發布 URL、事件 URL、停止 URL 及上游工作階段識別碼。MediaMTX 授權回呼目前直接呼叫推論服務，且只允許作用中的工作階段路徑。

### 已確認決策

- 首版涵蓋聊天／SSE、視訊上傳與即時工作階段控制；RTSP／RTMP 媒體資料流直接連接 MediaMTX。
- Gateway 對外使用 mage-vl-4b；上游請求使用 microsoft/Mage-VL，對外模型清單和聊天回應維持專案別名。
- 首版以單一 Gateway 實例為目標，租戶限流狀態可保存在程序記憶體中。
- 客戶端 API Key 與租戶對應由環境變數或掛載的安全檔案提供；密鑰不得提交至 Git。
- Gateway 錯誤使用 OpenAI 風格 JSON。
- 即時工作階段控制 API 使用租戶綁定的簽名工作階段句柄；RTSP／RTMP URL 是隨工作階段失效的能力憑證。
- 端到端性能測試先建立實測基線，不在首版寫入硬性 SLA。

### 技術假設

- Gateway 預設監聽 127.0.0.1:8000；監聽位址及連接埠由 Gateway YAML 設定，推論上游位址亦由該設定載入。
- 部署時由外部入口或反向代理終止 TLS；若日後要求 Gateway 直接對公網提供 TLS，應另行補充部署與憑證規格。
- 推論服務的本文限制仍由其 inference.yml 管理；Gateway 設定的請求本文上限不得超過推論端相應上限。

## 方案與取捨

| 方案 | 收益 | 成本／風險 |
|---|---|---|
| 單實例 Gateway、程序內限流、租戶綁定簽名句柄 | 不需新增共享資料庫；符合目前部署與規模假設 | 多實例時須改用共享限流與工作階段狀態方案 |
| 首版使用共享狀態服務 | 可直接擴展多實例 | 增加服務依賴、部署及維運工作，目前沒有需求依據 |
| 讓 RTSP／RTMP 媒體封包經 Gateway | 可集中處理網路入口 | 改變 MediaMTX 架構並增加長連線與頻寬負擔，不符合既定元件邊界 |

建議採用第一列。這是已確認的首版部署取捨，不代表專案已具備多實例能力。

## 影響面、依賴與風險

- 主要影響：src/gateway/ 的後續 Go 實作、src/gateway/configs/gateway.yml、Gateway API 使用者文件及 tests/ 下的分層測試。
- 上游依賴：目前的 Mage-VL-4B API 與 MediaMTX；Gateway 不管理推論服務或 MediaMTX 的啟動。
- 安全風險：API Key、簽名密鑰、請求本文、圖片、視訊及含媒體 URL 的完整日誌都可能暴露敏感資料；設計明確禁止記錄這些內容。
- 兼容風險：推論端目前要求 Content-Length；Gateway 不應為了支援未知長度串流而將大型視訊整段緩存在記憶體。
- 運維風險：程序內限流在 Gateway 重啟後重置；單實例不是高可用部署。
- 回復方式：停止或撤下 Gateway 入口並還原 Gateway YAML；推論服務與 MediaMTX 不需改動。仍作用中的媒體工作階段依推論服務的工作階段生命週期結束；正常切換前應先呼叫停止 API。

## 未決問題

沒有阻擋規格實作的產品決策。部署者需在 Gateway YAML 中提供 API Key 來源、簽名密鑰、請求限制、限流值、逾時及上游位址；性能目標待基線結果出爐後再決定。

## 變更記錄

| 日期 | 內容 | 決定／依據 |
|---|---|---|
| 2026-10-20 | 建立 Gateway 跨邊界規格草稿 | 使用者已確認首版 API 範圍、模型別名、單實例、錯誤格式、密鑰來源、性能基線與即時流授權邊界 |
| 2026-10-20 | 將路由、SSE、上游 HTTP 及影片 multipart 代理拆至 internal/route、stream、upstream、videoproxy | 讓實作模組與設計責任一致；透過既有 Gateway mock 整合測試確認外部行為 |

