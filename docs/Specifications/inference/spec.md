# 行為規格：推論服務現況基線

> 關聯提案：[proposal.md](proposal.md)  
> 技術設計：[design.md](design.md)  
> 任務追蹤：[tasks.md](tasks.md)  
> 狀態：現況基線／待評審；需求與驗收條件描述目前程式行為，未執行的 runtime 驗證不視為通過。  
> 相關外部入口：[Go Gateway 規格](../go-gateway/spec.md)

## 範圍和術語

本文描述 src/inference/launcher.py、src/inference/model_registry.yaml、src/inference/mage_vl_4b/ 及其 YAML 設定。推論服務管理模型權重、處理器、GPU 生成、圖片／視訊輸入及即時視訊工作階段。Gateway 負責未來的外部客戶端認證與政策；目前推論 API 不實作客戶端 API Key 或租戶認證。

- 專案模型別名：mage-vl-4b。
- 上游模型 repo_id：microsoft/Mage-VL。
- 推論配置：模型登錄表所指向的 inference.yml。
- 影格模式：以 OpenCV 均勻抽取有限影格，供模型理解視訊。
- codec 模式：使用 ffprobe／FFmpeg 與 codec-video-prep 前處理視訊片段。
- 本文是程式現況基線，不是新增功能批准。標註 [待確認] 的契約必須由維護者決定。

## 需求與驗收條件

### REQ-01：模型登錄與中央啟動器

**需求**：launcher.py 應根據模型登錄表列出模型設定並啟動所選服務入口。

**驗收條件**

- **AC-01.1** 給定 model_registry.yaml 目前內容，當執行 launcher.py list，則輸出 mage-vl-4b、預設後端 transformers、設定路徑、服務入口及檔案狀態；此狀態檢查檔案存在，不代表模型權重已載入。
- **AC-01.2** 給定執行 launcher.py start 未提供 --model，當啟動器選擇模型，則使用 default_model；指定未知別名或缺少 default_backend 時，則在啟動服務前失敗。
- **AC-01.3** 給定模型登錄項目缺少設定檔或服務入口，當啟動器解析選擇，則回報缺少檔案或設定錯誤，不啟動 Uvicorn。

**驗證方式**：以暫存 YAML／檔案的 launcher 單元測試；不載入模型。

### REQ-02：推論設定選擇與環境變數

**需求**：啟動器應依登錄表載入設定，允許指定設定檔及 runtime 覆寫，並防止通用環境欄位覆蓋專用受保護欄位。

**驗收條件**

- **AC-02.1** 給定相對設定路徑，當啟動器解析路徑，則以專案根目錄為基準；絕對路徑可直接使用，路徑不存在時啟動失敗。
- **AC-02.2** 給定 start 命令帶有 --inference-config、--network-config 或 --media-config，當啟動器解析設定，則只覆蓋相應登錄路徑；--host、--port 與 --cuda-visible-devices 覆蓋對應 runtime 值。
- **AC-02.3** 給定 runtime.environment，當啟動器套用環境變數，則名稱須符合大寫環境變數格式、值須為字串／數字／布林值；CUDA_VISIBLE_DEVICES、HF_HOME、HF_ENDPOINT、HF_HUB_DISABLE_XET、HF_HUB_DOWNLOAD_TIMEOUT、INFERENCE_CONFIG 及 INFERENCE_NETWORK_CONFIG 不可由此欄位覆蓋。
- **AC-02.4** 給定網路 YAML 的 host 或 port 不合法，當啟動器驗證監聽參數，則拒絕空白或非法 host，並要求 port 位於 1 至 65535。

**驗證方式**：設定解析與 CLI 覆寫測試。

### REQ-03：固定 revision 權重取得與模型生命週期

**需求**：推論服務應以模型 YAML 指定的固定 repo_id 與 revision 載入處理器及模型，並在服務停止時釋放即時工作階段與 GPU 資源。

**驗收條件**

- **AC-03.1** 給定本機快取缺少指定模型 revision 且 runtime.local_files_only 為 false，當推論服務準備模型路徑，則從設定的 Hugging Face endpoint 取得快照並放入 HF_HOME/hub；local_files_only 為 true 時不得下載。
- **AC-03.2** 給定服務開始 lifespan，當載入模型，則要求 CUDA 可用，並讓 AutoProcessor 與 AutoModelForCausalLM 使用同一 revision；載入模型後進入 eval 狀態。
- **AC-03.3** 給定量化 method 設為 none、prequantized、nf4 或 int8，當建立模型載入參數，則使用相應的 Transformers／BitsAndBytes 路徑；其他 method 或無效設定導致啟動錯誤。
- **AC-03.4** 給定 lifespan 關閉，當服務釋放資源，則停止所有 live session、清除模型及 processor 引用，並在 CUDA 可用時清理 CUDA cache。

**驗證方式**：以 mock 驗證下載策略、載入參數與 lifespan 關閉；另需有模型及目標 GPU 才能完成 runtime 驗收。

### REQ-04：健康與模型清單 API

**需求**：推論 API 應提供模型狀態及目前服務模型識別碼。

**驗收條件**

- **AC-04.1** 給定模型尚未完成載入，當呼叫 GET /health，則回應 status 為 starting；模型載入後 status 為 ok。
- **AC-04.2** 給定服務已載入，當呼叫 GET /health，則回應 backend、repo_id、revision、quantization、live_sessions 及 live_video_enabled；CUDA 可用時另回報 GPU 名稱及 VRAM 使用資訊。
- **AC-04.3** 給定呼叫 GET /v1/models，當回傳模型卡，則 id 為 microsoft/Mage-VL，而非專案別名 mage-vl-4b；目前兩個路由均不要求 API Key。

**驗證方式**：API schema 測試及載入／未載入狀態模擬。

### REQ-05：聊天請求及公開識別碼現況

**需求**：聊天路由應驗證目前的 ChatCompletionRequest，且只有在 request.model 為 null 或等於上游 repo_id 時接受。

**驗收條件**

- **AC-05.1** 給定有效聊天請求，當 messages 包含至少一項訊息且 role 是 system、user 或 assistant，則接受字串內容或文字／圖片內容部分。
- **AC-05.2** 給定 request.model 為 null 或 microsoft/Mage-VL，當呼叫 POST /v1/chat/completions，則進入生成流程；model 為 mage-vl-4b 或其他值時，則回 HTTP 404 Unknown model。
- **AC-05.3** 給定 messages 缺少、為空或結構不符，當 FastAPI 驗證本文，則回傳請求驗證錯誤；目前 Pydantic 額外欄位採 extra=ignore。
- **AC-05.4** 給定所有訊息均沒有 image_url，當服務整理訊息，則目前程式以 HTTP 400 拒絕；[待確認] README 所述「文字／圖片請求」是否應包含純文字請求。

**驗證方式**：API schema 與路由測試；AC-05.4 同時核對文件契約待決事項。

### REQ-06：圖片 URL 取得、解碼與限制

**需求**：聊天圖片內容應支援 base64 data URL 或 HTTP(S) URL，並受目前設定的大小、像素、數量與縮放限制。

**驗收條件**

- **AC-06.1** 給定 data URL，當服務載入圖片，則要求 base64 編碼且嚴格解碼；格式錯誤或非 base64 data URL 回 HTTP 400。
- **AC-06.2** 給定 image_url.url 為 HTTP(S)，當服務取得圖片，則以設定的 image_url_timeout_seconds 等待、要求 HTTP 200，並最多讀取 max_image_bytes 加一個 byte。
- **AC-06.3** 給定圖片超過 max_image_bytes、max_image_pixels 或 max_images_per_request，當服務檢查圖片，則回 HTTP 413；無法識別或解碼的圖片回 HTTP 400。
- **AC-06.4** 給定有效圖片，當處理器收到圖片，則轉成 RGB 並依設定的 resampling 與 max_image_side 縮放；目前值分別為 lanczos 與 896。

**驗證方式**：使用小型本機圖片 fixture 的圖片解析測試；遠端 URL 測試使用可控 mock server，不連外抓取。

### REQ-07：生成參數、上下文與完成回應

**需求**：推論服務應合併 generation.yml 區段的預設值與請求參數，並套用 token、範圍及上下文限制。

**驗收條件**

- **AC-07.1** 給定未提供 max_tokens 或 max_completion_tokens，當建立生成參數，則使用 generation.max_tokens_default；目前預設為 128，API 上限 max_new_tokens 為 512。
- **AC-07.2** 給定 temperature、top_p、top_k、min_p、repetition_penalty、no_repeat_ngram_size 或 num_beams 超出程式限制，當驗證生成參數，則回 HTTP 400；temperature 為 0 時不啟用 sampling。
- **AC-07.3** 給定 prompt token 數加 max_new_tokens 大於 max_context_tokens，當生成前驗證上下文，則回 HTTP 413；目前 max_context_tokens 為 8192。
- **AC-07.4** 給定非串流聊天成功，當生成完成，則回 chat.completion 物件、assistant 內容、finish_reason、usage token 數及 model=microsoft/Mage-VL。

**驗證方式**：生成參數邊界與 mock model response 測試。

### REQ-08：聊天 SSE 輸出與錯誤

**需求**：stream=true 時服務應以 SSE 輸出文字 token、最終 usage 與完成標記；串流錯誤採事件內 JSON 表達。

**驗收條件**

- **AC-08.1** 給定有效 stream=true 請求，當模型生成文字，則回傳 chat.completion.chunk，包含 role／content delta、finish_reason 與最終 usage，並以 data: [DONE] 結束。
- **AC-08.2** 給定輸入驗證、上下文檢查或模型生成發生 HTTPException／工作者錯誤，當 SSE 已建立，則輸出 data 內的 error 物件並以 [DONE] 結束；HTTP 狀態已是串流回應狀態。
- **AC-08.3** 給定串流正常結束或 generator 收尾，當執行 finally 區段，則等待仍在執行的工作執行緒並釋放 generation_lock；目前 streamer queue timeout 設為 300 秒，但 timeout 例外未正規化為 SSE error，需由後續測試確認其實際中斷表現。

**驗證方式**：SSE 事件順序、錯誤事件、DONE 與鎖釋放測試。

### REQ-09：POST 本文長度限制

**需求**：API middleware 應在解析指定 POST 本文前檢查 Content-Length。

**驗收條件**

- **AC-09.1** 給定 POST /v1/chat/completions、POST /v1/videos/understand、POST /v1/live/sessions 或 POST /v1/live/mediamtx/auth 未提供 Content-Length，當 middleware 檢查請求，則回 HTTP 411。
- **AC-09.2** 給定 Content-Length 不是非負整數，當 middleware 檢查請求，則回 HTTP 400；超過相應上限則回 HTTP 413。
- **AC-09.3** 給定目前 inference.yml，當檢查路由上限，則聊天為 15,728,640 bytes、上傳視訊為 210,763,776 bytes、即時建立及 MediaMTX 回呼各為 65,536 bytes。

**驗證方式**：HTTP middleware 邊界測試，不傳送大型模型或大型影片。

### REQ-10：上傳視訊 API 與回應

**需求**：視訊理解端點應接收 multipart 視訊與問題，完成單次理解後回傳帶有視訊中繼資料、時間與 GPU 記憶體資訊的 JSON。

**驗收條件**

- **AC-10.1** 給定 multipart 欄位 video、question 及支援的選用欄位，當呼叫 POST /v1/videos/understand，則接受 video_backend frames／codec、codec_engine hevc／dcvc-rt，以及 max_tokens、temperature、top_p。
- **AC-10.2** 給定 question 為空白、影片檔為空、超過 max_video_bytes 或超過 max_video_duration_seconds，當服務處理上傳，則以對應的 HTTP 400 或 HTTP 413 拒絕並刪除暫存工作目錄。
- **AC-10.3** 給定視訊理解成功，當服務回應，則 object 為 magevl4b.video.understanding，並包含 video 中繼資料、timing_seconds、gpu_memory_bytes、usage 及文字 answer。
- **AC-10.4** 給定推論鎖已滿，當再呼叫視訊端點，則不等待並回 HTTP 429；取得推論鎖後，無論成功或處理失敗，finally 區段皆關閉上傳檔案並釋放鎖。

**驗證方式**：合成小型影片 fixture、臨時目錄與 mock model；真機 GPU 測試另列，不在本文件任務執行。

### REQ-11：視訊前處理模式

**需求**：上傳視訊的 frames 與 codec 模式應按各自依賴及設定處理輸入，並回報採樣／編碼中繼資料。

**驗收條件**

- **AC-11.1** 給定 video_backend 為 frames，當服務解析有效影片，則驗證 fps／影格數／尺寸／時長，均勻取樣不超過 video_max_frames，並將影格縮放至設定的最大邊長。
- **AC-11.2** 給定 video_backend 為 codec，當服務前處理影片，則要求 ffprobe、FFmpeg 與 codec-video-prep 可用；缺少執行環境時回 HTTP 503，前處理逾時回 HTTP 504。
- **AC-11.3** 給定 codec_engine 使用 hevc 或 dcvc-rt，當 codec 模式生成輸入，則遵循 codec canvas、max_pixels 與 timeout 設定，並在回應 video metadata 中標示 backend、engine 及 canvas 數。
- **AC-11.4** 給定影片時長超過 api.max_video_duration_seconds，當任一模式檢查 metadata，則回 HTTP 413；目前時長上限為 120 秒。

**驗證方式**：影格採樣與中繼資料單元測試；codec 前處理使用合成素材及隔離環境測試。

### REQ-12：推論併發限制

**需求**：聊天、視訊與即時工作階段應共用由 api.max_concurrent_requests 設定的非阻塞模型推論名額。

**驗收條件**

- **AC-12.1** 給定可用推論名額，當聊天、視訊或 live 分析進入模型，則持有 generation_lock 直至該次模型工作完成。
- **AC-12.2** 給定非串流聊天或視訊請求沒有名額，當請求進入，則立即回 HTTP 429，不排隊等待 GPU。
- **AC-12.3** 給定 SSE 聊天沒有名額，當串流開始，則輸出 server_error JSON SSE 事件與 [DONE]；live 分析沒有名額時以 waiting_for_model 狀態等待，不突破鎖的容量。

**驗證方式**：以 fake semaphore/model 驗證競爭、拒絕及錯誤路徑。

### REQ-13：即時工作階段建立

**需求**：即時工作階段 API 應建立暫時推流路徑並依 YAML 設定開始背景擷取與分析工作。

**驗收條件**

- **AC-13.1** 給定 POST /v1/live/sessions 請求含有效 question，當服務建立工作階段，則回傳 id、status、model、RTSP／RTMP publish URL、events_url、stop_url 及有效 settings。
- **AC-13.2** 給定即時參數，當服務驗證，則 sample_interval_seconds 位於 0.25 至 60、window_seconds 位於 1 至 60、max_duration_seconds 不超過設定上限；frames 模式要求 window 不小於取樣間隔。
- **AC-13.3** 給定已達 max_sessions，當再建立工作階段，則回 HTTP 429；目前 max_sessions 為 1，最長期間設定為 3600 秒。
- **AC-13.4** 給定 codec live 模式所需的二進位檔、套件或 processor 能力缺失，當建立工作階段，則回報相應的 HTTP 503 或設定錯誤，不回傳成功工作階段。

**驗證方式**：工作階段設定驗證、最大數量及缺少依賴測試；不啟動真實 GPU 串流。

### REQ-14：即時視訊擷取與分析生命週期

**需求**：live 工作階段應透過 MediaMTX RTSP/TCP 擷取影格或 codec 片段，依視窗設定分析並發布狀態與分析事件。

**驗收條件**

- **AC-14.1** 給定 frames 模式，當 RTSP 串流可讀，則依 sample_interval_seconds 抽樣、縮放影格並保留 window_seconds 內最多 max_frames 張影格。
- **AC-14.2** 給定 RTSP 尚未發布或連線中斷，當 capture loop 執行，則發布 waiting_for_stream／reconnecting 狀態並依 reconnect_interval_seconds 重試；目前 open/read timeout 為 10／5 秒。
- **AC-14.3** 給定 codec 模式片段到達速度高於推論速度，當等待佇列超過 codec_max_pending_segments，則丟棄過期片段並發布 segment_dropped，不形成無界佇列。
- **AC-14.4** 給定 live 推論發生錯誤、使用者停止或到達 max_duration_seconds，當工作階段結束，則發布 error／stopping／stopped／completed 相應事件並清理片段與暫存目錄。

**驗證方式**：以 fake capture、時鐘及 analyzer 驗證重連、佇列邊界及收尾。

### REQ-15：即時事件、停止與 MediaMTX 授權

**需求**：即時工作階段應提供可續接 SSE 事件、停止操作及 MediaMTX HTTP callback 授權。

**驗收條件**

- **AC-15.1** 給定 GET /v1/live/sessions/{session_id}/events，當客戶端連線，則 SSE 使用遞增 event ID、event 與 JSON data，支援 Last-Event-ID，並每 15 秒送出 keep-alive；單一工作階段事件環形佇列容量為 256。
- **AC-15.2** 給定客戶端斷線或工作階段完成且事件讀取完畢，當 event stream 迴圈檢查狀態，則結束串流；不存在或已清除的工作階段回 HTTP 404。
- **AC-15.3** 給定 DELETE /v1/live/sessions/{session_id} 指向現存工作階段，當服務處理停止請求，則發出停止訊號並回傳 status=stopping；完成超過 900 秒的工作階段可在後續建立新工作階段時清除。
- **AC-15.4** 給定 MediaMTX callback JSON，當 action 是 publish/read、protocol 是 rtsp/rtmp 且 path 指向作用中工作階段，則回 HTTP 204；其他 action、protocol、格式或無效 path 不授權。

**驗證方式**：SSE event replay、停止清理與 MediaMTX callback 契約測試。

### REQ-16：設定邊界、網路暴露與安全責任

**需求**：模型／推論設定、網路端點與 MediaMTX sidecar 設定應由各自檔案管理；推論服務不得冒充外部 API Gateway。

**驗收條件**

- **AC-16.1** 給定目前服務設定，當維護者調整模型、生成、API 請求或影片推論參數，則修改 mage_vl_4b/configs/inference.yml；API host／port 與串流網路位址修改 network.yml；MediaMTX listener 與 auth callback 修改 mediamtx.yml。
- **AC-16.2** 給定目前模型登錄表與 API，當外部系統查詢模型或呼叫 API，則必須區分 mage-vl-4b 專案別名與 microsoft/Mage-VL 上游 repo_id；Inference 不提供 API Key／租戶認證，預設只監聽 127.0.0.1:8001。
- **AC-16.3** 給定安全審查任意 image_url，當服務直接抓取 HTTP(S) 圖片，則明確記錄目前沒有 host/IP allowlist；不得把現況解讀為已具 SSRF 防護。
- **AC-16.4** 給定硬體或 runtime YAML 欄位未被讀取，當規格記錄目前行為，則 expected_gpu、conda_env、python_venv 與 startup_* 欄位標示為未生效，不宣稱服務已驗證這些值。

**驗證方式**：設定責任及網路暴露檢視、安全測試與設定引用靜態檢查。

## 不變項與相容要求

- Gateway 與推論服務分離；推論服務處理模型載入、GPU、視訊前處理與生成。
- 專案別名 mage-vl-4b 與上游 repo_id microsoft/Mage-VL 不可混為一談；目前 Inference API 接受上游識別碼，不接受專案別名。
- 推論 API 預設 127.0.0.1:8001；實際 host／port 由 network.yml 或 launcher 覆寫。
- 聊天 SSE 是文字 token 串流；視訊上傳是 multipart 單次 JSON 回應；live API 以 SSE 發送狀態／分析事件。
- MediaMTX 只供即時推流；文字／圖片聊天與上傳視訊不要求啟動 MediaMTX。
- 模型權重及快取不入 Git；checkpoints/val.md 與 checkpoints/val.xlsx 必須保留。

## 資料處理和保留

- 上傳影片寫入 TemporaryDirectory，推論完成或錯誤後清理；上傳檔案關閉。
- live codec 模式的片段及工作目錄在工作階段結束時清理；事件佇列與工作階段資料保留於程序記憶體。
- 模型快取位置為 runtime.hf_home 下的 Hub cache；目前預設 runtime.hf_home 為 checkpoints。
- API 不持久保存聊天紀錄或視訊理解結果；回應內容由呼叫者接收。

## 明確不做

- 不將推論 API 認證、API Key、租戶管理或限流等 Gateway 職責混入此現況規格。
- 不把 SGLang、vLLM 或第二模型描述為目前可用後端。
- 不新增資料集精度測試或模型評分；本組後續任務是 API／設定／生命週期的行為驗證。
- tests/inference 的靜態來源契約檢查可在無模型環境執行；推論 API runtime 測試、權重下載、GPU 或 MediaMTX 驗證不在本階段完成。

## 需求追蹤

| 現況依據／條款 | REQ | 模組／服務 | 任務 ID | AC | 驗收證據／狀態 |
|---|---|---|---|---|---|
| model_registry.yaml 與 launcher list/start | REQ-01 | launcher.py、model_registry.yaml | T-01 | AC-01.1、AC-01.2、AC-01.3 | 原始碼已檢視；測試待建 |
| YAML 路徑、CLI 覆寫及 runtime environment | REQ-02 | launcher.py、configs/ | T-01 | AC-02.1、AC-02.2、AC-02.3、AC-02.4 | 原始碼已檢視；測試待建 |
| revision、CUDA 模型載入與關閉 | REQ-03 | service.py、inference.yml | T-01、T-07 | AC-03.1、AC-03.2、AC-03.3、AC-03.4 | 原始碼已檢視；真機未驗 |
| 健康及模型清單 | REQ-04 | main.py、service.py | T-01 | AC-04.1、AC-04.2、AC-04.3 | 原始碼已檢視；測試待建 |
| ChatCompletionRequest 與 text-only 差異 | REQ-05 | main.py、service.py、README.md | T-02、T-06 | AC-05.1、AC-05.2、AC-05.3、AC-05.4 | 文件差異已記錄；契約待確認 |
| 圖片下載、解碼、縮放與風險 | REQ-06 | service.py、inference.yml | T-02、T-06 | AC-06.1、AC-06.2、AC-06.3、AC-06.4 | 原始碼已檢視；安全測試待建 |
| 生成參數、上下文與輸出 | REQ-07 | service.py、inference.yml | T-02、T-07 | AC-07.1、AC-07.2、AC-07.3、AC-07.4 | 原始碼已檢視；測試待建 |
| SSE chunks、錯誤事件及完成標記 | REQ-08 | main.py、service.py | T-03 | AC-08.1、AC-08.2、AC-08.3 | 原始碼已檢視；測試待建 |
| POST Content-Length middleware | REQ-09 | main.py、service.py、inference.yml | T-03、T-08 | AC-09.1、AC-09.2、AC-09.3 | 靜態來源檢查涵蓋 middleware 與上限值；runtime 測試待建 |
| multipart 視訊理解與回應 schema | REQ-10 | main.py、service.py | T-04 | AC-10.1、AC-10.2、AC-10.3、AC-10.4 | 原始碼已檢視；測試待建 |
| frames／codec 前處理 | REQ-11 | service.py、inference.yml、requirements | T-04 | AC-11.1、AC-11.2、AC-11.3、AC-11.4 | 原始碼已檢視；環境測試待建 |
| generation_lock 併發行為 | REQ-12 | service.py、inference.yml | T-03、T-05 | AC-12.1、AC-12.2、AC-12.3 | 原始碼已檢視；測試待建 |
| 即時工作階段建立與限制 | REQ-13 | service.py、inference.yml、network.yml | T-05 | AC-13.1、AC-13.2、AC-13.3、AC-13.4 | 原始碼已檢視；測試待建 |
| RTSP capture、重連、片段佇列與結束 | REQ-14 | service.py、inference.yml | T-05 | AC-14.1、AC-14.2、AC-14.3、AC-14.4 | 原始碼已檢視；測試待建 |
| live SSE、stop 與 MediaMTX callback | REQ-15 | service.py、mediamtx.yml | T-05 | AC-15.1、AC-15.2、AC-15.3、AC-15.4 | 原始碼已檢視；整合測試待建 |
| 設定分層、網路邊界與安全差異 | REQ-16 | configs/、launcher.py、main.py | T-06 | AC-16.1、AC-16.2、AC-16.3、AC-16.4 | 原始碼已檢視；決策待確認 |

## 假設與未決問題

- [待確認] README／AGENTS 所述聊天支援文字／圖片，是否代表純文字請求應成功；現況程式明確要求至少一張圖片。
- [待確認] image_url 是否需新增網域／IP 白名單或其他 SSRF 防護。
- [假設] 本規格以目前 checkout 的程式、YAML 為實際現況依據，並以 runtime 測試結果補充後續證據。
- 除上述事項外，本文不對現有 API 行為做產品化改寫。

## 變更記錄

| 日期 | 變更 |
|---|---|
| 2026-10-20 | 建立推論服務現況規格基線，涵蓋 16 項需求 |

