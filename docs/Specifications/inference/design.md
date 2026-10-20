# 技術設計：推論服務現況基線

> 關聯規格：[spec.md](spec.md)  
> 關聯提案：[proposal.md](proposal.md)  
> 任務追蹤：[tasks.md](tasks.md)  
> 狀態：現況描述，非新功能設計。

## 當前實作與責任邊界

| 元件 | 當前責任 |
|---|---|
| src/inference/launcher.py | 讀取模型登錄表，解析路徑、套用 runtime 環境及命令列覆寫，動態載入服務 app 並啟動 Uvicorn |
| src/inference/model_registry.yaml | 提供 default_model、default_backend 與每模型 inference/network/media 設定路徑及 service_entrypoint |
| src/inference/mage_vl_4b/main.py | FastAPI 路由、請求 schema 綁定及 Content-Length middleware |
| src/inference/mage_vl_4b/service.py | 模型載入／生成、輸入整理、圖片／影片前處理、live session 與 MediaMTX auth |
| configs/inference.yml | 模型 repo/revision、權重載入、執行環境、API 限制、影片前處理、生成預設及最佳化 |
| configs/network.yml | API host/port、MediaMTX RTSP／RTMP 內部位址與公開 host |
| configs/mediamtx.yml | RTSP／RTMP listener、HTTP auth callback 及停用的 sidecar 功能 |
| requirements/requirements-lock.txt | 固定 Transformers 推論環境依賴 |

推論服務不提供 API Key 或租戶驗證。預設綁定 loopback，面向外部客戶端的認證與限流屬於 Gateway 責任。

## 啟動與設定解析

1. 從專案根目錄執行 launcher.py list 或 start。
2. 啟動器讀取 src/inference/model_registry.yaml；相對路徑以專案根目錄解析，路徑必須存在。
3. start 以 --model 或 default_model 選擇別名，取用登錄表 default_backend；命令列的 --inference-config、--network-config、--media-config 分別覆寫對應設定路徑。
4. 推論 YAML 載入後，runtime.environment 套用一般環境變數；受保護環境變數必須用專屬設定。runtime.cuda_visible_devices、CLI --cuda-visible-devices 用於限制 GPU 可見裝置。
5. launcher 將 INFERENCE_CONFIG 與 INFERENCE_NETWORK_CONFIG 設為絕對路徑，服務模組讀取兩份 YAML。network.api.host/port 可由 CLI --host/--port 覆寫。
6. 模型服務 app 動態載入後交由 Uvicorn 常駐執行。現在設定 workers 為 1、host 127.0.0.1、port 8001。
7. 模型快照路徑於服務模組載入時解析；模型與 processor 在 FastAPI lifespan 啟動時載入。首次快取缺少時可能先下載，下載／載入完成前 API 尚未就緒。

launcher.py 的 list 只檢查登錄表所指檔案是否存在，不檢查權重快取、CUDA、模型載入或 API 可用性。inference.yml 內 expected_gpu、conda_env、python_venv 及 startup_* 欄位目前未被 launcher.py／service.py 消費。

## API 契約與回應

| 方法與路徑 | 當前行為 |
|---|---|
| GET /health | 回報 starting／ok、backend、模型 repo_id、revision、量化方式、live 工作階段數與功能旗標；模型就緒且 CUDA 可用時回報 GPU 與 VRAM |
| GET /v1/models | 回傳一張 id=microsoft/Mage-VL 的模型卡；此 API 不回傳專案別名 |
| POST /v1/chat/completions | 接受 ChatCompletionRequest；model 可缺省或等於 microsoft/Mage-VL；支援一般 JSON 或 stream=true SSE |
| POST /v1/videos/understand | 接收 multipart video、question、video_backend、codec_engine、max_tokens、temperature、top_p；回傳一次性理解 JSON |
| POST /v1/live/sessions | 建立短期工作階段並回傳 publish、events_url、stop_url 與處理設定 |
| GET /v1/live/sessions/{session_id}/events | SSE 傳送工作階段 status、analysis、error、completed 事件，支援 Last-Event-ID |
| DELETE /v1/live/sessions/{session_id} | 發出停止訊號並回傳 stopping 狀態 |
| POST /v1/live/mediamtx/auth | 僅供 MediaMTX HTTP auth callback；驗證 action、protocol 與作用中路徑 |

Inference API 的聊天 model 欄位使用上游 repo_id；Gateway 規格定義的公開別名映射屬於 Gateway 行為，不應反寫為 Inference 現況。除 MediaMTX callback 外，live client routes 目前沒有 API Key 認證。

## 模型載入與推論資源

- 使用 model.id 與固定 model.revision 從 Hugging Face 準備快照，快取根目錄為 runtime.hf_home/hub；runtime.local_files_only 控制是否可下載。
- AutoProcessor 與 AutoModelForCausalLM 以相同 revision 載入，trust_remote_code 為 true，模型進入 eval 狀態。
- CUDA 不可用時 lifespan 啟動失敗。當前 YAML 指定 bfloat16、device_map auto、quantization.method none、gpu_max_memory 22GiB，expected_gpu 欄位目前不會檢查實際裝置。
- 設定可選 nf4、int8、prequantized 等量化方式；可選 offload、kernel、attention implementation 與 torch.compile。當前 torch.compile.enabled 為 false。
- generation_lock 由 api.max_concurrent_requests 建立，聊天、視訊與 live 分析共用；一般聊天及視訊採非阻塞取得，名額滿時回 429。
- lifespan 關閉先停止 live session，再解除 model／processor 引用；CUDA 可用時呼叫 empty_cache。

## 聊天與圖片處理

- Pydantic ChatCompletionRequest 至少要求一筆 messages；role 僅 system、user、assistant；content 可為字串或 text／image_url parts。額外欄位忽略。
- 路由只接受 model 為 null 或 microsoft/Mage-VL；傳入 mage-vl-4b 會被判定為 Unknown model。
- 目前 normalize_messages 要求至少一個 image_url，故純文字對話會在服務層回 400。README 說明與此處不同，待產品負責人確認預期契約。
- image_url 支援嚴格 base64 data URL 或由服務端 GET 的 HTTP／HTTPS URL。圖片位元組、像素、數量、最大邊長及重採樣方法由 inference.yml 控制。
- 目前有效設定：單一請求最多 1 張圖、每圖 10 MiB、最多 16,777,216 像素、縮放最大邊長 896、遠端 URL timeout 10 秒。
- 圖片使用 Pillow 解碼並轉 RGB；無效格式回 400，大小／像素／數量限制回 413。遠端 URL 不設來源主機或 IP allowlist，這是對外暴露前需評估的安全缺口。

## 生成參數與 SSE

- max_tokens 優先於 max_completion_tokens；兩者皆缺省時使用 generation.max_tokens_default。API 上限為 api.max_new_tokens。
- temperature、top_p、top_k、min_p、repetition_penalty、no_repeat_ngram_size、num_beams 與 cache 行為由請求及 generation YAML 合併；各參數的實際界限見 REQ-07。
- prompt tokens 加預計輸出 tokens 不可超過 api.max_context_tokens。
- 一般完成回應包含 chat.completion、repo_id、choices、finish_reason 與 usage。
- SSE 使用 TextIteratorStreamer，由背景執行緒生成；每次輸出 JSON chunk，成功時附最終 usage chunk 及 data: [DONE]。
- SSE 開始後發生 HTTPException 或模型工作者錯誤時，服務將 error JSON 作為 data 事件傳送，再送 [DONE]；此錯誤不再改變已送出的 HTTP 狀態。

## 視訊理解流程

1. API middleware 依 Content-Length 在解析本文前拒絕缺少長度、無效長度或超過本文上限的 POST。
2. FastAPI 接收 multipart 上傳；服務檢查非空 question，取用設定或請求指定的 backend／codec。
3. 上傳先以 1 MiB 區塊寫入唯一 TemporaryDirectory；檔案大小上限為 200 MiB。finally 路徑關閉上傳檔並釋放推論名額，臨時目錄離開作用域後清理。
4. frames 模式以 OpenCV 讀取 fps、影格數、尺寸與時長；均勻涵蓋影片抽取最多 16 張影格，每張縮放至最大邊長 896。
5. codec 模式先用 ffprobe 讀中繼資料，再由 codec-video-prep/FFmpeg 路徑建立模型輸入；目前允許 hevc 與 dcvc-rt。
6. 成功回應含 magevl4b.video.understanding、回答文字、token usage、來源及採樣／codec metadata、staging_and_decode／processor／generation／server_total 時間與 GPU peak allocated/reserved bytes。

目前限制：檔案 209,715,200 bytes，總請求 210,763,776 bytes，live 建立及 MediaMTX callback 本文各 65,536 bytes，時長 120 秒；這些上限與 frames/codec 前處理參數均在 inference.yml。影片 API 是單次 JSON 回應，不是 SSE。

## 即時視訊工作階段

- 工作階段資料、事件環形佇列及擷取／推論執行緒均存在推論程序記憶體；session ID 使用 secrets.token_urlsafe(24)。
- 建立請求含 question、生成參數及可覆寫的 video_backend、codec_engine、sample_interval_seconds、window_seconds、max_duration_seconds。question 長度不超過 2000；採樣間隔 0.25 至 60 秒，視窗 1 至 60 秒，frames 模式視窗需至少等於採樣間隔。
- 預設最大工作階段數為 1、最長 3600 秒；發起新工作階段前會清除已完成超過 900 秒的工作階段。
- frames 模式使用 OpenCV 從 MediaMTX RTSP/TCP 擷取並縮放影格；codec 模式使用 FFmpeg 切段並維持有限 pending segment queue。串流中斷時依設定重連。
- 即時事件包含 status、analysis、error、completed；事件 ID 遞增，單一工作階段最多保留 256 筆。SSE 每 15 秒 keep-alive，離線客戶端可用 Last-Event-ID 續讀尚保留事件。
- POST /v1/live/mediamtx/auth 接受 publish/read 與 rtsp/rtmp 組合，只授權仍作用中的 session path，成功回 HTTP 204。
- live 建立及 MediaMTX callback 共用 api.max_live_request_bytes；目前 inference.yml 設為 65,536 bytes。
- API 關閉時 stop_all 發出停止訊號並等待推論執行緒完成，再卸載模型。程序中斷會丟失 live session 與未讀事件。

## 設定檔分工與目前注意事項

| 檔案 | 現況責任 |
|---|---|
| model_registry.yaml | 模型別名、default_model、default_backend、推論／網路／媒體設定路徑、服務入口 |
| inference.yml | 固定模型 revision、載入與量化、生成、請求限制、圖片／視訊／live 參數、PyTorch 最佳化 |
| network.yml | API host 127.0.0.1、port 8001、MediaMTX 主機／連接埠及 public_host |
| mediamtx.yml | HTTP auth callback 指向 http://127.0.0.1:8001/v1/live/mediamtx/auth；啟用 RTSP/TCP 與 RTMP，停用 HLS、WebRTC、SRT 等 |

目前 network.yml 是網路位址唯一來源；調整時須同步檢視 MediaMTX callback 與外部推流 URL。模型推論參數留在 inference.yml，勿搬到 Gateway。model_registry 頂層 default_backend 目前是 transformers，不代表其他 backend 已存在。

## 安全、失敗及相容風險

- inference API 沒有客戶端認證；預設 loopback 綁定是目前的網路隔離，對外暴露必須經受控 Gateway／網路政策。
- 任意 HTTP(S) image_url 由服務端下載，未實作 SSRF allowlist；遠端圖片需 timeout 與大小上限，但這不足以替代來源限制。
- trust_remote_code 允許載入固定 revision 中的模型程式碼；改 revision 前須審查程式來源與相依版本。
- API 錯誤一般採 FastAPI detail JSON；SSE 例外採 data 中 error JSON。Gateway 若代理，需按 Gateway 規格轉換，而不應聲稱推論 API 本身已提供 OpenAI error envelope。
- GPU、codec 套件、ffmpeg／ffprobe、CUDA 與權重快取缺失會導致啟動或特定 video backend 失敗；health 只回報模型引用是否就緒，並非所有 codec／MediaMTX 依賴的完整探測。
- hardware.expected_gpu 與 runtime 的部分環境／啟動紀錄欄位目前未被程式使用，文件不可把它們表述為已驗證能力。

## 驗收與回復策略

- 靜態規格驗收只證明文件結構及 ID 追蹤，不證明程式行為或真機結果。
- 單元／API 測試置於 tests/inference/，以 mock model、mock downloader、synthetic image/video fixture 驗證，不下載大型權重。
- 有權重與 GPU 的載入／真機驗證另行標明硬體、CUDA、套件版本及模型 revision；未執行時不得標記通過。
- 即時端到端流程需要 inference API 與 MediaMTX；不得將該依賴放入一般 unit test。
- 推論程式可回退至上一版程式及 YAML；權重 revision 若需恢復，使用原固定 revision 與既有 checkpoints cache。任何清理 cache 的操作須先核對目標目錄。

## 未決項

- 純文字聊天是否屬於預期 API 能力。
- image_url 是否增加 host/IP allowlist 或網路出口限制。
- unused hardware/runtime 欄位應清理、實作或保留作為人工註記。

## 變更記錄

| 日期 | 變更 |
|---|---|
| 2026-10-20 | 建立推論服務現況技術設計 |

