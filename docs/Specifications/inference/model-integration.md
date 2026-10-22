# 新模型接入操作指南

本指南說明如何將模型接入共用 Inference Engine。新模型只新增模型專屬程式、設定與登錄項目；不建立自己的 API app 或 Engine Core。程式目錄使用 `models/<model_alias>/`，不把 `src/` 或 `inference/` 當成 Python package。

## 接入路徑

先盤點模型的輸入模態、逐步生成方式、Cache 所有權、輸出型態及模型權重格式，再選擇整合方式：

| 路徑 | 適用條件 | 模型包工作 |
|---|---|---|
| `standard` | 常見 decoder-only causal LM；固定 revision 提供文字 Chat Template，forward 支援逐步輸入並回傳可延續的 `past_key_values`；部署為單一已登記裝置 | 建立模型 YAML 和 Registry 項目；可省略 Runner／Adapter／InputProcessor，由共用預設實作處理 |
| `custom` | 模型需要專屬計算圖、權重映射、Tokenizer／Chat Template、影音前處理、Cache 或採樣邏輯 | 在模型目錄實作 Runner／Adapter 及所需元件，明確擁有模型專屬狀態 |

`standard` 是一個受限的共用文字執行路徑，不代表任意 `AutoModel` 都相容。預設元件在 `api/default_chat_processor.py` 和 `engine/standard_text.py`：API 用固定 revision Tokenizer 的 `chat_template` 產生 token IDs，Worker 按 Engine 的 prefill／decode work item 逐步呼叫 causal LM forward 並保留每個 request 的 `past_key_values`。目前 `execute_batch()` 逐項 forward，不做跨請求合併；Engine KV block 是容量租約，不是實體分頁 Cache。遇到多模態、encoder-decoder、多裝置、特殊 Cache 或生成流程時改用 `custom`，不得用整段 `generate()` 假裝符合 Scheduler。

目前 Mage-VL 與 Qwen3-0.6B 仍登記為 `custom`：Mage-VL 有圖片／影片／Live Processor 及模型 Cache；Qwen3 使用原生 PyTorch 計算圖和 GPU KV page pool。Qwen snapshot 已驗證標準預設 Runner；2026-10-22 另以隔離臨時 Registry 將 Qwen 改列 `standard`、省略模型工廠並啟動完整 API，確認預設路徑可服務非串流及文字 SSE。正式 Registry 仍保持原 profile；實測報告見 `tests/end2end/results/qwen3-0.6b-standard-profile-api-smoke-2026-10-22.json`。

## 目錄

```text
src/inference/models/<model_alias>/
├── __init__.py
├── adapter.py                 # 將 Engine 工作轉為 Runner 操作
├── model_runner.py            # 載入模型並執行生成
├── input_processor.py         # 選用：Tokenizer／Chat Template 等 API 輸入處理
├── components/                # 選用：模型拓撲、算子及權重載入
├── utils/                     # 選用：上游格式、Tokenizer 或 revision 輔助工具
├── configs/
│   ├── inference.yml
│   ├── network.yml
│   └── mediamtx.yml            # 選用：只有使用 Live 時需要
└── requirements/               # 選用：模型專屬依賴
```

`adapter.py` 和 `model_runner.py` 是目前逐步生成鏈路的模型實作入口。`config.py`、Processor、權重 loader、`components/` 與 `utils/` 依模型實際需要新增；不需複製 Mage-VL 或 Qwen 的整套目錄。

標準路徑的模型包可以只保留 `configs/` 和依賴清單；Runner／Adapter／InputProcessor 工廠不需建立。若後續要覆寫預設實作，只為確有差異的元件增加明確工廠，其他元件仍走共用預設。

標準預設 Runner 需要共用環境安裝 PyTorch、`transformers` 和 `accelerate`；模型 YAML 的固定 revision 與 tokenizer／權重檔案需預先存在本機。各模型仍應將依賴版本寫入自己的 `requirements/`，並在目標部署環境完成固定 snapshot smoke，不能只依賴無權重預檢。

## 1. 建立 Runner 與 Adapter

Runner 在 GPU Worker 子程序建立模型。共用 Worker 會呼叫 `load()`、`gpu_status()` 和 `close()`；Adapter 需要提供 Engine 的操作介面，並管理每個 request 的模型狀態。

選擇 `standard` 時無需實作下方 class；預設 Runner／Adapter 已提供這些操作。要覆寫時才實作相同方法，並將工廠登記在 Registry。`custom` 必須提供自己的 Runner 和 Adapter。

```python
class ExampleModelRunner:
    def load(self) -> None:
        ...

    def gpu_status(self) -> dict:
        ...

    def close(self) -> None:
        ...


class ExampleModelAdapter:
    def __init__(self, model_runner: ExampleModelRunner) -> None:
        self.model_runner = model_runner

    def health_info(self) -> dict:
        ...

    def generation_options(self, request_body: dict) -> dict:
        ...

    def prepare_chat_generation(self, request_id: str, request_body: dict) -> dict:
        ...

    def bind_generation_cache(self, request_id: str, block_ids: tuple[int, ...]) -> None:
        ...

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> Sequence[WorkerResult]:
        ...

    def finish_chat_generation(self, request_id: str) -> InferenceResult:
        ...

    def cancel_generation(self, request_id: str) -> None:
        ...
```

`execute_batch()` 按 `request_id` 回傳獨立結果，遵守既有 prefill／decode 契約。Adapter／Runner 負責模型 Cache 的建立、追加、finish 和 cancel；共用 Engine 的 KV Manager 只按模型已實作的語義管理容量租約。標準預設 Runner 逐項執行 batch 中的 work item，不承諾一次 forward 處理多個 request。若模型不能逐步 prefill／decode，先設計並驗收相容方案，不得把整段 `generate()` 假稱為排程式生成。

多模態模型依實際路由增加 `prepare_video_generation()`、`finish_video_generation()` 或 Live 操作；不支援的路由不登記。模型回傳文字增量時沿用現有文字 SSE；若模型回傳 token ID，需在 API 登記輸入處理器，讓 API 負責增量 Detokenize。兩種 SSE 都必須保留文字內容、finish usage 和 `[DONE]`。

Adapter／Runner 最小方法集可由 [`tests/inference/model_contract.py`](../../../tests/inference/model_contract.py) 讀取檢查；宣告 `video_upload` 或 `live_video` 的模型還必須實作相應 Adapter 操作。此檢查解析 Python AST，不 import 模型、不讀權重、不建立 CUDA context。

## 2. 新增 YAML 設定

將模型識別碼與固定 revision、權重載入、生成參數、API 請求上限、Scheduler／KV 設定及 Worker／Engine timeout 放在 `configs/inference.yml`。API 位址放在 `configs/network.yml`；Live 模型才需要 MediaMTX 設定。新模型的特殊可調參數應留在模型 YAML，不散落於共用程式碼。

## 3. 登記模型工廠

在 `src/inference/model_registry.yaml` 新增模型項目。符合標準文字契約時，Registry 只需路徑、識別碼和能力；可省略 `components` 與 `api.input_processor`：

```yaml
models:
  example-text-model:
    display_name: Example Text Model
    integration_profile: standard
    backend: standard_text
    inference_config: src/inference/models/example_text_model/configs/inference.yml
    network_config: src/inference/models/example_text_model/configs/network.yml
    api:
      model_id: example-text-model
      capabilities:
        chat_inputs: [text]
        streaming: true
        output_mode: token_ids
        video_upload: false
        live_video: false
```

標準路徑 YAML 必須有固定 40 位 commit revision、`model.id`、支援的 `dtype` 和 `runtime.local_files_only`；預設實作使用 `AutoModelForCausalLM`、單一 `torch_cuda` 裝置及模型 `chat_template`。執行 `validate` 時不會載入 Transformers 或權重。只有完成該固定 revision 的實際模型 smoke 後，才可將模型登記為 `standard`。

若不符合標準文字契約，登記 `custom` 並使用模型專屬工廠；下例示範此路徑，工廠格式為 `module:attribute`：

```yaml
models:
  example-model:
    display_name: Example Model
    integration_profile: custom
    backend: example_backend
    inference_config: src/inference/models/example_model/configs/inference.yml
    network_config: src/inference/models/example_model/configs/network.yml
    api:
      model_id: example-model
      input_processor: models.example_model.input_processor:ExampleInputProcessor
      capabilities:
        chat_inputs: [text]
        streaming: true
        output_mode: token_ids
        video_upload: false
        live_video: false
    components:
      runner: models.example_model.model_runner:ExampleModelRunner
      adapter: models.example_model.adapter:ExampleModelAdapter
```

`api.model_id` 是推論 API 接受及回傳的識別碼；它與 `inference.yml` 中的上游 `model.id`／repo ID 分開設定。直連 API 時可使用專案別名；若 Gateway 會改寫上游請求，則依 Gateway route 登記的上游識別碼設定。`api.capabilities` 是推論 API 的路由契約：`chat_inputs` 只接受 `text`／`image`，`output_mode` 為 `text_delta` 或 `token_ids`，並明確宣告串流、影片上傳及 Live 能力。自訂 `token_ids` 模型必須登記提供非串流與增量輸出解碼的 `input_processor`；標準模型會使用共用預設 Processor。`live_video` 必須登記 Live manager 和 MediaMTX YAML。影片及 Live 的本文上限仍放在模型 `inference.yml`。

Gateway 的 `configs/gateway.yml` 另登記同一專案別名、上游 `model_id`、推論服務 URL 和公開能力。Gateway 與 Inference 保持部署／程式邊界分離；不得在兩邊各自宣告後任其漂移。`tests/end2end/test_config_compatibility.py` 會比對兩份設定的別名、上游識別碼、API 能力和服務位址。

API、Engine Core、Executor 和 Worker 均由 `api/service_app.py`、`engine/model_bootstrap.py` 建立。新模型不可在共用 route 或 Launcher 中加入 alias 分支，也不需要註冊 `service_entrypoint`。

## 4. 檢查並啟動

從專案根目錄執行：

```bash
python src/inference/launcher.py list
python src/inference/launcher.py validate
python src/inference/launcher.py validate --model example-model
python src/inference/launcher.py start --model example-model
```

`list` 顯示登錄狀態；`validate` 檢查整合 profile、能力宣告、YAML、factory 路徑及其 class methods（以 AST 解析），不 import 模型工廠、不下載或載入權重，也不啟動 GPU。`tests/inference/model_contract.py` 將同一套方法契約測試套用到每個登錄模型；共用 Worker 在模型權重載入前再檢查 Runner／Adapter 實例的方法。兩者都不能證明固定模型 Cache 數值行為，該項仍需 snapshot smoke。`start` 建立共用 API app、Engine Core 和 Worker。若環境允許自動取得權重，第一次啟動可能準備上游 snapshot；離線部署則先將指定 revision 放入 `runtime.local_model_path`。

依模型路由檢查 `/health`、`/v1/models` 和 `/v1/chat/completions`。對逐 token 模型，確認 SSE 增量合併結果等於非串流回答，且包含 finish usage 和 `[DONE]`；對多模態模型，檢查其登記的影片／Live 路由及本文上限。這些驗收依賴權重及目標硬體，不能由 `launcher.py list` 取代。

若透過 Go Gateway 對外提供多模型，還需更新 Gateway YAML 並通過跨設定檔相容測試。聊天以 JSON `model` 選模型；影片上傳使用 `POST /v1/videos/understand?model=<alias>`（省略時採 Gateway 的 `defaults.video_model`），避免解析／緩存 multipart 本文；Live 建立以 JSON `model` 選模型，後續 SSE／停止路由依簽名句柄回到建立時的同一個上游。Gateway 不支援的輸入或功能會在轉送前拒絕。

## 5. 以 Mage-VL-4B 驗證 custom 路徑

Mage-VL 維持 `custom`，因其圖片／影片輸入、專屬 Processor、Transformers Cache 及 Live 工作階段不符合標準純文字 Runner。它仍完整使用共用 API、Engine Core、Scheduler、Executor、Worker、KV lease 和文字 SSE；模型專屬能力留在 Mage 包：

| 工作 | Mage-VL 實作 | 接入檢查 |
|---|---|---|
| 模型載入及逐步 Cache | `models/mage_vl_4b/model_runner.py`、`components/model_loader.py`、`components/stepwise_generation.py` | Worker／Runner contract；固定 revision 在 Worker ready 前驗證 |
| 圖片／影片處理與生成參數 | `models/mage_vl_4b/processor.py`、`adapter.py` | `chat_inputs: [image]`、`video_upload: true`；檢查本文及模型 YAML 上限 |
| Live frame／video adapter | `models/mage_vl_4b/adapter.py`、`live.py` | `live_video: true`、Live manager 工廠及 `mediamtx.yml` 均必須登記 |
| 組件替換及推論設定 | `models/mage_vl_4b/components/`、`configs/inference.yml` | 替換項目先經固定 registry 驗證；不在通用 Engine 加模型分支 |
| 服務 API／網路／依賴 | `configs/network.yml`、`requirements/`、`model_registry.yaml` | `python src/inference/launcher.py validate --model mage-vl-4b` |

Mage 路徑的 Gateway mock capability matrix 覆蓋圖片聊天 JSON／SSE、影片上傳、Live create/events/stop，並拒絕純文字聊天。Qwen matrix 覆蓋文字聊天 JSON／SSE，並拒絕圖片、影片和 Live；用例位於 `tests/gateway/handler_test.go`。2026-10-22 的真實 Go Gateway smoke 已完成 Qwen 文字 JSON／SSE、Mage-VL 固定 snapshot 圖片 JSON／SSE／視訊上傳，以及 Mage Live／MediaMTX 合成 RTSP。Mage 圖片 JSON 報告見 [`mage JSON smoke`](../../../tests/end2end/results/mage-vl-4b-gateway-json-smoke-2026-10-22.json)，Live 建立、事件接收、停止均成功，首次 `analysis` 約 36.4 秒；報告見 [`mage live smoke`](../../../tests/end2end/results/mage-vl-4b-live-smoke-position-fix-2026-10-22.json)。Qwen JSON／SSE 報告亦置於 [`tests/end2end/results/`](../../../tests/end2end/results/)。這些低樣本 smoke 不構成正式性能基線。

## 6. 分層驗收

- `tests/inference/`：預檢、Runner／Adapter／InputProcessor contract、Engine lifecycle、batch、Cache 與 token/text SSE 事件。CPU synthetic test 不讀模型權重。
- `tests/gateway/`：兩個登錄模型所有已宣告能力的正向路由與不支援能力拒絕；使用 Go `httptest` mock 上游。
- `tests/end2end/`：保留 config／客戶端契約測試與真實服務性能工具。2026-10-22 已透過 Gateway 對 Qwen3-0.6B 執行 JSON／SSE、串行／並發文字 SSE 與客戶端斷流回收 smoke；Mage-VL 圖片 JSON／SSE、視訊上傳各完成 1 次，並完成 Live／MediaMTX 合成 RTSP 單次 smoke。目前樣本不足以建立性能基線；每個結果需標註模型、推理設定與負載，不把 mock 結果當成部署驗收。
- 真實模型 smoke 必須使用本機固定 revision，記錄裝置與依賴版本。4060 應先單請求、短輸出、低併發；若 GPU 正忙則先用 CPU 或延後，不為了 smoke 擴大 KV pool。此項不做資料集精度評測，也不以 mock 結果替代。

## 服務入口邊界

Mage-VL 與 Qwen 均由 Launcher 載入共用 API／Engine 裝配器，模型目錄不建立或保留 `app.py`／`engine_core.py` 包裝檔。新增模型只需提供已登記的 Runner／Adapter 與選用模型元件。Qwen 原生 Worker、Tokenizer、token ID event 和 API 增量 Detokenizer 維持原責任邊界。
