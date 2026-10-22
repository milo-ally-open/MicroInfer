package gateway_test

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gitee.com/milo-ally-intern/LYGCloudModelAPI/src/gateway"
)

const (
	apiKeyA = "test-api-key-for-tenant-a-0001"
	apiKeyB = "test-api-key-for-tenant-b-0002"
)

func newConfig(upstream string) gateway.Config {
	var cfg gateway.Config
	cfg.Listener.Host = "127.0.0.1"
	cfg.Listener.Port = 8000
	cfg.PublicBaseURL = "http://gateway.test:8000"
	cfg.Upstream.BaseURL = upstream
	cfg.Defaults.VideoModel = "mage-vl-4b"
	cfg.Defaults.LiveModel = "mage-vl-4b"
	cfg.Upstream.MaxIdleConnections = 10
	cfg.Upstream.MaxIdleConnectionsPerHost = 5
	cfg.Upstream.IdleConnectionSeconds = 30
	cfg.Upstream.TCPKeepAliveSeconds = 30
	cfg.Upstream.ExpectContinueSeconds = 1
	cfg.Models = map[string]gateway.ModelRoute{
		"mage-vl-4b": {UpstreamModel: "microsoft/Mage-VL", BaseURL: upstream, Capabilities: gateway.ModelCapabilities{ImageChat: true, Streaming: true, VideoUpload: true, LiveVideo: true}},
		"qwen3-0.6b": {UpstreamModel: "qwen3-0.6b", BaseURL: upstream, Capabilities: gateway.ModelCapabilities{TextChat: true, Streaming: true}},
	}
	cfg.Limits.ChatRequestBytes = 1024 * 1024
	cfg.Limits.VideoRequestBytes = 2 * 1024 * 1024
	cfg.Limits.LiveRequestBytes = 64 * 1024
	cfg.Limits.JSONResponseBytes = 1024 * 1024
	cfg.Limits.SSEEventBytes = 64 * 1024
	cfg.Limits.MaxHeaderBytes = 1024 * 1024
	cfg.Limits.StreamBufferBytes = 32 * 1024
	cfg.Timeouts.ConnectSeconds = 2
	cfg.Timeouts.RequestSeconds = 2
	cfg.Timeouts.VideoRequestSeconds = 2
	cfg.Timeouts.StreamIdleSeconds = 2
	cfg.Timeouts.StreamMaxSeconds = 4
	cfg.Timeouts.ReadinessSeconds = 1
	cfg.Timeouts.ReadHeaderSeconds = 2
	cfg.Timeouts.IdleConnectionSeconds = 2
	cfg.Timeouts.ShutdownGraceSeconds = 2
	cfg.RateLimit.RequestsPerSecond = 100
	cfg.RateLimit.Burst = 100
	cfg.Auth.APIKeysFileEnv = "GATEWAY_API_KEYS_FILE"
	cfg.Session.HMACKeysEnv = "GATEWAY_SESSION_HMAC_KEYS"
	cfg.Session.HandleTTLSeconds = 3600
	cfg.Observability.LogLevel = "info"
	cfg.Observability.MetricsEnabled = true
	cfg.Observability.MetricsPath = "/metrics"
	return cfg
}

func newHandler(t *testing.T, cfg gateway.Config, keys [][]byte) http.Handler {
	t.Helper()
	handler, err := gateway.NewHandler(cfg, gateway.Options{
		APIKeys: []gateway.APIKey{
			{ID: "key-a", TenantID: "tenant-a", Secret: apiKeyA},
			{ID: "key-b", TenantID: "tenant-b", Secret: apiKeyB},
		},
		HMACKeys: keys,
		Logger:   slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

func postJSON(t *testing.T, handler http.Handler, path, token, body string) *httptest.ResponseRecorder {
	t.Helper()
	request := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func TestModelsAuthAndPublicAlias(t *testing.T) {
	upstreamCalls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected upstream path %q", r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["model"] != "microsoft/Mage-VL" {
			t.Errorf("upstream model = %v", body["model"])
		}
		if r.Header.Get("Authorization") != "" {
			t.Error("client API key was forwarded to inference")
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"completion-1","model":"microsoft/Mage-VL","choices":[]}`)
	}))
	defer upstream.Close()
	handler := newHandler(t, newConfig(upstream.URL), [][]byte{bytes.Repeat([]byte{7}, 32)})

	models := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	models.Header.Set("Authorization", "Bearer "+apiKeyA)
	modelResponse := httptest.NewRecorder()
	handler.ServeHTTP(modelResponse, models)
	if modelResponse.Code != http.StatusOK || !strings.Contains(modelResponse.Body.String(), "mage-vl-4b") || !strings.Contains(modelResponse.Body.String(), "qwen3-0.6b") || strings.Contains(modelResponse.Body.String(), "microsoft/Mage-VL") {
		t.Fatalf("unexpected models response (%d): %s", modelResponse.Code, modelResponse.Body.String())
	}

	response := postJSON(t, handler, "/v1/chat/completions", apiKeyA, `{"model":"mage-vl-4b","messages":[{"role":"user","content":[{"type":"text","text":"Describe this"},{"type":"image_url","image_url":{"url":"https://example.test/image.png"}}]}],"stream":false}`)
	if response.Code != http.StatusOK {
		t.Fatalf("chat status = %d: %s", response.Code, response.Body.String())
	}
	var result map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result["model"] != "mage-vl-4b" {
		t.Fatalf("public model = %v", result["model"])
	}
	if upstreamCalls != 1 {
		t.Fatalf("upstream calls = %d, want 1", upstreamCalls)
	}
}

func TestAuthenticationAndUnknownModelAreRejectedBeforeUpstream(t *testing.T) {
	calls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls++; w.WriteHeader(http.StatusOK) }))
	defer upstream.Close()
	handler := newHandler(t, newConfig(upstream.URL), [][]byte{bytes.Repeat([]byte{3}, 32)})

	unauthorized := postJSON(t, handler, "/v1/chat/completions", "", `{"model":"mage-vl-4b"}`)
	if unauthorized.Code != http.StatusUnauthorized || !strings.Contains(unauthorized.Body.String(), `"code":"invalid_api_key"`) {
		t.Fatalf("unexpected unauthorized response: %d %s", unauthorized.Code, unauthorized.Body.String())
	}
	unknown := postJSON(t, handler, "/v1/chat/completions", apiKeyA, `{"model":"future-model"}`)
	if unknown.Code != http.StatusBadRequest || !strings.Contains(unknown.Body.String(), `"code":"model_not_found"`) {
		t.Fatalf("unexpected model response: %d %s", unknown.Code, unknown.Body.String())
	}
	if calls != 0 {
		t.Fatalf("upstream calls = %d, want 0", calls)
	}
}

func TestQwenTextChatStreamingAndImageCapabilityRouting(t *testing.T) {
	var calls int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["model"] != "qwen3-0.6b" {
			t.Errorf("upstream model = %v", body["model"])
		}
		if body["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, "data: {\"model\":\"qwen3-0.6b\",\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: [DONE]\n\n")
			return
		}
		io.WriteString(w, `{"model":"qwen3-0.6b","choices":[]}`)
	}))
	defer upstream.Close()
	mageCalls := 0
	mageUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mageCalls++
		w.WriteHeader(http.StatusOK)
	}))
	defer mageUpstream.Close()
	cfg := newConfig(mageUpstream.URL)
	cfg.Models["qwen3-0.6b"] = gateway.ModelRoute{
		UpstreamModel: "qwen3-0.6b", BaseURL: upstream.URL,
		Capabilities: gateway.ModelCapabilities{TextChat: true, Streaming: true},
	}
	handler := newHandler(t, cfg, [][]byte{bytes.Repeat([]byte{0x13}, 32)})

	ordinary := postJSON(t, handler, "/v1/chat/completions", apiKeyA, `{"model":"qwen3-0.6b","messages":[{"role":"user","content":"hello"}]}`)
	if ordinary.Code != http.StatusOK || !strings.Contains(ordinary.Body.String(), `"model":"qwen3-0.6b"`) {
		t.Fatalf("Qwen text chat failed: %d %s", ordinary.Code, ordinary.Body.String())
	}
	streamed := postJSON(t, handler, "/v1/chat/completions", apiKeyA, `{"model":"qwen3-0.6b","messages":[{"role":"user","content":"hello"}],"stream":true}`)
	if streamed.Code != http.StatusOK || !strings.Contains(streamed.Body.String(), "data: [DONE]\n\n") || !strings.Contains(streamed.Body.String(), `"model":"qwen3-0.6b"`) {
		t.Fatalf("Qwen SSE failed: %d %s", streamed.Code, streamed.Body.String())
	}
	image := postJSON(t, handler, "/v1/chat/completions", apiKeyA, `{"model":"qwen3-0.6b","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.test/image.png"}}]}]}`)
	if image.Code != http.StatusBadRequest || !strings.Contains(image.Body.String(), `"code":"unsupported_model_capability"`) {
		t.Fatalf("Qwen image request should be rejected: %d %s", image.Code, image.Body.String())
	}
	mageText := postJSON(t, handler, "/v1/chat/completions", apiKeyA, `{"model":"mage-vl-4b","messages":[{"role":"user","content":"text only"}]}`)
	if mageText.Code != http.StatusBadRequest || !strings.Contains(mageText.Body.String(), `"code":"unsupported_model_capability"`) {
		t.Fatalf("Mage text-only request should be rejected: %d %s", mageText.Code, mageText.Body.String())
	}
	if calls != 2 {
		t.Fatalf("upstream calls = %d, want 2 successful Qwen requests", calls)
	}
	if mageCalls != 0 {
		t.Fatalf("Qwen requests were sent to Mage upstream %d times", mageCalls)
	}
}

func TestChatSSERewritesAliasAndPreservesDone(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		io.WriteString(w, "data: {\"model\":\"microsoft/Mage-VL\",\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n")
		flusher.Flush()
		io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	handler := newHandler(t, newConfig(upstream.URL), [][]byte{bytes.Repeat([]byte{5}, 32)})
	response := postJSON(t, handler, "/v1/chat/completions", apiKeyA, `{"model":"mage-vl-4b","messages":[{"role":"user","content":[{"type":"text","text":"Describe this"},{"type":"image_url","image_url":{"url":"https://example.test/image.png"}}]}],"stream":true}`)
	if response.Code != http.StatusOK {
		t.Fatalf("stream status = %d: %s", response.Code, response.Body.String())
	}
	body := response.Body.String()
	if !strings.Contains(body, `"model":"mage-vl-4b"`) || !strings.Contains(body, "data: [DONE]\n\n") {
		t.Fatalf("unexpected SSE body: %s", body)
	}
}

func TestVideoMultipartStreamsThroughWithOriginalBoundary(t *testing.T) {
	var received []byte
	var contentType string
	var mageCalls int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mageCalls++
		contentType = r.Header.Get("Content-Type")
		var err error
		received, err = io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"model":"microsoft/Mage-VL","text":"ok"}`)
	}))
	defer upstream.Close()
	qwenCalls := 0
	qwenUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		qwenCalls++
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer qwenUpstream.Close()
	cfg := newConfig(upstream.URL)
	qwenRoute := cfg.Models["qwen3-0.6b"]
	qwenRoute.BaseURL = qwenUpstream.URL
	cfg.Models["qwen3-0.6b"] = qwenRoute
	handler := newHandler(t, cfg, [][]byte{bytes.Repeat([]byte{9}, 32)})
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	file, err := writer.CreateFormFile("video", "sample.mp4")
	if err != nil {
		t.Fatal(err)
	}
	file.Write([]byte("synthetic-video-bytes"))
	writer.WriteField("question", "What happens?")
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/v1/videos/understand?model=mage-vl-4b", bytes.NewReader(body.Bytes()))
	request.Header.Set("Content-Type", writer.FormDataContentType())
	request.Header.Set("Authorization", "Bearer "+apiKeyA)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("video status = %d: %s", response.Code, response.Body.String())
	}
	if !strings.HasPrefix(contentType, "multipart/form-data; boundary=") {
		t.Fatalf("content type lost: %q", contentType)
	}
	if !bytes.Equal(received, body.Bytes()) {
		t.Fatal("multipart body or boundary changed while proxying")
	}
	if !strings.Contains(response.Body.String(), `"model":"mage-vl-4b"`) {
		t.Fatalf("video response did not use the public alias: %s", response.Body.String())
	}
	if mageCalls != 1 || qwenCalls != 0 {
		t.Fatalf("video route calls: Mage=%d Qwen=%d, want Mage=1 Qwen=0", mageCalls, qwenCalls)
	}
}

func TestModelsWithoutVideoOrLiveCapabilityAreRejected(t *testing.T) {
	calls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()
	cfg := newConfig(upstream.URL)
	cfg.Models["qwen3-0.6b"] = gateway.ModelRoute{
		UpstreamModel: "qwen3-0.6b", BaseURL: upstream.URL,
		Capabilities: gateway.ModelCapabilities{TextChat: true, Streaming: true},
	}
	handler := newHandler(t, cfg, [][]byte{bytes.Repeat([]byte{0x14}, 32)})

	video := httptest.NewRequest(http.MethodPost, "/v1/videos/understand?model=qwen3-0.6b", strings.NewReader("video"))
	video.Header.Set("Content-Type", "multipart/form-data; boundary=test")
	video.Header.Set("Authorization", "Bearer "+apiKeyA)
	videoResponse := httptest.NewRecorder()
	handler.ServeHTTP(videoResponse, video)
	if videoResponse.Code != http.StatusBadRequest || !strings.Contains(videoResponse.Body.String(), `"code":"unsupported_model_capability"`) {
		t.Fatalf("Qwen video request should be rejected: %d %s", videoResponse.Code, videoResponse.Body.String())
	}

	live := postJSON(t, handler, "/v1/live/sessions", apiKeyA, `{"model":"qwen3-0.6b","question":"describe"}`)
	if live.Code != http.StatusBadRequest || !strings.Contains(live.Body.String(), `"code":"unsupported_model_capability"`) {
		t.Fatalf("Qwen live request should be rejected: %d %s", live.Code, live.Body.String())
	}
	if calls != 0 {
		t.Fatalf("unsupported requests reached upstream %d times", calls)
	}
}

func TestReadinessRequiresEveryConfiguredModelUpstream(t *testing.T) {
	healthy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"status":"ok"}`)
	}))
	defer healthy.Close()
	unhealthy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		io.WriteString(w, `{"status":"starting"}`)
	}))
	defer unhealthy.Close()
	cfg := newConfig(healthy.URL)
	qwenRoute := cfg.Models["qwen3-0.6b"]
	qwenRoute.BaseURL = unhealthy.URL
	cfg.Models["qwen3-0.6b"] = qwenRoute
	handler := newHandler(t, cfg, [][]byte{bytes.Repeat([]byte{0x15}, 32)})
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if response.Code != http.StatusServiceUnavailable || !strings.Contains(response.Body.String(), `"code":"upstream_not_ready"`) {
		t.Fatalf("readiness should fail when one model upstream is unavailable: %d %s", response.Code, response.Body.String())
	}
}

func TestLiveSessionHandleBindsTenantAndForwardsLastEventID(t *testing.T) {
	var eventsCalled int
	var modelLeak bool
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/live/sessions":
			var input map[string]any
			if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
				t.Error(err)
			}
			modelLeak = input["model"] != nil
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"id":"upstream-session-1","status":"running","model":"microsoft/Mage-VL","publish":{"rtsp":"rtsp://media:8554/upstream-session-1","rtmp":"rtmp://media:1935/upstream-session-1"},"events_url":"http://inference/v1/live/sessions/upstream-session-1/events","stop_url":"http://inference/v1/live/sessions/upstream-session-1"}`)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/live/sessions/upstream-session-1/events":
			eventsCalled++
			if r.Header.Get("Last-Event-ID") != "4" {
				t.Errorf("Last-Event-ID = %q", r.Header.Get("Last-Event-ID"))
			}
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, "id: 5\ndata: {\"status\":\"running\"}\n\n")
		case r.Method == http.MethodDelete && r.URL.Path == "/v1/live/sessions/upstream-session-1":
			io.WriteString(w, `{"id":"upstream-session-1","status":"stopping"}`)
		default:
			t.Errorf("unexpected upstream request: %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer upstream.Close()
	qwenCalls := 0
	qwen := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		qwenCalls++
		http.NotFound(w, r)
	}))
	defer qwen.Close()
	keys := [][]byte{bytes.Repeat([]byte{1}, 32)}
	cfg := newConfig(upstream.URL)
	qwenRoute := cfg.Models["qwen3-0.6b"]
	qwenRoute.BaseURL = qwen.URL
	cfg.Models["qwen3-0.6b"] = qwenRoute
	handler := newHandler(t, cfg, keys)
	created := postJSON(t, handler, "/v1/live/sessions", apiKeyA, `{"model":"mage-vl-4b","question":"Describe the scene"}`)
	if created.Code != http.StatusOK {
		t.Fatalf("create status = %d: %s", created.Code, created.Body.String())
	}
	var session map[string]any
	if err := json.Unmarshal(created.Body.Bytes(), &session); err != nil {
		t.Fatal(err)
	}
	handle, _ := session["id"].(string)
	if handle == "upstream-session-1" || !strings.Contains(session["events_url"].(string), "/v1/live/sessions/"+handle+"/events") {
		t.Fatalf("upstream ID or URL was exposed: %#v", session)
	}
	if session["model"] != "mage-vl-4b" {
		t.Fatalf("live session public model = %v", session["model"])
	}
	if modelLeak {
		t.Fatal("Gateway model alias was forwarded to the inference session schema")
	}
	if session["publish"].(map[string]any)["rtsp"] != "rtsp://media:8554/upstream-session-1" {
		t.Fatal("MediaMTX publish URL was changed")
	}

	crossTenant := httptest.NewRequest(http.MethodGet, "/v1/live/sessions/"+handle+"/events", nil)
	crossTenant.Header.Set("Authorization", "Bearer "+apiKeyB)
	crossTenantResponse := httptest.NewRecorder()
	handler.ServeHTTP(crossTenantResponse, crossTenant)
	if crossTenantResponse.Code != http.StatusNotFound {
		t.Fatalf("cross-tenant status = %d", crossTenantResponse.Code)
	}

	events := httptest.NewRequest(http.MethodGet, "/v1/live/sessions/"+handle+"/events", nil)
	events.Header.Set("Authorization", "Bearer "+apiKeyA)
	events.Header.Set("Last-Event-ID", "4")
	eventsResponse := httptest.NewRecorder()
	handler.ServeHTTP(eventsResponse, events)
	if eventsResponse.Code != http.StatusOK || !strings.Contains(eventsResponse.Body.String(), "id: 5") {
		t.Fatalf("event proxy failed: %d %s", eventsResponse.Code, eventsResponse.Body.String())
	}
	if eventsCalled != 1 {
		t.Fatalf("events upstream calls = %d, want 1", eventsCalled)
	}
	stop := httptest.NewRequest(http.MethodDelete, "/v1/live/sessions/"+handle, nil)
	stop.Header.Set("Authorization", "Bearer "+apiKeyA)
	stopResponse := httptest.NewRecorder()
	handler.ServeHTTP(stopResponse, stop)
	if stopResponse.Code != http.StatusOK || !strings.Contains(stopResponse.Body.String(), `"status":"stopping"`) || !strings.Contains(stopResponse.Body.String(), handle) {
		t.Fatalf("stop proxy failed: %d %s", stopResponse.Code, stopResponse.Body.String())
	}
	if qwenCalls != 0 {
		t.Fatalf("Mage live session was routed to Qwen upstream %d times", qwenCalls)
	}
}

func TestContentLengthLimitRateLimitAndReadiness(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			io.WriteString(w, `{"status":"ok"}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"object":"list","data":[]}`)
	}))
	defer upstream.Close()
	cfg := newConfig(upstream.URL)
	cfg.Limits.ChatRequestBytes = 10
	cfg.RateLimit.Burst = 1
	cfg.RateLimit.RequestsPerSecond = 0.01
	handler := newHandler(t, cfg, [][]byte{bytes.Repeat([]byte{2}, 32)})

	ready := httptest.NewRecorder()
	handler.ServeHTTP(ready, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if ready.Code != http.StatusOK {
		t.Fatalf("ready status = %d", ready.Code)
	}
	health := httptest.NewRecorder()
	handler.ServeHTTP(health, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if health.Code != http.StatusOK {
		t.Fatalf("health status = %d", health.Code)
	}

	large := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`))
	large.ContentLength = cfg.Limits.ChatRequestBytes + 1
	large.Header.Set("Authorization", "Bearer "+apiKeyA)
	largeResponse := httptest.NewRecorder()
	handler.ServeHTTP(largeResponse, large)
	if largeResponse.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("large body status = %d", largeResponse.Code)
	}

	first := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	first.Header.Set("Authorization", "Bearer "+apiKeyB)
	firstResponse := httptest.NewRecorder()
	handler.ServeHTTP(firstResponse, first)
	second := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	second.Header.Set("Authorization", "Bearer "+apiKeyB)
	secondResponse := httptest.NewRecorder()
	handler.ServeHTTP(secondResponse, second)
	if secondResponse.Code != http.StatusTooManyRequests || secondResponse.Header().Get("Retry-After") == "" {
		t.Fatalf("rate limit response = %d %s", secondResponse.Code, secondResponse.Body.String())
	}
}

func TestMissingContentLengthUses411(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { t.Error("unexpected upstream call") }))
	defer upstream.Close()
	handler := newHandler(t, newConfig(upstream.URL), [][]byte{bytes.Repeat([]byte{4}, 32)})
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("{}"))
	request.ContentLength = -1
	request.Header.Set("Authorization", "Bearer "+apiKeyA)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusLengthRequired {
		t.Fatalf("missing Content-Length status = %d", response.Code)
	}
}

func TestCheckedInGatewayYAMLLoads(t *testing.T) {
	path := filepath.Join("..", "..", "src", "gateway", "configs", "gateway.yml")
	cfg, err := gateway.LoadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Listener.Port != 8000 || cfg.Upstream.BaseURL != "http://127.0.0.1:8001" {
		t.Fatalf("unexpected gateway defaults: %#v", cfg)
	}
	if cfg.Models["mage-vl-4b"].UpstreamModel != "microsoft/Mage-VL" || cfg.Models["qwen3-0.6b"].UpstreamModel != "qwen3-0.6b" {
		t.Fatal("public alias is not mapped to the upstream model ID")
	}
	if !cfg.Models["mage-vl-4b"].Capabilities.ImageChat || !cfg.Models["mage-vl-4b"].Capabilities.VideoUpload || !cfg.Models["mage-vl-4b"].Capabilities.LiveVideo {
		t.Fatal("Mage-VL Gateway capabilities are incomplete")
	}
	if !cfg.Models["qwen3-0.6b"].Capabilities.TextChat || cfg.Models["qwen3-0.6b"].Capabilities.ImageChat || cfg.Models["qwen3-0.6b"].Capabilities.VideoUpload || cfg.Models["qwen3-0.6b"].Capabilities.LiveVideo {
		t.Fatal("Qwen3-0.6B Gateway capabilities do not match its text-only API")
	}
}

func TestModelCapabilitiesMustBeExplicitInGatewayYAML(t *testing.T) {
	path := filepath.Join("..", "..", "src", "gateway", "configs", "gateway.yml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	modified := strings.Replace(string(data), "      streaming: true\n", "", 1)
	if modified == string(data) {
		t.Fatal("test fixture did not find a streaming capability")
	}
	configPath := filepath.Join(t.TempDir(), "gateway.yml")
	if err := os.WriteFile(configPath, []byte(modified), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := gateway.LoadConfig(configPath); err == nil || !strings.Contains(err.Error(), `model capability "streaming" must be explicitly set`) {
		t.Fatalf("missing model capability should be rejected, got %v", err)
	}
}

func TestUpstreamUnavailableReturns502WithoutInternalDetails(t *testing.T) {
	cfg := newConfig("http://127.0.0.1:1")
	handler := newHandler(t, cfg, [][]byte{bytes.Repeat([]byte{8}, 32)})
	response := postJSON(t, handler, "/v1/chat/completions", apiKeyA, `{"model":"mage-vl-4b","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.test/image.png"}}]}]}`)
	if response.Code != http.StatusBadGateway {
		t.Fatalf("unavailable upstream status = %d", response.Code)
	}
	if strings.Contains(response.Body.String(), "127.0.0.1:1") {
		t.Fatal("internal upstream address leaked in error response")
	}
}

func TestLogsDoNotContainAPIKeyOrRequestBody(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"model":"microsoft/Mage-VL","choices":[]}`)
	}))
	defer upstream.Close()
	var logs bytes.Buffer
	cfg := newConfig(upstream.URL)
	keys := []gateway.APIKey{{ID: "key-a", TenantID: "tenant-a", Secret: apiKeyA}}
	handler, err := gateway.NewHandler(cfg, gateway.Options{APIKeys: keys, HMACKeys: [][]byte{bytes.Repeat([]byte{6}, 32)}, Logger: slog.New(slog.NewTextHandler(&logs, nil))})
	if err != nil {
		t.Fatal(err)
	}
	const prompt = "private prompt text that must not be logged"
	response := postJSON(t, handler, "/v1/chat/completions", apiKeyA, `{"model":"mage-vl-4b","messages":[{"role":"user","content":[{"type":"text","text":"`+prompt+`"},{"type":"image_url","image_url":{"url":"https://example.test/image.png"}}]}]}`)
	if response.Code != http.StatusOK {
		t.Fatalf("chat status = %d: %s", response.Code, response.Body.String())
	}
	if strings.Contains(logs.String(), apiKeyA) || strings.Contains(logs.String(), prompt) {
		t.Fatal("sensitive request data was written to logs")
	}
	if !strings.Contains(logs.String(), "request_id") || !strings.Contains(logs.String(), "upstream_ms") {
		t.Fatal("request log is missing request_id")
	}
}

func TestMetricsExposeSafeLatencyHistograms(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/chat/completions":
			if r.Header.Get("Content-Type") == "application/json" {
				var request map[string]any
				_ = json.NewDecoder(r.Body).Decode(&request)
				if request["stream"] == true {
					w.Header().Set("Content-Type", "text/event-stream")
					io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\ndata: [DONE]\n\n")
					return
				}
			}
			io.WriteString(w, `{"model":"microsoft/Mage-VL","choices":[]}`)
		case "/v1/videos/understand":
			_, _ = io.Copy(io.Discard, r.Body)
			io.WriteString(w, `{"model":"microsoft/Mage-VL","text":"ok"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer upstream.Close()
	handler := newHandler(t, newConfig(upstream.URL), [][]byte{bytes.Repeat([]byte{0x12}, 32)})
	chat := postJSON(t, handler, "/v1/chat/completions", apiKeyA, `{"model":"mage-vl-4b","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.test/image.png"}}]}],"stream":true}`)
	if chat.Code != http.StatusOK {
		t.Fatalf("chat status = %d", chat.Code)
	}
	videoRequest := httptest.NewRequest(http.MethodPost, "/v1/videos/understand", strings.NewReader("video-body"))
	videoRequest.Header.Set("Content-Type", "multipart/form-data; boundary=test-boundary")
	videoRequest.Header.Set("Authorization", "Bearer "+apiKeyA)
	videoResponse := httptest.NewRecorder()
	handler.ServeHTTP(videoResponse, videoRequest)
	if videoResponse.Code != http.StatusOK {
		t.Fatalf("video status = %d: %s", videoResponse.Code, videoResponse.Body.String())
	}
	metricsRequest := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	metricsResponse := httptest.NewRecorder()
	handler.ServeHTTP(metricsResponse, metricsRequest)
	metrics := metricsResponse.Body.String()
	for _, name := range []string{"gateway_http_requests_total", "gateway_sse_time_to_first_token_seconds", "gateway_video_upload_duration_seconds"} {
		if !strings.Contains(metrics, name) {
			t.Errorf("metrics response is missing %s", name)
		}
	}
	if strings.Contains(metrics, apiKeyA) || strings.Contains(metrics, "tenant-a") {
		t.Fatal("metrics expose an API key or tenant identifier")
	}
}

func TestRotatedHMACKeyCanVerifyExistingHandle(t *testing.T) {
	firstKey := bytes.Repeat([]byte{1}, 32)
	secondKey := bytes.Repeat([]byte{2}, 32)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/live/sessions" {
			io.WriteString(w, `{"id":"session-before-rotation","status":"running","publish":{"rtsp":"rtsp://media/live","rtmp":"rtmp://media/live"}}`)
			return
		}
		if r.URL.Path == "/v1/live/sessions/session-before-rotation/events" {
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, "data: ok\n\n")
			return
		}
		http.NotFound(w, r)
	}))
	defer upstream.Close()
	createdHandler := newHandler(t, newConfig(upstream.URL), [][]byte{firstKey})
	created := postJSON(t, createdHandler, "/v1/live/sessions", apiKeyA, `{"question":"test"}`)
	var payload map[string]any
	if err := json.Unmarshal(created.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	handle := payload["id"].(string)
	rotatedHandler := newHandler(t, newConfig(upstream.URL), [][]byte{secondKey, firstKey})
	request := httptest.NewRequest(http.MethodGet, "/v1/live/sessions/"+handle+"/events", nil)
	request.Header.Set("Authorization", "Bearer "+apiKeyA)
	response := httptest.NewRecorder()
	rotatedHandler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("rotated key could not validate existing handle: %d %s", response.Code, response.Body.String())
	}
	separator := strings.LastIndex(handle, ".")
	signature := handle[separator+1:]
	replacement := "a"
	if signature[0] == 'a' {
		replacement = "b"
	}
	tampered := handle[:separator+1] + replacement + signature[1:]
	tamperRequest := httptest.NewRequest(http.MethodDelete, "/v1/live/sessions/"+tampered, nil)
	tamperRequest.Header.Set("Authorization", "Bearer "+apiKeyA)
	tamperResponse := httptest.NewRecorder()
	rotatedHandler.ServeHTTP(tamperResponse, tamperRequest)
	if tamperResponse.Code != http.StatusNotFound {
		t.Fatalf("tampered handle status = %d", tamperResponse.Code)
	}
}
