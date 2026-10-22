package gateway

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"gitee.com/milo-ally-intern/LYGCloudModelAPI/src/gateway/internal/auth"
	"gitee.com/milo-ally-intern/LYGCloudModelAPI/src/gateway/internal/config"
	"gitee.com/milo-ally-intern/LYGCloudModelAPI/src/gateway/internal/middleware"
	"gitee.com/milo-ally-intern/LYGCloudModelAPI/src/gateway/internal/observability"
	"gitee.com/milo-ally-intern/LYGCloudModelAPI/src/gateway/internal/route"
	streaming "gitee.com/milo-ally-intern/LYGCloudModelAPI/src/gateway/internal/stream"
	"gitee.com/milo-ally-intern/LYGCloudModelAPI/src/gateway/internal/upstream"
	"gitee.com/milo-ally-intern/LYGCloudModelAPI/src/gateway/internal/videoproxy"
)

type Config = config.Config
type ModelRoute = config.ModelRoute
type ModelCapabilities = config.ModelCapabilities
type APIKey = auth.APIKey

type Options struct {
	APIKeys  []APIKey
	HMACKeys [][]byte
	Logger   *slog.Logger
}

func LoadConfig(path string) (Config, error) { return config.Load(path) }

func LoadAPIKeys(path string) ([]APIKey, error) {
	return auth.ReadAPIKeys(path)
}

func ParseHMACKeys(value string) ([][]byte, error) { return auth.ParseHMACKeys(value) }

type Application struct {
	cfg            Config
	keys           *auth.KeySet
	hmac           [][]byte
	upstreamClient *upstream.Client
	limiter        *middleware.Limiter
	metrics        *observability.Metrics
	logger         *slog.Logger
	router         *route.Router
	streamer       *streaming.Forwarder
	videoProxy     *videoproxy.Proxy
}

func NewHandler(cfg Config, options Options) (http.Handler, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if len(options.HMACKeys) == 0 {
		return nil, fmt.Errorf("at least one session HMAC key is required")
	}
	for _, key := range options.HMACKeys {
		if len(key) < 32 {
			return nil, fmt.Errorf("session HMAC keys must contain at least 32 bytes")
		}
	}
	keys, err := auth.NewKeySet(options.APIKeys)
	for index := range options.APIKeys {
		options.APIKeys[index].Secret = ""
	}
	if err != nil {
		return nil, err
	}
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}
	metrics := observability.NewMetrics()
	upstreamClient := upstream.NewClient(cfg)
	app := &Application{
		cfg: cfg, keys: keys, hmac: options.HMACKeys,
		upstreamClient: upstreamClient,
		limiter:        middleware.NewLimiter(cfg.RateLimit.RequestsPerSecond, cfg.RateLimit.Burst),
		metrics:        metrics,
		logger:         logger,
	}
	app.streamer = streaming.New(streaming.Config{
		EventBytes:  cfg.Limits.SSEEventBytes,
		BufferBytes: cfg.Limits.StreamBufferBytes,
		IdleTimeout: time.Duration(cfg.Timeouts.StreamIdleSeconds) * time.Second,
	}, metrics, logger, writeError)
	app.videoProxy = videoproxy.New(upstreamClient, cfg.Limits.VideoRequestBytes, time.Duration(cfg.Timeouts.VideoRequestSeconds)*time.Second, metrics)
	app.router = route.New(cfg.Models, route.Handlers{
		Models:     app.models,
		Chat:       app.chat,
		Video:      app.video,
		CreateLive: app.createLiveSessionRoute,
		LiveEvents: app.liveEventsRoute,
		StopLive:   app.stopLiveSessionRoute,
	}, writeError)
	return app, nil
}

type requestInfo struct {
	id       string
	tenant   string
	started  time.Time
	upstream time.Duration
}

type infoKey struct{}

type statusWriter struct {
	http.ResponseWriter
	status int
	bytes  int64
}

func (writer *statusWriter) WriteHeader(status int) {
	if writer.status != 0 {
		return
	}
	writer.status = status
	writer.ResponseWriter.WriteHeader(status)
}

func (writer *statusWriter) Write(data []byte) (int, error) {
	if writer.status == 0 {
		writer.WriteHeader(http.StatusOK)
	}
	count, err := writer.ResponseWriter.Write(data)
	writer.bytes += int64(count)
	return count, err
}

func (writer *statusWriter) Flush() {
	if writer.status == 0 {
		writer.WriteHeader(http.StatusOK)
	}
	if flusher, ok := writer.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (writer *statusWriter) Unwrap() http.ResponseWriter { return writer.ResponseWriter }

func (app *Application) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	info := &requestInfo{id: requestID(), started: time.Now()}
	r = r.WithContext(context.WithValue(r.Context(), infoKey{}, info))
	w.Header().Set("X-Request-ID", info.id)
	recorder := &statusWriter{ResponseWriter: w}
	started := time.Now()
	app.dispatch(recorder, r, info)
	if recorder.status == 0 {
		recorder.status = http.StatusOK
	}
	elapsed := time.Since(started)
	routeName := route.RouteName(r)
	app.metrics.ObserveRequest(routeName, recorder.status, elapsed.Seconds())
	app.logger.Info("gateway request completed",
		"request_id", info.id, "route", routeName, "method", r.Method,
		"tenant_hash", safeTenantHash(info.tenant), "status", recorder.status,
		"upstream_ms", info.upstream.Milliseconds(),
		"response_bytes", recorder.bytes, "duration_ms", elapsed.Milliseconds())
}

func (app *Application) dispatch(w http.ResponseWriter, r *http.Request, info *requestInfo) {
	if r.URL.Path == "/healthz" {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "invalid_request_error", "method_not_allowed", "Method not allowed.")
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
		return
	}
	if r.URL.Path == "/readyz" {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "invalid_request_error", "method_not_allowed", "Method not allowed.")
			return
		}
		app.readiness(w, r)
		return
	}
	if app.cfg.Observability.MetricsEnabled && r.URL.Path == app.cfg.Observability.MetricsPath {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "invalid_request_error", "method_not_allowed", "Method not allowed.")
			return
		}
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		app.metrics.WritePrometheus(w)
		return
	}
	if !strings.HasPrefix(r.URL.Path, "/v1/") {
		writeError(w, http.StatusNotFound, "invalid_request_error", "not_found", "The requested endpoint was not found.")
		return
	}
	tenant, _, ok := app.keys.Authenticate(r.Header.Get("Authorization"))
	if !ok {
		writeError(w, http.StatusUnauthorized, "authentication_error", "invalid_api_key", "A valid Bearer API key is required.")
		return
	}
	info.tenant = tenant
	allowed, retryAfter := app.limiter.Allow(tenant, time.Now())
	if !allowed {
		w.Header().Set("Retry-After", strconv.Itoa(retryAfter))
		writeError(w, http.StatusTooManyRequests, "rate_limit_error", "rate_limit_exceeded", "The tenant request rate limit has been exceeded.")
		return
	}
	app.router.ServeHTTP(w, r)
}

func (app *Application) models(w http.ResponseWriter, _ *http.Request) {
	models := make([]map[string]any, 0, len(app.cfg.Models))
	for _, alias := range app.router.Models() {
		models = append(models, map[string]any{"id": alias, "object": "model", "created": 0, "owned_by": "local"})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": models})
}

func (app *Application) createLiveSessionRoute(w http.ResponseWriter, r *http.Request) {
	app.createLiveSession(w, r, requestTenant(r))
}

func (app *Application) liveEventsRoute(w http.ResponseWriter, r *http.Request) {
	handle := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/v1/live/sessions/"), "/events")
	app.liveEvents(w, r, requestTenant(r), handle)
}

func (app *Application) stopLiveSessionRoute(w http.ResponseWriter, r *http.Request) {
	handle := strings.TrimPrefix(r.URL.Path, "/v1/live/sessions/")
	app.stopLiveSession(w, r, requestTenant(r), handle)
}

func (app *Application) readiness(w http.ResponseWriter, r *http.Request) {
	checked := make(map[string]struct{})
	for _, modelRoute := range app.cfg.Models {
		baseURL := routeBaseURL(app.cfg, modelRoute)
		if _, exists := checked[baseURL]; exists {
			continue
		}
		checked[baseURL] = struct{}{}
		response, _, cancel, err := app.forward(r, http.MethodGet, "/health", baseURL, nil, 0, "", app.cfg.Timeouts.ReadinessSeconds)
		if err != nil {
			app.metrics.AddUpstreamError("readiness")
			writeError(w, http.StatusServiceUnavailable, "service_unavailable_error", "upstream_not_ready", "A configured inference service is not ready.")
			return
		}
		var payload struct {
			Status string `json:"status"`
		}
		ready := response.StatusCode == http.StatusOK && json.NewDecoder(io.LimitReader(response.Body, 64*1024)).Decode(&payload) == nil && payload.Status == "ok"
		_ = response.Body.Close()
		cancel()
		if !ready {
			writeError(w, http.StatusServiceUnavailable, "service_unavailable_error", "upstream_not_ready", "A configured inference service is not ready.")
			return
		}
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

func (app *Application) chat(w http.ResponseWriter, r *http.Request) {
	if !checkContentLength(w, r, app.cfg.Limits.ChatRequestBytes) {
		return
	}
	body, ok := readRequestBody(w, r, app.cfg.Limits.ChatRequestBytes)
	if !ok {
		return
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil || payload == nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_json", "The request body must be a JSON object.")
		return
	}
	modelName, _ := payload["model"].(string)
	modelRoute, exists := app.cfg.Models[modelName]
	if !exists {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "model_not_found", "The model field must name a configured public model.")
		return
	}
	imageInput := containsImageInput(payload["messages"])
	if imageInput && !modelRoute.Capabilities.ImageChat {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "unsupported_model_capability", "The selected model does not accept image input.")
		return
	}
	if !imageInput && !modelRoute.Capabilities.TextChat {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "unsupported_model_capability", "The selected model does not accept text-only chat input.")
		return
	}
	streamingResponse, _ := payload["stream"].(bool)
	if streamingResponse && !modelRoute.Capabilities.Streaming {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "unsupported_model_capability", "The selected model does not support streaming output.")
		return
	}
	payload["model"] = modelRoute.UpstreamModel
	upstreamBody, err := json.Marshal(payload)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_json", "The request body could not be processed.")
		return
	}
	baseURL := routeBaseURL(app.cfg, modelRoute)
	timeout := app.cfg.Timeouts.RequestSeconds
	if streamingResponse {
		timeout = app.cfg.Timeouts.StreamMaxSeconds
	}
	response, ctx, cancel, err := app.forward(r, http.MethodPost, "/v1/chat/completions", baseURL, bytes.NewReader(upstreamBody), int64(len(upstreamBody)), "application/json", timeout)
	if err != nil {
		app.writeUpstreamFailure(w, r, err)
		return
	}
	defer cancel()
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		app.writeUpstreamError(w, response)
		return
	}
	if streamingResponse {
		app.streamer.Chat(w, r, ctx, response, modelName, requestIDFrom(r), requestStarted(r))
		return
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, app.cfg.Limits.JSONResponseBytes+1))
	if err != nil || int64(len(data)) > app.cfg.Limits.JSONResponseBytes {
		writeError(w, http.StatusBadGateway, "upstream_error", "invalid_upstream_response", "The inference service returned an invalid response.")
		return
	}
	var result map[string]any
	if json.Unmarshal(data, &result) != nil || result == nil {
		writeError(w, http.StatusBadGateway, "upstream_error", "invalid_upstream_response", "The inference service returned an invalid response.")
		return
	}
	result["model"] = modelName
	copyResponseHeaders(w, response.Header, false)
	writeJSON(w, response.StatusCode, result)
}

func (app *Application) video(w http.ResponseWriter, r *http.Request) {
	modelName := r.URL.Query().Get("model")
	if modelName == "" {
		modelName = app.cfg.Defaults.VideoModel
	}
	modelRoute, exists := app.cfg.Models[modelName]
	if !exists {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "model_not_found", "The model query parameter must name a configured public model.")
		return
	}
	if !modelRoute.Capabilities.VideoUpload {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "unsupported_model_capability", "The selected model does not support video uploads.")
		return
	}
	response, _, cancel, err := app.videoProxy.Forward(r, requestIDFrom(r), routeBaseURL(app.cfg, modelRoute))
	if err != nil {
		var requestError *videoproxy.RequestError
		if errors.As(err, &requestError) {
			writeError(w, requestError.Status, "invalid_request_error", requestError.Code, requestError.Message)
			return
		}
		app.writeUpstreamFailure(w, r, err)
		return
	}
	defer cancel()
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		app.writeUpstreamError(w, response)
		return
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, app.cfg.Limits.JSONResponseBytes+1))
	if err != nil || int64(len(data)) > app.cfg.Limits.JSONResponseBytes {
		writeError(w, http.StatusBadGateway, "upstream_error", "invalid_upstream_response", "The inference service returned an invalid response.")
		return
	}
	var result map[string]any
	if json.Unmarshal(data, &result) != nil || result == nil {
		writeError(w, http.StatusBadGateway, "upstream_error", "invalid_upstream_response", "The inference service returned an invalid response.")
		return
	}
	if _, ok := result["model"]; ok {
		result["model"] = modelName
	}
	copyResponseHeaders(w, response.Header, false)
	writeJSON(w, response.StatusCode, result)
}

func (app *Application) createLiveSession(w http.ResponseWriter, r *http.Request, tenant string) {
	if !checkContentLength(w, r, app.cfg.Limits.LiveRequestBytes) {
		return
	}
	body, ok := readRequestBody(w, r, app.cfg.Limits.LiveRequestBytes)
	if !ok {
		return
	}
	var input map[string]any
	if json.Unmarshal(body, &input) != nil || input == nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_json", "The request body must be a JSON object.")
		return
	}
	modelName := app.cfg.Defaults.LiveModel
	if requested, provided := input["model"]; provided {
		var valid bool
		modelName, valid = requested.(string)
		if !valid || strings.TrimSpace(modelName) == "" {
			writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_model", "The model field must be a configured public model alias.")
			return
		}
		delete(input, "model")
		body, _ = json.Marshal(input)
	}
	modelRoute, exists := app.cfg.Models[modelName]
	if !exists {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "model_not_found", "The model field must name a configured public model.")
		return
	}
	if !modelRoute.Capabilities.LiveVideo {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "unsupported_model_capability", "The selected model does not support live video sessions.")
		return
	}
	response, _, cancel, err := app.forward(r, http.MethodPost, "/v1/live/sessions", routeBaseURL(app.cfg, modelRoute), bytes.NewReader(body), int64(len(body)), "application/json", app.cfg.Timeouts.RequestSeconds)
	if err != nil {
		app.writeUpstreamFailure(w, r, err)
		return
	}
	defer cancel()
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		app.writeUpstreamError(w, response)
		return
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, app.cfg.Limits.JSONResponseBytes+1))
	if err != nil || int64(len(data)) > app.cfg.Limits.JSONResponseBytes {
		writeError(w, http.StatusBadGateway, "upstream_error", "invalid_upstream_response", "The inference service returned an invalid response.")
		return
	}
	var result map[string]any
	if json.Unmarshal(data, &result) != nil || result == nil {
		writeError(w, http.StatusBadGateway, "upstream_error", "invalid_upstream_response", "The inference service returned an invalid response.")
		return
	}
	upstreamID, _ := result["id"].(string)
	if upstreamID == "" {
		writeError(w, http.StatusBadGateway, "upstream_error", "invalid_upstream_response", "The inference service returned an invalid session response.")
		return
	}
	expires := time.Now().Add(time.Duration(app.cfg.Session.HandleTTLSeconds) * time.Second)
	handle, err := auth.SignModelSession(tenant, modelName, upstreamID, expires, app.hmac)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", "session_handle_failed", "The live session could not be secured.")
		return
	}
	publicURL := strings.TrimRight(app.cfg.PublicBaseURL, "/")
	result["id"] = handle
	result["model"] = modelName
	result["events_url"] = publicURL + "/v1/live/sessions/" + handle + "/events"
	result["stop_url"] = publicURL + "/v1/live/sessions/" + handle
	copyResponseHeaders(w, response.Header, false)
	writeJSON(w, response.StatusCode, result)
}

func (app *Application) liveEvents(w http.ResponseWriter, r *http.Request, tenant, handle string) {
	session, valid := auth.VerifyModelSession(handle, tenant, app.hmac, time.Now())
	if !valid {
		writeError(w, http.StatusNotFound, "invalid_request_error", "session_not_found", "The live session was not found or has expired.")
		return
	}
	modelRoute, exists := app.cfg.Models[session.ModelAlias]
	if session.ModelAlias == "" {
		modelRoute, exists = app.cfg.Models[app.cfg.Defaults.LiveModel]
	}
	if !exists || !modelRoute.Capabilities.LiveVideo {
		writeError(w, http.StatusNotFound, "invalid_request_error", "session_not_found", "The live session was not found or has expired.")
		return
	}
	path := "/v1/live/sessions/" + url.PathEscape(session.UpstreamID) + "/events"
	response, ctx, cancel, err := app.forward(r, http.MethodGet, path, routeBaseURL(app.cfg, modelRoute), nil, 0, "", app.cfg.Timeouts.StreamMaxSeconds)
	if err != nil {
		app.writeUpstreamFailure(w, r, err)
		return
	}
	defer cancel()
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		app.writeUpstreamError(w, response)
		return
	}
	copyResponseHeaders(w, response.Header, true)
	if response.Header.Get("Content-Type") == "" {
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	}
	w.WriteHeader(response.StatusCode)
	app.streamer.Copy(w, r, ctx, response.Body, "/v1/live/sessions/{session}/events", requestIDFrom(r))
}

func (app *Application) stopLiveSession(w http.ResponseWriter, r *http.Request, tenant, handle string) {
	session, valid := auth.VerifyModelSession(handle, tenant, app.hmac, time.Now())
	if !valid {
		writeError(w, http.StatusNotFound, "invalid_request_error", "session_not_found", "The live session was not found or has expired.")
		return
	}
	modelRoute, exists := app.cfg.Models[session.ModelAlias]
	if session.ModelAlias == "" {
		modelRoute, exists = app.cfg.Models[app.cfg.Defaults.LiveModel]
	}
	if !exists || !modelRoute.Capabilities.LiveVideo {
		writeError(w, http.StatusNotFound, "invalid_request_error", "session_not_found", "The live session was not found or has expired.")
		return
	}
	path := "/v1/live/sessions/" + url.PathEscape(session.UpstreamID)
	response, _, cancel, err := app.forward(r, http.MethodDelete, path, routeBaseURL(app.cfg, modelRoute), nil, 0, "", app.cfg.Timeouts.RequestSeconds)
	if err != nil {
		app.writeUpstreamFailure(w, r, err)
		return
	}
	defer cancel()
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		app.writeUpstreamError(w, response)
		return
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, app.cfg.Limits.JSONResponseBytes+1))
	if err != nil || int64(len(data)) > app.cfg.Limits.JSONResponseBytes {
		writeError(w, http.StatusBadGateway, "upstream_error", "invalid_upstream_response", "The inference service returned an invalid response.")
		return
	}
	var result map[string]any
	if json.Unmarshal(data, &result) != nil || result == nil {
		result = map[string]any{"status": "stopping"}
	}
	result["id"] = handle
	copyResponseHeaders(w, response.Header, false)
	writeJSON(w, response.StatusCode, result)
}

func (app *Application) forward(r *http.Request, method, path, baseURL string, body io.Reader, contentLength int64, contentType string, timeoutSeconds int) (*http.Response, context.Context, context.CancelFunc, error) {
	response, ctx, cancel, elapsed, err := app.upstreamClient.Forward(r, requestIDFrom(r), method, path, baseURL, body, contentLength, contentType, time.Duration(timeoutSeconds)*time.Second)
	if info, ok := r.Context().Value(infoKey{}).(*requestInfo); ok {
		info.upstream = elapsed
	}
	if err != nil {
		return nil, nil, nil, err
	}
	return response, ctx, cancel, nil
}

func (app *Application) writeUpstreamFailure(w http.ResponseWriter, r *http.Request, err error) {
	if r.Context().Err() != nil {
		return
	}
	if isTimeout(err) {
		app.metrics.AddUpstreamError("timeout")
		writeError(w, http.StatusGatewayTimeout, "upstream_error", "upstream_timeout", "The inference service request timed out.")
		return
	}
	app.metrics.AddUpstreamError("unavailable")
	writeError(w, http.StatusBadGateway, "upstream_error", "upstream_unavailable", "The inference service is unavailable.")
}

func (app *Application) writeUpstreamError(w http.ResponseWriter, response *http.Response) {
	data, _ := io.ReadAll(io.LimitReader(response.Body, app.cfg.Limits.JSONResponseBytes))
	message := "The inference service rejected the request."
	if response.StatusCode >= 500 {
		message = "The inference service failed to process the request."
	} else {
		var payload map[string]any
		if json.Unmarshal(data, &payload) == nil {
			if detail, ok := payload["detail"].(string); ok && detail != "" {
				message = detail
			}
			if nested, ok := payload["error"].(map[string]any); ok {
				if value, ok := nested["message"].(string); ok && value != "" {
					message = value
				}
			}
		}
	}
	if response.StatusCode < 400 || response.StatusCode > 599 {
		response.StatusCode = http.StatusBadGateway
	}
	writeError(w, response.StatusCode, "upstream_error", "upstream_rejected_request", message)
}

func routeBaseURL(cfg Config, route config.ModelRoute) string {
	return upstream.BaseURL(cfg, route)
}

func checkContentLength(w http.ResponseWriter, r *http.Request, limit int64) bool {
	if r.ContentLength < -1 {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_content_length", "Content-Length must be a non-negative integer.")
		return false
	}
	if r.ContentLength == -1 {
		writeError(w, http.StatusLengthRequired, "invalid_request_error", "content_length_required", "Content-Length is required for this endpoint.")
		return false
	}
	if r.ContentLength > limit {
		writeError(w, http.StatusRequestEntityTooLarge, "invalid_request_error", "request_too_large", "The request body exceeds the configured size limit.")
		return false
	}
	return true
}

func readRequestBody(w http.ResponseWriter, r *http.Request, limit int64) ([]byte, bool) {
	data, err := io.ReadAll(io.LimitReader(r.Body, limit+1))
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "request_read_failed", "The request body could not be read.")
		return nil, false
	}
	if int64(len(data)) > limit {
		writeError(w, http.StatusRequestEntityTooLarge, "invalid_request_error", "request_too_large", "The request body exceeds the configured size limit.")
		return nil, false
	}
	return data, true
}

type apiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
	Param   any    `json:"param"`
}

func writeError(w http.ResponseWriter, status int, kind, code, message string) {
	writeJSON(w, status, map[string]any{"error": apiError{Message: message, Type: kind, Code: code, Param: nil}})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func copyResponseHeaders(w http.ResponseWriter, source http.Header, streaming bool) {
	copySelectedHeaders(w, source, "Content-Type", "Cache-Control", "Retry-After", "X-Accel-Buffering")
	if streaming {
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
	}
}

func copySelectedHeaders(w http.ResponseWriter, source http.Header, names ...string) {
	for _, name := range names {
		if value := source.Values(name); len(value) > 0 {
			w.Header()[http.CanonicalHeaderKey(name)] = append([]string(nil), value...)
		}
	}
}

func isTimeout(err error) bool {
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var networkError net.Error
	return errors.As(err, &networkError) && networkError.Timeout()
}

func containsImageInput(value any) bool {
	switch item := value.(type) {
	case map[string]any:
		if kind, _ := item["type"].(string); kind == "image_url" {
			return true
		}
		if _, exists := item["image_url"]; exists {
			return true
		}
		for _, child := range item {
			if containsImageInput(child) {
				return true
			}
		}
	case []any:
		for _, child := range item {
			if containsImageInput(child) {
				return true
			}
		}
	}
	return false
}

func requestID() string {
	var value [16]byte
	if _, err := rand.Read(value[:]); err != nil {
		return strconv.FormatInt(time.Now().UnixNano(), 16)
	}
	return hex.EncodeToString(value[:])
}

func requestIDFrom(r *http.Request) string {
	if info, ok := r.Context().Value(infoKey{}).(*requestInfo); ok {
		return info.id
	}
	return ""
}

func requestTenant(r *http.Request) string {
	if info, ok := r.Context().Value(infoKey{}).(*requestInfo); ok {
		return info.tenant
	}
	return ""
}

func requestStarted(r *http.Request) time.Time {
	if info, ok := r.Context().Value(infoKey{}).(*requestInfo); ok && !info.started.IsZero() {
		return info.started
	}
	return time.Now()
}

func safeTenantHash(tenant string) string {
	if tenant == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(tenant))
	return hex.EncodeToString(sum[:6])
}
