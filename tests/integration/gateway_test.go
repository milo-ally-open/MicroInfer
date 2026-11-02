package integration_test

import (
	"bytes"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"runtime"
	"testing"

	gatewayapp "example.com/videostreamapi/gateway"
)

func gatewayConfigPath(t *testing.T) string {
	t.Helper()
	_, sourceFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller() could not locate this test file")
	}
	return filepath.Join(filepath.Dir(sourceFile), "..", "..", "src", "gateway", "configs", "gateway.yml")
}

func TestPublicV1RoutesRequireBearerAPIKey(t *testing.T) {
	cfg, err := gatewayapp.LoadConfig(gatewayConfigPath(t))
	if err != nil {
		t.Fatal(err)
	}
	handler, err := gatewayapp.NewHandler(cfg, gatewayapp.Options{
		APIKeys:  []gatewayapp.APIKey{{ID: "client-a", TenantID: "tenant-a", Secret: "0123456789abcdef0123456789abcdef"}},
		HMACKeys: [][]byte{bytes.Repeat([]byte{1}, 32)},
		Logger:   slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatal(err)
	}

	routes := []struct {
		method string
		path   string
	}{
		{method: http.MethodGet, path: "/v1/models"},
		{method: http.MethodPost, path: "/v1/chat/completions"},
		{method: http.MethodPost, path: "/v1/videos/understand"},
		{method: http.MethodPost, path: "/v1/live/sessions"},
		{method: http.MethodGet, path: "/v1/live/sessions/fake-handle/events"},
		{method: http.MethodDelete, path: "/v1/live/sessions/fake-handle"},
		{method: http.MethodGet, path: "/v1/unregistered"},
	}
	for _, route := range routes {
		for _, authorization := range []string{"", "Bearer invalid-key"} {
			request := httptest.NewRequest(route.method, route.path, nil)
			if authorization != "" {
				request.Header.Set("Authorization", authorization)
			}
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != http.StatusUnauthorized {
				t.Errorf("%s %s with authorization %q returned %d, want %d", route.method, route.path, authorization, response.Code, http.StatusUnauthorized)
			}
		}
	}
}

func TestHealthEndpointRemainsUnauthenticated(t *testing.T) {
	cfg, err := gatewayapp.LoadConfig(gatewayConfigPath(t))
	if err != nil {
		t.Fatal(err)
	}
	handler, err := gatewayapp.NewHandler(cfg, gatewayapp.Options{
		APIKeys:  []gatewayapp.APIKey{{ID: "client-a", TenantID: "tenant-a", Secret: "0123456789abcdef0123456789abcdef"}},
		HMACKeys: [][]byte{bytes.Repeat([]byte{1}, 32)},
		Logger:   slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if response.Code != http.StatusOK {
		t.Fatalf("GET /healthz returned %d, want %d", response.Code, http.StatusOK)
	}
}
