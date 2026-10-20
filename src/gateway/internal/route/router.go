package route

import (
	"net/http"
	"sort"
	"strings"

	"gitee.com/milo-ally-intern/LYGCloudModelAPI/src/gateway/internal/config"
)

type ErrorWriter func(http.ResponseWriter, int, string, string, string)

type Handlers struct {
	Models     http.HandlerFunc
	Chat       http.HandlerFunc
	Video      http.HandlerFunc
	CreateLive http.HandlerFunc
	LiveEvents http.HandlerFunc
	StopLive   http.HandlerFunc
}

type Router struct {
	models     []string
	handlers   Handlers
	writeError ErrorWriter
}

func New(models map[string]config.ModelRoute, handlers Handlers, writeError ErrorWriter) *Router {
	aliases := make([]string, 0, len(models))
	for alias := range models {
		aliases = append(aliases, alias)
	}
	sort.Strings(aliases)
	return &Router{models: aliases, handlers: handlers, writeError: writeError}
}

func (router *Router) Models() []string {
	return append([]string(nil), router.models...)
}

func (router *Router) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/v1/models":
		router.dispatchMethod(w, r, http.MethodGet, router.handlers.Models)
	case "/v1/chat/completions":
		router.dispatchMethod(w, r, http.MethodPost, router.handlers.Chat)
	case "/v1/videos/understand":
		router.dispatchMethod(w, r, http.MethodPost, router.handlers.Video)
	case "/v1/live/sessions":
		router.dispatchMethod(w, r, http.MethodPost, router.handlers.CreateLive)
	default:
		const prefix = "/v1/live/sessions/"
		if strings.HasPrefix(r.URL.Path, prefix) {
			remainder := strings.TrimPrefix(r.URL.Path, prefix)
			if strings.HasSuffix(remainder, "/events") && r.Method == http.MethodGet {
				router.handlers.LiveEvents(w, r)
				return
			}
			if !strings.Contains(remainder, "/") && r.Method == http.MethodDelete {
				router.handlers.StopLive(w, r)
				return
			}
		}
		router.writeError(w, http.StatusNotFound, "invalid_request_error", "not_found", "The requested endpoint was not found.")
	}
}

func (router *Router) dispatchMethod(w http.ResponseWriter, r *http.Request, method string, handler http.HandlerFunc) {
	if r.Method != method {
		router.writeError(w, http.StatusMethodNotAllowed, "invalid_request_error", "method_not_allowed", "Method not allowed.")
		return
	}
	handler(w, r)
}

func RouteName(r *http.Request) string {
	switch r.URL.Path {
	case "/healthz", "/readyz", "/metrics", "/v1/models", "/v1/chat/completions", "/v1/videos/understand", "/v1/live/sessions":
		return r.URL.Path
	}
	if strings.HasPrefix(r.URL.Path, "/v1/live/sessions/") {
		if strings.HasSuffix(r.URL.Path, "/events") {
			return "/v1/live/sessions/{session}/events"
		}
		return "/v1/live/sessions/{session}"
	}
	return "unmatched"
}
