package config

import (
	"fmt"
	"net"
	"net/url"
	"os"
	"regexp"
	"strings"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Listener struct {
		Host string `yaml:"host"`
		Port int    `yaml:"port"`
	} `yaml:"listener"`
	PublicBaseURL string `yaml:"public_base_url"`
	Upstream      struct {
		BaseURL                   string `yaml:"base_url"`
		MaxIdleConnections        int    `yaml:"max_idle_connections"`
		MaxIdleConnectionsPerHost int    `yaml:"max_idle_connections_per_host"`
		IdleConnectionSeconds     int    `yaml:"idle_connection_seconds"`
		TCPKeepAliveSeconds       int    `yaml:"tcp_keepalive_seconds"`
		ExpectContinueSeconds     int    `yaml:"expect_continue_seconds"`
	} `yaml:"upstream"`
	Models map[string]ModelRoute `yaml:"models"`
	Limits struct {
		ChatRequestBytes  int64 `yaml:"chat_request_bytes"`
		VideoRequestBytes int64 `yaml:"video_request_bytes"`
		LiveRequestBytes  int64 `yaml:"live_request_bytes"`
		JSONResponseBytes int64 `yaml:"json_response_bytes"`
		SSEEventBytes     int64 `yaml:"sse_event_bytes"`
		MaxHeaderBytes    int   `yaml:"max_header_bytes"`
		StreamBufferBytes int   `yaml:"stream_buffer_bytes"`
	} `yaml:"limits"`
	Timeouts struct {
		ConnectSeconds        int `yaml:"connect_seconds"`
		RequestSeconds        int `yaml:"request_seconds"`
		VideoRequestSeconds   int `yaml:"video_request_seconds"`
		StreamIdleSeconds     int `yaml:"stream_idle_seconds"`
		StreamMaxSeconds      int `yaml:"stream_max_seconds"`
		ReadinessSeconds      int `yaml:"readiness_seconds"`
		ReadHeaderSeconds     int `yaml:"read_header_seconds"`
		IdleConnectionSeconds int `yaml:"idle_connection_seconds"`
		ShutdownGraceSeconds  int `yaml:"shutdown_grace_seconds"`
	} `yaml:"timeouts"`
	RateLimit struct {
		RequestsPerSecond float64 `yaml:"requests_per_second"`
		Burst             int     `yaml:"burst"`
	} `yaml:"rate_limit"`
	Auth struct {
		APIKeysFileEnv string `yaml:"api_keys_file_env"`
	} `yaml:"auth"`
	Session struct {
		HMACKeysEnv      string `yaml:"hmac_keys_env"`
		HandleTTLSeconds int    `yaml:"handle_ttl_seconds"`
	} `yaml:"session"`
	Observability struct {
		LogLevel       string `yaml:"log_level"`
		MetricsPath    string `yaml:"metrics_path"`
		MetricsEnabled bool   `yaml:"metrics_enabled"`
	} `yaml:"observability"`
}

type ModelRoute struct {
	UpstreamModel string `yaml:"upstream_model"`
	BaseURL       string `yaml:"base_url,omitempty"`
}

var envName = regexp.MustCompile(`^[A-Z_][A-Z0-9_]*$`)
var hostName = regexp.MustCompile(`^[A-Za-z0-9.-]+$`)

func Load(path string) (Config, error) {
	var cfg Config
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, fmt.Errorf("read gateway config: %w", err)
	}
	decoder := yaml.NewDecoder(strings.NewReader(string(data)))
	decoder.KnownFields(true)
	if err := decoder.Decode(&cfg); err != nil {
		return cfg, fmt.Errorf("parse gateway config: %w", err)
	}
	if err := cfg.Validate(); err != nil {
		return cfg, err
	}
	return cfg, nil
}

func (cfg Config) Validate() error {
	if cfg.Listener.Host == "" || cfg.Listener.Host != strings.TrimSpace(cfg.Listener.Host) {
		return fmt.Errorf("listener.host is required")
	}
	if strings.Contains(cfg.Listener.Host, ":") {
		if net.ParseIP(cfg.Listener.Host) == nil {
			return fmt.Errorf("listener.host must be an IP address or hostname")
		}
	} else if !hostName.MatchString(cfg.Listener.Host) {
		return fmt.Errorf("listener.host must be an IP address or hostname")
	}
	if cfg.Listener.Port < 1 || cfg.Listener.Port > 65535 {
		return fmt.Errorf("listener.port must be between 1 and 65535")
	}
	if err := validateHTTPURL("upstream.base_url", cfg.Upstream.BaseURL); err != nil {
		return err
	}
	for name, value := range map[string]int{
		"upstream.max_idle_connections":          cfg.Upstream.MaxIdleConnections,
		"upstream.max_idle_connections_per_host": cfg.Upstream.MaxIdleConnectionsPerHost,
		"upstream.idle_connection_seconds":       cfg.Upstream.IdleConnectionSeconds,
		"upstream.tcp_keepalive_seconds":         cfg.Upstream.TCPKeepAliveSeconds,
		"upstream.expect_continue_seconds":       cfg.Upstream.ExpectContinueSeconds,
	} {
		if value < 0 || (name != "upstream.expect_continue_seconds" && value == 0) {
			return fmt.Errorf("%s must be positive (or zero for Expect-Continue timeout)", name)
		}
	}
	if err := validateHTTPURL("public_base_url", cfg.PublicBaseURL); err != nil {
		return err
	}
	if len(cfg.Models) == 0 {
		return fmt.Errorf("models must contain at least one public model route")
	}
	for alias, route := range cfg.Models {
		if strings.TrimSpace(alias) == "" || strings.ContainsAny(alias, " /\\") {
			return fmt.Errorf("invalid public model alias %q", alias)
		}
		if strings.TrimSpace(route.UpstreamModel) == "" {
			return fmt.Errorf("models.%s.upstream_model is required", alias)
		}
		if route.BaseURL != "" {
			if err := validateHTTPURL("models."+alias+".base_url", route.BaseURL); err != nil {
				return err
			}
		}
	}
	for name, value := range map[string]int64{
		"limits.chat_request_bytes":  cfg.Limits.ChatRequestBytes,
		"limits.video_request_bytes": cfg.Limits.VideoRequestBytes,
		"limits.live_request_bytes":  cfg.Limits.LiveRequestBytes,
		"limits.json_response_bytes": cfg.Limits.JSONResponseBytes,
		"limits.sse_event_bytes":     cfg.Limits.SSEEventBytes,
		"limits.max_header_bytes":    int64(cfg.Limits.MaxHeaderBytes),
		"limits.stream_buffer_bytes": int64(cfg.Limits.StreamBufferBytes),
	} {
		if value <= 0 {
			return fmt.Errorf("%s must be greater than zero", name)
		}
	}
	for name, value := range map[string]int{
		"timeouts.connect_seconds":         cfg.Timeouts.ConnectSeconds,
		"timeouts.request_seconds":         cfg.Timeouts.RequestSeconds,
		"timeouts.video_request_seconds":   cfg.Timeouts.VideoRequestSeconds,
		"timeouts.stream_idle_seconds":     cfg.Timeouts.StreamIdleSeconds,
		"timeouts.stream_max_seconds":      cfg.Timeouts.StreamMaxSeconds,
		"timeouts.readiness_seconds":       cfg.Timeouts.ReadinessSeconds,
		"timeouts.read_header_seconds":     cfg.Timeouts.ReadHeaderSeconds,
		"timeouts.idle_connection_seconds": cfg.Timeouts.IdleConnectionSeconds,
		"timeouts.shutdown_grace_seconds":  cfg.Timeouts.ShutdownGraceSeconds,
	} {
		if value <= 0 {
			return fmt.Errorf("%s must be greater than zero", name)
		}
	}
	if cfg.RateLimit.RequestsPerSecond <= 0 || cfg.RateLimit.Burst < 1 {
		return fmt.Errorf("rate_limit.requests_per_second and rate_limit.burst must be positive")
	}
	if !envName.MatchString(cfg.Auth.APIKeysFileEnv) || !envName.MatchString(cfg.Session.HMACKeysEnv) {
		return fmt.Errorf("auth.api_keys_file_env and session.hmac_keys_env must name environment variables")
	}
	if cfg.Session.HandleTTLSeconds < 1 {
		return fmt.Errorf("session.handle_ttl_seconds must be positive")
	}
	if cfg.Observability.MetricsEnabled && (cfg.Observability.MetricsPath == "" || !strings.HasPrefix(cfg.Observability.MetricsPath, "/")) {
		return fmt.Errorf("observability.metrics_path must be an absolute URL path when metrics are enabled")
	}
	return nil
}

func validateHTTPURL(name, raw string) error {
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return fmt.Errorf("%s must be an absolute http or https URL without credentials, query, or fragment", name)
	}
	return nil
}
