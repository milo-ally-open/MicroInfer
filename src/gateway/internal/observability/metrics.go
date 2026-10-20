package observability

import (
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
	"sync"
)

var durationBounds = []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 300, 900}

type histogram struct {
	count   uint64
	sum     float64
	buckets []uint64
}

type Metrics struct {
	mu             sync.Mutex
	requests       map[string]uint64
	durations      map[string]*histogram
	upstreamErrors map[string]uint64
}

func NewMetrics() *Metrics {
	return &Metrics{requests: map[string]uint64{}, durations: map[string]*histogram{}, upstreamErrors: map[string]uint64{}}
}

func (metrics *Metrics) ObserveRequest(route string, status int, seconds float64) {
	metrics.mu.Lock()
	defer metrics.mu.Unlock()
	key := route + "\x00" + strconv.Itoa(status)
	metrics.requests[key]++
	metrics.observeHistogram("http_request_duration_seconds", route, seconds)
}

func (metrics *Metrics) Observe(name, route string, seconds float64) {
	metrics.mu.Lock()
	defer metrics.mu.Unlock()
	metrics.observeHistogram(name, route, seconds)
}

func (metrics *Metrics) AddUpstreamError(kind string) {
	metrics.mu.Lock()
	defer metrics.mu.Unlock()
	metrics.upstreamErrors[kind]++
}

func (metrics *Metrics) observeHistogram(name, route string, seconds float64) {
	key := name + "\x00" + route
	h := metrics.durations[key]
	if h == nil {
		h = &histogram{buckets: make([]uint64, len(durationBounds))}
		metrics.durations[key] = h
	}
	h.count++
	h.sum += seconds
	for i, bound := range durationBounds {
		if seconds <= bound {
			h.buckets[i]++
		}
	}
}

func (metrics *Metrics) WritePrometheus(writer io.Writer) {
	metrics.mu.Lock()
	defer metrics.mu.Unlock()
	fmt.Fprintln(writer, "# HELP gateway_http_requests_total Completed HTTP requests.")
	fmt.Fprintln(writer, "# TYPE gateway_http_requests_total counter")
	requestKeys := make([]string, 0, len(metrics.requests))
	for key := range metrics.requests {
		requestKeys = append(requestKeys, key)
	}
	sort.Strings(requestKeys)
	for _, key := range requestKeys {
		parts := strings.Split(key, "\x00")
		fmt.Fprintf(writer, "gateway_http_requests_total{route=%q,status=%q} %d\n", parts[0], parts[1], metrics.requests[key])
	}
	metricKeys := make([]string, 0, len(metrics.durations))
	for key := range metrics.durations {
		metricKeys = append(metricKeys, key)
	}
	sort.Strings(metricKeys)
	typeWritten := make(map[string]bool)
	for _, key := range metricKeys {
		parts := strings.Split(key, "\x00")
		name, route, h := parts[0], parts[1], metrics.durations[key]
		if !typeWritten[name] {
			fmt.Fprintf(writer, "# HELP gateway_%s Gateway latency measurement in seconds.\n", name)
			fmt.Fprintf(writer, "# TYPE gateway_%s histogram\n", name)
			typeWritten[name] = true
		}
		for i, bound := range durationBounds {
			fmt.Fprintf(writer, "gateway_%s_bucket{route=%q,le=%q} %d\n", name, route, strconv.FormatFloat(bound, 'f', -1, 64), h.buckets[i])
		}
		fmt.Fprintf(writer, "gateway_%s_bucket{route=%q,le=\"+Inf\"} %d\n", name, route, h.count)
		fmt.Fprintf(writer, "gateway_%s_sum{route=%q} %g\n", name, route, h.sum)
		fmt.Fprintf(writer, "gateway_%s_count{route=%q} %d\n", name, route, h.count)
	}
	fmt.Fprintln(writer, "# HELP gateway_upstream_errors_total Upstream request failures by safe error class.")
	fmt.Fprintln(writer, "# TYPE gateway_upstream_errors_total counter")
	errorKinds := make([]string, 0, len(metrics.upstreamErrors))
	for key := range metrics.upstreamErrors {
		errorKinds = append(errorKinds, key)
	}
	sort.Strings(errorKinds)
	for _, kind := range errorKinds {
		fmt.Fprintf(writer, "gateway_upstream_errors_total{kind=%q} %d\n", kind, metrics.upstreamErrors[kind])
	}
}
