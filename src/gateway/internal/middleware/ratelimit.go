package middleware

import (
	"math"
	"sync"
	"time"
)

type bucket struct {
	tokens float64
	last   time.Time
}

type Limiter struct {
	mu      sync.Mutex
	rate    float64
	burst   float64
	buckets map[string]bucket
}

func NewLimiter(rate float64, burst int) *Limiter {
	return &Limiter{rate: rate, burst: float64(burst), buckets: make(map[string]bucket)}
}

func (limiter *Limiter) Allow(key string, now time.Time) (bool, int) {
	limiter.mu.Lock()
	defer limiter.mu.Unlock()
	state, exists := limiter.buckets[key]
	if !exists {
		state = bucket{tokens: limiter.burst, last: now}
	}
	state.tokens = math.Min(limiter.burst, state.tokens+now.Sub(state.last).Seconds()*limiter.rate)
	state.last = now
	if state.tokens >= 1 {
		state.tokens--
		limiter.buckets[key] = state
		return true, 0
	}
	limiter.buckets[key] = state
	return false, int(math.Ceil((1 - state.tokens) / limiter.rate))
}
