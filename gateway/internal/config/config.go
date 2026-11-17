// Package config loads gateway settings from environment variables.
package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

type Config struct {
	Addr           string
	RAGServiceURL  string
	RequestTimeout time.Duration
	MaxRetries     int
	RateLimitRPS   float64
	RateLimitBurst int
	CacheTTL       time.Duration
	CacheSize      int
	MaxQuestionLen int
}

func Load() (Config, error) {
	c := Config{
		Addr:           getenv("GATEWAY_ADDR", ":8080"),
		RAGServiceURL:  getenv("RAG_SERVICE_URL", "http://localhost:8001"),
		RequestTimeout: 60 * time.Second,
		MaxRetries:     2,
		RateLimitRPS:   2,
		RateLimitBurst: 10,
		CacheTTL:       10 * time.Minute,
		CacheSize:      1000,
		MaxQuestionLen: 1000,
	}
	var err error
	if c.RequestTimeout, err = durationEnv("GATEWAY_REQUEST_TIMEOUT", c.RequestTimeout); err != nil {
		return c, err
	}
	if c.CacheTTL, err = durationEnv("GATEWAY_CACHE_TTL", c.CacheTTL); err != nil {
		return c, err
	}
	if c.MaxRetries, err = intEnv("GATEWAY_MAX_RETRIES", c.MaxRetries); err != nil {
		return c, err
	}
	if c.RateLimitBurst, err = intEnv("GATEWAY_RATE_LIMIT_BURST", c.RateLimitBurst); err != nil {
		return c, err
	}
	if c.CacheSize, err = intEnv("GATEWAY_CACHE_SIZE", c.CacheSize); err != nil {
		return c, err
	}
	if v := os.Getenv("GATEWAY_RATE_LIMIT_RPS"); v != "" {
		if c.RateLimitRPS, err = strconv.ParseFloat(v, 64); err != nil {
			return c, fmt.Errorf("GATEWAY_RATE_LIMIT_RPS: %w", err)
		}
	}
	return c, nil
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func durationEnv(key string, def time.Duration) (time.Duration, error) {
	v := os.Getenv(key)
	if v == "" {
		return def, nil
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return 0, fmt.Errorf("%s: %w", key, err)
	}
	return d, nil
}

func intEnv(key string, def int) (int, error) {
	v := os.Getenv(key)
	if v == "" {
		return def, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, fmt.Errorf("%s: %w", key, err)
	}
	return n, nil
}
