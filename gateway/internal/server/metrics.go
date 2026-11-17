package server

import "github.com/prometheus/client_golang/prometheus"

type metrics struct {
	requests    *prometheus.CounterVec
	latency     *prometheus.HistogramVec
	cacheHits   prometheus.Counter
	rateLimited prometheus.Counter
	upstreamErr *prometheus.CounterVec
}

func newMetrics(reg prometheus.Registerer) *metrics {
	m := &metrics{
		requests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_requests_total", Help: "HTTP requests by route and status code.",
		}, []string{"route", "code"}),
		latency: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "gateway_request_duration_seconds",
			Help:    "HTTP request latency.",
			Buckets: []float64{0.005, 0.025, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60},
		}, []string{"route"}),
		cacheHits: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "gateway_cache_hits_total", Help: "Queries served from the answer cache.",
		}),
		rateLimited: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "gateway_rate_limited_total", Help: "Requests rejected by the rate limiter.",
		}),
		upstreamErr: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_upstream_errors_total", Help: "Failed calls to the RAG service.",
		}, []string{"kind"}),
	}
	reg.MustRegister(m.requests, m.latency, m.cacheHits, m.rateLimited, m.upstreamErr)
	return m
}
