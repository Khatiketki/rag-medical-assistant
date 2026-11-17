// Command gateway is the public API for the medical RAG assistant.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"

	"github.com/medrag/gateway/internal/cache"
	"github.com/medrag/gateway/internal/config"
	"github.com/medrag/gateway/internal/ragclient"
	"github.com/medrag/gateway/internal/ratelimit"
	"github.com/medrag/gateway/internal/server"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	cfg, err := config.Load()
	if err != nil {
		logger.Error("invalid config", "err", err)
		os.Exit(1)
	}

	reg := prometheus.NewRegistry()
	reg.MustRegister(collectors.NewGoCollector(), collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))

	limiter := ratelimit.New(cfg.RateLimitRPS, cfg.RateLimitBurst)
	srv := server.New(
		ragclient.New(cfg.RAGServiceURL, cfg.RequestTimeout, cfg.MaxRetries),
		server.Options{
			MaxQuestionLen: cfg.MaxQuestionLen,
			Limiter:        limiter,
			Cache:          cache.New(cfg.CacheSize, cfg.CacheTTL),
			Registry:       reg,
			Logger:         logger,
		},
	)

	httpServer := &http.Server{
		Addr:              cfg.Addr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		WriteTimeout:      cfg.RequestTimeout + 5*time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		ticker := time.NewTicker(time.Minute)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				limiter.Sweep(10 * time.Minute)
			case <-ctx.Done():
				return
			}
		}
	}()

	go func() {
		logger.Info("gateway listening", "addr", cfg.Addr, "rag_service", cfg.RAGServiceURL)
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("server failed", "err", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	logger.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		logger.Error("graceful shutdown failed", "err", err)
	}
}
