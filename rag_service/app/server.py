"""FastAPI service exposing the RAG pipeline.

Run:  uvicorn app.server:app --port 8001
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field

from app import ingest
from app.config import PROFILES, get_settings
from app.rag import RAGPipeline, build_pipeline

log = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    profile: str | None = None


def create_app(pipelines: dict[str, RAGPipeline] | None = None, default_profile: str | None = None) -> FastAPI:
    settings = get_settings()
    default = default_profile or settings.default_profile

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = None
        if pipelines is None:
            # Embedded Qdrant allows one client per path, so all pipelines share it.
            client = ingest.make_qdrant_client(settings)
            loaded = {}
            for name in PROFILES:
                try:
                    loaded[name] = build_pipeline(settings, name, client=client)
                except RuntimeError as exc:
                    log.warning("profile %s unavailable: %s", name, exc)
            app.state.pipelines = loaded
        else:
            app.state.pipelines = pipelines
        log.info("loaded profiles: %s", sorted(app.state.pipelines))
        yield
        if client is not None:
            client.close()

    app = FastAPI(title="Medical RAG Service", version="1.0.0", lifespan=lifespan)
    app.mount("/metrics", make_asgi_app())

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        if default not in app.state.pipelines:
            raise HTTPException(503, f"default profile {default!r} not loaded")
        return {"status": "ready", "profiles": sorted(app.state.pipelines)}

    @app.post("/query")
    async def query(req: QueryRequest):
        name = req.profile or default
        pipeline = app.state.pipelines.get(name)
        if pipeline is None:
            raise HTTPException(400, f"profile {name!r} is not available")
        # The pipeline is synchronous (embedding + HTTP to the LLM); keep the event loop free.
        result = await run_in_threadpool(pipeline.answer, req.question.strip())
        return result.to_dict()

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()
