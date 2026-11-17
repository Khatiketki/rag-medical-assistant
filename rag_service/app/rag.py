"""RAG pipeline: retrieve -> build context -> prompt -> generate -> post-process."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from prometheus_client import Counter, Histogram

from app import ingest
from app.config import RetrievalProfile, Settings, get_profile
from app.llm import LLM, make_llm
from app.prompts import render
from app.retriever import BM25Retriever, DenseRetriever, RetrievedChunk, Retriever
from app.safety import DISCLAIMER, EMERGENCY_NOTICE, is_emergency

QUERIES = Counter("rag_queries_total", "Queries answered", ["profile", "outcome"])
STAGE_LATENCY = Histogram(
    "rag_stage_latency_seconds", "Latency per pipeline stage", ["stage"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
TOKENS = Counter("rag_llm_tokens_total", "LLM tokens used", ["direction"])


@dataclass
class Source:
    id: int
    doc_id: str
    title: str
    section: str
    score: float
    snippet: str


@dataclass
class Answer:
    answer: str
    sources: list[Source]
    profile: str
    model: str
    emergency: bool
    disclaimer: str
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        heading = f"{c.title} - {c.section}" if c.section else c.title
        blocks.append(f"[{i}] {heading}\n{c.text}")
    return "\n\n".join(blocks)


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: LLM):
        self.retriever = retriever
        self.llm = llm
        self.profile: RetrievalProfile = retriever.profile

    def answer(self, question: str) -> Answer:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        with STAGE_LATENCY.labels("retrieve").time():
            chunks = self.retriever.retrieve(question)
        timings["retrieve"] = (time.perf_counter() - t0) * 1000

        system, user = render(self.profile.prompt_version, question, format_context(chunks))

        t1 = time.perf_counter()
        with STAGE_LATENCY.labels("generate").time():
            gen = self.llm.generate(system, user)
        timings["generate"] = (time.perf_counter() - t1) * 1000
        TOKENS.labels("input").inc(gen.input_tokens)
        TOKENS.labels("output").inc(gen.output_tokens)

        emergency = is_emergency(question)
        text = f"{EMERGENCY_NOTICE}\n\n{gen.text}" if emergency else gen.text
        QUERIES.labels(self.profile.name, "refused" if gen.refused else "ok").inc()

        return Answer(
            answer=text,
            sources=[Source(id=i, doc_id=c.doc_id, title=c.title, section=c.section,
                            score=round(c.score, 4), snippet=c.text[:240])
                     for i, c in enumerate(chunks, start=1)],
            profile=self.profile.name,
            model=gen.model,
            emergency=emergency,
            disclaimer=DISCLAIMER,
            timings_ms={k: round(v, 1) for k, v in timings.items()},
        )


def build_pipeline(settings: Settings, profile_name: str, client=None, llm: LLM | None = None) -> RAGPipeline:
    """Wire a pipeline against an already-ingested collection."""
    profile = get_profile(profile_name)
    client = client or ingest.make_qdrant_client(settings)
    collection = ingest.collection_name(settings, profile)
    if not client.collection_exists(collection):
        raise RuntimeError(f"collection {collection!r} not found; run `python -m app.ingest` first")

    dense = DenseRetriever(client, collection, ingest.make_embed_model(settings, profile))
    sparse = BM25Retriever.from_jsonl(ingest.nodes_path(settings, profile)) if profile.hybrid else None
    llm = llm or make_llm(settings.llm_provider, settings.claude_model,
                          settings.claude_effort, settings.claude_max_tokens)
    return RAGPipeline(Retriever(profile, dense, sparse), llm)
