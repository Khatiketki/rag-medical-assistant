"""Retrieval: dense vector search (LlamaIndex + Qdrant), BM25, and hybrid fusion."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from app.config import RetrievalProfile

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it my of on or should "
    "that the to what when which who why with you your".split()
)
RRF_K = 60  # standard reciprocal-rank-fusion damping constant


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    score: float


class DenseRetriever:
    def __init__(self, client: QdrantClient, collection: str, embed_model: BaseEmbedding):
        store = QdrantVectorStore(client=client, collection_name=collection)
        self._index = VectorStoreIndex.from_vector_store(store, embed_model=embed_model)

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        results = self._index.as_retriever(similarity_top_k=k).retrieve(query)
        return [_from_metadata(r.node.metadata, r.node.get_content(), r.score or 0.0) for r in results]


class BM25Retriever:
    def __init__(self, records: list[dict]):
        self._records = records
        self._bm25 = BM25Okapi([tokenize(_bm25_text(r)) for r in records])

    @classmethod
    def from_jsonl(cls, path: Path) -> BM25Retriever:
        with path.open(encoding="utf-8") as f:
            return cls([json.loads(line) for line in f])

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [_from_metadata(self._records[i]["metadata"], self._records[i]["text"], float(scores[i]))
                for i in ranked if scores[i] > 0]


def _bm25_text(record: dict) -> str:
    meta = record["metadata"]
    return f"{meta.get('title', '')} {meta.get('section', '')} {record['text']}"


def _from_metadata(meta: dict, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=meta.get("chunk_id", ""),
        doc_id=meta.get("doc_id", ""),
        title=meta.get("title", ""),
        section=meta.get("section", ""),
        text=text,
        score=score,
    )


def reciprocal_rank_fusion(result_lists: list[list[RetrievedChunk]], k: int) -> list[RetrievedChunk]:
    fused: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            chunks.setdefault(chunk.chunk_id, chunk)
    ordered = sorted(fused, key=fused.get, reverse=True)[:k]
    return [RetrievedChunk(**{**chunks[c].__dict__, "score": fused[c]}) for c in ordered]


class Retriever:
    """Profile-aware retriever: dense-only, or hybrid dense + BM25 with RRF."""

    def __init__(self, profile: RetrievalProfile, dense: DenseRetriever, sparse: BM25Retriever | None):
        if profile.hybrid and sparse is None:
            raise ValueError("hybrid profile requires a BM25 retriever")
        self.profile = profile
        self._dense = dense
        self._sparse = sparse

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        k = self.profile.top_k
        if not self.profile.hybrid:
            return self._dense.retrieve(query, k)
        # Over-fetch from each retriever so fusion has candidates to rerank.
        return reciprocal_rank_fusion(
            [self._dense.retrieve(query, k * 3), self._sparse.retrieve(query, k * 3)], k
        )
