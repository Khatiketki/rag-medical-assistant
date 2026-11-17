from __future__ import annotations

import pytest
from llama_index.core.embeddings import MockEmbedding
from qdrant_client import QdrantClient

from app import ingest
from app.config import REPO_ROOT, Settings, get_profile
from app.llm import ExtractiveLLM
from app.rag import RAGPipeline
from app.retriever import BM25Retriever, DenseRetriever, Retriever


@pytest.fixture(scope="session")
def docs():
    return ingest.load_documents(REPO_ROOT / "data" / "corpus")


@pytest.fixture()
def settings(tmp_path):
    return Settings(storage_dir=tmp_path, llm_provider="extractive")


@pytest.fixture()
def tuned_pipeline(docs):
    """Full pipeline over an in-memory Qdrant with a mock (constant) embedding, so
    tests are offline and fast. Hybrid fusion lets BM25 drive the ranking."""
    profile = get_profile("tuned")
    nodes = ingest.chunk_documents(docs, profile)
    client = QdrantClient(location=":memory:")
    embed = MockEmbedding(embed_dim=8)
    ingest.build_index(nodes, client, "test_tuned", embed)
    sparse = BM25Retriever([{"text": n.text, "metadata": n.metadata} for n in nodes])
    retriever = Retriever(profile, DenseRetriever(client, "test_tuned", embed), sparse)
    yield RAGPipeline(retriever, ExtractiveLLM())
    client.close()
