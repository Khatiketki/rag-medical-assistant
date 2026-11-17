"""Service settings and retrieval profiles.

A *profile* bundles every knob that affects answer quality (chunking, embedding
text, retrieval strategy, prompt version). The evaluation harness runs the same
test set against several profiles so changes can be measured, not guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=REPO_ROOT / ".env", extra="ignore")

    corpus_dir: Path = REPO_ROOT / "data" / "corpus"
    storage_dir: Path = REPO_ROOT / "storage"

    # Vector DB: a Qdrant server URL, or empty to use embedded on-disk Qdrant.
    qdrant_url: str = ""
    collection_prefix: str = "medical"

    embed_model: str = "BAAI/bge-small-en-v1.5"

    # "claude" calls the Anthropic API; "extractive" is an offline, no-key
    # fallback used for tests and CI.
    llm_provider: str = "claude"
    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "medium"
    claude_max_tokens: int = 4096

    default_profile: str = "tuned"


@dataclass(frozen=True)
class RetrievalProfile:
    name: str
    chunk_size: int  # characters
    chunk_overlap: int
    split_on_headers: bool  # split on markdown sections before size-based splitting
    embed_metadata: bool  # prepend title/section to the text that gets embedded
    query_instruction: str | None  # BGE query-side instruction
    top_k: int
    hybrid: bool  # fuse dense + BM25 results with reciprocal rank fusion
    prompt_version: str


PROFILES: dict[str, RetrievalProfile] = {
    # First working version: big fixed-size chunks, dense-only, minimal prompt.
    "baseline": RetrievalProfile(
        name="baseline",
        chunk_size=1500,
        chunk_overlap=0,
        split_on_headers=False,
        embed_metadata=False,
        query_instruction=None,
        top_k=2,
        hybrid=False,
        prompt_version="v1",
    ),
    # Tuned: section-aware smaller chunks with contextual headers, BGE query
    # instruction, hybrid dense+BM25 retrieval, grounded prompt with citations.
    "tuned": RetrievalProfile(
        name="tuned",
        chunk_size=600,
        chunk_overlap=100,
        split_on_headers=True,
        embed_metadata=True,
        query_instruction="Represent this sentence for searching relevant passages: ",
        top_k=4,
        hybrid=True,
        prompt_version="v2",
    ),
}


def get_profile(name: str) -> RetrievalProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}") from None


def get_settings() -> Settings:
    return Settings()
