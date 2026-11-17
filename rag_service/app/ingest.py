"""Data pipeline: load -> clean -> chunk -> embed -> store in Qdrant.

Run:  python -m app.ingest --profile tuned
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import RetrievalProfile, Settings, get_profile, get_settings

log = logging.getLogger(__name__)

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class SourceDocument:
    doc_id: str
    title: str
    category: str
    text: str


def parse_document(path: Path) -> SourceDocument:
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    match = _FRONTMATTER.match(raw)
    if match:
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            if value:
                meta[key.strip()] = value.strip()
        raw = raw[match.end():]
    text = re.sub(r"[ \t]+", " ", raw).strip()
    return SourceDocument(
        doc_id=path.stem,
        title=meta.get("title", path.stem.replace("_", " ").title()),
        category=meta.get("category", "general"),
        text=text,
    )


def load_documents(corpus_dir: Path) -> list[SourceDocument]:
    paths = sorted(corpus_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no .md documents found in {corpus_dir}")
    return [parse_document(p) for p in paths]


def chunk_documents(docs: list[SourceDocument], profile: RetrievalProfile) -> list[TextNode]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=profile.chunk_size,
        chunk_overlap=profile.chunk_overlap,
        separators=["\n## ", "\n\n", "\n", ". ", " "],
    )
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=True
    )

    nodes: list[TextNode] = []
    for doc in docs:
        if profile.split_on_headers:
            sections = [(s.metadata.get("section", "Overview"), s.page_content)
                        for s in header_splitter.split_text(doc.text)]
        else:
            sections = [("", doc.text)]

        for section, section_text in sections:
            for chunk in splitter.split_text(section_text):
                chunk_id = f"{doc.doc_id}-{len(nodes)}"
                excluded = [] if profile.embed_metadata else ["title", "section"]
                nodes.append(TextNode(
                    id_=_stable_uuid(profile.name, chunk_id),
                    text=chunk,
                    metadata={"doc_id": doc.doc_id, "title": doc.title,
                              "section": section, "category": doc.category,
                              "chunk_id": chunk_id},
                    excluded_embed_metadata_keys=["doc_id", "category", "chunk_id", *excluded],
                    excluded_llm_metadata_keys=["doc_id", "category", "chunk_id"],
                ))
    return nodes


def _stable_uuid(*parts: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "/".join(parts)))


def make_qdrant_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.storage_dir / "qdrant"))


def make_embed_model(settings: Settings, profile: RetrievalProfile) -> BaseEmbedding:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=settings.embed_model,
        query_instruction=profile.query_instruction,
        normalize=True,
    )


def collection_name(settings: Settings, profile: RetrievalProfile) -> str:
    return f"{settings.collection_prefix}_{profile.name}"


def nodes_path(settings: Settings, profile: RetrievalProfile) -> Path:
    return settings.storage_dir / f"nodes_{profile.name}.jsonl"


def build_index(
    nodes: list[TextNode],
    client: QdrantClient,
    collection: str,
    embed_model: BaseEmbedding,
) -> VectorStoreIndex:
    if client.collection_exists(collection):
        client.delete_collection(collection)
    store = QdrantVectorStore(client=client, collection_name=collection)
    storage = StorageContext.from_defaults(vector_store=store)
    return VectorStoreIndex(nodes, storage_context=storage, embed_model=embed_model)


def save_nodes(nodes: list[TextNode], path: Path) -> None:
    """Persist chunk text for the sparse (BM25) retriever."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for n in nodes:
            f.write(json.dumps({"id": n.node_id, "text": n.text, "metadata": n.metadata}) + "\n")


def run(profile_name: str, settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    profile = get_profile(profile_name)
    start = time.perf_counter()

    docs = load_documents(settings.corpus_dir)
    nodes = chunk_documents(docs, profile)
    client = make_qdrant_client(settings)
    try:
        build_index(nodes, client, collection_name(settings, profile), make_embed_model(settings, profile))
    finally:
        client.close()
    save_nodes(nodes, nodes_path(settings, profile))

    log.info("ingested profile=%s docs=%d chunks=%d in %.1fs",
             profile.name, len(docs), len(nodes), time.perf_counter() - start)
    return len(nodes)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest the medical corpus into Qdrant.")
    parser.add_argument("--profile", action="append",
                        help="profile(s) to build; default: all profiles")
    args = parser.parse_args()

    from app.config import PROFILES
    for name in args.profile or list(PROFILES):
        run(name)


if __name__ == "__main__":
    main()
