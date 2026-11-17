from app import ingest
from app.config import get_profile


def test_parse_document_reads_frontmatter(tmp_path):
    path = tmp_path / "sample_doc.md"
    path.write_text("---\ntitle: Sample Title\ncategory: test\n---\n# Heading\n\nBody   text.\n")
    doc = ingest.parse_document(path)
    assert doc.doc_id == "sample_doc"
    assert doc.title == "Sample Title"
    assert doc.category == "test"
    assert "---" not in doc.text
    assert "Body text." in doc.text


def test_load_documents_finds_corpus(docs):
    assert len(docs) >= 10
    assert all(d.text and d.title for d in docs)


def test_tuned_chunks_respect_size_and_carry_sections(docs):
    profile = get_profile("tuned")
    nodes = ingest.chunk_documents(docs, profile)
    assert all(len(n.text) <= profile.chunk_size for n in nodes)
    assert all(n.metadata["section"] for n in nodes)
    # Contextual headers: title/section are part of the embedded text.
    sample = nodes[0]
    embedded = sample.get_content(metadata_mode="embed")
    assert sample.metadata["title"] in embedded


def test_baseline_chunks_embed_text_only(docs):
    nodes = ingest.chunk_documents(docs, get_profile("baseline"))
    sample = nodes[0]
    assert sample.metadata["title"] not in sample.get_content(metadata_mode="embed").split("\n")[0]


def test_chunk_ids_are_unique_and_stable(docs):
    profile = get_profile("tuned")
    first = [n.node_id for n in ingest.chunk_documents(docs, profile)]
    second = [n.node_id for n in ingest.chunk_documents(docs, profile)]
    assert first == second
    assert len(set(first)) == len(first)
