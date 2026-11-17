from app.retriever import BM25Retriever, RetrievedChunk, reciprocal_rank_fusion, tokenize


def _chunk(cid: str, doc: str = "d") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, doc_id=doc, title="", section="", text=cid, score=0.0)


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is the A1C, for diabetes?") == ["a1c", "diabetes"]


def test_rrf_rewards_agreement_between_retrievers():
    dense = [_chunk("a"), _chunk("b")]
    sparse = [_chunk("b"), _chunk("c")]
    fused = reciprocal_rank_fusion([dense, sparse], k=2)
    # "b" is second in both lists but beats "a" and "c", which each appear once at rank 1.
    assert fused[0].chunk_id == "b"
    assert len(fused) == 2 and fused[0].score > fused[1].score


def test_bm25_ranks_keyword_match_first():
    records = [
        {"text": "Metformin is first-line for type 2 diabetes.", "metadata": {"chunk_id": "1", "doc_id": "dm"}},
        {"text": "Albuterol is a rescue inhaler for asthma.", "metadata": {"chunk_id": "2", "doc_id": "asthma"}},
        {"text": "Oseltamivir treats influenza.", "metadata": {"chunk_id": "3", "doc_id": "flu"}},
    ]
    results = BM25Retriever(records).retrieve("which inhaler for asthma", k=3)
    assert results[0].doc_id == "asthma"
    assert len(results) == 1  # zero-score records are dropped
