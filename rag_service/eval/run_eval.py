"""Evaluate retrieval and answer quality for one or more profiles.

Run:  python -m eval.run_eval --profiles baseline tuned [--llm extractive|claude]

Metrics
  hit@k         expected document appears among retrieved chunks
  mrr           mean reciprocal rank of the expected document
  accuracy      answer contains every required fact (in-scope) or correctly
                abstains (out-of-scope); this is the headline number
  abstention    share of out-of-scope questions the system declined to answer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app import ingest
from app.config import REPO_ROOT, get_settings
from app.llm import make_llm
from app.rag import Answer, build_pipeline

TESTSET = REPO_ROOT / "data" / "eval" / "testset.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class Case:
    id: str
    question: str
    expected_doc: str | None
    must_include: list[list[str]]


def load_cases(path: Path = TESTSET) -> list[Case]:
    with path.open(encoding="utf-8") as f:
        return [Case(**json.loads(line)) for line in f if line.strip()]


def abstained(answer_text: str) -> bool:
    return "don't have enough information" in answer_text.lower()


def facts_present(answer_text: str, must_include: list[list[str]]) -> bool:
    """Each inner list is a set of acceptable alternatives; every group must match."""
    text = answer_text.lower()
    return all(any(alt.lower() in text for alt in group) for group in must_include)


def score_case(case: Case, answer: Answer) -> dict:
    docs = [s.doc_id for s in answer.sources]
    if case.expected_doc is None:
        return {"id": case.id, "in_scope": False, "correct": abstained(answer.answer),
                "abstained": abstained(answer.answer)}
    rank = docs.index(case.expected_doc) + 1 if case.expected_doc in docs else 0
    return {
        "id": case.id,
        "in_scope": True,
        "hit": rank > 0,
        "rr": 1.0 / rank if rank else 0.0,
        "correct": not abstained(answer.answer) and facts_present(answer.answer, case.must_include),
    }


def summarize(rows: list[dict]) -> dict:
    in_scope = [r for r in rows if r["in_scope"]]
    oos = [r for r in rows if not r["in_scope"]]
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else 0.0  # noqa: E731
    return {
        "n": len(rows),
        "hit@k": mean([r["hit"] for r in in_scope]),
        "mrr": mean([r["rr"] for r in in_scope]),
        "answer_accuracy_in_scope": mean([r["correct"] for r in in_scope]),
        "abstention_rate_oos": mean([r["abstained"] for r in oos]),
        "accuracy": mean([r["correct"] for r in rows]),
    }


def evaluate(profile: str, llm_provider: str, cases: list[Case], client) -> dict:
    settings = get_settings()
    llm = make_llm(llm_provider, settings.claude_model, settings.claude_effort, settings.claude_max_tokens)
    pipeline = build_pipeline(settings, profile, client=client, llm=llm)

    rows, latencies = [], []
    for case in cases:
        start = time.perf_counter()
        answer = pipeline.answer(case.question)
        latencies.append(time.perf_counter() - start)
        row = score_case(case, answer)
        row["answer"] = answer.answer
        row["retrieved"] = [s.doc_id for s in answer.sources]
        rows.append(row)

    summary = summarize(rows)
    latencies.sort()
    summary["p50_latency_s"] = round(latencies[len(latencies) // 2], 3)
    summary["p95_latency_s"] = round(latencies[int(len(latencies) * 0.95) - 1], 3)
    return {"profile": profile, "llm": llm_provider, "summary": summary, "cases": rows}


def print_table(reports: list[dict]) -> None:
    cols = ["hit@k", "mrr", "answer_accuracy_in_scope", "abstention_rate_oos", "accuracy", "p50_latency_s"]
    print(f"\n{'profile':<10}" + "".join(f"{c:>26}" for c in cols))
    for r in reports:
        print(f"{r['profile']:<10}" + "".join(f"{r['summary'][c]:>26}" for c in cols))
    if len(reports) >= 2:
        base, best = reports[0]["summary"]["accuracy"], reports[-1]["summary"]["accuracy"]
        print(f"\naccuracy: {base:.1%} -> {best:.1%} "
              f"({(best - base) * 100:+.1f} pts, {((best - base) / base if base else 0):+.0%} relative)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", default=["baseline", "tuned"])
    parser.add_argument("--llm", default="extractive", choices=["extractive", "claude"])
    parser.add_argument("--min-accuracy", type=float, default=None,
                        help="exit non-zero if the last profile scores below this (CI gate)")
    args = parser.parse_args()

    settings = get_settings()
    client = ingest.make_qdrant_client(settings)
    cases = load_cases()
    try:
        reports = [evaluate(p, args.llm, cases, client) for p in args.profiles]
    finally:
        client.close()

    print_table(reports)
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"eval_{args.llm}_{stamp}.json"
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nfull report: {out}")

    if args.min_accuracy is not None and reports[-1]["summary"]["accuracy"] < args.min_accuracy:
        print(f"FAIL: accuracy below {args.min_accuracy}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
