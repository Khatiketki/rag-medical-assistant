from app.prompts import NO_ANSWER
from app.rag import Answer, Source
from eval.run_eval import Case, facts_present, load_cases, score_case, summarize


def _answer(text: str, docs: list[str]) -> Answer:
    sources = [Source(i, d, "", "", 0.0, "") for i, d in enumerate(docs, 1)]
    return Answer(text, sources, "tuned", "test", False, "")


def test_testset_is_well_formed():
    cases = load_cases()
    assert len(cases) >= 30
    assert len({c.id for c in cases}) == len(cases)
    assert all(c.must_include for c in cases if c.expected_doc)


def test_facts_present_requires_every_group():
    groups = [["140"], ["90"]]
    assert facts_present("Stage 2 is 140/90 or higher", groups)
    assert not facts_present("Stage 2 is 140 systolic", groups)


def test_score_in_scope_and_out_of_scope():
    case = Case("x", "q", "uti", [["coli"]])
    row = score_case(case, _answer("E. coli causes most UTIs [1]", ["gerd", "uti"]))
    assert row == {"id": "x", "in_scope": True, "hit": True, "rr": 0.5, "correct": True}

    oos = Case("y", "q", None, [])
    assert score_case(oos, _answer(NO_ANSWER, ["gerd"]))["correct"]
    assert not score_case(oos, _answer("Psoriasis is treated with...", ["gerd"]))["correct"]


def test_summarize():
    rows = [
        {"in_scope": True, "hit": True, "rr": 1.0, "correct": True},
        {"in_scope": True, "hit": False, "rr": 0.0, "correct": False},
        {"in_scope": False, "abstained": True, "correct": True},
    ]
    s = summarize(rows)
    assert s["hit@k"] == 0.5 and s["answer_accuracy_in_scope"] == 0.5
    assert s["abstention_rate_oos"] == 1.0 and s["accuracy"] == round(2 / 3, 4)
