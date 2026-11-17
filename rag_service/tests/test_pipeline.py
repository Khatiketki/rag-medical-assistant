from app.llm import ExtractiveLLM
from app.prompts import NO_ANSWER, render
from app.rag import format_context
from app.retriever import RetrievedChunk
from app.safety import EMERGENCY_NOTICE, is_emergency


def test_v2_prompt_includes_rules_context_and_question():
    system, user = render("v2", "What is metformin?", "[1] Diabetes\nMetformin is first-line.")
    assert "Cite the passage numbers" in system
    assert NO_ANSWER in system
    assert "[1] Diabetes" in user and user.endswith("What is metformin?")


def test_format_context_numbers_passages():
    chunks = [RetrievedChunk("c1", "dm", "Diabetes", "Treatment", "Metformin.", 1.0),
              RetrievedChunk("c2", "htn", "Hypertension", "", "DASH diet.", 0.5)]
    assert format_context(chunks) == "[1] Diabetes - Treatment\nMetformin.\n\n[2] Hypertension\nDASH diet."


def test_extractive_llm_cites_best_sentence():
    _, user = render("v2", "Which drug is first-line for diabetes?",
                     "[1] Diabetes - Treatment\nExercise helps. Metformin is the first-line drug for diabetes.")
    out = ExtractiveLLM().generate("", user)
    assert out.text.startswith("Metformin is the first-line drug for diabetes. [1]")


def test_extractive_llm_abstains_without_support():
    _, user = render("v2", "How is psoriasis treated?", "[1] Asthma\nAlbuterol opens airways.")
    assert ExtractiveLLM().generate("", user).text == NO_ANSWER


def test_emergency_detection():
    assert is_emergency("My dad has chest pain and his arm is numb")
    assert is_emergency("she can't breathe")
    assert not is_emergency("what foods trigger heartburn")


def test_pipeline_answers_with_sources(tuned_pipeline):
    result = tuned_pipeline.answer("What A1C level is used to diagnose diabetes?")
    assert result.sources[0].doc_id == "type2_diabetes"
    assert "6.5" in result.answer
    assert result.profile == "tuned"
    assert set(result.timings_ms) == {"retrieve", "generate"}


def test_pipeline_prepends_emergency_notice(tuned_pipeline):
    result = tuned_pipeline.answer("I have crushing chest pain, is it a heart attack?")
    assert result.emergency
    assert result.answer.startswith(EMERGENCY_NOTICE)
