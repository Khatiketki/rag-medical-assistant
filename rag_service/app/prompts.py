"""Versioned prompt templates (LangChain). Each version is evaluated separately."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

NO_ANSWER = "I don't have enough information in my medical knowledge base to answer that."

# v1: the first prompt that worked. No grounding rules, no citation format.
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful medical assistant."),
    ("human", "Context:\n{context}\n\nQuestion: {question}\nAnswer:"),
])

# v2: grounded answers with inline citations, explicit abstention, and safety framing.
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system",
     "You are a medical information assistant for patients and caregivers. You answer "
     "questions using only the numbered reference passages supplied with each question.\n\n"
     "Rules:\n"
     "- Base every statement on the passages. Cite the passage numbers you used in square "
     "brackets, e.g. [1] or [2][3].\n"
     "- Include the specific numbers, thresholds, drug names, and timeframes the passages give; "
     "these are what the reader most needs.\n"
     "- If the passages do not contain the answer, reply exactly: \"" + NO_ANSWER + "\" and do "
     "not guess from general knowledge.\n"
     "- Lead with the direct answer in one or two sentences, then add supporting detail. Keep "
     "the whole answer under 200 words and use plain language.\n"
     "- If the question describes possible emergency symptoms, tell the reader to call 911 or "
     "their local emergency number first.\n"
     "- You provide general information, not a diagnosis; do not tell someone to start or stop "
     "a medication."),
    ("human", "Reference passages:\n{context}\n\nQuestion: {question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


def render(version: str, question: str, context: str) -> tuple[str, str]:
    """Return (system, user) strings for the given prompt version."""
    messages = PROMPTS[version].format_messages(question=question, context=context)
    system = "\n\n".join(m.content for m in messages if isinstance(m, SystemMessage))
    user = "\n\n".join(m.content for m in messages if isinstance(m, HumanMessage))
    return system, user
