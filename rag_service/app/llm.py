"""LLM backends: Claude (Anthropic SDK) and an offline extractive fallback."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol

import anthropic

from app.prompts import NO_ANSWER
from app.retriever import tokenize

log = logging.getLogger(__name__)


@dataclass
class Generation:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    refused: bool = False


class LLM(Protocol):
    def generate(self, system: str, user: str) -> Generation: ...


class ClaudeLLM:
    def __init__(self, model: str, effort: str, max_tokens: int, client: anthropic.Anthropic | None = None):
        # Credentials resolve from ANTHROPIC_API_KEY (or an `ant auth login` profile).
        self._client = client or anthropic.Anthropic()
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens

    def generate(self, system: str, user: str) -> Generation:
        # Server-side fallback: if a safety classifier declines a benign medical
        # question, the API re-runs it on a fallback model inside the same call.
        response = self._client.beta.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self._effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            log.warning("claude refused request_id=%s", response._request_id)
            return Generation(text=NO_ANSWER, model=response.model, refused=True,
                              input_tokens=response.usage.input_tokens,
                              output_tokens=response.usage.output_tokens)
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return Generation(text=text, model=response.model,
                          input_tokens=response.usage.input_tokens,
                          output_tokens=response.usage.output_tokens)


_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_PASSAGE = re.compile(r"^\[(\d+)\][^\n]*\n(.*?)(?=^\[\d+\]|\Z)", re.DOTALL | re.MULTILINE)


class ExtractiveLLM:
    """Offline stand-in for an LLM: returns the passage sentences that best match
    the question. Lets the full pipeline, tests, and eval run without an API key."""

    def __init__(self, max_sentences: int = 3, min_overlap: float = 0.34):
        self._max_sentences = max_sentences
        self._min_overlap = min_overlap

    def generate(self, system: str, user: str) -> Generation:
        context, _, question = user.rpartition("Question:")
        q_terms = set(tokenize(question))
        scored: list[tuple[float, str, str]] = []
        for num, body in _PASSAGE.findall(context):
            for sentence in _SENTENCE.split(body.strip()):
                overlap = len(q_terms & set(tokenize(sentence)))
                if overlap:
                    scored.append((overlap / max(len(q_terms), 1), sentence.strip(), num))
        scored.sort(key=lambda s: s[0], reverse=True)
        if not scored or scored[0][0] < self._min_overlap:
            return Generation(text=NO_ANSWER, model="extractive")
        top = scored[: self._max_sentences]
        return Generation(text=" ".join(f"{s} [{n}]" for _, s, n in top), model="extractive")


def make_llm(provider: str, model: str, effort: str, max_tokens: int) -> LLM:
    if provider == "claude":
        return ClaudeLLM(model=model, effort=effort, max_tokens=max_tokens)
    if provider == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"unknown llm provider {provider!r}")
