"""Safety layer: emergency-symptom detection and the standard disclaimer."""

from __future__ import annotations

import re

DISCLAIMER = (
    "This is general health information, not medical advice. "
    "Talk to a qualified clinician about your situation."
)

EMERGENCY_NOTICE = (
    "If you or someone else may be having a medical emergency, call 911 "
    "(or your local emergency number) now."
)

_EMERGENCY_PATTERNS = [
    r"chest (pain|pressure|tightness)",
    r"(can'?t|cannot|unable to|trouble|difficulty) breath",
    r"short(ness)? of breath",
    r"face (is )?droop",
    r"slurred speech",
    r"(sudden|worst) (severe )?headache",
    r"thunderclap",
    r"(unconscious|unresponsive|passed out|fainted)",
    r"seizure",
    r"(suicid|kill myself|end my life|overdos)",
    r"(lips|fingernails) (are |turning )?(blue|gray|grey)",
    r"vomiting blood",
]
_EMERGENCY_RE = re.compile("|".join(_EMERGENCY_PATTERNS), re.IGNORECASE)


def is_emergency(text: str) -> bool:
    return bool(_EMERGENCY_RE.search(text))
