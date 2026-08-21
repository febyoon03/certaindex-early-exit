"""Answer extraction — one place, not copied per script.

The original addition scripts scored TRUE/FALSE against numeric labels, so
baseline accuracy was structurally 0 even when the model wrote ANSWER: 37.
"""

from __future__ import annotations

import re
from typing import Any

TRUE_FALSE_RE = re.compile(r"\b(TRUE|FALSE)\b", re.IGNORECASE)
FINAL_RE = re.compile(r"FINAL:\s*(TRUE|FALSE)", re.IGNORECASE)
ANSWER_INT_RE = re.compile(r"ANSWER:\s*(-?\d+)")
COUNT_RE = re.compile(r"COUNT:\s*(-?\d+)")
BARE_INT_RE = re.compile(r"-?\d+")


def extract_tf(text: str) -> str | None:
    if not text:
        return None
    m = FINAL_RE.search(text)
    if m:
        return m.group(1).upper()
    matches = TRUE_FALSE_RE.findall(text)
    if matches:
        return matches[-1].upper()
    return None


def extract_int(text: str) -> int | None:
    """Prefer ANSWER:, then COUNT:, then the first integer."""
    if not text:
        return None
    for pattern in (ANSWER_INT_RE, COUNT_RE):
        m = pattern.search(text)
        if m:
            return int(m.group(1))
    m = BARE_INT_RE.search(text)
    return int(m.group(0)) if m else None


def extract_answer(text: str, kind: str) -> Any:
    if kind == "tf":
        return extract_tf(text)
    if kind == "int":
        return extract_int(text)
    raise ValueError(f"unknown extract kind: {kind!r}")


def answer_complete_int(text: str) -> tuple[int | None, bool]:
    """Parse-stop accept rule.

    ANSWER_INT_RE is greedy on digits, so a match that reaches the end of
    the buffer may still be mid-number. We accept only when a non-digit
    has arrived after the match, or when the caller says the stream ended.
    """
    if not text:
        return None, False
    m = ANSWER_INT_RE.search(text)
    if not m:
        return None, False
    value = int(m.group(1))
    terminated = m.end() < len(text)
    return value, terminated
