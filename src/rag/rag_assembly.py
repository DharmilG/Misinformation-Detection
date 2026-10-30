"""Bounded prompt assembly; retrieved text remains untrusted user data."""
from __future__ import annotations

from dataclasses import dataclass
import re

try:
    from .rag_prompts import SYSTEM_PROMPT, format_evidence_passages
except ImportError:
    from rag_prompts import SYSTEM_PROMPT, format_evidence_passages


@dataclass(frozen=True)
class AssembledPrompt:
    system_prompt: str
    user_prompt: str
    token_count: int
    truncated: bool
    omitted_passages: int = 0


def _words(text: str) -> list[str]:
    return text.split()


def assemble_prompt(query: str, retrieval_result: dict, max_tokens: int = 2048, classifier_context: str = ""):
    passages = list(retrieval_result.get("passages", ()))
    if not passages:
        return None
    selected: list[dict] = []
    truncated = False
    omitted = 0
    if max_tokens <= 500 and len(passages) > 10:
        omitted = len(passages) - 10
        passages = passages[:10]
        truncated = True
    base = f"<claim>{query}</claim>\n{classifier_context}\n"
    budget = max(1, max_tokens - len(_words(base)) - 8)
    used = 0
    for passage in passages:
        text = str(passage.get("text", passage.get("excerpt", "")))
        words = _words(text)
        remaining = budget - used
        if remaining <= 0:
            omitted += 1
            truncated = True
            continue
        if len(words) > remaining:
            passage = dict(passage)
            passage["text"] = " ".join(words[:remaining])
            truncated = True
            omitted += 1
        selected.append(passage)
        used += len(_words(str(passage.get("text", passage.get("excerpt", "")))))
    user = base + format_evidence_passages(selected)
    count = len(_words(SYSTEM_PROMPT)) + len(_words(user))
    if count > max_tokens:
        user_words = _words(user)[:max_tokens - len(_words(SYSTEM_PROMPT))]
        user = " ".join(user_words)
        count = max_tokens
        truncated = True
    return AssembledPrompt(SYSTEM_PROMPT, user, count, truncated, omitted)
