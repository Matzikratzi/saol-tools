from __future__ import annotations

import re

HEADWORD = re.compile(r"^[\^]?([A-Za-zÅÄÖåäöÀÁÉàáé][A-Za-zÅÄÖåäöÀÁÉàáé-]*)")
VALID_WORD = re.compile(r"^[a-zåäöàáé-]+$", re.IGNORECASE)

RUNEberg_METADATA = re.compile(
    r"(?:Below is the raw OCR text|Do you see an error|Proofread the page now|"
    r"Här nedan syns maskintolkade texten|Ser du något fel|Korrekturläs sidan nu|"
    r"This page has .*proofread|Denna sida har .*korrekturlästs|Project Runeberg)",
    re.IGNORECASE,
)


def normalize_word(word: str) -> str:
    return word.strip().lstrip("^").lower()


def candidate_words(text: str) -> list[str]:
    """Extract cautious headword candidates from Runeberg OCR."""
    words: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.isdigit() or len(line) > 180:
            continue
        # Defence in depth: metadata must never become candidate words even if
        # Runeberg changes its HTML and the earlier OCR cleanup misses a line.
        if RUNEberg_METADATA.search(line):
            continue
        match = HEADWORD.match(line)
        if not match:
            continue
        word = normalize_word(match.group(1))
        if len(word) < 2 or word in seen:
            continue
        seen.add(word)
        words.append(word)
    return words


def suspicious_word(word: str) -> bool:
    word = normalize_word(word)
    if not VALID_WORD.fullmatch(word):
        return True
    if "iii" in word or "lll" in word or "vvv" in word:
        return True
    if word.startswith("-") or word.endswith("-") or "--" in word:
        return True
    return False
