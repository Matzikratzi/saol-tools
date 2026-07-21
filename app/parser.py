from __future__ import annotations

import re

VALID_WORD = re.compile(r"^[a-zåäöàáé-]+$", re.IGNORECASE)

RUNEberg_METADATA = re.compile(
    r"(?:Below is the raw OCR text|Do you see an error|Proofread the page now|"
    r"Här nedan syns maskintolkade texten|Ser du något fel|Korrekturläs sidan nu|"
    r"This page has .*proofread|Denna sida har .*korrekturlästs|Project Runeberg)",
    re.IGNORECASE,
)


def normalize_word(word: str) -> str:
    return re.sub(r"\s+", " ", word.strip().lstrip("^")).lower()


def candidate_words(text: str) -> list[str]:
    """Normalize already selected bold headword groups.

    The image OCR selects typographically bold spans first. A span may be a
    compound fragment or a multiword headword, so it must not be truncated to
    the first token.
    """
    words: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.isdigit() or len(line) > 180:
            continue
        if RUNEberg_METADATA.search(line):
            continue
        word = normalize_word(line)
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
