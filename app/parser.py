from __future__ import annotations

import re

VALID_WORD = re.compile(r"^[a-zåäöàáé-]+$", re.IGNORECASE)

RUNEberg_METADATA = re.compile(
    r"(?:Below is the raw OCR text|Do you see an error|Proofread the page now|"
    r"Här nedan syns maskintolkade texten|Ser du något fel|Korrekturläs sidan nu|"
    r"This page has .*proofread|Denna sida har .*korrekturlästs|Project Runeberg)",
    re.IGNORECASE,
)

SUPERSCRIPT_SENSE_PREFIX = re.compile(r"^[⁰¹²³⁴⁵⁶⁷⁸⁹]+\s*")
OCR_SENSE_PREFIX = re.compile(r"^[1-9](?=[a-zåäöàáé])", re.IGNORECASE)


def normalize_word(word: str) -> str:
    normalized = re.sub(r"\s+", " ", word.strip().lstrip("^")).lower()
    # SAOL prints a raised number before homonymous headwords, for example
    # ¹a, ²a and ³a. Tesseract may preserve the superscript glyph or flatten it
    # to an ordinary digit (1a). The number marks meaning/homonymy and is not
    # part of the headword itself.
    normalized = SUPERSCRIPT_SENSE_PREFIX.sub("", normalized)
    normalized = OCR_SENSE_PREFIX.sub("", normalized)
    return normalized.strip()


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
        if len(word) < 1 or word in seen:
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
