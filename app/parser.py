from __future__ import annotations

import re

VALID_WORD = re.compile(r"^[a-zåäöàáé-]+$", re.IGNORECASE)
RUNEberg_METADATA = re.compile(
    r"(?:Below is the raw OCR text|Do you see an error|Proofread the page now|"
    r"Här nedan syns maskintolkade texten|Ser du något fel|Korrekturläs sidan nu|"
    r"This page has .*proofread|Denna sida har .*korrekturlästs|Project Runeberg)",
    re.IGNORECASE,
)
SUPERSCRIPT_TRANSLATION = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
SUPERSCRIPT_PREFIX = re.compile(r"^([⁰¹²³⁴⁵⁶⁷⁸⁹]+)\s*(.*)$")
FLAT_SENSE_PREFIX = re.compile(r"^([1-9][0-9]?)\s*([a-zåäöàáé].*)$", re.IGNORECASE)
APOSTROPHE_ONE_PREFIX = re.compile(r"^[\'’`´]\s*([a-zåäöàáé].*)$", re.IGNORECASE)

# Very small raised digits are repeatedly misread in the first SAOL articles.
# These aliases are only interpreted by split_headword_marker(), i.e. while a
# token is already being considered as an article-start token. They are not
# global spelling corrections and therefore cannot silently alter article text.
OCR_HEADWORD_ALIASES: dict[str, tuple[int, str]] = {
    "åå": (2, "a"),   # printed ²a
    "'&": (3, "a"),  # printed ³a
    "’&": (3, "a"),
    "`&": (3, "a"),
    "´&": (3, "a"),
}

ARTICLE_LABELS = {
    "best.",
    "obest.",
    "pl.",
    "sing.",
    "pres.",
    "pret.",
    "perf.",
    "sup.",
    "imp.",
    "inf.",
    "part.",
    "komp.",
    "superl.",
    "neutr.",
    "mask.",
    "fem.",
    "gen.",
    "dat.",
    "ack.",
    "äv.",
    "el.",
}


def normalize_word(word: str) -> str:
    return re.sub(r"\s+", " ", word.strip().lstrip("^")).lower()


def _is_article_label(text: str) -> bool:
    normalized = normalize_word(text)
    return normalized in ARTICLE_LABELS or normalized.endswith(".")


def split_headword_marker(text: str) -> tuple[int | None, str]:
    """Split a printed homonym number from a headword without losing either.

    Tesseract sometimes reads a tiny raised ``¹`` as an apostrophe, producing
    tokens such as ``'a`` or ``’a``. It also repeatedly reads the following
    printed entries ``²a`` and ``³a`` as ``åå`` and ``'&``. Since this function
    is called only for an article-start candidate, those known glyph confusions
    can safely be converted to homonym metadata here.
    """
    normalized = normalize_word(text)
    if _is_article_label(normalized):
        return None, ""

    alias = OCR_HEADWORD_ALIASES.get(normalized)
    if alias is not None:
        return alias

    match = SUPERSCRIPT_PREFIX.match(normalized)
    if match:
        word = normalize_word(match.group(2))
        if _is_article_label(word):
            return None, ""
        return int(match.group(1).translate(SUPERSCRIPT_TRANSLATION)), word

    match = APOSTROPHE_ONE_PREFIX.match(normalized)
    if match:
        word = normalize_word(match.group(1))
        if _is_article_label(word):
            return None, ""
        return 1, word

    match = FLAT_SENSE_PREFIX.match(normalized)
    if match:
        word = normalize_word(match.group(2))
        if _is_article_label(word):
            return None, ""
        return int(match.group(1)), word

    return None, normalized


def normalize_forms(forms: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in forms:
        for part in re.split(r"[,;\n]+", value):
            form = normalize_word(part)
            if not form or form in seen:
                continue
            seen.add(form)
            result.append(form)
    return result


def candidate_words(text: str) -> list[str]:
    words: list[str] = []
    seen: set[tuple[str, int | None]] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.isdigit() or len(line) > 180:
            continue
        if RUNEberg_METADATA.search(line):
            continue
        sense_number, word = split_headword_marker(line)
        key = (word, sense_number)
        if not word or key in seen:
            continue
        seen.add(key)
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
