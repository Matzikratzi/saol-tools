from __future__ import annotations

"""Find all bold and semibold SAOL lemma candidates inside grouped articles."""

import argparse
import difflib
import html
import json
import re
import statistics
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scripts.article_start_ml import DEFAULT_CACHE, extract_page


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FACIT = ROOT / "data" / "lemma_review_facit.json"
RULE_HITS: Counter[str] = Counter()


def rule_hit(name: str, count: int = 1) -> None:
    """Record that one generic parsing or correction rule affected output."""
    RULE_HITS[name] += count


def rule_stats() -> dict[str, int]:
    """Return stable, non-zero rule counters for reports and comparisons."""
    return dict(sorted(RULE_HITS.items()))


POS = {"adj", "adv", "interj", "prep", "pron", "s", "v"}
GRAMMAR_MARKERS = POS | {
    "best", "el", "eller", "komp", "oböjl", "pl", "superl", "äv",
}
# "-lt" is Tesseract's recurring reading of a thin separator plus adjective -t.
NON_LEMMA_SUFFIXES = {
    "-a", "-ad", "-ade", "-an", "-ar", "-are", "-at", "-de", "-dde",
    "-e", "-en", "-er", "-et", "-la", "-lt", "-na", "-n", "-or", "-r", "-ra", "-t",
    "-te",
}


def normalize_lemma(value: str) -> str:
    value = value.casefold().replace("|", "").replace("¦", "")
    value = re.sub(r"[^a-zåäöàáé -]+", "", value)
    return re.sub(r"\s+", " ", value).strip(" -")


def expand_compound(base: str, suffix: str) -> str:
    base = normalize_lemma(base)
    suffix = suffix.strip().strip(";,:.()")
    if not suffix.startswith("-"):
        return normalize_lemma(suffix)
    return normalize_lemma(base + suffix[1:])


def plural_of_previous(previous: str, candidate: str) -> bool:
    """Recognize an -a noun's plural, e.g. afrikanska -> afrikanskor."""
    previous = normalize_lemma(previous)
    candidate = normalize_lemma(candidate)
    return (
        len(previous) > 2
        and previous.endswith("a")
        and candidate == previous[:-1] + "or"
    )



def present_form_of_previous(previous: str, candidate: str) -> bool:
    """Recognize a safe full-token present form, e.g. affischera -> affischer."""
    previous = normalize_lemma(previous)
    candidate = normalize_lemma(candidate)
    return (
        len(previous) > 4
        and previous.endswith("era")
        and candidate == previous[:-1]
    )


def inflection_of_previous(previous: str, candidate: str) -> bool:
    """Recognize a full OCR token that is only an inflected previous lemma."""
    previous = normalize_lemma(previous)
    candidate = normalize_lemma(candidate)
    if not previous:
        return False
    suffix_inflection = any(
        candidate == previous + suffix[1:]
        for suffix in NON_LEMMA_SUFFIXES
    )
    return suffix_inflection or present_form_of_previous(previous, candidate)

def runeberg_short_inflection(raw: str, head: dict) -> bool:
    """Use aligned Runeberg grammar to reject a corrupt short inflection."""
    if (
        not any(character.isupper() for character in raw)
        or float(head.get("runeberg_match_score", 0.0)) < 0.80
    ):
        return False
    candidate = normalize_lemma(raw)
    if not candidate or len(candidate) > 4:
        return False
    suffixes = [
        normalize_lemma(value)
        for value in re.findall(
            r"-[A-Za-zÅÄÖåäöÀÁÉàáé]+\.?",
            head.get("runeberg_line", ""),
        )
    ]
    return any(
        suffix
        and len(suffix) <= 4
        and difflib.SequenceMatcher(
            None, candidate, suffix
        ).ratio() >= 0.60
        for suffix in suffixes
    )


def merged_pos_inflection(
    raw: str, normalized_suffix: str, bold_score: float
) -> bool:
    """Recognize e.g. '-ers.' or '-rs.' as inflection plus noun marker 's.'."""
    compact = raw.strip().casefold()
    return (
        normalized_suffix.endswith("s")
        and normalized_suffix[:-1] in NON_LEMMA_SUFFIXES
        and (compact.endswith("s.") or bold_score < 0.25)
    )


def weak_alternative_suffix(previous_raw: str, bold_score: float) -> bool:
    """Reject an ordinary-text '-suffix' after 'el.' or 'eller'."""
    return (
        normalize_lemma(previous_raw) in {"el", "eller"}
        and bold_score < 0.45
    )


def optional_parenthesis_variants(value: str) -> list[str]:
    """Include SAOL's parenthesized ending without inventing a short variant."""
    match = re.match(r"^(.*)\(([^()]*)\)$", value)
    if not match or not match.group(1) or not match.group(2):
        return [value]
    return [match.group(1) + match.group(2)]


def repair_initial_i_suffix_from_order(
    value: str,
    current_base: str,
    previous_lemma: str,
    following_tokens: list[dict],
) -> str:
    """Repair OCR -I... to -l... when both alphabetical neighbours prove it."""
    cleaned = value.strip().strip(";,:.()[]{}")
    if not cleaned.startswith("-I"):
        return value
    next_suffix = next(
        (
            token.get("text", "").strip().strip(";,:.()[]{}")
            for token in following_tokens
            if token.get("text", "").strip().strip(";,:.()[]{}").startswith("-")
        ),
        "",
    )
    if not next_suffix:
        return value
    previous = normalize_lemma(previous_lemma)
    wrong = expand_compound(current_base, cleaned)
    repaired = "-l" + cleaned[2:]
    corrected = expand_compound(current_base, repaired)
    following = expand_compound(current_base, next_suffix)
    if (
        previous
        and previous < corrected < following
        and not previous < wrong < following
    ):
        return repaired
    return value


def repair_mixed_case_duplicate(value: str) -> str:
    """Collapse an OCR duplicate such as -mMässighet to -Mässighet."""
    match = re.match(r"^(-?)([a-zåäö])([A-ZÅÄÖ])(.*)$", value)
    if match and match.group(2).casefold() == match.group(3).casefold():
        return match.group(1) + match.group(3) + match.group(4)
    return value


def repair_intrusion_before_boundary(value: str, article_head: str) -> str:
    """Remove an OCR-inserted j immediately before a visible or misread |."""
    cleaned = value.strip().strip(";,:.()[]{}")
    head = normalize_lemma(article_head)
    for marker in ("|", "¦"):
        if marker not in cleaned:
            continue
        base, tail = cleaned.split(marker, 1)
        normalized_base = normalize_lemma(base)
        if (
            normalized_base.endswith("j")
            and len(normalized_base) >= 5
            and head.startswith(normalized_base[:-1])
            and not head.startswith(normalized_base)
        ):
            return normalized_base[:-1] + "|" + normalize_lemma(tail)
        return cleaned

    normalized = normalize_lemma(cleaned)
    common_length = 0
    for left, right in zip(normalized, head):
        if left != right:
            break
        common_length += 1
    if (
        common_length >= 4
        and normalized[common_length : common_length + 2] == "jl"
        and head.startswith(normalized[:common_length])
    ):
        return (
            normalized[:common_length]
            + "|"
            + normalized[common_length + 2 :]
        )
    return cleaned


def repair_compacted_multiword_boundary(
    value: str, article_head: str
) -> str:
    """Recover a compact compound base derived from a spaced headword.

    Tesseract may render ``à jour|föra`` as ``åjourl|föra``: the accent is
    confused with å and a thin stroke immediately before | becomes l.
    """
    cleaned = value.strip().strip(";,:.()[]{}")
    head = normalize_lemma(article_head)
    if " " not in head or not any(marker in cleaned for marker in ("|", "¦")):
        return cleaned

    marker = "|" if "|" in cleaned else "¦"
    raw_base, tail = cleaned.split(marker, 1)
    expected = re.sub(r"[-\s]+", "", head).translate(
        str.maketrans({"à": "a", "á": "a", "é": "e"})
    )
    observed = normalize_lemma(raw_base).translate(
        str.maketrans({"à": "a", "á": "a", "å": "a", "é": "e"})
    )
    if observed.endswith("l") and observed[:-1] == expected:
        return expected + "|" + tail
    if observed == expected:
        return expected + "|" + tail
    return cleaned


def suffix_base(value: str) -> str:
    """Return the repeatable stem before SAOL's vertical boundary marker."""
    for marker in ("|", "¦"):
        if marker in value:
            return normalize_lemma(value.split(marker, 1)[0])
    return normalize_lemma(value)



def infer_era_boundary_from_verb_grammar(
    value: str, following_tokens: list[dict]
) -> str:
    """Recover ...er|a when the following grammar is '-ade v.'."""
    if len(following_tokens) < 2:
        return ""
    first = following_tokens[0].get("text", "").strip().strip(";,:.()[]{}")
    second = normalize_lemma(following_tokens[1].get("text", ""))
    if normalize_lemma(first) != "ade" or not first.startswith("-") or second != "v":
        return ""
    cleaned = value.strip().strip(";,:.()[]{}")
    if cleaned.casefold().endswith("erl|a"):
        return cleaned[:-3] + "|a"
    normalized = normalize_lemma(cleaned)
    if normalized.endswith("erla"):
        return normalized[:-2] + "|a"
    return ""


def infer_a_boundary_from_noun_grammar(
    value: str, following_tokens: list[dict]
) -> str:
    """Recover a final |a read as la before -an/-or noun grammar."""
    if not pronunciation_then_inflection(following_tokens):
        return ""
    suffixes = {
        token.get("text", "").strip().strip(";,:.()[]{}")
        for token in following_tokens
        if token.get("text", "").strip().startswith("-")
    }
    has_singular = "-an" in suffixes
    has_plural = any(suffix.startswith("-or") for suffix in suffixes)
    cleaned = value.strip().strip(";,:.()[]{}")
    normalized = normalize_lemma(cleaned)
    if (
        has_singular
        and has_plural
        and len(normalized) >= 5
        and normalized.endswith("la")
    ):
        return normalized[:-2] + "|a"
    return ""


def infer_boundary_from_previous(value: str, previous: str) -> str:
    """Recover previous|ending when OCR renders the boundary as l."""
    normalized = normalize_lemma(value)
    previous = normalize_lemma(previous)
    marker_prefix = previous + "l"
    if (
        len(previous) >= 3
        and normalized.startswith(marker_prefix)
        and len(normalized) > len(marker_prefix)
    ):
        return previous + "|" + normalized[len(marker_prefix):]
    return ""




def infer_suffix_boundary_from_series(value: str, prefix: str) -> str:
    """Recover -prefix|tail when OCR renders the series boundary as l."""
    cleaned = value.strip().strip(";,:.()[]{}")
    prefix = normalize_lemma(prefix)
    marker_prefix = "-" + prefix + "l"
    if (
        prefix
        and cleaned.casefold().startswith(marker_prefix)
        and len(cleaned) > len(marker_prefix)
    ):
        return "-" + prefix + "|" + cleaned[len(marker_prefix):]
    return ""

def infer_boundary_at_article_divergence(
    value: str, article_head: str, following_tokens: list[dict]
) -> str:
    """Recover a | read as l where a related word diverges before a suffix."""
    if (
        not following_tokens
        or not following_tokens[0].get("text", "").strip().startswith("-")
    ):
        return ""
    normalized = normalize_lemma(value)
    head = normalize_lemma(article_head)
    common_length = 0
    for left, right in zip(normalized, head):
        if left != right:
            break
        common_length += 1
    if (
        common_length >= 4
        and common_length < len(normalized) - 1
        and normalized[common_length] == "l"
    ):
        return (
            normalized[:common_length]
            + "|"
            + normalized[common_length + 1 :]
        )
    return ""


def infer_boundary_from_article_family(value: str, article_head: str) -> str:
    """Recover family|ending when OCR renders SAOL's boundary as l."""
    normalized = normalize_lemma(value)
    head = normalize_lemma(article_head)
    family = head[:-1] if len(head) > 5 else head
    marker_prefix = family + "l"
    if (
        len(family) >= 3
        and normalized.startswith(marker_prefix)
        and len(normalized) > len(marker_prefix)
    ):
        return family + "|" + normalized[len(marker_prefix):]
    return ""

def infer_boundary_from_repeated_suffix(
    value: str, following_tokens: list[dict]
) -> str:
    """Recover a | misread as l when a later dash repeats the printed tail."""
    normalized = normalize_lemma(value)
    suffixes = {
        normalize_lemma(
            token["text"].strip().strip(";,:.()[]{}")[1:]
        )
        for token in following_tokens
        if token["text"].strip().strip(";,:.()[]{}").startswith("-")
    }
    for index, character in enumerate(normalized):
        if character not in "li1":
            continue
        base = normalized[:index]
        tail = normalized[index + 1 :]
        exact_repeat = tail in suffixes
        pronounced_derivative = (
            len(tail) >= 3
            and bool(following_tokens)
            and following_tokens[0].get("text", "").strip().startswith("(")
            and any(
                suffix.startswith(tail) and len(suffix) > len(tail)
                for suffix in suffixes
            )
        )
        if len(base) >= 3 and (exact_repeat or pronounced_derivative):
            return base + "|" + tail
    return ""


def infer_compound_series_boundary(
    value: str, article_head: str, next_suffix: str
) -> str:
    """Recover a | misread as l when neighbouring compounds prove the order."""
    normalized = normalize_lemma(value)
    head = normalize_lemma(article_head)
    following = normalize_lemma(next_suffix)
    for base in (head + "s", head):
        marker_prefix = base + "l"
        if not normalized.startswith(marker_prefix):
            continue
        tail_with_l = normalized[len(base):]
        repaired_tail = normalized[len(marker_prefix):]
        if (
            repaired_tail
            and following
            and repaired_tail <= following < tail_with_l
        ):
            return base + "|" + repaired_tail
    return ""



def pronunciation_then_inflection(tokens: list[dict]) -> bool:
    """Recognize a new lemma followed by pronunciation and inflection."""
    if not tokens or not tokens[0].get("text", "").strip().startswith("("):
        return False
    depth = 0
    for index, token in enumerate(tokens):
        text = token.get("text", "").strip()
        depth += text.count("(") - text.count(")")
        if depth <= 0:
            return (
                index + 1 < len(tokens)
                and tokens[index + 1].get("text", "").strip().startswith("-")
            )
    return False


def inflections_then_part_of_speech(tokens: list[dict]) -> bool:
    """Recognize a lemma followed by one or more inflections and a POS marker."""
    if not tokens or not tokens[0].get("text", "").strip().startswith("-"):
        return False
    index = 0
    while index < len(tokens):
        raw = tokens[index].get("text", "").strip()
        if not raw.startswith("-"):
            break
        cleaned = raw.strip(";,:.()[]{}")
        normalized = "-" + normalize_lemma(cleaned[1:])
        if merged_pos_inflection(raw, normalized, 0.0):
            return True
        index += 1
    return (
        index < len(tokens)
        and normalize_lemma(tokens[index].get("text", "")) in POS
    )

def _token_score(token: dict, ordinary: float, bold: float) -> float:
    density = float(token.get("ink_density", 0.0))
    span = max(0.005, bold - ordinary)
    return max(0.0, min(1.0, (density - ordinary) / span))


def _references(payload: dict) -> tuple[float, float]:
    all_values = []
    head_values = []
    for article in payload["articles"]:
        for line in article["lines"]:
            all_values.extend(
                float(token.get("ink_density", 0.0))
                for token in line.get("tokens", [])
                if float(token.get("ink_density", 0.0)) > 0
            )
        tokens = sorted(
            article["lines"][0].get("tokens", []), key=lambda token: token["left"]
        )
        lexical = [
            token
            for token in tokens
            if re.search(r"[A-Za-zÅÄÖåäöÀÁÉàáé]", token.get("text", ""))
            and normalize_lemma(token.get("text", ""))
        ]
        if lexical:
            head_values.append(float(lexical[0].get("ink_density", 0.0)))
    ordinary = statistics.median(all_values) if all_values else 0.0
    bold = statistics.median(head_values) if head_values else ordinary + 0.05
    if bold <= ordinary:
        bold = ordinary + max(0.02, ordinary * 0.20)
    return ordinary, bold


def _after_inflection_prefix(tokens: list[dict]) -> list[dict]:
    for index, token in enumerate(tokens):
        plain = re.sub(r"[^a-zåäö]+", "", token["text"].casefold())
        if plain == "ss":
            plain = "s"
        if plain in POS:
            for alias_index in range(2, index):
                previous = normalize_lemma(
                    tokens[alias_index - 1].get("text", "")
                )
                alias = tokens[alias_index].get("text", "").strip()
                if (
                    previous in {"el", "eller", "äv"}
                    and normalize_lemma(alias)
                    and not alias.startswith("-")
                ):
                    return tokens[alias_index:]
            return tokens[index + 1 :]
    return tokens[1:]



def swedish_sort_key(value: str) -> str:
    """Return a comparison key using Swedish a–z, å, ä, ö order.

    Spaces and hyphens do not occupy alphabetic positions in SAOL.  Ignoring
    them also lets a following multiword headword bound ordinary prose from
    the preceding article, e.g. ajabaja < à jour < av.
    """
    normalized = normalize_lemma(value)
    translated = normalized.translate(
        str.maketrans(
            {
                "à": "a",
                "á": "a",
                "é": "e",
                "å": "{a",
                "ä": "{b",
                "ö": "{c",
            }
        )
    )
    return re.sub(r"[-\s]+", "", translated)


def same_lexical_family(headword: str, candidate: str) -> bool:
    """Recognize words that retain a useful prefix of the article headword."""
    headword = normalize_lemma(headword)
    candidate = normalize_lemma(candidate)
    required = min(4, len(headword))
    common = 0
    for left, right in zip(headword, candidate):
        if left != right:
            break
        common += 1
    return bool(required) and common >= required


def remove_alphabetic_family_outliers(
    items: list[dict], heads: dict[int, dict]
) -> list[dict]:
    """Drop non-family definition words between two lexical family anchors."""
    by_article: dict[int, list[dict]] = {}
    for item in items:
        by_article.setdefault(item["article_number"], []).append(item)

    rejected_ids = set()
    article_order = list(by_article)
    for article_index, article_number in enumerate(article_order):
        article_items = by_article[article_number]
        headword = normalize_lemma(heads[article_number]["headword"])
        previous_anchor = headword
        pending: list[dict] = []
        for item in article_items:
            lemma = normalize_lemma(item["lemma"])
            if (
                item["method"] == "artikelhuvud"
                or same_lexical_family(headword, lemma)
            ):
                if pending:
                    lower = swedish_sort_key(previous_anchor)
                    upper = swedish_sort_key(lemma)
                    for candidate in pending:
                        candidate_key = swedish_sort_key(
                            candidate["lemma"]
                        )
                        if (
                            lower > upper
                            or not lower <= candidate_key <= upper
                        ):
                            rejected_ids.add(id(candidate))
                            rule_hit("filter.alfabetisk_ordning")
                pending.clear()
                previous_anchor = lemma
            else:
                pending.append(item)

        if pending and article_index + 1 < len(article_order):
            next_article = article_order[article_index + 1]
            next_headword = normalize_lemma(
                heads[next_article]["headword"]
            )
            lower = swedish_sort_key(previous_anchor)
            upper = swedish_sort_key(next_headword)
            if lower <= upper:
                for candidate in pending:
                    candidate_key = swedish_sort_key(candidate["lemma"])
                    if not lower <= candidate_key <= upper:
                        rejected_ids.add(id(candidate))
                        rule_hit("filter.alfabetisk_ordning")

    return [item for item in items if id(item) not in rejected_ids]


def recover_runeberg_boundary_series(
    items: list[dict], heads: dict[int, dict]
) -> list[dict]:
    """Recover explicit lodstreck words from a well-aligned Runeberg line."""
    letters = "A-Za-zÅÄÖåäöÀÁÉàáé"
    boundary_pattern = re.compile(
        rf"(?<![{letters}-])-?[{letters}]*[|¦][{letters}]+"
    )
    token_pattern = re.compile(rf"-?[{letters}]+")
    for article_number, head in heads.items():
        if float(head.get("runeberg_match_score", 0.0)) < 0.80:
            continue
        runeberg_text = " ".join(
            head.get("runeberg_article_lines")
            or [head.get("runeberg_line", "")]
        )
        matches = list(boundary_pattern.finditer(runeberg_text))
        if not matches:
            continue
        article_items = [
            item for item in items
            if int(item["article_number"]) == int(article_number)
        ]
        head_item = next(
            (
                item for item in article_items
                if item.get("method") == "artikelhuvud"
            ),
            None,
        )
        if head_item is None:
            continue

        compound_base = suffix_base(
            head.get("runeberg_stem_headword")
            or head.get("stem_headword")
            or head["headword"]
        )
        anchor = head_item
        cursor = 0
        used_ids = set()
        for match in matches:
            for token_value in token_pattern.findall(
                runeberg_text[cursor : match.start()]
            ):
                possible = (
                    expand_compound(compound_base, token_value)
                    if token_value.startswith("-")
                    else normalize_lemma(token_value)
                )
                preceding = next(
                    (
                        item for item in article_items
                        if item["lemma"] == possible
                    ),
                    None,
                )
                if preceding is not None:
                    anchor = preceding

            raw = match.group(0)
            lemma = (
                expand_compound(compound_base, raw)
                if raw.startswith("-")
                else normalize_lemma(raw)
            )
            cursor = match.end()
            if lemma == normalize_lemma(head["headword"]):
                anchor = head_item
                continue

            recovered = next(
                (
                    item for item in article_items
                    if item["lemma"] == lemma and id(item) not in used_ids
                ),
                None,
            )
            if recovered is not None:
                recovered["stem_lemma"] = lemma
                recovered["raw"] = raw
            else:
                candidates = [
                    item
                    for item in article_items
                    if (
                        id(item) not in used_ids
                        and item.get("method") != "artikelhuvud"
                        and item.get("raw", "").strip().startswith("-")
                    )
                ]
                similar = max(
                    candidates,
                    key=lambda item: difflib.SequenceMatcher(
                        None, lemma, item["lemma"]
                    ).ratio(),
                    default=None,
                )
                similarity = (
                    difflib.SequenceMatcher(
                        None, lemma, similar["lemma"]
                    ).ratio()
                    if similar is not None
                    else 0.0
                )
                if similar is not None and similarity >= 0.82:
                    recovered = similar
                    recovered["lemma"] = lemma
                    recovered["stem_lemma"] = lemma
                    recovered["raw"] = raw
                    recovered["method"] = (
                        "Runebergkorrigerad lodstrecksserie"
                    )
                    recovered["reasons"] = [
                        reason
                        for reason in recovered.get("reasons", [])
                        if reason != "svag halvfetssignal"
                    ]
                    rule_hit("runeberg.korrigerad_lodstrecksserie")
                else:
                    recovered = head_item.copy()
                    recovered.update(
                        {
                            "lemma": lemma,
                            "stem_lemma": lemma,
                            "raw": raw,
                            "method": "Runebergs lodstrecksserie",
                            "bold_score": 0.0,
                            "status": "osäker",
                            "reasons": [
                                "saknades i bild-OCR; återställd från Runebergs parallella OCR"
                            ],
                        }
                    )
                    article_items.append(recovered)
                    rule_hit("runeberg.återställd_lodstrecksserie")

            used_ids.add(id(recovered))
            if recovered is anchor:
                continue
            anchor_index = next(
                (
                    index for index, item in enumerate(items)
                    if item is anchor
                ),
                None,
            )
            if anchor_index is None:
                anchor = head_item
                anchor_index = next(
                    index for index, item in enumerate(items)
                    if item is head_item
                )
            existing_index = next(
                (
                    index for index, item in enumerate(items)
                    if item is recovered
                ),
                None,
            )
            if existing_index is not None:
                items.pop(existing_index)
                if existing_index < anchor_index:
                    anchor_index -= 1
            if recovered.get("method") == "Runebergs lodstrecksserie":
                source_left = float(
                    anchor.get(
                        "source_right",
                        anchor.get("source_left", 0.0),
                    )
                ) + 12.0
                recovered.update(
                    {
                        "source_page": int(anchor.get("source_page", 0)),
                        "source_column": int(
                            anchor.get("source_column", 0)
                        ),
                        "source_top": float(anchor.get("source_top", 0.0)),
                        "source_bottom": float(
                            anchor.get("source_bottom", 0.0)
                        ),
                        "source_left": source_left,
                        "source_right": source_left
                        + max(60.0, len(recovered["lemma"]) * 16.0),
                    }
                )
            items.insert(anchor_index + 1, recovered)
            anchor = recovered
    return items


def extract_candidates(articles_payload: dict, heads_payload: dict) -> list[dict]:
    RULE_HITS.clear()
    heads = {item["article_number"]: item for item in heads_payload["headwords"]}
    ordinary, bold = _references(articles_payload)
    result = []
    seen = set()

    def add(article, lemma, raw, method, score, stem="", line=None, token=None):
        lemma = normalize_lemma(lemma)
        if not lemma:
            return
        key = (article["number"], lemma)
        if key in seen:
            return
        seen.add(key)
        reasons = []
        if method != "artikelhuvud" and score < 0.45:
            reasons.append("svag halvfetssignal")
        if len(lemma) == 1 and method != "artikelhuvud":
            reasons.append("ovanligt kort kandidat")
        source_line = line or {}
        source_token = token or {}
        source_left = float(source_token.get("left", 0.0))
        source_top = float(
            source_token.get(
                "top", source_line.get("top", article.get("start_y", 0.0))
            )
        )
        source_width = float(source_token.get("width", 0.0))
        line_height = max(
            1.0,
            float(source_line.get("bottom", source_top + 1.0))
            - float(source_line.get("top", source_top)),
        )
        source_height = float(source_token.get("height", line_height))
        result.append(
            {
                "article_number": article["number"],
                "homonym": heads[article["number"]].get("homonym"),
                "homonym_marker_detected": heads[article["number"]].get(
                    "homonym_marker_detected", False
                ),
                "page": article["start_page"],
                "column": article["start_column"],
                "lemma": lemma,
                "stem_lemma": stem or lemma,
                "raw": raw,
                "method": method,
                "bold_score": score,
                "status": "osäker" if reasons else "kandidat",
                "reasons": reasons,
                "source_page": int(source_line.get("page", article["start_page"])),
                "source_column": int(
                    source_line.get("column", article["start_column"])
                ),
                "source_left": source_left,
                "source_top": source_top,
                "source_right": float(
                    source_token.get("right", source_left + source_width)
                ),
                "source_bottom": float(
                    source_token.get("bottom", source_top + source_height)
                ),
            }
        )

    for article in articles_payload["articles"]:
        head = heads[article["number"]]
        structured_head = head.get("stem_headword", "")
        current_head = (
            normalize_lemma(structured_head)
            if "|" in structured_head or "¦" in structured_head
            else head["headword"]
        )
        current_base = suffix_base(structured_head or current_head)
        article_family = normalize_lemma(current_head)
        if len(article_family) > 5:
            article_family = article_family[:-1]
        suffix_series_prefix = ""
        last_lookup_lemma = normalize_lemma(current_head)
        first_line = article["lines"][0]
        head_tokens = sorted(first_line.get("tokens", []), key=lambda token: token["left"])
        head_token = next(
            (
                token
                for token in head_tokens
                if normalize_lemma(token.get("text", ""))
            ),
            head_tokens[0] if head_tokens else None,
        )
        add(
            article,
            current_head,
            head.get("raw_headword", current_head),
            "artikelhuvud",
            1.0,
            structured_head or current_head,
            first_line,
            head_token,
        )
        for line_index, line in enumerate(article["lines"]):
            tokens = sorted(line.get("tokens", []), key=lambda token: token["left"])
            if line_index == 0:
                tokens = _after_inflection_prefix(tokens)
            previous_separator = False
            at_line_start = True
            parenthesis_depth = 0
            for token_index, token in enumerate(tokens):
                raw = token["text"].strip()
                if not raw:
                    continue
                opens = raw.count("(")
                closes = raw.count(")")
                pronunciation_token = (
                    parenthesis_depth > 0 or raw.startswith("(")
                )
                if pronunciation_token:
                    parenthesis_depth = max(
                        0, parenthesis_depth + opens - closes
                    )
                    previous_separator = False
                    continue
                if raw in {"—", "–", "--"}:
                    previous_separator = True
                    continue
                score = _token_score(token, ordinary, bold)
                cleaned = raw.strip(";,:.()[]{}")
                if (
                    len(cleaned) > 1
                    and cleaned.startswith(("—", "–", "~"))
                ):
                    cleaned = "-" + cleaned[1:]
                same_line_following = list(tokens[token_index + 1 :])
                following_tokens = list(same_line_following)
                for following_line in article["lines"][line_index + 1 :]:
                    following_tokens.extend(
                        sorted(
                            following_line.get("tokens", []),
                            key=lambda candidate: candidate["left"],
                        )
                    )
                intrusion_boundary = repair_intrusion_before_boundary(
                    cleaned, current_head
                )
                multiword_boundary = repair_compacted_multiword_boundary(
                    cleaned, current_head
                )
                era_boundary = infer_era_boundary_from_verb_grammar(
                    cleaned, following_tokens
                )
                noun_a_boundary = infer_a_boundary_from_noun_grammar(
                    cleaned, following_tokens
                )
                previous_boundary = infer_boundary_from_previous(
                    cleaned, last_lookup_lemma
                )
                divergence_boundary = infer_boundary_at_article_divergence(
                    cleaned, current_head, same_line_following
                )
                family_boundary = infer_boundary_from_article_family(
                    cleaned, current_head
                )
                repeated_boundary = infer_boundary_from_repeated_suffix(
                    cleaned, same_line_following
                )
                if intrusion_boundary != cleaned:
                    cleaned = intrusion_boundary
                    rule_hit("repair.intrång_före_lodstreck")
                elif multiword_boundary != cleaned:
                    cleaned = multiword_boundary
                    rule_hit("repair.sammanskrivet_flerordsuttryck")
                elif era_boundary:
                    cleaned = era_boundary
                    rule_hit("infer.verb_era_lodstreck")
                elif noun_a_boundary:
                    cleaned = noun_a_boundary
                    rule_hit("infer.substantiv_a_lodstreck")
                elif previous_boundary:
                    cleaned = previous_boundary
                    rule_hit("infer.föregående_lemma_lodstreck")
                elif divergence_boundary:
                    cleaned = divergence_boundary
                    rule_hit("infer.artikelavvikelse_lodstreck")
                elif family_boundary:
                    cleaned = family_boundary
                    rule_hit("infer.artikelfamilj_lodstreck")
                elif repeated_boundary:
                    cleaned = repeated_boundary
                    rule_hit("infer.upprepad_suffix_lodstreck")
                series_position = line_index == 0 and at_line_start
                inferred = ""
                if series_position:
                    next_suffix = next(
                        (
                            candidate["text"].strip().strip(";,:.()[]{}")[1:]
                            for candidate in following_tokens
                            if candidate["text"].strip().strip(";,:.()[]{}").startswith("-")
                        ),
                        "",
                    )
                    inferred = infer_compound_series_boundary(
                        cleaned, current_head, next_suffix
                    )
                    if inferred:
                        cleaned = inferred
                        rule_hit("infer.sammansättningsserie_lodstreck")
                series_first = bool(inferred)
                lexical = bool(re.search(r"[A-Za-zÅÄÖåäöÀÁÉàáé]", cleaned))
                if not lexical:
                    previous_separator = False
                    at_line_start = False
                    continue
                if cleaned.startswith("-"):
                    if "[" in raw or "]" in raw:
                        rule_hit("filter.hakparentesböjning")
                        previous_separator = False
                        at_line_start = False
                        continue
                    previous_raw = (
                        tokens[token_index - 1].get("text", "")
                        if token_index > 0
                        else ""
                    )
                    if weak_alternative_suffix(previous_raw, score):
                        rule_hit("filter.svag_eller_suffix")
                        previous_separator = False
                        at_line_start = False
                        continue
                    before_initial_repair = cleaned
                    cleaned = repair_initial_i_suffix_from_order(
                        cleaned,
                        current_base,
                        last_lookup_lemma,
                        same_line_following,
                    )
                    if cleaned != before_initial_repair:
                        rule_hit("repair.initial_i_via_ordning")
                    following_series_prefix = next(
                        (
                            candidate["text"]
                            .strip()
                            .strip(";,:.()[]{}")
                            .split("|", 1)[0]
                            .split("¦", 1)[0]
                            .lstrip("-")
                            for candidate in following_tokens
                            if (
                                candidate["text"]
                                .strip()
                                .strip(";,:.()[]{}")
                                .startswith("-")
                                and (
                                    "|" in candidate["text"]
                                    or "¦" in candidate["text"]
                                )
                            )
                        ),
                        "",
                    )
                    inferred_suffix_boundary = (
                        infer_suffix_boundary_from_series(
                            cleaned,
                            suffix_series_prefix
                            or following_series_prefix,
                        )
                    )
                    if inferred_suffix_boundary:
                        cleaned = inferred_suffix_boundary
                        rule_hit("infer.suffixserie_lodstreck")
                    runeberg_inflection = (
                        line_index == 0
                        and score < 0.25
                        and runeberg_short_inflection(raw, head)
                    )
                    if runeberg_inflection:
                        rule_hit("filter.runeberg_kort_böjning")
                        previous_separator = False
                        at_line_start = False
                        continue
                    repaired_suffix = repair_mixed_case_duplicate(cleaned)
                    if repaired_suffix != cleaned:
                        rule_hit("repair.dubblerad_blandad_versal")
                    for marker in ("|", "¦"):
                        if marker in repaired_suffix:
                            suffix_series_prefix = repaired_suffix.split(
                                marker, 1
                            )[0].lstrip("-")
                            break
                    suffix_variants = optional_parenthesis_variants(
                        repaired_suffix
                    )
                    for suffix_variant in suffix_variants:
                        normalized_suffix = (
                            "-" + normalize_lemma(suffix_variant[1:])
                        )
                        suffix_word = normalize_lemma(suffix_variant[1:])
                        repeated_full_word = (
                            suffix_word
                            and normalize_lemma(current_base).endswith(
                                suffix_word
                            )
                        )
                        if (
                            normalized_suffix not in NON_LEMMA_SUFFIXES
                            and not merged_pos_inflection(
                                raw, normalized_suffix, score
                            )
                            and len(normalized_suffix) > 2
                            and not repeated_full_word
                        ):
                            lemma = expand_compound(
                                current_base, suffix_variant
                            )
                            if (
                                plural_of_previous(
                                    last_lookup_lemma, lemma
                                )
                                or inflection_of_previous(
                                    current_head, lemma
                                )
                            ):
                                continue
                            add(
                                article,
                                lemma,
                                cleaned,
                                "sammansättningssuffix",
                                score,
                                line=line,
                                token=token,
                            )
                            last_lookup_lemma = lemma
                            rule_hit("extract.sammansättningssuffix")
                    previous_separator = False
                    at_line_start = False
                    continue
                plausible_position = previous_separator or at_line_start
                clearly_semibold = score >= 0.70
                has_stem_boundary = "|" in cleaned or "¦" in cleaned
                has_lemma_grammar = pronunciation_then_inflection(
                    following_tokens
                )
                followed_by_pos = bool(
                    following_tokens
                    and normalize_lemma(
                        following_tokens[0].get("text", "")
                    ) in POS
                )
                preceded_by_pos = (
                    token_index > 0
                    and normalize_lemma(
                        tokens[token_index - 1].get("text", "")
                    ) in POS
                )
                normalized_cleaned = normalize_lemma(cleaned)
                normalized_head = normalize_lemma(current_head)
                capitalized_hyphenated_head = (
                    line_index == 0
                    and bool(
                        re.fullmatch(
                            r"[A-ZÅÄÖ]-[A-Za-zÅÄÖåäö]+",
                            cleaned,
                        )
                    )
                )
                first_line_family_word = (
                    line_index == 0
                    and preceded_by_pos
                    and len(normalized_head) >= 2
                    and len(normalized_cleaned) > len(normalized_head) + 2
                    and normalized_cleaned.startswith(normalized_head)
                )
                followed_by_inflection_grammar = (
                    bool(same_line_following)
                    and same_line_following[0].get(
                        "text", ""
                    ).strip().startswith("-")
                    and inflections_then_part_of_speech(
                        following_tokens
                    )
                )
                if (
                    (plausible_position and score >= 0.45)
                    or previous_separator
                    or clearly_semibold
                    or has_stem_boundary
                    or has_lemma_grammar
                    or followed_by_pos
                    or followed_by_inflection_grammar
                    or first_line_family_word
                    or capitalized_hyphenated_head
                    or series_first
                ):
                    lemma = normalize_lemma(cleaned)
                    bare_inflection_before_pos = (
                        followed_by_pos
                        and f"-{lemma}" in NON_LEMMA_SUFFIXES
                        and score < 0.25
                    )
                    embedded_head_inflection = (
                        followed_by_pos
                        and normalized_cleaned.startswith(
                            normalized_head + "-"
                        )
                        and (
                            "-"
                            + normalized_cleaned[
                                len(normalized_head) + 1 :
                            ]
                        )
                        in NON_LEMMA_SUFFIXES
                    )
                    # Only suppress a full token when the preceding lemma's
                    # morphology makes the reading unambiguous.  The broader
                    # suffix rule would wrongly remove real lemmas such as
                    # afghanska after afghansk and aforistiker after aforistik.
                    full_word_inflection = present_form_of_previous(
                        last_lookup_lemma, lemma
                    )
                    same_article_family = lemma.startswith(
                        article_family
                    )
                    unsupported_definition_before_inflection = (
                        followed_by_inflection_grammar
                        and not same_article_family
                        and not previous_separator
                        and not clearly_semibold
                        and not has_stem_boundary
                        and not has_lemma_grammar
                        and not followed_by_pos
                        and not series_first
                        and not (
                            plausible_position
                            and score >= 0.45
                        )
                    )
                    if (
                        lemma
                        and lemma not in GRAMMAR_MARKERS
                        and len(lemma) > 1
                        and not bare_inflection_before_pos
                        and not embedded_head_inflection
                        and not full_word_inflection
                        and not unsupported_definition_before_inflection
                    ):
                        add(
                            article, lemma, cleaned, "halvfet token",
                            score, line=line, token=token
                        )
                        rule_hit("extract.friliggande_lemma")
                        same_article_family = lemma.startswith(
                            article_family
                        )
                        structurally_new_base = (
                            previous_separator
                            or has_stem_boundary
                            or has_lemma_grammar
                            or followed_by_pos
                            or followed_by_inflection_grammar
                            or first_line_family_word
                            or capitalized_hyphenated_head
                            or series_first
                            or (
                                at_line_start
                                and score >= 0.45
                                and same_article_family
                            )
                        )
                        if structurally_new_base:
                            last_lookup_lemma = lemma
                            current_base = suffix_base(cleaned)
                previous_separator = False
                at_line_start = False
    recover_runeberg_boundary_series(result, heads)
    return remove_alphabetic_family_outliers(result, heads)


def facit_signature(item: dict) -> dict:
    """Stable interpretation stored for one approved candidate."""
    return {
        "article_number": int(item["article_number"]),
        "lemma": item["lemma"],
    }


def apply_manual_insertions(items: list[dict], facit: dict) -> list[dict]:
    """Insert printed lemmas that OCR omitted, at a facit-anchored position."""
    for insertion in facit.get("manual_insertions", []):
        anchor_signature = _signature_tuple(insertion["after"])
        candidate = insertion["candidate"]
        anchors = [
            index
            for index, item in enumerate(items)
            if _signature_tuple(item) == anchor_signature
        ]
        if not anchors:
            anchors = [
                index
                for index, item in enumerate(items)
                if item["lemma"] == insertion["after"]["lemma"]
            ]
        if len(anchors) != 1:
            continue
        anchor_index = anchors[0]
        anchor = items[anchor_index]
        if any(
            int(item["article_number"]) == int(anchor["article_number"])
            and item["lemma"] == candidate["lemma"]
            for item in items
        ):
            continue
        source_left = float(anchor.get("source_right", anchor.get("source_left", 0.0))) + 12.0
        inserted = anchor.copy()
        inserted.update(candidate)
        inserted["article_number"] = anchor["article_number"]
        inserted.update(
            {
                "homonym": anchor.get("homonym"),
                "homonym_marker_detected": anchor.get(
                    "homonym_marker_detected", False
                ),
                "page": anchor.get("page", anchor.get("source_page")),
                "column": anchor.get("column", anchor.get("source_column")),
                "method": "facitinsättning",
                "bold_score": 1.0,
                "status": "kandidat",
                "reasons": ["saknades i OCR; tillagd från korrekturfacit"],
                "source_page": int(anchor.get("source_page", 0)),
                "source_column": int(anchor.get("source_column", 0)),
                "source_top": float(anchor.get("source_top", 0.0)),
                "source_bottom": float(anchor.get("source_bottom", 0.0)),
                "source_left": source_left,
                "source_right": source_left
                + max(60.0, len(candidate["lemma"]) * 16.0),
            }
        )
        inserted.setdefault("stem_lemma", inserted["lemma"])
        inserted.setdefault("raw", inserted["lemma"])
        items.insert(anchor_index + 1, inserted)
        rule_hit("special.manuell_lemma_infogning")
    return items


def _signature_tuple(item: dict) -> tuple[int, str]:
    return int(item["article_number"]), item["lemma"]


def _physical_tuple(item: dict) -> tuple[int, int, float, float]:
    return (
        int(item.get("source_page", 0)),
        int(item.get("source_column", 0)),
        float(item.get("source_top", 0.0)),
        float(item.get("source_left", 0.0)),
    )


def _facit_match_key(item: dict) -> tuple:
    """Match reviewed words independently of run-local article numbering."""
    if int(item.get("source_page", 0)):
        return (*_physical_tuple(item), item["lemma"])
    return ("legacy", *_signature_tuple(item))


def _at_or_before_boundary(item: dict, boundary: dict) -> bool:
    """Compare an item with a stored physical boundary if its signature changed."""
    return _physical_tuple(item) <= _physical_tuple(boundary)


def _at_or_after_boundary(item: dict, boundary: dict) -> bool:
    return _physical_tuple(item) >= _physical_tuple(boundary)


def apply_review_facit(
    items: list[dict], facit: dict
) -> list[dict]:
    """Mark approved matches and return missing approved interpretations."""
    pages = facit.get("pages", {})
    missing = []
    for item in items:
        item["review_state"] = "unread"
    for page_text, page_data in pages.items():
        page = int(page_text)
        expected = {
            _signature_tuple(value)
            for value in page_data.get("candidates", [])
        }
        current_items = [
            item for item in items if int(item["source_page"]) == page
        ]
        current = {_signature_tuple(item) for item in current_items}
        for item in current_items:
            item["review_state"] = (
                "approved"
                if _signature_tuple(item) in expected
                else "facit_new"
            )
        for article_number, lemma in sorted(expected - current):
            missing.append(
                {
                    "page": page,
                    "article_number": article_number,
                    "lemma": lemma,
                }
            )

    reviewed_ranges = list(facit.get("reviewed_ranges", []))
    prefix = facit.get("reviewed_prefix")
    if prefix:
        reviewed_ranges.append(prefix)
    for reviewed_range in reviewed_ranges:
        boundary = reviewed_range["through"]
        expected_values = reviewed_range.get("candidates", [])
        start_boundary = (
            reviewed_range.get("from")
            or (expected_values[0] if expected_values else boundary)
        )
        boundary_signature = _facit_match_key(boundary)
        start_signature = _facit_match_key(start_boundary)
        boundary_indexes = [
            index
            for index, item in enumerate(items)
            if _facit_match_key(item) == boundary_signature
        ]
        start_indexes = [
            index
            for index, item in enumerate(items)
            if _facit_match_key(item) == start_signature
        ]
        if (
            len(start_indexes) == 1
            and len(boundary_indexes) == 1
            and start_indexes[0] <= boundary_indexes[0]
        ):
            current_items = items[
                start_indexes[0] : boundary_indexes[0] + 1
            ]
        else:
            current_items = [
                item
                for item in items
                if _at_or_after_boundary(item, start_boundary)
                if _at_or_before_boundary(item, boundary)
            ]
        expected = {_facit_match_key(value) for value in expected_values}
        current = {_facit_match_key(item) for item in current_items}
        for item in current_items:
            item["review_state"] = (
                "approved"
                if _facit_match_key(item) in expected
                else "facit_new"
            )
        stored_by_signature = {
            _facit_match_key(value): value for value in expected_values
        }
        for signature in sorted(expected - current):
            stored = stored_by_signature[signature]
            missing.append(
                {
                    "page": int(stored.get("source_page", 0)),
                    "article_number": int(stored["article_number"]),
                    "lemma": stored["lemma"],
                }
            )

    deduplicated = {
        (value["page"], value["article_number"], value["lemma"]): value
        for value in missing
    }
    return [deduplicated[key] for key in sorted(deduplicated)]


def approve_through(facit: dict, items: list[dict], lemma: str) -> dict:
    """Snapshot the exact interpreted sequence through one requested lemma."""
    requested = normalize_lemma(lemma)
    matches = [
        index
        for index, item in enumerate(items)
        if normalize_lemma(item["lemma"]) == requested
    ]
    if not matches:
        raise ValueError(f"grundformen finns inte i körningen: {requested}")
    if len(matches) > 1:
        articles = ", ".join(
            str(items[index]["article_number"]) for index in matches
        )
        raise ValueError(
            f"grundformen är inte entydig: {requested} (artiklar {articles})"
        )
    boundary_index = matches[0]
    existing = facit.get("reviewed_prefix", {})
    existing_values = existing.get("candidates", [])
    start_boundary = (
        existing.get("from")
        or (existing_values[0] if existing_values else None)
    )
    start_index = 0
    if start_boundary is not None:
        start_matches = [
            index
            for index, item in enumerate(items)
            if _facit_match_key(item) == _facit_match_key(start_boundary)
        ]
        if len(start_matches) == 1:
            start_index = start_matches[0]
    if start_index > boundary_index:
        raise ValueError("slutordet ligger före facitets startord")
    candidates = []
    for item in items[start_index : boundary_index + 1]:
        value = facit_signature(item)
        value.update(
            {
                "source_page": int(item.get("source_page", 0)),
                "source_column": int(item.get("source_column", 0)),
                "source_top": float(item.get("source_top", 0.0)),
                "source_left": float(item.get("source_left", 0.0)),
            }
        )
        candidates.append(value)
    facit["version"] = max(2, int(facit.get("version", 1)))
    facit["reviewed_prefix"] = {
        "from": candidates[0].copy(),
        "through": candidates[-1].copy(),
        "candidates": candidates,
    }
    return facit


def approve_pages(
    facit: dict, items: list[dict], pages: list[int]
) -> dict:
    """Snapshot the current interpretation for explicitly approved pages."""
    stored_pages = facit.setdefault("pages", {})
    for page in pages:
        candidates = [
            facit_signature(item)
            for item in items
            if int(item["source_page"]) == page
        ]
        candidates.sort(
            key=lambda value: (
                value["article_number"],
                value["lemma"],
            )
        )
        stored_pages[str(page)] = {"candidates": candidates}
    return facit


def display_lemma(item: dict) -> str:
    """Add review and homonym markers without changing the lemma itself."""
    prefix = ""
    if item.get("review_state") == "approved":
        prefix = "✓ "
    elif item.get("review_state") == "facit_new":
        prefix = "⚠ "
    homonym = item.get("homonym")
    if homonym is not None and item.get("method") == "artikelhuvud":
        return f"{prefix}[H{homonym}] {item['lemma']}"
    return prefix + item["lemma"]


def _review_font(size: int = 28):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _items_by_printed_row(items: list[dict]) -> list[list[dict]]:
    """Group slightly uneven OCR boxes into physical printed rows."""
    rows: list[list[dict]] = []
    for item in sorted(
        items,
        key=lambda value: (
            (value["source_top"] + value["source_bottom"]) / 2,
            value["source_left"],
        ),
    ):
        centre = (item["source_top"] + item["source_bottom"]) / 2
        height = max(1.0, item["source_bottom"] - item["source_top"])
        if rows:
            previous_centres = [
                (value["source_top"] + value["source_bottom"]) / 2
                for value in rows[-1]
            ]
            previous_height = max(
                max(1.0, value["source_bottom"] - value["source_top"])
                for value in rows[-1]
            )
            row_centre = statistics.median(previous_centres)
            if abs(centre - row_centre) <= max(height, previous_height) * 0.55:
                rows[-1].append(item)
                continue
        rows.append([item])
    return [
        sorted(row, key=lambda value: value["source_left"])
        for row in rows
    ]


def _items_in_reading_order(items: list[dict]) -> list[dict]:
    return [
        item
        for row in _items_by_printed_row(items)
        for item in row
    ]


def render_review_images(
    items: list[dict], pages: list[int], cache_dir: Path, image_dir: Path
) -> list[Path]:
    """Render candidate rows beside the same physical rows in the facsimile."""
    image_dir.mkdir(parents=True, exist_ok=True)
    font = _review_font()
    outputs = []
    separator = "  ·  "
    for page in pages:
        source = cache_dir / f"page-{page:04d}-deskewed.png"
        if not source.exists():
            continue
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        split = image.width // 2
        page_items = [item for item in items if item["source_page"] == page]
        for column, (left, right) in enumerate(((0, split), (split, image.width)), 1):
            crop = image.crop((left, 0, right, image.height))
            column_rows = _items_by_printed_row(
                [
                    item
                    for item in page_items
                    if item["source_column"] == column
                ]
            )
            measuring = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
            separator_box = measuring.textbbox((0, 0), separator, font=font)
            separator_width = separator_box[2] - separator_box[0]
            row_widths = []
            for row in column_rows:
                widths = [
                    measuring.textbbox((0, 0), display_lemma(item), font=font)[2]
                    for item in row
                ]
                row_widths.append(
                    sum(widths) + separator_width * max(0, len(widths) - 1)
                )
            margin = max(460, crop.width // 3, max(row_widths, default=0) + 40)
            canvas = Image.new("RGB", (crop.width + margin, crop.height), "white")
            canvas.paste(crop, (margin, 0))
            draw = ImageDraw.Draw(canvas)
            last_label_y = -100
            for row, row_width in zip(column_rows, row_widths):
                source_y = int(
                    statistics.median(
                        (item["source_top"] + item["source_bottom"]) / 2
                        for item in row
                    )
                )
                label_y = max(source_y - 15, last_label_y + 34)
                label_y = min(label_y, canvas.height - 34)
                last_label_y = label_y
                label_x = max(8, margin - 18 - row_width)
                for index, item in enumerate(row):
                    if item.get("review_state") == "approved":
                        color = "#777777"
                    elif (
                        item.get("review_state") == "facit_new"
                        or item["status"] == "osäker"
                    ):
                        color = "#c62828"
                    else:
                        color = "#00695c"
                    visible_lemma = display_lemma(item)
                    text_box = draw.textbbox(
                        (0, 0), visible_lemma, font=font
                    )
                    text_width = text_box[2] - text_box[0]
                    source_x = margin + int(
                        max(0, item["source_right"] - left)
                    )
                    draw.line(
                        (
                            label_x + text_width / 2,
                            label_y + 31,
                            source_x,
                            source_y,
                        ),
                        fill=color,
                        width=2,
                    )
                    draw.ellipse(
                        (
                            source_x - 4,
                            source_y - 4,
                            source_x + 4,
                            source_y + 4,
                        ),
                        fill=color,
                    )
                    draw.text(
                        (label_x, label_y),
                        visible_lemma,
                        font=font,
                        fill=color,
                    )
                    label_x += text_width
                    if index < len(row) - 1:
                        draw.text(
                            (label_x, label_y),
                            separator,
                            font=font,
                            fill="#777777",
                        )
                        label_x += separator_width
            output = image_dir / f"page-{page:04d}-column-{column}.png"
            canvas.save(output, format="PNG")
            outputs.append(output)
    return outputs

def report_html(
    items: list[dict],
    images: list[Path] | None = None,
    missing: list[dict] | None = None,
) -> str:
    missing = missing or []
    rows = []
    state_labels = {
        "approved": "✓ tidigare godkänd",
        "facit_new": "⚠ nytillkommen/ändrad",
        "unread": "oläst",
    }
    for item in items:
        state = item.get("review_state", "unread")
        if state == "facit_new":
            css = "mismatch"
        elif state == "approved":
            css = "approved"
        elif item["status"] == "osäker":
            css = "uncertain"
        else:
            css = ""
        rows.append(
            '<tr class="%s"><td>%d</td><td>%d:%d</td><td><b>%s</b></td>'
            '<td>%s</td><td>%s</td><td>%.2f</td><td><code>%s</code></td>'
            '<td>%s</td></tr>' % (
                css,
                item["article_number"],
                item["page"],
                item["column"],
                html.escape(display_lemma(item)),
                html.escape(state_labels.get(state, state)),
                html.escape(item["method"]),
                item["bold_score"],
                html.escape(item["raw"]),
                html.escape("; ".join(item["reasons"]) or "—"),
            )
        )
    uncertain = sum(item["status"] == "osäker" for item in items)
    unique = {item["lemma"] for item in items}
    approved = sum(
        item.get("review_state") == "approved" for item in items
    )
    changed = sum(
        item.get("review_state") == "facit_new" for item in items
    )
    missing_rows = "".join(
        "<li>Sida %d, artikel %d: <b>%s</b> saknas nu</li>"
        % (
            value["page"],
            value["article_number"],
            html.escape(value["lemma"]),
        )
        for value in missing
    )
    mismatch_block = (
        "<h2>Facitavvikelser</h2><ul>%s</ul>" % missing_rows
        if missing_rows
        else ""
    )
    first_review = next(
        (
            item
            for item in items
            if item.get("review_state", "unread") != "approved"
        ),
        None,
    )
    first_target = (
        (int(first_review["source_page"]), int(first_review["source_column"]))
        if first_review
        else None
    )
    jump_button = (
        '<p><a class="jump" href="#first-review">Gå till första nya eller '
        'felaktiga ordet: %s</a></p>'
        % html.escape(display_lemma(first_review))
        if first_review
        else ""
    )
    image_parts = []
    for path in images or []:
        match = re.search(r"page-(\d+)-column-(\d+)", path.stem)
        figure_id = ""
        review_anchor = ""
        if match:
            path_target = (int(match.group(1)), int(match.group(2)))
            figure_id = ' id="review-page-%d-column-%d"' % path_target
            if path_target == first_target:
                if path.exists():
                    with Image.open(path) as review_image:
                        image_height = max(1, review_image.height)
                else:
                    image_height = max(
                        1.0,
                        float(first_review.get("source_bottom", 0.0)),
                        float(first_review.get("source_top", 0.0)) + 1.0,
                    )
                anchor_top = max(
                    0.0,
                    min(
                        100.0,
                        100.0
                        * float(first_review["source_top"])
                        / image_height,
                    ),
                )
                review_anchor = (
                    '<span id="first-review" class="review-anchor" '
                    'style="top:%.3f%%"></span>' % anchor_top
                )
        image_parts.append(
            '<figure%s>%s<img src="%s" loading="lazy"><figcaption>%s</figcaption></figure>'
            % (
                figure_id,
                review_anchor,
                html.escape(str(path)),
                html.escape(path.stem.replace("-", " ")),
            )
        )
    image_blocks = "".join(image_parts)
    return f"""<!doctype html><html lang="sv"><head><meta charset="utf-8">
<title>SAOL – grundformskandidater</title><style>
body{{font:15px system-ui;margin:24px;max-width:1800px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}}th{{position:sticky;top:0;background:#eee}}tr.uncertain{{background:#fff3cd}}tr.approved{{background:#eee;color:#666}}tr.mismatch{{background:#ffd6d6}}code{{white-space:pre-wrap}}figure{{margin:24px 0;border:1px solid #aaa;padding:10px;background:#eee}}figure img{{display:block;width:100%;height:auto}}figcaption{{margin-top:6px;color:#555}}.jump{{display:inline-block;padding:10px 14px;background:#1769e0;color:white;text-decoration:none;border-radius:7px;font-weight:700}}figure[id]{{position:relative}}.review-anchor{{position:absolute;left:0;scroll-margin-top:12px}}
</style></head><body><h1>Grundformskandidater</h1>
<p>{len(items)} träffar; {len(unique)} unika grundformer; {uncertain} osäkra kandidater.</p>
<p>{approved} poster stämmer med tidigare facit. {changed + len(missing)} facitavvikelser.</p>
<p>Grått med ✓ är redan godkänt och oförändrat. Rött med ⚠ är nytt eller ändrat på en godkänd sida. Olästa poster visas som tidigare.</p>
{jump_button}
{mismatch_block}
{image_blocks}
<h2>Alla kandidater som tabell</h2>
<table><thead><tr><th>Artikel</th><th>Sida:spalt</th><th>Grundform</th><th>Granskning</th><th>Metod</th><th>Fet</th><th>OCR</th><th>Anmärkning</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""


def write_review_bundle(bundle: Path, paths: list[Path]) -> Path:
    """Package the three review JSON files for convenient sharing."""
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        bundle, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--articles",
        type=Path,
        default=Path("article-text-review.json"),
    )
    parser.add_argument(
        "--headwords",
        type=Path,
        default=Path("headword-review.json"),
    )
    parser.add_argument("--json", type=Path, default=Path("lemma-review.json"))
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("saol-review-json.zip"),
        help="ZIP med artikel-, huvudords- och lemma-JSON",
    )
    parser.add_argument(
        "--report", type=Path, default=Path("lemma-review.html")
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--image-dir", type=Path, default=Path("lemma-review-pages")
    )
    parser.add_argument("--facit", type=Path, default=DEFAULT_FACIT)
    parser.add_argument(
        "--approve-page",
        type=int,
        action="append",
        default=[],
        help="Spara aktuell tolkning av sidan som korrekturläst facit",
    )
    parser.add_argument(
        "--approve-through",
        metavar="GRUNDFORM",
        help="Spara aktuell tolkning till och med grundformen som facit",
    )
    args = parser.parse_args()
    if args.approve_page and args.approve_through:
        parser.error("--approve-page och --approve-through kan inte kombineras")
    articles = json.loads(args.articles.read_text(encoding="utf-8"))
    heads = json.loads(args.headwords.read_text(encoding="utf-8"))
    for page in articles.get("pages", []):
        extract_page(page, args.cache_dir, False)
    items = extract_candidates(articles, heads)

    if args.facit.exists():
        facit = json.loads(args.facit.read_text(encoding="utf-8"))
    else:
        facit = {"version": 1, "pages": {}}
    apply_manual_insertions(items, facit)
    available_pages = {int(page) for page in articles.get("pages", [])}
    unknown_pages = set(args.approve_page) - available_pages
    if unknown_pages:
        parser.error(
            "kan inte godkänna sidor som inte ingår i körningen: "
            + ", ".join(map(str, sorted(unknown_pages)))
        )
    if args.approve_page:
        approve_pages(facit, items, args.approve_page)
    if args.approve_through:
        try:
            approve_through(facit, items, args.approve_through)
        except ValueError as error:
            parser.error(str(error))
    if args.approve_page or args.approve_through:
        args.facit.parent.mkdir(parents=True, exist_ok=True)
        args.facit.write_text(
            json.dumps(facit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.approve_page:
        print(
            "Facit sparat för sida: "
            + ", ".join(map(str, sorted(set(args.approve_page))))
        )
    if args.approve_through:
        print(
            "Facit sparat till och med: "
            + facit["reviewed_prefix"]["through"]["lemma"]
        )
    missing = apply_review_facit(items, facit)
    images = render_review_images(
        items, articles.get("pages", []), args.cache_dir, args.image_dir
    )
    facit_new_count = sum(
        item.get("review_state") == "facit_new" for item in items
    )
    output = {
        "candidate_count": len(items),
        "unique_lemma_count": len({item["lemma"] for item in items}),
        "uncertain_count": sum(
            item["status"] == "osäker" for item in items
        ),
        "approved_pages": sorted(
            int(page) for page in facit.get("pages", {})
        ),
        "approved_through": (
            facit.get("reviewed_prefix", {}).get("through", {}).get("lemma")
        ),
        "facit_new_count": facit_new_count,
        "facit_missing": missing,
        "rule_stats": rule_stats(),
        "candidates": items,
    }
    args.json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.report.write_text(
        report_html(items, images, missing),
        encoding="utf-8",
    )
    write_review_bundle(
        args.bundle,
        [args.articles, args.headwords, args.json],
    )
    print(f"Kandidater: {output['candidate_count']}")
    print(f"Unika grundformer: {output['unique_lemma_count']}")
    print(f"Osäkra: {output['uncertain_count']}")
    print(
        "Godkända sidor: "
        + (
            ", ".join(map(str, output["approved_pages"]))
            if output["approved_pages"]
            else "—"
        )
    )
    print(
        "Godkänt till och med: "
        + (output["approved_through"] or "—")
    )
    print(f"Facitavvikelser: {facit_new_count + len(missing)}")
    print("Regelträffar:")
    for name, count in output["rule_stats"].items():
        print(f"  {name}: {count}")
    for value in missing:
        print(
            "  SAKNAS sida=%d artikel=%d lemma=%s"
            % (
                value["page"],
                value["article_number"],
                value["lemma"],
            )
        )
    print(f"Data: {args.json.resolve()}")
    print(f"Rapport: {args.report.resolve()}")
    print(f"Bildsidor: {args.image_dir.resolve()}")
    print(f"ZIP för uppladdning: {args.bundle.resolve()}")


if __name__ == "__main__":
    main()
