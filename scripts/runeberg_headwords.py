from __future__ import annotations

"""Secondary headword evidence from Project Runeberg's independent OCR."""

import difflib
import re

from scripts import debug_runeberg_ocr as debug


STOP_WORDS = {
    "adj", "adv", "best", "el", "eller", "i", "interj", "jfr", "komp",
    "ei", "mus", "n", "oböjl", "pl", "prep", "pron", "s", "se", "ss", "subst",
    "v", "vard", "äv", "åld",
}


def _line_key(value: str) -> str:
    value = value.casefold().replace("|", "")
    return "".join(character for character in value if character.isalnum())


def raw_headword(value: str) -> str:
    selected = []
    for token in value.split():
        text = token.strip()
        if not selected:
            text = text.lstrip("^'\"“”„`´•123456789")
        plain = re.sub(r"[^a-zåäöàáé]+", "", text.casefold())
        if selected and (
            text.startswith(("(", "[", "-", "—", "–"))
            or plain in STOP_WORDS
            or text[0].isdigit()
        ):
            break
        if not selected and not plain:
            continue
        if not selected and plain in STOP_WORDS:
            return ""
        selected.append(text)
    value = " ".join(selected).casefold().replace("|", "").replace("¦", "")
    value = re.sub(r"[^a-zåäöàáé0-9 -]+", "", value)
    return re.sub(r"\s+", " ", value).strip(" -")


def align_lines(items: list[dict], raw_lines: list[str]) -> list[tuple[int, float]]:
    """Monotonically align article starts to Runeberg OCR lines."""
    count, line_count = len(items), len(raw_lines)
    negative = -1e9
    scores = [[negative] * (line_count + 1) for _ in range(count + 1)]
    choices = [[False] * (line_count + 1) for _ in range(count + 1)]
    for column in range(line_count + 1):
        scores[0][column] = 0.0
    for item_index in range(1, count + 1):
        source = _line_key(items[item_index - 1]["source_line"])
        for line_index in range(1, line_count + 1):
            skip = scores[item_index][line_index - 1]
            candidate = _line_key(raw_lines[line_index - 1])
            similarity = difflib.SequenceMatcher(None, source, candidate).ratio()
            match = scores[item_index - 1][line_index - 1] + similarity
            if match > skip:
                scores[item_index][line_index] = match
                choices[item_index][line_index] = True
            else:
                scores[item_index][line_index] = skip
    result = []
    item_index, line_index = count, line_count
    while item_index:
        if line_index <= 0:
            raise ValueError("Kunde inte linjera artiklar mot Runebergs OCR")
        if choices[item_index][line_index]:
            source = _line_key(items[item_index - 1]["source_line"])
            candidate = _line_key(raw_lines[line_index - 1])
            score = difflib.SequenceMatcher(None, source, candidate).ratio()
            result.append((line_index - 1, score))
            item_index -= 1
            line_index -= 1
        else:
            line_index -= 1
    return list(reversed(result))


def fetch_and_enrich(items: list[dict]) -> None:
    module = debug._load_base_module()
    headers = {"User-Agent": "saol-tools/headword-review"}
    for page in sorted({item["page"] for item in items}):
        page_items = [item for item in items if item["page"] == page]
        source_url, _image_url = module.page_urls(page)
        response = module.httpx.get(
            source_url, timeout=60.0, follow_redirects=True, headers=headers
        )
        response.raise_for_status()
        raw_lines = [
            line.strip()
            for line in module._runeberg_ocr_text(response.text).splitlines()
            if line.strip()
        ]
        for item, (line_index, score) in zip(page_items, align_lines(page_items, raw_lines)):
            raw_line = raw_lines[line_index]
            secondary = raw_headword(raw_line)
            item["runeberg_line"] = raw_line
            item["runeberg_headword"] = secondary
            item["runeberg_match_score"] = score
            if score < 0.62 or not secondary:
                continue
            missing = not item["headword"]
            explicit_boundary = "|" in raw_line
            only_low_confidence = item["reasons"] and all(
                reason.startswith("låg OCR-säkerhet") for reason in item["reasons"]
            )
            same_headword = secondary == item["headword"]
            if same_headword and score >= 0.85:
                item["reasons"] = [
                    reason for reason in item["reasons"]
                    if not reason.startswith("låg OCR-säkerhet")
                ]
                item["status"] = "osäker" if item["reasons"] else "preliminär"
                continue
            headword_similarity = difflib.SequenceMatcher(
                None, item["headword"], secondary
            ).ratio()
            strong_boundary_correction = (
                explicit_boundary and score >= 0.75 and headword_similarity >= 0.70
            )
            strong_single_word_correction = (
                only_low_confidence
                and score >= 0.85
                and " " not in secondary
            )
            if missing or strong_boundary_correction or strong_single_word_correction:
                old = item["headword"]
                item["corrected_from"] = old
                item["correction_method"] = "Runebergs parallella OCR"
                item["headword"] = secondary
                item["reasons"] = []
                item["status"] = "preliminär"
