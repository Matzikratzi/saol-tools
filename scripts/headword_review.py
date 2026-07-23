from __future__ import annotations

"""Extract provisional headwords and homonym numbers from grouped articles.

Run ``article_text_review.py`` first so its JSON contains token-level OCR data.
This stage is deliberately conservative: uncertain or missing heads are listed
for review instead of being silently invented.
"""

import argparse
import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORRECTIONS = ROOT / "data" / "headword_corrections.json"
MARKERS = set("'\"“”„`´¹²³⁴⁵⁶⁷⁸⁹")
SUPERSCRIPT = {character: index for index, character in enumerate("¹²³⁴⁵⁶⁷⁸⁹", 1)}
STOP_WORDS = {
    "adj", "adv", "best", "el", "eller", "i", "interj", "jfr", "komp",
    "mus", "n", "oböjl", "pl", "prep", "pron", "s", "se", "ss", "subst",
    "v", "vard", "äv", "åld",
}


def normalize_headword(value: str) -> str:
    value = value.casefold().replace("|", "").replace("¦", "")
    value = re.sub(r"[^a-zåäöàáé0-9 -]+", "", value)
    return re.sub(r"\s+", " ", value).strip(" -")


def _plain_token(value: str) -> str:
    return normalize_headword(value).replace("-", "")


def swedish_sort_key(value: str) -> tuple[int, ...]:
    """Collation key where accented a/e sort with a/e and å follows z."""
    value = normalize_headword(value).translate(str.maketrans({"à": "a", "á": "a", "é": "e"}))
    order = {character: index for index, character in enumerate("abcdefghijklmnopqrstuvwxyzåäö", 1)}
    return tuple(order[character] for character in value if character in order)


def repair_alphabetic_accents(items: list[dict]) -> None:
    """Repair OCR accents only when the surrounding article order proves it."""
    index = 1
    while index < len(items) - 1:
        if not items[index]["headword"].startswith("å "):
            index += 1
            continue
        end = index
        while end < len(items) and items[end]["headword"].startswith("å "):
            end += 1
        if end >= len(items):
            break
        originals = [items[position]["headword"] for position in range(index, end)]
        candidates = ["à " + value[2:] for value in originals]
        previous_key = swedish_sort_key(items[index - 1]["headword"])
        following_key = swedish_sort_key(items[end]["headword"])
        original_keys = [swedish_sort_key(value) for value in originals]
        candidate_keys = [swedish_sort_key(value) for value in candidates]
        originals_fit = (
            previous_key <= original_keys[0]
            and original_keys == sorted(original_keys)
            and original_keys[-1] <= following_key
        )
        candidates_fit = (
            previous_key <= candidate_keys[0]
            and candidate_keys == sorted(candidate_keys)
            and candidate_keys[-1] <= following_key
        )
        if candidates_fit and not originals_fit:
            for position, candidate in zip(range(index, end), candidates):
                items[position]["corrected_from"] = items[position]["headword"]
                items[position]["correction_method"] = "alfabetisk ordning"
                items[position]["headword"] = candidate
        index = end


def extract_head(article: dict) -> dict:
    line = article["lines"][0]
    tokens = sorted(line.get("tokens", []), key=lambda token: token["left"])
    reasons: list[str] = []
    homonym: int | None = None
    marker_unclear = False
    index = 0

    if not tokens:
        return {
            "headword": "",
            "raw_headword": "",
            "homonym": None,
            "status": "osäker",
            "reasons": ["tokenmetadata saknas; kör om OCR med aktuell cacheversion"],
        }

    first = tokens[0]["text"].strip()
    if first and all(character in MARKERS or character.isdigit() for character in first):
        if first.isdigit():
            homonym = int(first)
        elif first in SUPERSCRIPT:
            homonym = SUPERSCRIPT[first]
        else:
            marker_unclear = True
        index = 1
    elif first:
        if first[0].isdigit() and len(first) > 1:
            homonym = int(first[0])
            tokens[0] = dict(tokens[0], text=first[1:])
        elif first[0] in SUPERSCRIPT and len(first) > 1:
            homonym = SUPERSCRIPT[first[0]]
            tokens[0] = dict(tokens[0], text=first[1:])
        elif first[0] in MARKERS and len(first) > 1:
            marker_unclear = True
            tokens[0] = dict(tokens[0], text=first[1:])

    selected = []
    for token in tokens[index:]:
        text = token["text"].strip()
        plain = _plain_token(text)
        if not text:
            continue
        if selected and (
            text.startswith(("(", "[", "-", "—", "–"))
            or plain in STOP_WORDS
            or text[0].isdigit()
        ):
            break
        if not selected and (
            text.startswith(("(", "[", "-", "—", "–"))
            or plain in STOP_WORDS
        ):
            reasons.append("själva huvudordet saknas i Tesseract")
            break
        selected.append(token)

    raw = " ".join(token["text"].strip() for token in selected).strip()
    headword = normalize_headword(raw)
    if marker_unclear:
        reasons.append("homonymtecknet känns igen som citattecken men inte som säker siffra")
    if not headword:
        reasons.append("inget huvudord kunde tas ut")
    if selected:
        confidence = sum(float(token.get("confidence", 0.0)) for token in selected) / len(selected)
        if confidence < 60:
            reasons.append(f"låg OCR-säkerhet ({confidence:.0f})")
    return {
        "headword": headword,
        "raw_headword": raw,
        "homonym": homonym,
        "status": "osäker" if reasons else "preliminär",
        "reasons": reasons,
    }


def infer_homonym_runs(items: list[dict]) -> None:
    position = 0
    while position < len(items):
        end = position + 1
        key = items[position]["headword"]
        while end < len(items) and key and items[end]["headword"] == key:
            end += 1
        run = items[position:end]
        if len(run) > 1:
            for number, item in enumerate(run, 1):
                if item["homonym"] is None:
                    item["homonym"] = number
                    item["homonym_inferred"] = True
                item["reasons"] = [
                    reason for reason in item["reasons"]
                    if not reason.startswith("homonymtecknet")
                ]
                item["status"] = "osäker" if item["reasons"] else "preliminär"
        position = end


def extract_heads(
    payload: dict, corrections: dict[str, str] | None = None
) -> list[dict]:
    corrections = corrections or {}
    result = []
    for article in payload["articles"]:
        item = {
            "article_number": article["number"],
            "page": article["start_page"],
            "column": article["start_column"],
            "source_line": article["headword_ocr"],
            "homonym_inferred": False,
            **extract_head(article),
        }
        item["corrected_from"] = ""
        item["correction_method"] = ""
        result.append(item)
    repair_alphabetic_accents(result)
    for item in result:
        corrected = corrections.get(item["headword"])
        if corrected:
            item["corrected_from"] = item["headword"]
            item["correction_method"] = "manuell korrektionsfil"
            item["headword"] = corrected
    infer_homonym_runs(result)
    return result


def report_html(items: list[dict]) -> str:
    rows = []
    for item in items:
        notes = list(item["reasons"])
        if item.get("corrected_from"):
            notes.append(
                f"rättad från {item['corrected_from']} via {item['correction_method']}"
            )
        reasons = "; ".join(notes) or "—"
        homonym = "—" if item["homonym"] is None else str(item["homonym"])
        css = "uncertain" if item["status"] == "osäker" else ""
        rows.append(
            '<tr class="%s"><td>%d</td><td>%d:%d</td><td><b>%s</b></td>'
            '<td>%s%s</td><td>%s</td><td>%s</td></tr>' % (
                css,
                item["article_number"], item["page"], item["column"],
                html.escape(item["headword"] or "—"), homonym,
                " (tolkad)" if item["homonym_inferred"] else "",
                html.escape(item["source_line"]), html.escape(reasons),
            )
        )
    uncertain = sum(item["status"] == "osäker" for item in items)
    return f"""<!doctype html><html lang="sv"><head><meta charset="utf-8">
<title>SAOL – huvudordsgranskning</title><style>
body{{font:15px system-ui;margin:24px;max-width:1500px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}}th{{position:sticky;top:0;background:#eee}}tr.uncertain{{background:#fff3cd}}
</style></head><body><h1>Preliminära huvudord</h1>
<p>{len(items)} artiklar; {uncertain} gula poster behöver granskas eller förbättrad OCR.</p>
<table><thead><tr><th>#</th><th>Sida:spalt</th><th>Huvudord</th><th>Homonym</th><th>OCR-rad</th><th>Anmärkning</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, default=Path("article-text-review.json"))
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--json", type=Path, default=Path("headword-review.json"))
    parser.add_argument("--report", type=Path, default=Path("headword-review.html"))
    args = parser.parse_args()
    payload = json.loads(args.articles.read_text(encoding="utf-8"))
    corrections = json.loads(args.corrections.read_text(encoding="utf-8"))
    items = extract_heads(payload, corrections)
    output = {"article_count": len(items), "headwords": items}
    args.json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(report_html(items), encoding="utf-8")
    print(f"Artiklar: {len(items)}")
    print(f"Preliminära huvudord: {sum(bool(item['headword']) for item in items)}")
    print(f"Osäkra: {sum(item['status'] == 'osäker' for item in items)}")
    print(f"Data: {args.json.resolve()}")
    print(f"Rapport: {args.report.resolve()}")


if __name__ == "__main__":
    main()
