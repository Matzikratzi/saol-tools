from __future__ import annotations

"""Parse explicitly printed SAOL inflections conservatively.

The parser expands only forms directly supported by the article's initial
inflection notation. Omitted regular paradigms are reported, not guessed.
"""

import argparse
import html
import json
import re
from pathlib import Path


POS = {"adj", "adv", "interj", "prep", "pron", "s", "v"}
CONTROL = {"best", "el", "eller", "n", "pl", "sing", "äv"}


def normalize_form(value: str) -> str:
    value = value.casefold().replace("¦", "|")
    value = re.sub(r"[^a-zåäöàáé| -]+", "", value)
    return re.sub(r"\s+", " ", value).strip(" -")


def bracket_variants(value: str) -> list[str]:
    match = re.search(r"\[([^]]+)\]", value)
    if not match:
        return [value]
    before, optional, after = value[: match.start()], match.group(1), value[match.end() :]
    result = []
    for replacement in ("", optional):
        result.extend(bracket_variants(before + replacement + after))
    return list(dict.fromkeys(result))


def apply_ending(stem_headword: str, headword: str, notation: str) -> list[str]:
    notation = notation.strip()
    if notation == "=":
        return [normalize_form(headword)]
    if not notation.startswith("-"):
        value = normalize_form(notation)
        return [value] if value else []
    ending = notation[1:]
    stem = normalize_form(stem_headword)
    base = normalize_form(headword)
    prefix = stem.split("|", 1)[0] if "|" in stem else base
    return [
        normalize_form(prefix + variant)
        for variant in bracket_variants(ending)
        if normalize_form(prefix + variant)
    ]


def _prepare_line(value: str) -> str:
    # Tesseract frequently joins the final suffix and the following ``s.``.
    value = re.sub(
        r"(-[A-Za-zÅÄÖåäöÀÁÉàáé\[\]]+)s\.(?=\s|$)",
        r"\1 s.",
        value,
    )
    return value


def parse_inflection(article: dict, head: dict) -> dict:
    source = _prepare_line(article["headword_ocr"])
    tokens = source.split()
    pos = ""
    pos_index = len(tokens)
    for index, token in enumerate(tokens):
        plain = re.sub(r"[^a-zåäö]+", "", token.casefold())
        if plain == "ss":
            plain = "s"
        if plain in POS:
            pos, pos_index = plain, index
            break

    start = None
    for index, token in enumerate(tokens[:pos_index]):
        plain = token.strip(";,:()[]")
        if plain.startswith(("-", "~")) or plain == "=" or plain.casefold() in {"pl.", "n."}:
            start = index
            break
    notation_tokens = tokens[start:pos_index] if start is not None else []
    notation = " ".join(notation_tokens)

    candidates = []
    saw_form = False
    for raw in notation_tokens:
        token = raw.strip(";,:()")
        plain = re.sub(r"[^a-zåäö]+", "", token.casefold())
        if plain in CONTROL or token.isdigit():
            continue
        if token == "=":
            candidates.append(token)
            saw_form = True
            continue
        if token.startswith("-"):
            token = re.sub(r"[^A-Za-zÅÄÖåäöÀÁÉàáé\[\]-]+$", "", token)
            if token and token != "-":
                candidates.append(token)
                saw_form = True
            continue
        if saw_form and re.fullmatch(r"[A-Za-zÅÄÖåäöÀÁÉàáé]+", token):
            candidates.append(token)

    forms = [normalize_form(head["headword"])]
    evidence = [{"form": forms[0], "notation": "uppslagsord", "kind": "headword"}]
    for candidate in candidates:
        for form in apply_ending(
            head.get("stem_headword") or head["headword"],
            head["headword"],
            candidate,
        ):
            if form and form not in forms:
                forms.append(form)
                evidence.append(
                    {"form": form, "notation": candidate, "kind": "explicit"}
                )

    reasons = []
    if not pos:
        reasons.append("ordklass saknas eller kunde inte läsas")
    if not candidates and pos in {"s", "adj", "v", "pron"}:
        reasons.append("ingen uttrycklig böjningsform hittades")
    if "~" in notation:
        reasons.append("tilde-notation behöver separat analys")
    suspicious = [token for token in notation_tokens if "?" in token or "€" in token]
    if suspicious:
        reasons.append("misstänkta OCR-tecken i böjningen")
    return {
        "article_number": article["number"],
        "page": article["start_page"],
        "column": article["start_column"],
        "headword": head["headword"],
        "stem_headword": head.get("stem_headword") or head["headword"],
        "homonym": head.get("homonym"),
        "part_of_speech": pos,
        "source_line": article["headword_ocr"],
        "notation": notation,
        "forms": forms,
        "evidence": evidence,
        "status": "osäker" if reasons else "uttrycklig",
        "reasons": reasons,
    }


def parse_all(articles_payload: dict, heads_payload: dict) -> list[dict]:
    heads = {item["article_number"]: item for item in heads_payload["headwords"]}
    return [
        parse_inflection(article, heads[article["number"]])
        for article in articles_payload["articles"]
    ]


def report_html(items: list[dict]) -> str:
    rows = []
    for item in items:
        css = "uncertain" if item["status"] == "osäker" else ""
        homonym = "" if item["homonym"] is None else f"<sup>{item['homonym']}</sup>"
        rows.append(
            '<tr class="%s"><td>%d</td><td>%d:%d</td><td>%s<b>%s</b></td>'
            '<td>%s</td><td><code>%s</code></td><td>%s</td><td>%s</td></tr>' % (
                css, item["article_number"], item["page"], item["column"], homonym,
                html.escape(item["headword"]), html.escape(item["part_of_speech"] or "—"),
                html.escape(item["notation"] or "—"),
                html.escape(", ".join(item["forms"])),
                html.escape("; ".join(item["reasons"]) or "—"),
            )
        )
    uncertain = sum(item["status"] == "osäker" for item in items)
    forms = {form for item in items for form in item["forms"]}
    return f"""<!doctype html><html lang="sv"><head><meta charset="utf-8">
<title>SAOL – böjningsgranskning</title><style>
body{{font:15px system-ui;margin:24px;max-width:1600px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}}th{{position:sticky;top:0;background:#eee}}tr.uncertain{{background:#fff3cd}}code{{white-space:pre-wrap}}
</style></head><body><h1>Uttryckliga böjningsformer</h1>
<p>{len(items)} artiklar; {len(forms)} unika preliminära former; {uncertain} gula poster behöver fortsatt regel- eller OCR-analys.</p>
<table><thead><tr><th>#</th><th>Sida:spalt</th><th>Huvudord</th><th>Ordklass</th><th>Notation</th><th>Former</th><th>Anmärkning</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, default=Path("article-text-review.json"))
    parser.add_argument("--headwords", type=Path, default=Path("headword-review.json"))
    parser.add_argument("--json", type=Path, default=Path("inflection-review.json"))
    parser.add_argument("--report", type=Path, default=Path("inflection-review.html"))
    args = parser.parse_args()
    articles = json.loads(args.articles.read_text(encoding="utf-8"))
    heads = json.loads(args.headwords.read_text(encoding="utf-8"))
    items = parse_all(articles, heads)
    output = {
        "article_count": len(items),
        "unique_form_count": len({form for item in items for form in item["forms"]}),
        "uncertain_count": sum(item["status"] == "osäker" for item in items),
        "inflections": items,
    }
    args.json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(report_html(items), encoding="utf-8")
    print(f"Artiklar: {len(items)}")
    print(f"Unika preliminära former: {output['unique_form_count']}")
    print(f"Osäkra: {output['uncertain_count']}")
    print(f"Data: {args.json.resolve()}")
    print(f"Rapport: {args.report.resolve()}")


if __name__ == "__main__":
    main()
