from __future__ import annotations

"""Find all bold and semibold SAOL lemma candidates inside grouped articles."""

import argparse
import html
import json
import re
import statistics
from pathlib import Path


POS = {"adj", "adv", "interj", "prep", "pron", "s", "v"}
NON_LEMMA_SUFFIXES = {
    "-a", "-ad", "-ade", "-an", "-ar", "-are", "-at", "-de", "-dde",
    "-e", "-en", "-er", "-et", "-la", "-na", "-n", "-or", "-ra", "-t",
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
        tokens = article["lines"][0].get("tokens", [])
        if tokens:
            head_values.append(float(min(tokens, key=lambda token: token["left"])["ink_density"]))
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
            return tokens[index + 1 :]
    return tokens[1:]


def extract_candidates(articles_payload: dict, heads_payload: dict) -> list[dict]:
    heads = {item["article_number"]: item for item in heads_payload["headwords"]}
    ordinary, bold = _references(articles_payload)
    result = []
    seen = set()

    def add(article, lemma, raw, method, score, stem=""):
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
        result.append(
            {
                "article_number": article["number"],
                "page": article["start_page"],
                "column": article["start_column"],
                "lemma": lemma,
                "stem_lemma": stem or lemma,
                "raw": raw,
                "method": method,
                "bold_score": score,
                "status": "osäker" if reasons else "kandidat",
                "reasons": reasons,
            }
        )

    for article in articles_payload["articles"]:
        head = heads[article["number"]]
        current_base = head["headword"]
        add(
            article,
            current_base,
            head.get("raw_headword", current_base),
            "artikelhuvud",
            1.0,
            head.get("stem_headword", current_base),
        )
        for line_index, line in enumerate(article["lines"]):
            tokens = sorted(line.get("tokens", []), key=lambda token: token["left"])
            if line_index == 0:
                tokens = _after_inflection_prefix(tokens)
            previous_separator = False
            at_line_start = True
            for token in tokens:
                raw = token["text"].strip()
                if not raw:
                    continue
                if raw in {"—", "–", "--"}:
                    previous_separator = True
                    continue
                score = _token_score(token, ordinary, bold)
                cleaned = raw.strip(";,:.()[]{}")
                lexical = bool(re.search(r"[A-Za-zÅÄÖåäöÀÁÉàáé]", cleaned))
                if not lexical:
                    previous_separator = False
                    at_line_start = False
                    continue
                if cleaned.startswith("-"):
                    normalized_suffix = "-" + normalize_lemma(cleaned[1:])
                    if (
                        normalized_suffix not in NON_LEMMA_SUFFIXES
                        and len(normalized_suffix) > 2
                        and score >= 0.25
                    ):
                        lemma = expand_compound(current_base, cleaned)
                        add(article, lemma, cleaned, "sammansättningssuffix", score)
                    previous_separator = False
                    at_line_start = False
                    continue
                plausible_position = previous_separator or at_line_start
                if plausible_position and score >= 0.35:
                    lemma = normalize_lemma(cleaned)
                    if lemma and lemma not in POS:
                        add(article, lemma, cleaned, "halvfet token", score)
                        current_base = lemma
                previous_separator = False
                at_line_start = False
    return result


def report_html(items: list[dict]) -> str:
    rows = []
    for item in items:
        css = "uncertain" if item["status"] == "osäker" else ""
        rows.append(
            '<tr class="%s"><td>%d</td><td>%d:%d</td><td><b>%s</b></td>'
            '<td>%s</td><td>%.2f</td><td><code>%s</code></td><td>%s</td></tr>' % (
                css, item["article_number"], item["page"], item["column"],
                html.escape(item["lemma"]), html.escape(item["method"]),
                item["bold_score"], html.escape(item["raw"]),
                html.escape("; ".join(item["reasons"]) or "—"),
            )
        )
    uncertain = sum(item["status"] == "osäker" for item in items)
    unique = {item["lemma"] for item in items}
    return f"""<!doctype html><html lang="sv"><head><meta charset="utf-8">
<title>SAOL – grundformskandidater</title><style>
body{{font:15px system-ui;margin:24px;max-width:1500px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}}th{{position:sticky;top:0;background:#eee}}tr.uncertain{{background:#fff3cd}}code{{white-space:pre-wrap}}
</style></head><body><h1>Grundformskandidater</h1>
<p>{len(items)} träffar; {len(unique)} unika grundformer; {uncertain} gula kandidater.</p>
<table><thead><tr><th>Artikel</th><th>Sida:spalt</th><th>Grundform</th><th>Metod</th><th>Fet</th><th>OCR</th><th>Anmärkning</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, default=Path("article-text-review.json"))
    parser.add_argument("--headwords", type=Path, default=Path("headword-review.json"))
    parser.add_argument("--json", type=Path, default=Path("lemma-review.json"))
    parser.add_argument("--report", type=Path, default=Path("lemma-review.html"))
    args = parser.parse_args()
    articles = json.loads(args.articles.read_text(encoding="utf-8"))
    heads = json.loads(args.headwords.read_text(encoding="utf-8"))
    items = extract_candidates(articles, heads)
    output = {
        "candidate_count": len(items),
        "unique_lemma_count": len({item["lemma"] for item in items}),
        "uncertain_count": sum(item["status"] == "osäker" for item in items),
        "candidates": items,
    }
    args.json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(report_html(items), encoding="utf-8")
    print(f"Kandidater: {output['candidate_count']}")
    print(f"Unika grundformer: {output['unique_lemma_count']}")
    print(f"Osäkra: {output['uncertain_count']}")
    print(f"Data: {args.json.resolve()}")
    print(f"Rapport: {args.report.resolve()}")


if __name__ == "__main__":
    main()
