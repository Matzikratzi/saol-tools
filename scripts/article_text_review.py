from __future__ import annotations

"""Group detected article starts with their continuation rows.

This is the second experimental step in the SAOL extraction pipeline. It uses
the geometric T rule from ``article_start_ml.py`` and deliberately runs on
pages without ground truth.

Usage:
    PYTHONPATH=. python3 scripts/article_text_review.py
    PYTHONPATH=. python3 scripts/article_text_review.py --pages 23 24 25
"""

import argparse
import html
import json
from pathlib import Path

from scripts.article_start_ml import DEFAULT_CACHE, extract_page


DEFAULT_PAGES = list(range(23, 31))


def reading_order(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (row["page"], row["column"], row["top"]))


def merge_overlapping_rows(rows: list[dict]) -> list[dict]:
    """Join Tesseract fragments that occupy the same physical printed row."""
    merged: list[dict] = []
    for row in reading_order(rows):
        if merged and (merged[-1]["page"], merged[-1]["column"]) == (
            row["page"],
            row["column"],
        ):
            previous = merged[-1]
            overlap = min(previous["bottom"], row["bottom"]) - max(
                previous["top"], row["top"]
            )
            previous_height = max(1.0, previous["bottom"] - previous["top"])
            row_height = max(1.0, row["bottom"] - row["top"])
            shorter = min(previous_height, row_height)
            previous_centre = (previous["top"] + previous["bottom"]) / 2
            row_centre = (row["top"] + row["bottom"]) / 2
            same_printed_row = (
                overlap >= shorter * 0.65
                and abs(previous_centre - row_centre) <= shorter * 0.30
            )
            if same_printed_row:
                fragments = sorted(
                    (previous, row), key=lambda item: (item["left"], item["top"])
                )
                combined = dict(fragments[0])
                combined["top"] = min(previous["top"], row["top"])
                combined["bottom"] = max(previous["bottom"], row["bottom"])
                combined["left"] = min(previous["left"], row["left"])
                combined["right"] = max(previous["right"], row["right"])
                combined["text"] = " ".join(
                    fragment["text"].strip() for fragment in fragments
                )
                combined["match_text"] = " ".join(
                    fragment.get("match_text", "") for fragment in fragments
                ).strip()
                for key in (
                    "baseline",
                    "chapter_heading",
                    "ocr_reaches_left",
                    "pixel_reaches_left",
                ):
                    combined[key] = bool(previous.get(key) or row.get(key))
                combined["left_ink"] = previous.get("left_ink", 0) + row.get(
                    "left_ink", 0
                )
                merged[-1] = combined
                continue
        merged.append(dict(row))
    return merged


def group_articles(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Return articles, unattached lines and excluded chapter headings."""
    articles: list[dict] = []
    unattached: list[dict] = []
    headings: list[dict] = []
    current: dict | None = None

    for row in merge_overlapping_rows(rows):
        if row.get("chapter_heading"):
            headings.append(row)
            current = None
            continue
        if row.get("baseline"):
            current = {
                "number": len(articles) + 1,
                "start_page": row["page"],
                "start_column": row["column"],
                "start_y": row["top"],
                "headword_ocr": row["text"],
                "lines": [],
            }
            articles.append(current)
        if current is None:
            unattached.append(row)
            continue
        current["lines"].append(
            {
                "page": row["page"],
                "column": row["column"],
                "top": row["top"],
                "bottom": row["bottom"],
                "text": row["text"],
                "article_start": bool(row.get("baseline")),
                "ocr_reaches_left": bool(row.get("ocr_reaches_left")),
                "pixel_reaches_left": bool(row.get("pixel_reaches_left")),
                "mean_confidence": float(row["features"][15]),
            }
        )

    for article in articles:
        article["text"] = " ".join(line["text"].strip() for line in article["lines"])
        article["end_page"] = article["lines"][-1]["page"]
        article["end_column"] = article["lines"][-1]["column"]
    return articles, unattached, headings


def _location(row: dict) -> str:
    return f's. {row["page"]}, sp. {row["column"]}, y={row["top"]:.0f}'


def review_html(articles: list[dict], unattached: list[dict], headings: list[dict]) -> str:
    article_blocks = []
    for article in articles:
        lines = "".join(
            '<div class="line%s"><span>%s</span><code>%s</code></div>'
            % (
                " start" if line["article_start"] else "",
                html.escape(_location(line)),
                html.escape(line["text"]),
            )
            for line in article["lines"]
        )
        crosses = article["start_page"] != article["end_page"] or article["start_column"] != article["end_column"]
        article_blocks.append(
            '<details%s><summary><b>%d.</b> %s <small>%s%s</small></summary>%s</details>'
            % (
                " open" if article["number"] <= 10 else "",
                article["number"],
                html.escape(article["headword_ocr"]),
                html.escape(_location(article["lines"][0])),
                " · spalt-/sidbrytning" if crosses else "",
                lines,
            )
        )
    orphan_rows = "".join(
        "<tr><td>%s</td><td>%s</td></tr>" % (html.escape(_location(row)), html.escape(row["text"]))
        for row in unattached
    )
    heading_rows = "".join(
        "<tr><td>%s</td><td>%s</td></tr>" % (html.escape(_location(row)), html.escape(row["text"]))
        for row in headings
    )
    return f"""<!doctype html>
<html lang="sv"><head><meta charset="utf-8"><title>SAOL – grupperade artiklar</title>
<style>
body{{font:15px system-ui;margin:24px;max-width:1200px}}details{{border:1px solid #bbb;border-radius:6px;margin:8px 0;padding:8px}}summary{{cursor:pointer}}small{{color:#666;margin-left:10px}}.line{{display:grid;grid-template-columns:180px 1fr;gap:12px;padding:3px 8px;border-top:1px solid #eee}}.line.start{{border-left:5px solid #198754;background:#eaf7ef;font-weight:700}}code{{white-space:pre-wrap;font:14px ui-monospace,monospace}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;text-align:left}}.warning{{background:#fff3cd;padding:10px}}
</style></head><body>
<h1>Grupperade SAOL-artiklar</h1>
<p>{len(articles)} artiklar. Grön rad är artikelstart; följande rader tillhör samma artikel.</p>
<p class="warning">Kontrollera särskilt poster märkta spalt-/sidbrytning och de fristående raderna nedan.</p>
{''.join(article_blocks)}
<h2>Fristående rader ({len(unattached)})</h2>
<table><thead><tr><th>Plats</th><th>OCR</th></tr></thead><tbody>{orphan_rows or '<tr><td colspan="2">Inga.</td></tr>'}</tbody></table>
<h2>Borttagna kapitelrubriker ({len(headings)})</h2>
<table><thead><tr><th>Plats</th><th>OCR</th></tr></thead><tbody>{heading_rows or '<tr><td colspan="2">Inga.</td></tr>'}</tbody></table>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", nargs="+", type=int, default=DEFAULT_PAGES)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--json", type=Path, default=Path("article-text-review.json"))
    parser.add_argument("--report", type=Path, default=Path("article-text-review.html"))
    args = parser.parse_args()

    rows = []
    for page in args.pages:
        print(f"OCR sida {page} ...", flush=True)
        rows.extend(extract_page(page, args.cache_dir, args.refresh))
    articles, unattached, headings = group_articles(rows)
    payload = {
        "pages": args.pages,
        "article_count": len(articles),
        "unattached_count": len(unattached),
        "chapter_heading_count": len(headings),
        "articles": articles,
        "unattached": unattached,
        "chapter_headings": headings,
    }
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(review_html(articles, unattached, headings), encoding="utf-8")
    print(f"Artiklar: {len(articles)}")
    print(f"Fristående rader: {len(unattached)}")
    print(f"Borttagna kapitelrubriker: {len(headings)}")
    print(f"Data: {args.json.resolve()}")
    print(f"Rapport: {args.report.resolve()}")


if __name__ == "__main__":
    main()
