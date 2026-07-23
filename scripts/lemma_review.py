from __future__ import annotations

"""Find all bold and semibold SAOL lemma candidates inside grouped articles."""

import argparse
import html
import json
import re
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scripts.article_start_ml import DEFAULT_CACHE, extract_page


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
            return tokens[index + 1 :]
    return tokens[1:]


def extract_candidates(articles_payload: dict, heads_payload: dict) -> list[dict]:
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
        current_base = head["headword"]
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
            current_base,
            head.get("raw_headword", current_base),
            "artikelhuvud",
            1.0,
            head.get("stem_headword", current_base),
            first_line,
            head_token,
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
                        add(
                            article, lemma, cleaned, "sammansättningssuffix",
                            score, line=line, token=token
                        )
                    previous_separator = False
                    at_line_start = False
                    continue
                plausible_position = previous_separator or at_line_start
                if plausible_position and score >= 0.35:
                    lemma = normalize_lemma(cleaned)
                    if lemma and lemma not in POS:
                        add(
                            article, lemma, cleaned, "halvfet token",
                            score, line=line, token=token
                        )
                        current_base = lemma
                previous_separator = False
                at_line_start = False
    return result


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


def render_review_images(
    items: list[dict], pages: list[int], cache_dir: Path, image_dir: Path
) -> list[Path]:
    """Render each printed column beside labels aligned to their source rows."""
    image_dir.mkdir(parents=True, exist_ok=True)
    font = _review_font()
    outputs = []
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
            margin = max(460, crop.width // 3)
            canvas = Image.new("RGB", (crop.width + margin, crop.height), "white")
            canvas.paste(crop, (margin, 0))
            draw = ImageDraw.Draw(canvas)
            column_items = sorted(
                (
                    item
                    for item in page_items
                    if item["source_column"] == column
                ),
                key=lambda item: (item["source_top"], item["source_left"]),
            )
            last_label_y = -100
            for item in column_items:
                source_y = int((item["source_top"] + item["source_bottom"]) / 2)
                label_y = max(source_y - 15, last_label_y + 34)
                label_y = min(label_y, canvas.height - 34)
                last_label_y = label_y
                color = "#c62828" if item["status"] == "osäker" else "#00695c"
                source_x = margin + int(max(0, item["source_right"] - left))
                label_right = margin - 18
                text_box = draw.textbbox((0, 0), item["lemma"], font=font)
                text_width = text_box[2] - text_box[0]
                label_x = max(8, label_right - text_width)
                draw.line(
                    (label_right + 6, label_y + 14, source_x, source_y),
                    fill=color,
                    width=3,
                )
                draw.ellipse(
                    (source_x - 4, source_y - 4, source_x + 4, source_y + 4),
                    fill=color,
                )
                draw.text((label_x, label_y), item["lemma"], font=font, fill=color)
            output = image_dir / f"page-{page:04d}-column-{column}.png"
            canvas.save(output, format="PNG")
            outputs.append(output)
    return outputs


def report_html(items: list[dict], images: list[Path] | None = None) -> str:
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
    image_blocks = "".join(
        '<figure><img src="%s" loading="lazy"><figcaption>%s</figcaption></figure>'
        % (html.escape(str(path)), html.escape(path.stem.replace("-", " ")))
        for path in (images or [])
    )
    return f"""<!doctype html><html lang="sv"><head><meta charset="utf-8">
<title>SAOL – grundformskandidater</title><style>
body{{font:15px system-ui;margin:24px;max-width:1800px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}}th{{position:sticky;top:0;background:#eee}}tr.uncertain{{background:#fff3cd}}code{{white-space:pre-wrap}}figure{{margin:24px 0;border:1px solid #aaa;padding:10px;background:#eee}}figure img{{display:block;width:100%;height:auto}}figcaption{{margin-top:6px;color:#555}}
</style></head><body><h1>Grundformskandidater</h1>
<p>{len(items)} träffar; {len(unique)} unika grundformer; {uncertain} röda kandidater.</p>
<p>Grönt är en säker kandidat. Rött bör kontrolleras. Linjen visar den tryckta källa som kandidaten kommer från.</p>
{image_blocks}
<h2>Alla kandidater som tabell</h2>
<table><thead><tr><th>Artikel</th><th>Sida:spalt</th><th>Grundform</th><th>Metod</th><th>Fet</th><th>OCR</th><th>Anmärkning</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, default=Path("article-text-review.json"))
    parser.add_argument("--headwords", type=Path, default=Path("headword-review.json"))
    parser.add_argument("--json", type=Path, default=Path("lemma-review.json"))
    parser.add_argument("--report", type=Path, default=Path("lemma-review.html"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--image-dir", type=Path, default=Path("lemma-review-pages"))
    args = parser.parse_args()
    articles = json.loads(args.articles.read_text(encoding="utf-8"))
    heads = json.loads(args.headwords.read_text(encoding="utf-8"))
    for page in articles.get("pages", []):
        extract_page(page, args.cache_dir, False)
    items = extract_candidates(articles, heads)
    images = render_review_images(
        items, articles.get("pages", []), args.cache_dir, args.image_dir
    )
    output = {
        "candidate_count": len(items),
        "unique_lemma_count": len({item["lemma"] for item in items}),
        "uncertain_count": sum(item["status"] == "osäker" for item in items),
        "candidates": items,
    }
    args.json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(report_html(items, images), encoding="utf-8")
    print(f"Kandidater: {output['candidate_count']}")
    print(f"Unika grundformer: {output['unique_lemma_count']}")
    print(f"Osäkra: {output['uncertain_count']}")
    print(f"Data: {args.json.resolve()}")
    print(f"Rapport: {args.report.resolve()}")
    print(f"Bildsidor: {args.image_dir.resolve()}")


if __name__ == "__main__":
    main()
