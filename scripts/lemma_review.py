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


def plural_of_previous(previous: str, candidate: str) -> bool:
    """Recognize an -a noun's plural, e.g. afrikanska -> afrikanskor."""
    previous = normalize_lemma(previous)
    candidate = normalize_lemma(candidate)
    return (
        len(previous) > 2
        and previous.endswith("a")
        and candidate == previous[:-1] + "or"
    )


def merged_pos_inflection(
    raw: str, normalized_suffix: str, bold_score: float
) -> bool:
    """Recognize '-ers.' as non-bold inflection '-er' plus noun marker 's.'."""
    compact = raw.strip().casefold()
    return (
        normalized_suffix.endswith("s")
        and normalized_suffix[:-1] in NON_LEMMA_SUFFIXES
        and (compact.endswith("s.") or bold_score < 0.25)
    )


def optional_parenthesis_variants(value: str) -> list[str]:
    """Include SAOL's parenthesized ending without inventing a short variant."""
    match = re.match(r"^(.*)\(([^()]*)\)$", value)
    if not match or not match.group(1) or not match.group(2):
        return [value]
    return [match.group(1) + match.group(2)]


def repair_mixed_case_duplicate(value: str) -> str:
    """Collapse an OCR duplicate such as -mMässighet to -Mässighet."""
    match = re.match(r"^(-?)([a-zåäö])([A-ZÅÄÖ])(.*)$", value)
    if match and match.group(2).casefold() == match.group(3).casefold():
        return match.group(1) + match.group(3) + match.group(4)
    return value


def suffix_base(value: str) -> str:
    """Return the repeatable stem before SAOL's vertical boundary marker."""
    for marker in ("|", "¦"):
        if marker in value:
            return normalize_lemma(value.split(marker, 1)[0])
    return normalize_lemma(value)


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
        if len(base) >= 3 and tail in suffixes:
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


def remove_alphabetic_family_outliers(
    items: list[dict], heads: dict[int, dict]
) -> list[dict]:
    """Drop definition words before the text returns to the headword family."""
    kept = []
    by_article: dict[int, list[dict]] = {}
    for item in items:
        by_article.setdefault(item["article_number"], []).append(item)
    rejected_ids = set()
    for article_number, article_items in by_article.items():
        headword = normalize_lemma(heads[article_number]["headword"])
        family = headword[:-1] if len(headword) > 5 else headword
        for index, item in enumerate(article_items):
            lemma = normalize_lemma(item["lemma"])
            if item["method"] == "artikelhuvud" or lemma.startswith(family):
                continue
            later_family = next(
                (
                    normalize_lemma(following["lemma"])
                    for following in article_items[index + 1 :]
                    if normalize_lemma(following["lemma"]).startswith(family)
                ),
                "",
            )
            if later_family and lemma > later_family:
                rejected_ids.add(id(item))
    for item in items:
        if id(item) not in rejected_ids:
            kept.append(item)
    return kept


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
        structured_head = head.get("stem_headword", "")
        current_head = (
            normalize_lemma(structured_head)
            if "|" in structured_head or "¦" in structured_head
            else head["headword"]
        )
        current_base = suffix_base(structured_head or current_head)
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
                following_tokens = list(tokens[token_index + 1 :])
                for following_line in article["lines"][line_index + 1 :]:
                    following_tokens.extend(
                        sorted(
                            following_line.get("tokens", []),
                            key=lambda candidate: candidate["left"],
                        )
                    )
                previous_boundary = infer_boundary_from_previous(
                    cleaned, last_lookup_lemma
                )
                repeated_boundary = infer_boundary_from_repeated_suffix(
                    cleaned, following_tokens
                )
                if previous_boundary:
                    cleaned = previous_boundary
                elif repeated_boundary:
                    cleaned = repeated_boundary
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
                series_first = bool(inferred)
                lexical = bool(re.search(r"[A-Za-zÅÄÖåäöÀÁÉàáé]", cleaned))
                if not lexical:
                    previous_separator = False
                    at_line_start = False
                    continue
                if cleaned.startswith("-"):
                    repaired_suffix = repair_mixed_case_duplicate(cleaned)
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
                            if plural_of_previous(
                                last_lookup_lemma, lemma
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
                    previous_separator = False
                    at_line_start = False
                    continue
                plausible_position = previous_separator or at_line_start
                clearly_semibold = score >= 0.70
                has_stem_boundary = "|" in cleaned or "¦" in cleaned
                if (
                    (plausible_position and score >= 0.35)
                    or previous_separator
                    or clearly_semibold
                    or has_stem_boundary
                    or series_first
                ):
                    lemma = normalize_lemma(cleaned)
                    if lemma and lemma not in POS and len(lemma) > 1:
                        add(
                            article, lemma, cleaned, "halvfet token",
                            score, line=line, token=token
                        )
                        last_lookup_lemma = lemma
                        current_base = suffix_base(cleaned)
                previous_separator = False
                at_line_start = False
    return remove_alphabetic_family_outliers(result, heads)


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
                    measuring.textbbox((0, 0), item["lemma"], font=font)[2]
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
                    color = (
                        "#c62828"
                        if item["status"] == "osäker"
                        else "#00695c"
                    )
                    text_box = draw.textbbox(
                        (0, 0), item["lemma"], font=font
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
                        item["lemma"],
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
