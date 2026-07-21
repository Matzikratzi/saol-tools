from __future__ import annotations

import argparse
import base64
import html
import io
import re
import statistics
import webbrowser
from dataclasses import dataclass, replace
from pathlib import Path

import httpx
from PIL import Image

from app.runeberg import (
    _align_lines,
    _normalized_observation_line,
    _observation_line_indices,
    _runeberg_ocr_lines,
    _runeberg_ocr_text,
    _runeberg_ocr_tokens,
    extract_observations,
    ocr_image_url,
    page_urls,
    reconcile_contextual_observations,
)

WORD_RE = re.compile(r"[A-Za-zÅÄÖåäöÀÁÉàáé]")
COMBINED_HOMONYM_RE = re.compile(r"^([1-9])([A-Za-zÅÄÖåäöÀÁÉàáé].*)$")
LEADING_MARKER_RE = re.compile(r"^([1-9Iil|])([A-Za-zÅÄÖåäöÀÁÉàáé].*)$")
SUPERSCRIPT = str.maketrans("123456789", "¹²³⁴⁵⁶⁷⁸⁹")
SUPERSCRIPT_DIGITS = "¹²³⁴⁵⁶⁷⁸⁹"


@dataclass(frozen=True)
class PrintedLine:
    column: int
    items: tuple[object, ...]
    text: str
    first: object
    left: float
    top: float
    right: float
    bottom: float
    indent: float = 0.0
    start_score: float = 0.0


@dataclass(frozen=True)
class Article:
    column: int
    headword: str
    raw_headword: str
    lines: tuple[PrintedLine, ...]
    score: float
    homonym_inferred: bool = False

    @property
    def left(self) -> int:
        return int(min(line.left for line in self.lines))

    @property
    def top(self) -> int:
        return int(min(line.top for line in self.lines))

    @property
    def right(self) -> int:
        return int(max(line.right for line in self.lines))

    @property
    def bottom(self) -> int:
        return int(max(line.bottom for line in self.lines))

    @property
    def width(self) -> int:
        return max(1, self.right - self.left)

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top)

    @property
    def article_text(self) -> str:
        return "\n".join(line.text for line in self.lines)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _image_data_url(content: bytes) -> tuple[str, int, int]:
    with Image.open(io.BytesIO(content)) as source:
        image = source.convert("RGB")
        width, height = image.size
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}", width, height


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[position]


def _homonym_text(number: str, word: str) -> str:
    return number.translate(SUPERSCRIPT) + word


def _merge_line_headword(items: tuple[object, ...], median_height: float) -> object:
    first = items[0]
    first_text = first.text.strip()

    if match := COMBINED_HOMONYM_RE.match(first_text):
        text = _homonym_text(match.group(1), match.group(2))
        return replace(first, text=text, ocr_tesseract=text)

    if first_text not in "123456789" or len(items) < 2:
        return first

    word = items[1]
    if not WORD_RE.match(word.text.strip()):
        return first

    gap = word.left - (first.left + first.width)
    digit_bottom = first.top + first.height
    is_small = first.height <= max(median_height * 0.95, word.height * 0.9)
    is_raised = digit_bottom <= word.top + word.height * 0.9 or first.top < word.top
    is_close = -3 <= gap <= max(10.0, word.height)
    if not (is_small and is_raised and is_close):
        return first

    left = min(first.left, word.left)
    top = min(first.top, word.top)
    right = max(first.left + first.width, word.left + word.width)
    bottom = max(first.top + first.height, word.top + word.height)
    area1 = first.width * first.height
    area2 = word.width * word.height
    total_area = max(1, area1 + area2)
    text = _homonym_text(first_text, word.text)
    return replace(
        word,
        text=text,
        left=left,
        top=top,
        width=right - left,
        height=bottom - top,
        confidence=min(first.confidence, word.confidence),
        ink_density=(first.ink_density * area1 + word.ink_density * area2) / total_area,
        ocr_tesseract=text,
    )


def _build_lines(observations: list, image_width: int, image_height: int) -> tuple[list[PrintedLine], dict[int, float], float]:
    if not observations:
        return [], {1: 0.0, 2: image_width / 2}, 1.0

    heights = [item.height for item in observations]
    median_height = statistics.median(heights) if heights else 1.0
    page_left = min(item.left for item in observations)
    page_right = max(item.left + item.width for item in observations)
    split = page_left + (page_right - page_left) / 2

    raw: dict[int, list[PrintedLine]] = {1: [], 2: []}
    for indices in _observation_line_indices(observations):
        ordered = tuple(sorted((observations[index] for index in indices), key=lambda item: item.left))
        if not ordered:
            continue
        left = min(item.left for item in ordered)
        top = min(item.top for item in ordered)
        right = max(item.left + item.width for item in ordered)
        bottom = max(item.top + item.height for item in ordered)
        center_y = (top + bottom) / 2
        if center_y < image_height * 0.07 or center_y > image_height * 0.93:
            continue
        column = 1 if left < split else 2
        first = _merge_line_headword(ordered, median_height)
        text = " ".join(item.text for item in ordered)
        raw[column].append(PrintedLine(column, ordered, text, first, left, top, right, bottom))

    margins: dict[int, float] = {}
    result: list[PrintedLine] = []
    indent_tolerance = max(5.0, median_height * 0.9)

    for column in (1, 2):
        lines = sorted(raw[column], key=lambda line: (line.top, line.left))
        if not lines:
            margins[column] = page_left if column == 1 else split
            continue

        lexical_lefts = [line.left for line in lines if WORD_RE.search(line.first.text)]
        low_cut = _quantile(lexical_lefts, 0.30)
        low_cluster = [value for value in lexical_lefts if value <= low_cut + indent_tolerance]
        margin = statistics.median(low_cluster or lexical_lefts or [lines[0].left])
        margins[column] = margin

        inks = [line.first.ink_density for line in lines if line.first.ink_density > 0]
        ordinary_ink = statistics.median(inks) if inks else 1.0
        bold_reference = max(_quantile(inks, 0.72) if inks else ordinary_ink, ordinary_ink, 1e-6)

        for line in lines:
            indent = max(0.0, line.left - margin)
            margin_score = max(0.0, 1.0 - indent / (indent_tolerance * 2.2))
            ink_ratio = line.first.ink_density / bold_reference
            bold_score = max(0.0, min(1.0, (ink_ratio - 0.72) / 0.38))
            height_score = max(0.0, min(1.0, line.first.height / max(median_height, 1.0) - 0.72))
            confidence_score = max(0.0, min(1.0, line.first.confidence / 100.0))
            score = 0.58 * margin_score + 0.27 * bold_score + 0.10 * height_score + 0.05 * confidence_score
            result.append(replace(line, indent=indent, start_score=score))

    return sorted(result, key=lambda line: (line.column, line.top, line.left)), margins, indent_tolerance


def _group_articles(lines: list[PrintedLine], threshold: float, indent_tolerance: float) -> list[Article]:
    articles: list[Article] = []
    for column in (1, 2):
        column_lines = [line for line in lines if line.column == column]
        current: list[PrintedLine] = []
        current_score = 0.0

        for line in column_lines:
            lexical = bool(WORD_RE.search(line.first.text))
            is_start = lexical and line.indent <= indent_tolerance * 1.15 and line.start_score >= threshold
            if is_start:
                if current:
                    first = current[0]
                    articles.append(Article(column, first.first.text, first.first.text, tuple(current), current_score))
                current = [line]
                current_score = line.start_score
            elif current:
                current.append(line)

        if current:
            first = current[0]
            articles.append(Article(column, first.first.text, first.first.text, tuple(current), current_score))

    return articles


def _base_headword(text: str) -> tuple[str | None, str]:
    token = text.strip()
    if token and token[0] in SUPERSCRIPT_DIGITS:
        return str(SUPERSCRIPT_DIGITS.index(token[0]) + 1), token[1:].casefold()
    if match := LEADING_MARKER_RE.match(token):
        marker = match.group(1)
        number = marker if marker.isdigit() else None
        return number, match.group(2).casefold()
    return None, token.casefold()


def _infer_homonym_series(articles: list[Article]) -> list[Article]:
    result = list(articles)
    for column in (1, 2):
        indices = [index for index, article in enumerate(result) if article.column == column]
        pos = 0
        while pos < len(indices):
            start = pos
            first_index = indices[pos]
            _, base = _base_headword(result[first_index].headword)
            pos += 1
            while pos < len(indices):
                _, next_base = _base_headword(result[indices[pos]].headword)
                if next_base != base:
                    break
                pos += 1

            run = indices[start:pos]
            if len(run) < 2 or not base:
                continue

            parsed = [_base_headword(result[index].headword)[0] for index in run]
            explicit = [int(value) for value in parsed if value is not None]
            plausible = not explicit or explicit == sorted(explicit)
            if not plausible:
                continue

            for sequence_number, index in enumerate(run, start=1):
                article = result[index]
                expected = _homonym_text(str(sequence_number), base)
                number, current_base = _base_headword(article.headword)
                if current_base != base:
                    continue
                inferred = number != str(sequence_number) or article.headword != expected
                result[index] = replace(article, headword=expected, homonym_inferred=inferred)

    return result


def _review_html(
    page: int,
    source_url: str,
    image_content: bytes,
    raw_text: str,
    runeberg_lines: list[list[str]],
    observations: list,
    articles: list[Article],
    pairs: list[tuple[int, int, float]],
    margins: dict[int, float],
    threshold: float,
) -> str:
    rows: list[str] = []
    for index, article in enumerate(articles):
        continuation_count = max(0, len(article.lines) - 1)
        inferred = "Ja" if article.homonym_inferred else ""
        rows.append(
            '<tr tabindex="0" data-index="%d" data-column="%d" '
            'data-left="%d" data-top="%d" data-width="%d" data-height="%d" '
            'onclick="selectRow(this)">'
            '<td>%d</td><td><strong>%s</strong></td><td>%s</td><td>%d</td>'
            '<td>%.0f%%</td><td>%s</td><td><pre>%s</pre></td></tr>'
            % (
                index, article.column, article.left, article.top, article.width, article.height,
                article.column, _escape(article.headword), _escape(article.raw_headword),
                continuation_count, article.score * 100, inferred, _escape(article.article_text),
            )
        )

    observation_lines = _observation_line_indices(observations)
    tesseract_lines = [_normalized_observation_line(observations, line) for line in observation_lines]
    line_rows = [
        "<tr><td>%d</td><td>%d</td><td>%.3f</td><td>%s</td><td>%s</td></tr>"
        % (left, right, score, _escape(" ".join(tesseract_lines[left])), _escape(" ".join(runeberg_lines[right])))
        for left, right, score in pairs
    ]
    pair_scores = [score for _, _, score in pairs]
    mean_similarity = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
    image_data, image_width, image_height = _image_data_url(image_content)

    return f"""<!doctype html>
<html lang="sv"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Artikelgranskning – sida {page}</title>
<style>
:root{{color-scheme:light dark;font-family:system-ui,sans-serif}}body{{margin:0;background:#f3f4f6;color:#111827}}header{{position:sticky;top:0;z-index:20;padding:14px 20px;background:#111827;color:white}}header h1{{margin:0 0 8px;font-size:1.25rem}}.summary{{display:flex;flex-wrap:wrap;gap:8px}}.badge{{padding:4px 9px;border-radius:999px;background:#374151;font-size:.88rem}}.note{{margin-top:10px;font-size:.9rem}}main{{max-width:1900px;margin:auto;padding:18px}}.grid{{display:grid;grid-template-columns:minmax(360px,48%) minmax(560px,52%);gap:18px;align-items:start}}.panel{{background:white;border:1px solid #d1d5db;border-radius:10px;overflow:hidden}}.panel h2{{margin:0;padding:12px 14px;font-size:1rem;background:#e5e7eb}}.image-toolbar{{display:flex;align-items:center;gap:7px;padding:8px 12px}}.image-wrap{{height:calc(100vh - 225px);min-height:420px;overflow:auto;padding:12px;background:#d1d5db}}.scan-stage{{position:relative;width:{image_width}px;height:{image_height}px}}.scan-stage img{{display:block;width:{image_width}px;height:{image_height}px}}.marker{{position:absolute;display:none;box-sizing:border-box;border:4px solid #dc2626;background:#ef444422;border-radius:3px;pointer-events:none;box-shadow:0 0 0 2px white,0 0 14px #0008}}.table-wrap{{height:calc(100vh - 165px);min-height:480px;overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:.84rem}}th{{position:sticky;top:0;z-index:2;background:#e5e7eb;text-align:left}}th,td{{padding:6px 8px;border-bottom:1px solid #e5e7eb;vertical-align:top}}tbody tr{{cursor:pointer}}tbody tr:hover,tbody tr.selected{{outline:3px solid #2563eb;outline-offset:-3px}}td pre{{margin:0;white-space:pre-wrap;font:inherit}}details{{margin-top:18px}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}@media(prefers-color-scheme:dark){{body{{background:#111827;color:#f9fafb}}.panel{{background:#1f2937}}.panel h2,th{{background:#374151}}}}
</style></head><body>
<header><h1>Artikelgranskning – sida {page}</h1><div class="summary"><span class="badge">Artiklar: {len(articles)}</span><span class="badge">Spalt 1: {margins.get(1,0):.0f}px</span><span class="badge">Spalt 2: {margins.get(2,0):.0f}px</span><span class="badge">Starttröskel: {threshold:.0%}</span><span class="badge">Radsimilaritet: {mean_similarity:.3f}</span></div><div class="note">Oindragen rad startar artikel. Efterföljande indragna rader följer med tills nästa artikelstart. ↑/↓ går mellan artiklar, ←/→ byter spalt, Home/End går till början/slutet.</div></header>
<main><div class="grid"><section class="panel"><h2>Faksimil – <a href="{_escape(source_url)}">Runeberg</a></h2><div class="image-toolbar"><button onclick="changeZoom(-.25)">−</button><button onclick="fitImage()">Anpassa</button><button onclick="changeZoom(.25)">+</button><span id="zoomLabel">100 %</span></div><div class="image-wrap" id="imageWrap"><div class="scan-stage" id="scanStage"><img id="scanImage" src="{image_data}"><div class="marker" id="marker"></div></div></div></section><section class="panel"><h2>Artiklar</h2><div class="table-wrap" id="tableWrap"><table><thead><tr><th>Spalt</th><th>Huvudord</th><th>OCR-rubrik</th><th>Forts.</th><th>Start</th><th>Homonym lagad</th><th>Hela artikeln</th></tr></thead><tbody id="articleRows">{''.join(rows) or '<tr><td colspan="7">Inga artiklar hittades.</td></tr>'}</tbody></table></div></section></div><details><summary>Debug: matchade OCR-rader ({len(pairs)})</summary><table><tbody>{''.join(line_rows)}</tbody></table></details><details><summary>Runebergs råa OCR</summary><pre>{_escape(raw_text)}</pre></details></main>
<script>
const naturalWidth={image_width},naturalHeight={image_height};
const imageWrap=document.getElementById('imageWrap'),scanStage=document.getElementById('scanStage'),scanImage=document.getElementById('scanImage'),marker=document.getElementById('marker');
const rows=[...document.querySelectorAll('#articleRows tr[data-index]')];let zoom=1,selected=-1;
function setZoom(v){{zoom=Math.max(.15,Math.min(3,v));scanStage.style.width=`${{naturalWidth*zoom}}px`;scanStage.style.height=`${{naturalHeight*zoom}}px`;scanImage.style.width=`${{naturalWidth*zoom}}px`;scanImage.style.height=`${{naturalHeight*zoom}}px`;document.getElementById('zoomLabel').textContent=`${{Math.round(zoom*100)}} %`;if(selected>=0)positionMarker(rows[selected])}}
function fitImage(){{setZoom(Math.min(1,Math.max(100,imageWrap.clientWidth-24)/naturalWidth));imageWrap.scrollTo(0,0)}}
function changeZoom(d){{setZoom(zoom+d)}}
function positionMarker(r){{const l=+r.dataset.left,t=+r.dataset.top,w=+r.dataset.width,h=+r.dataset.height,m=5;marker.style.left=`${{(l-m)*zoom}}px`;marker.style.top=`${{(t-m)*zoom}}px`;marker.style.width=`${{(w+2*m)*zoom}}px`;marker.style.height=`${{(h+2*m)*zoom}}px`;marker.style.display='block'}}
function selectIndex(i){{if(!rows.length)return;i=Math.max(0,Math.min(rows.length-1,i));if(selected>=0)rows[selected].classList.remove('selected');selected=i;const r=rows[i];r.classList.add('selected');r.focus({{preventScroll:true}});r.scrollIntoView({{block:'nearest'}});positionMarker(r);const l=+r.dataset.left*zoom,t=+r.dataset.top*zoom,w=+r.dataset.width*zoom,h=+r.dataset.height*zoom;imageWrap.scrollTo({{left:Math.max(0,l+w/2-imageWrap.clientWidth/2),top:Math.max(0,t+h/2-imageWrap.clientHeight/2),behavior:'smooth'}})}}
function selectRow(r){{selectIndex(rows.indexOf(r))}}
function nearestInColumn(column,targetY){{let best=-1,dist=Infinity;rows.forEach((r,i)=>{{if(+r.dataset.column!==column)return;const y=+r.dataset.top+(+r.dataset.height)/2,d=Math.abs(y-targetY);if(d<dist){{dist=d;best=i}}}});return best}}
document.addEventListener('keydown',e=>{{if(!rows.length)return;if(selected<0)selectIndex(0);let target=selected;if(e.key==='ArrowDown')target=selected+1;else if(e.key==='ArrowUp')target=selected-1;else if(e.key==='Home')target=0;else if(e.key==='End')target=rows.length-1;else if(e.key==='PageDown')target=selected+10;else if(e.key==='PageUp')target=selected-10;else if(e.key==='ArrowLeft'||e.key==='ArrowRight'){{const r=rows[selected],column=+r.dataset.column,wanted=e.key==='ArrowLeft'?1:2;if(column===wanted)return;target=nearestInColumn(wanted,+r.dataset.top+(+r.dataset.height)/2)}}else return;e.preventDefault();if(target>=0)selectIndex(target)}});
window.addEventListener('load',()=>{{fitImage();if(rows.length)selectIndex(0)}});
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Granska SAOL som artiklar utifrån tvåspaltslayout och indrag.")
    parser.add_argument("page", nargs="?", type=int, default=19)
    parser.add_argument("--html", nargs="?", const="", metavar="FIL")
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.55, help="Lägsta poäng för att en oindragen rad ska starta en artikel.")
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold måste ligga mellan 0 och 1")

    source_url, image_url = page_urls(args.page)
    headers = {"User-Agent": "saol-tools/debug"}
    source_response = httpx.get(source_url, timeout=60.0, follow_redirects=True, headers=headers)
    source_response.raise_for_status()
    tif_url = ocr_image_url(image_url)
    image_response = httpx.get(tif_url, timeout=60.0, follow_redirects=True, headers=headers)
    image_response.raise_for_status()

    raw_text = _runeberg_ocr_text(source_response.text)
    runeberg_tokens = _runeberg_ocr_tokens(source_response.text)
    runeberg_lines = _runeberg_ocr_lines(source_response.text)
    observations = extract_observations(image_response.content)
    corrected = reconcile_contextual_observations(observations, source_response.text)
    observation_lines = _observation_line_indices(observations)
    tesseract_lines = [_normalized_observation_line(observations, line) for line in observation_lines]
    runeberg_normalized = [[token.casefold() for token in line] for line in runeberg_lines]
    pairs = _align_lines(tesseract_lines, runeberg_normalized)

    with Image.open(io.BytesIO(image_response.content)) as source_image:
        image_width, image_height = source_image.size

    lines, margins, indent_tolerance = _build_lines(corrected, image_width, image_height)
    articles = _group_articles(lines, args.threshold, indent_tolerance)
    articles = _infer_homonym_series(articles)

    print(f"Runeberg-URL: {source_url}")
    print(f"OCR-bild: {tif_url}")
    print(f"Runeberg-token: {len(runeberg_tokens)}")
    print(f"Tesseract-token totalt: {len(observations)}")
    print(f"Tryckta rader: {len(lines)}")
    print(f"Artiklar: {len(articles)}")
    print(f"Detekterad spaltmarginal: vänster={margins.get(1,0):.0f}px, höger={margins.get(2,0):.0f}px")
    for article in articles[:50]:
        suffix = " [homonymserie]" if article.homonym_inferred else ""
        print(f"  spalt={article.column} y={article.top:4d} rader={len(article.lines):2d}: {article.headword!r}{suffix}")

    if args.html is not None or args.open:
        output = Path(args.html or f"page{args.page:04d}-review.html").resolve()
        output.write_text(
            _review_html(
                args.page, source_url, image_response.content, raw_text, runeberg_lines,
                observations, articles, pairs, margins, args.threshold,
            ),
            encoding="utf-8",
        )
        print(f"HTML-rapport: {output}")
        if args.open:
            webbrowser.open(output.as_uri())


if __name__ == "__main__":
    main()
