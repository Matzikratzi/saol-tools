from __future__ import annotations

import argparse
import base64
import html
import io
import math
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
    raw_start_x: float
    letter_start_x: float
    has_homonym_marker: bool
    x_class: str = "unknown"
    bold_score: float = 0.0


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


@dataclass(frozen=True)
class ColumnXModel:
    homonym_x: float | None
    article_x: float
    continuation_x: float
    boundary_x: float


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


def _homonym_text(number: str, word: str) -> str:
    return number.translate(SUPERSCRIPT) + word


def _linear_slope(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 3:
        return None
    mean_x = statistics.fmean(x for x, _ in points)
    mean_y = statistics.fmean(y for _, y in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator <= 1e-9:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def _estimate_skew_degrees(observations: list) -> float:
    slopes: list[float] = []
    for indices in _observation_line_indices(observations):
        items = sorted((observations[index] for index in indices), key=lambda item: item.left)
        if len(items) < 3:
            continue
        span = (items[-1].left + items[-1].width) - items[0].left
        median_height = statistics.median(item.height for item in items)
        if span < max(80.0, median_height * 5):
            continue
        points = [
            (item.left + item.width / 2, item.top + item.height / 2)
            for item in items
        ]
        slope = _linear_slope(points)
        if slope is not None and abs(slope) < math.tan(math.radians(3.0)):
            slopes.append(slope)
    if not slopes:
        return 0.0
    slope = statistics.median(slopes)
    return max(-2.0, min(2.0, math.degrees(math.atan(slope))))


def _deskew_image(content: bytes, observations: list) -> tuple[bytes, float]:
    skew_degrees = _estimate_skew_degrees(observations)
    if abs(skew_degrees) < 0.03:
        return content, 0.0
    with Image.open(io.BytesIO(content)) as source:
        image = source.convert("RGB")
        deskewed = image.rotate(-skew_degrees, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white")
        output = io.BytesIO()
        deskewed.save(output, format="PNG", optimize=True)
    return output.getvalue(), skew_degrees


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
    left, top = min(first.left, word.left), min(first.top, word.top)
    right = max(first.left + first.width, word.left + word.width)
    bottom = max(first.top + first.height, word.top + word.height)
    area1, area2 = first.width * first.height, word.width * word.height
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


def _line_start_positions(items: tuple[object, ...]) -> tuple[float, float, bool]:
    first = items[0]
    token = first.text.strip()
    raw_x = float(first.left)
    if token in "123456789" and len(items) > 1 and WORD_RE.search(items[1].text):
        return raw_x, float(items[1].left), True
    if token and token[0] in SUPERSCRIPT_DIGITS and len(token) > 1:
        return raw_x, raw_x + max(first.width * 0.16, first.height * 0.22), True
    if COMBINED_HOMONYM_RE.match(token):
        return raw_x, raw_x + max(first.width * 0.16, first.height * 0.22), True
    return raw_x, raw_x, False


def _kmeans_1d_two(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    ordered = sorted(values)
    left = ordered[max(0, round((len(ordered) - 1) * 0.2))]
    right = ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.8))]
    for _ in range(30):
        left_group = [value for value in values if abs(value - left) <= abs(value - right)]
        right_group = [value for value in values if abs(value - left) > abs(value - right)]
        new_left = statistics.fmean(left_group) if left_group else left
        new_right = statistics.fmean(right_group) if right_group else right
        if abs(new_left - left) + abs(new_right - right) < 0.01:
            break
        left, right = new_left, new_right
    return tuple(sorted((left, right)))


def _build_lines(
    observations: list,
    image_width: int,
    image_height: int,
) -> tuple[list[PrintedLine], dict[int, ColumnXModel]]:
    if not observations:
        empty = ColumnXModel(None, 0.0, 0.0, 0.0)
        return [], {1: empty, 2: empty}

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
        raw_x, letter_x, has_marker = _line_start_positions(ordered)
        text = " ".join(item.text for item in ordered)
        raw[column].append(
            PrintedLine(
                column=column,
                items=ordered,
                text=text,
                first=first,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                raw_start_x=raw_x,
                letter_start_x=letter_x,
                has_homonym_marker=has_marker,
            )
        )

    models: dict[int, ColumnXModel] = {}
    result: list[PrintedLine] = []
    for column in (1, 2):
        lines = sorted(raw[column], key=lambda line: (line.top, line.left))
        if not lines:
            fallback = page_left if column == 1 else split
            models[column] = ColumnXModel(None, fallback, fallback, fallback)
            continue

        lexical = [line for line in lines if WORD_RE.search(line.first.text)]
        letter_xs = [line.letter_start_x for line in lexical]
        article_x, continuation_x = _kmeans_1d_two(letter_xs)
        if continuation_x - article_x < max(3.0, median_height * 0.35):
            article_x = statistics.median(letter_xs) if letter_xs else lines[0].letter_start_x
            continuation_x = article_x + max(8.0, median_height * 0.8)
        boundary_x = (article_x + continuation_x) / 2
        marker_xs = [line.raw_start_x for line in lexical if line.has_homonym_marker]
        homonym_x = statistics.median(marker_xs) if marker_xs else None
        models[column] = ColumnXModel(homonym_x, article_x, continuation_x, boundary_x)

        inks = [line.first.ink_density for line in lexical if line.first.ink_density > 0]
        ordinary_ink = statistics.median(inks) if inks else 1.0
        bold_reference = max(
            sorted(inks)[min(len(inks) - 1, round((len(inks) - 1) * 0.75))] if inks else ordinary_ink,
            ordinary_ink,
            1e-6,
        )

        for line in lines:
            ink_ratio = line.first.ink_density / bold_reference
            bold_score = max(0.0, min(1.0, (ink_ratio - 0.70) / 0.34))
            x_class = "article" if line.letter_start_x <= boundary_x else "continuation"
            if line.has_homonym_marker:
                x_class = "homonym+" + x_class
            result.append(replace(line, x_class=x_class, bold_score=bold_score))

    return sorted(result, key=lambda line: (line.column, line.top, line.left)), models


def _group_articles(lines: list[PrintedLine], threshold: float) -> list[Article]:
    articles: list[Article] = []
    for column in (1, 2):
        current: list[PrintedLine] = []
        current_score = 0.0
        for line in (line for line in lines if line.column == column):
            lexical = bool(WORD_RE.search(line.first.text))
            at_article_x = line.x_class.endswith("article")
            is_start = lexical and at_article_x and line.bold_score >= threshold
            if is_start:
                if current:
                    first = current[0]
                    articles.append(Article(column, first.first.text, first.first.text, tuple(current), current_score))
                current = [line]
                current_score = line.bold_score
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
        return marker if marker.isdigit() else None, match.group(2).casefold()
    return None, token.casefold()


def _infer_homonym_series(articles: list[Article]) -> list[Article]:
    result = list(articles)
    for column in (1, 2):
        indices = [index for index, article in enumerate(result) if article.column == column]
        pos = 0
        while pos < len(indices):
            start = pos
            _, base = _base_headword(result[indices[pos]].headword)
            pos += 1
            while pos < len(indices) and _base_headword(result[indices[pos]].headword)[1] == base:
                pos += 1
            run = indices[start:pos]
            if len(run) < 2 or not base:
                continue
            parsed = [_base_headword(result[index].headword)[0] for index in run]
            explicit = [int(value) for value in parsed if value is not None]
            if explicit and explicit != sorted(explicit):
                continue
            for sequence_number, index in enumerate(run, start=1):
                article = result[index]
                expected = _homonym_text(str(sequence_number), base)
                number, current_base = _base_headword(article.headword)
                if current_base == base:
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
    x_models: dict[int, ColumnXModel],
    threshold: float,
    skew_degrees: float,
) -> str:
    rows: list[str] = []
    for index, article in enumerate(articles):
        first = article.lines[0]
        continuation_count = max(0, len(article.lines) - 1)
        inferred = "Ja" if article.homonym_inferred else ""
        rows.append(
            '<tr tabindex="0" data-index="%d" data-column="%d" '
            'data-left="%d" data-top="%d" data-width="%d" data-height="%d" onclick="selectRow(this)">'
            '<td>%d</td><td><strong>%s</strong></td><td>%s</td><td>%s</td><td>%.1f</td>'
            '<td>%d</td><td>%.0f%%</td><td>%s</td><td><pre>%s</pre></td></tr>'
            % (
                index, article.column, article.left, article.top, article.width, article.height,
                article.column, _escape(article.headword), _escape(article.raw_headword),
                _escape(first.x_class), first.letter_start_x, continuation_count,
                article.score * 100, inferred, _escape(article.article_text),
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

    model_badges = []
    for column in (1, 2):
        model = x_models[column]
        homonym = "–" if model.homonym_x is None else f"{model.homonym_x:.1f}"
        model_badges.append(
            f'<span class="badge">Spalt {column}: homonym {homonym}, artikel {model.article_x:.1f}, forts. {model.continuation_x:.1f}</span>'
        )

    return f"""<!doctype html>
<html lang="sv"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Artikelgranskning – sida {page}</title>
<style>
:root{{color-scheme:light dark;font-family:system-ui,sans-serif}}body{{margin:0;background:#f3f4f6;color:#111827}}header{{position:sticky;top:0;z-index:20;padding:14px 20px;background:#111827;color:white}}header h1{{margin:0 0 8px;font-size:1.25rem}}.summary{{display:flex;flex-wrap:wrap;gap:8px}}.badge{{padding:4px 9px;border-radius:999px;background:#374151;font-size:.88rem}}.note{{margin-top:10px;font-size:.9rem}}main{{max-width:1900px;margin:auto;padding:18px}}.grid{{display:grid;grid-template-columns:minmax(360px,48%) minmax(560px,52%);gap:18px;align-items:start}}.panel{{background:white;border:1px solid #d1d5db;border-radius:10px;overflow:hidden}}.panel h2{{margin:0;padding:12px 14px;font-size:1rem;background:#e5e7eb}}.image-toolbar{{display:flex;align-items:center;gap:7px;padding:8px 12px}}.image-wrap{{height:calc(100vh - 245px);min-height:420px;overflow:auto;padding:12px;background:#d1d5db}}.scan-stage{{position:relative;width:{image_width}px;height:{image_height}px}}.scan-stage img{{display:block;width:{image_width}px;height:{image_height}px}}.marker{{position:absolute;display:none;box-sizing:border-box;border:4px solid #dc2626;background:#ef444422;border-radius:3px;pointer-events:none;box-shadow:0 0 0 2px white,0 0 14px #0008}}.table-wrap{{height:calc(100vh - 185px);min-height:480px;overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:.82rem}}th{{position:sticky;top:0;z-index:2;background:#e5e7eb;text-align:left}}th,td{{padding:6px 8px;border-bottom:1px solid #e5e7eb;vertical-align:top}}tbody tr{{cursor:pointer}}tbody tr:hover,tbody tr.selected{{outline:3px solid #2563eb;outline-offset:-3px}}td pre{{margin:0;white-space:pre-wrap;font:inherit}}details{{margin-top:18px}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}@media(prefers-color-scheme:dark){{body{{background:#111827;color:#f9fafb}}.panel{{background:#1f2937}}.panel h2,th{{background:#374151}}}}
</style></head><body>
<header><h1>Artikelgranskning – sida {page}</h1><div class="summary"><span class="badge">Artiklar: {len(articles)}</span><span class="badge">Deskew: {skew_degrees:+.3f}°</span><span class="badge">Fetstilströskel: {threshold:.0%}</span><span class="badge">Radsimilaritet: {mean_similarity:.3f}</span>{''.join(model_badges)}</div><div class="note">Sidan rätas först upp. Första normala bokstavens x-värde delas därefter i artikelstart och fortsättningsrad; homonymindex mäts separat. Ny artikel kräver artikel-x och fetstil.</div></header>
<main><div class="grid"><section class="panel"><h2>Faksimil – <a href="{_escape(source_url)}">Runeberg</a></h2><div class="image-toolbar"><button onclick="changeZoom(-.25)">−</button><button onclick="fitImage()">Anpassa</button><button onclick="changeZoom(.25)">+</button><span id="zoomLabel">100 %</span></div><div class="image-wrap" id="imageWrap"><div class="scan-stage" id="scanStage"><img id="scanImage" src="{image_data}"><div class="marker" id="marker"></div></div></div></section><section class="panel"><h2>Artiklar</h2><div class="table-wrap" id="tableWrap"><table><thead><tr><th>Spalt</th><th>Huvudord</th><th>OCR-rubrik</th><th>x-klass</th><th>x</th><th>Forts.</th><th>Fet</th><th>Homonym lagad</th><th>Hela artikeln</th></tr></thead><tbody id="articleRows">{''.join(rows) or '<tr><td colspan="9">Inga artiklar hittades.</td></tr>'}</tbody></table></div></section></div><details><summary>Debug: matchade OCR-rader ({len(pairs)})</summary><table><tbody>{''.join(line_rows)}</tbody></table></details><details><summary>Runebergs råa OCR</summary><pre>{_escape(raw_text)}</pre></details></main>
<script>
const naturalWidth={image_width},naturalHeight={image_height};
const imageWrap=document.getElementById('imageWrap'),scanStage=document.getElementById('scanStage'),scanImage=document.getElementById('scanImage'),marker=document.getElementById('marker');
const rows=[...document.querySelectorAll('#articleRows tr[data-index]')];let zoom=1,selected=-1;
function setZoom(v){{zoom=Math.max(.15,Math.min(3,v));scanStage.style.width=`${{naturalWidth*zoom}}px`;scanStage.style.height=`${{naturalHeight*zoom}}px`;scanImage.style.width=`${{naturalWidth*zoom}}px`;scanImage.style.height=`${{naturalHeight*zoom}}px`;document.getElementById('zoomLabel').textContent=`${{Math.round(zoom*100)}} %`;if(selected>=0)positionMarker(rows[selected])}}
function fitImage(){{setZoom(Math.min(1,Math.max(100,imageWrap.clientWidth-24)/naturalWidth));imageWrap.scrollTo(0,0)}}
function changeZoom(d){{setZoom(zoom+d)}}
function markerBounds(r){{const l=+r.dataset.left,t=+r.dataset.top,w=+r.dataset.width,h=+r.dataset.height;const padX=Math.max(12,h*.35),padY=Math.max(9,h*.22);const left=Math.max(0,l-padX),top=Math.max(0,t-padY),right=Math.min(naturalWidth,l+w+padX),bottom=Math.min(naturalHeight,t+h+padY);return{{left,top,width:Math.max(1,right-left),height:Math.max(1,bottom-top)}}}}
function positionMarker(r){{const b=markerBounds(r);marker.style.left=`${{b.left*zoom}}px`;marker.style.top=`${{b.top*zoom}}px`;marker.style.width=`${{b.width*zoom}}px`;marker.style.height=`${{b.height*zoom}}px`;marker.style.display='block'}}
function selectIndex(i){{if(!rows.length)return;i=Math.max(0,Math.min(rows.length-1,i));if(selected>=0)rows[selected].classList.remove('selected');selected=i;const r=rows[i];r.classList.add('selected');r.focus({{preventScroll:true}});r.scrollIntoView({{block:'nearest'}});positionMarker(r);const b=markerBounds(r);imageWrap.scrollTo({{left:Math.max(0,(b.left+b.width/2)*zoom-imageWrap.clientWidth/2),top:Math.max(0,(b.top+b.height/2)*zoom-imageWrap.clientHeight/2),behavior:'smooth'}})}}
function selectRow(r){{selectIndex(rows.indexOf(r))}}
function nearestInColumn(column,targetY){{let best=-1,dist=Infinity;rows.forEach((r,i)=>{{if(+r.dataset.column!==column)return;const y=+r.dataset.top+(+r.dataset.height)/2,d=Math.abs(y-targetY);if(d<dist){{dist=d;best=i}}}});return best}}
document.addEventListener('keydown',e=>{{if(!rows.length)return;if(selected<0)selectIndex(0);let target=selected;if(e.key==='ArrowDown')target=selected+1;else if(e.key==='ArrowUp')target=selected-1;else if(e.key==='Home')target=0;else if(e.key==='End')target=rows.length-1;else if(e.key==='PageDown')target=selected+10;else if(e.key==='PageUp')target=selected-10;else if(e.key==='ArrowLeft'||e.key==='ArrowRight'){{const r=rows[selected],column=+r.dataset.column,wanted=e.key==='ArrowLeft'?1:2;if(column===wanted)return;target=nearestInColumn(wanted,+r.dataset.top+(+r.dataset.height)/2)}}else return;e.preventDefault();if(target>=0)selectIndex(target)}});
window.addEventListener('load',()=>{{fitImage();if(rows.length)selectIndex(0)}});
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Granska SAOL som artiklar utifrån deskew, x-kluster och fetstil.")
    parser.add_argument("page", nargs="?", type=int, default=19)
    parser.add_argument("--html", nargs="?", const="", metavar="FIL")
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.55, help="Lägsta fetstilspoäng för en rad i artikelstartsklustret.")
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

    initial_observations = extract_observations(image_response.content)
    deskewed_content, skew_degrees = _deskew_image(image_response.content, initial_observations)
    observations = extract_observations(deskewed_content) if deskewed_content is not image_response.content else initial_observations
    corrected = reconcile_contextual_observations(observations, source_response.text)

    observation_lines = _observation_line_indices(observations)
    tesseract_lines = [_normalized_observation_line(observations, line) for line in observation_lines]
    runeberg_normalized = [[token.casefold() for token in line] for line in runeberg_lines]
    pairs = _align_lines(tesseract_lines, runeberg_normalized)

    with Image.open(io.BytesIO(deskewed_content)) as source_image:
        image_width, image_height = source_image.size
    lines, x_models = _build_lines(corrected, image_width, image_height)
    articles = _infer_homonym_series(_group_articles(lines, args.threshold))

    print(f"Runeberg-URL: {source_url}")
    print(f"OCR-bild: {tif_url}")
    print(f"Deskew: {skew_degrees:+.3f}°")
    print(f"Runeberg-token: {len(runeberg_tokens)}")
    print(f"Tesseract-token totalt: {len(observations)}")
    print(f"Tryckta rader: {len(lines)}")
    print(f"Artiklar: {len(articles)}")
    for column in (1, 2):
        model = x_models[column]
        print(
            f"Spalt {column}: homonym={model.homonym_x}, "
            f"artikel={model.article_x:.1f}, fortsättning={model.continuation_x:.1f}, "
            f"gräns={model.boundary_x:.1f}"
        )
    for article in articles[:50]:
        suffix = " [homonymserie]" if article.homonym_inferred else ""
        first = article.lines[0]
        print(
            f"  spalt={article.column} y={article.top:4d} x={first.letter_start_x:6.1f} "
            f"klass={first.x_class:18s} fet={article.score:.2f} rader={len(article.lines):2d}: "
            f"{article.headword!r}{suffix}"
        )

    if args.html is not None or args.open:
        output = Path(args.html or f"page{args.page:04d}-review.html").resolve()
        output.write_text(
            _review_html(
                args.page,
                source_url,
                deskewed_content,
                raw_text,
                runeberg_lines,
                observations,
                articles,
                pairs,
                x_models,
                args.threshold,
                skew_degrees,
            ),
            encoding="utf-8",
        )
        print(f"HTML-rapport: {output}")
        if args.open:
            webbrowser.open(output.as_uri())


if __name__ == "__main__":
    main()
