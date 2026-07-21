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
SUPERSCRIPT = str.maketrans("123456789", "¹²³⁴⁵⁶⁷⁸⁹")


@dataclass(frozen=True)
class HeadwordCandidate:
    item: object
    column: int
    margin: float
    indent: float
    ink_ratio: float
    score: float
    line_text: str


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


def _merge_line_headword(observations: list, indices: list[int], median_height: float) -> tuple[object, set[int]]:
    """Return the first lexical unit on a printed line.

    A raised homonym digit may be part of the same OCR token (``1a``) or a
    separate token immediately before the word. Only the beginning of the line
    is considered, so ordinary digits later in an article are never merged.
    """
    ordered = sorted(indices, key=lambda index: observations[index].left)
    if not ordered:
        raise ValueError("tom OCR-rad")

    first_index = ordered[0]
    first = observations[first_index]
    first_text = first.text.strip()

    if match := COMBINED_HOMONYM_RE.match(first_text):
        text = _homonym_text(match.group(1), match.group(2))
        return replace(first, text=text, ocr_tesseract=text), {first_index}

    if first_text not in "123456789" or len(ordered) < 2:
        return first, {first_index}

    word_index = ordered[1]
    word = observations[word_index]
    if not WORD_RE.match(word.text.strip()):
        return first, {first_index}

    gap = word.left - (first.left + first.width)
    digit_bottom = first.top + first.height
    is_small = first.height <= max(median_height * 0.95, word.height * 0.9)
    is_raised = digit_bottom <= word.top + word.height * 0.9 or first.top < word.top
    is_close = -3 <= gap <= max(10.0, word.height)
    if not (is_small and is_raised and is_close):
        return first, {first_index}

    left = min(first.left, word.left)
    top = min(first.top, word.top)
    right = max(first.left + first.width, word.left + word.width)
    bottom = max(first.top + first.height, word.top + word.height)
    area1 = first.width * first.height
    area2 = word.width * word.height
    total_area = max(1, area1 + area2)
    text = _homonym_text(first_text, word.text)
    merged = replace(
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
    return merged, {first_index, word_index}


def _line_headwords(observations: list) -> list[tuple[object, str]]:
    """Build one possible article start from each printed OCR line."""
    if not observations:
        return []
    heights = [item.height for item in observations]
    median_height = statistics.median(heights) if heights else 1.0
    result: list[tuple[object, str]] = []
    for indices in _observation_line_indices(observations):
        if not indices:
            continue
        ordered = sorted(indices, key=lambda index: observations[index].left)
        line_text = " ".join(observations[index].text for index in ordered)
        first, _ = _merge_line_headword(observations, ordered, median_height)
        if WORD_RE.search(first.text):
            result.append((first, line_text))
    return result


def _headword_candidates(observations: list, image_width: int, image_height: int) -> tuple[list[HeadwordCandidate], dict[int, float]]:
    line_starts = _line_headwords(observations)
    if not line_starts:
        return [], {1: 0.0, 2: image_width / 2}

    page_left = min(item.left for item in observations)
    page_right = max(item.left + item.width for item in observations)
    split = page_left + (page_right - page_left) / 2
    heights = [item.height for item in observations]
    median_height = statistics.median(heights) if heights else 1.0
    indent_tolerance = max(5.0, median_height * 0.9)

    by_column: dict[int, list[tuple[object, str]]] = {1: [], 2: []}
    for item, line_text in line_starts:
        center_y = item.top + item.height / 2
        if center_y < image_height * 0.07 or center_y > image_height * 0.93:
            continue
        column = 1 if item.left < split else 2
        by_column[column].append((item, line_text))

    margins: dict[int, float] = {}
    candidates: list[HeadwordCandidate] = []
    for column in (1, 2):
        starts = by_column[column]
        if not starts:
            margins[column] = page_left if column == 1 else split
            continue

        lefts = [float(item.left) for item, _ in starts]
        low_cut = _quantile(lefts, 0.30)
        low_cluster = [value for value in lefts if value <= low_cut + indent_tolerance]
        margin = statistics.median(low_cluster or lefts)
        margins[column] = margin

        inks = [item.ink_density for item, _ in starts if item.ink_density > 0]
        ordinary_ink = statistics.median(inks) if inks else 1.0
        bold_reference = max(_quantile(inks, 0.72) if inks else ordinary_ink, ordinary_ink, 1e-6)

        for item, line_text in starts:
            indent = max(0.0, item.left - margin)
            margin_score = max(0.0, 1.0 - indent / (indent_tolerance * 2.2))
            ink_ratio = item.ink_density / bold_reference
            bold_score = max(0.0, min(1.0, (ink_ratio - 0.72) / 0.38))
            height_score = max(0.0, min(1.0, item.height / max(median_height, 1.0) - 0.72))
            confidence_score = max(0.0, min(1.0, item.confidence / 100.0))
            score = 0.58 * margin_score + 0.27 * bold_score + 0.10 * height_score + 0.05 * confidence_score
            if indent <= indent_tolerance * 2.2:
                candidates.append(HeadwordCandidate(item, column, margin, indent, ink_ratio, score, line_text))

    return candidates, margins


def _review_html(
    page: int,
    source_url: str,
    image_content: bytes,
    raw_text: str,
    runeberg_lines: list[list[str]],
    observations: list,
    displayed: list[HeadwordCandidate],
    pairs: list[tuple[int, int, float]],
    *,
    all_ocr: bool,
    threshold: float,
    margins: dict[int, float],
) -> str:
    conflicts = [candidate.item for candidate in displayed if candidate.item.ocr_conflict]
    accepted = [candidate.item for candidate in displayed if candidate.item.ocr_runeberg and not candidate.item.ocr_conflict]
    pair_scores = [score for _, _, score in pairs]
    mean_similarity = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0

    rows: list[str] = []
    for candidate in sorted(displayed, key=lambda value: (value.column, value.item.top, value.item.left)):
        item = candidate.item
        if item.ocr_conflict:
            status, css_class = "Konflikt", "conflict"
        elif item.ocr_runeberg:
            status, css_class = "Accepterad", "accepted"
        else:
            status, css_class = "Endast Tesseract", "tesseract-only"
        rows.append(
            '<tr class="%s" tabindex="0" data-left="%d" data-top="%d" data-width="%d" data-height="%d" onclick="focusObservation(this)">'
            '<td>%d</td><td>%s</td><td><strong>%s</strong></td><td>%s</td><td>%.0f%%</td><td>%.1f</td><td>%.2f</td><td>%s</td></tr>'
            % (
                css_class, item.left, item.top, item.width, item.height,
                candidate.column, _escape(status), _escape(item.text), _escape(item.ocr_runeberg),
                candidate.score * 100, candidate.indent, candidate.ink_ratio, _escape(candidate.line_text),
            )
        )

    observation_lines = _observation_line_indices(observations)
    tesseract_lines = [_normalized_observation_line(observations, line) for line in observation_lines]
    line_rows = [
        "<tr><td>%d</td><td>%d</td><td>%.3f</td><td>%s</td><td>%s</td></tr>"
        % (left, right, score, _escape(" ".join(tesseract_lines[left])), _escape(" ".join(runeberg_lines[right])))
        for left, right, score in pairs
    ]

    image_data, image_width, image_height = _image_data_url(image_content)
    mode = "alla radstarter nära marginalen" if all_ocr else f"artikelstarter ≥ {threshold:.0%}"
    heading = "Alla möjliga artikelstarter" if all_ocr else "Troliga grundord / artikelstarter"
    empty = '<tr><td colspan="8">Inga artikelstarter hittades över vald tröskel.</td></tr>'

    return f"""<!doctype html>
<html lang="sv"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grundordsgranskning – sida {page}</title>
<style>
:root{{color-scheme:light dark;font-family:system-ui,sans-serif}}body{{margin:0;background:#f3f4f6;color:#111827}}header{{position:sticky;top:0;z-index:20;padding:14px 20px;background:#111827;color:white}}header h1{{margin:0 0 8px;font-size:1.25rem}}.summary{{display:flex;flex-wrap:wrap;gap:8px}}.badge{{padding:4px 9px;border-radius:999px;background:#374151;font-size:.88rem}}.badge.good{{background:#166534}}.badge.bad{{background:#991b1b}}.note{{margin-top:10px;font-size:.9rem}}main{{max-width:1800px;margin:auto;padding:18px}}.grid{{display:grid;grid-template-columns:minmax(360px,48%) minmax(520px,52%);gap:18px}}.panel{{background:white;border:1px solid #d1d5db;border-radius:10px;overflow:hidden}}.panel h2{{margin:0;padding:12px 14px;font-size:1rem;background:#e5e7eb}}.image-toolbar{{display:flex;gap:7px;padding:8px 12px}}.image-wrap{{height:calc(100vh - 225px);min-height:420px;overflow:auto;padding:12px;background:#d1d5db}}.scan-stage{{position:relative;width:{image_width}px;height:{image_height}px}}.scan-stage img{{display:block;width:{image_width}px;height:{image_height}px}}.marker{{position:absolute;display:none;box-sizing:border-box;border:4px solid #dc2626;background:#ef444433;border-radius:3px;pointer-events:none}}.table-wrap{{height:calc(100vh - 165px);min-height:480px;overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:.86rem}}th{{position:sticky;top:0;background:#e5e7eb;text-align:left}}th,td{{padding:6px 8px;border-bottom:1px solid #e5e7eb;vertical-align:top}}tbody tr{{cursor:pointer}}tbody tr:hover,tbody tr.selected{{outline:3px solid #2563eb;outline-offset:-3px}}tr.accepted{{background:#dcfce7}}tr.conflict{{background:#fee2e2}}details{{margin-top:18px}}pre{{white-space:pre-wrap}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}@media(prefers-color-scheme:dark){{body{{background:#111827;color:#f9fafb}}.panel{{background:#1f2937}}.panel h2,th{{background:#374151}}tr.accepted{{background:#14532d}}tr.conflict{{background:#7f1d1d}}}}
</style></head><body>
<header><h1>Grundordsgranskning – sida {page}</h1><div class="summary"><span class="badge">Läge: {_escape(mode)}</span><span class="badge">Spalt 1: {margins.get(1,0):.0f}px</span><span class="badge">Spalt 2: {margins.get(2,0):.0f}px</span><span class="badge">Kandidater: {len(displayed)}</span><span class="badge good">OCR-eniga: {len(accepted)}</span><span class="badge bad">Konflikter: {len(conflicts)}</span><span class="badge">Radsimilaritet: {mean_similarity:.3f}</span></div><div class="note">Varje kandidat byggs från början av en hel tryckt rad. Upphöjda homonymnummer bevaras, så ¹a, ²a och ³a blir tre separata uppslagsord.</div></header>
<main><div class="grid"><section class="panel"><h2>Faksimil – <a href="{_escape(source_url)}">Runeberg</a></h2><div class="image-toolbar"><button onclick="changeZoom(-.25)">−</button><button onclick="fitImage()">Anpassa</button><button onclick="changeZoom(.25)">+</button><span id="zoomLabel">100 %</span></div><div class="image-wrap" id="imageWrap"><div class="scan-stage" id="scanStage"><img id="scanImage" src="{image_data}"><div class="marker" id="marker"></div></div></div></section><section class="panel"><h2>{heading}</h2><div class="table-wrap"><table><thead><tr><th>Spalt</th><th>Status</th><th>Grundord</th><th>Runeberg</th><th>Layout</th><th>Indrag</th><th>Svärta</th><th>Hel rad</th></tr></thead><tbody>{''.join(rows) or empty}</tbody></table></div></section></div><details><summary>Debug: matchade OCR-rader ({len(pairs)})</summary><table><tbody>{''.join(line_rows)}</tbody></table></details><details><summary>Runebergs råa OCR</summary><pre>{_escape(raw_text)}</pre></details></main>
<script>const naturalWidth={image_width},naturalHeight={image_height},imageWrap=document.getElementById('imageWrap'),scanStage=document.getElementById('scanStage'),scanImage=document.getElementById('scanImage'),marker=document.getElementById('marker'),zoomLabel=document.getElementById('zoomLabel');let zoom=1,selectedRow=null;function setZoom(v){{zoom=Math.max(.15,Math.min(3,v));scanStage.style.width=`${{naturalWidth*zoom}}px`;scanStage.style.height=`${{naturalHeight*zoom}}px`;scanImage.style.width=`${{naturalWidth*zoom}}px`;scanImage.style.height=`${{naturalHeight*zoom}}px`;zoomLabel.textContent=`${{Math.round(zoom*100)}} %`;if(selectedRow)positionMarker(selectedRow)}}function fitImage(){{setZoom(Math.min(1,Math.max(100,imageWrap.clientWidth-24)/naturalWidth));imageWrap.scrollTo(0,0)}}function changeZoom(d){{setZoom(zoom+d)}}function positionMarker(r){{const l=+r.dataset.left,t=+r.dataset.top,w=+r.dataset.width,h=+r.dataset.height,m=Math.max(4,h*.25);marker.style.left=`${{(l-m)*zoom}}px`;marker.style.top=`${{(t-m)*zoom}}px`;marker.style.width=`${{(w+2*m)*zoom}}px`;marker.style.height=`${{(h+2*m)*zoom}}px`;marker.style.display='block'}}function focusObservation(r){{if(selectedRow)selectedRow.classList.remove('selected');selectedRow=r;r.classList.add('selected');positionMarker(r);const l=+r.dataset.left*zoom,t=+r.dataset.top*zoom,w=+r.dataset.width*zoom,h=+r.dataset.height*zoom;imageWrap.scrollTo({{left:Math.max(0,l+w/2-imageWrap.clientWidth/2),top:Math.max(0,t+h/2-imageWrap.clientHeight/2),behavior:'smooth'}})}}window.addEventListener('load',fitImage);</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Granska SAOL-grundord utifrån tvåspaltslayout och tryckta rader.")
    parser.add_argument("page", nargs="?", type=int, default=19)
    parser.add_argument("--html", nargs="?", const="", metavar="FIL")
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--all-ocr", action="store_true", help="Visa alla radstarter nära spaltnas vänstermarginaler.")
    parser.add_argument("--threshold", type=float, default=0.55)
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

    raw_candidates, margins = _headword_candidates(observations, image_width, image_height)
    corrected_by_position = {(item.left, item.top, item.width, item.height): item for item in corrected}
    candidates = [
        replace(candidate, item=corrected_by_position.get((candidate.item.left, candidate.item.top, candidate.item.width, candidate.item.height), candidate.item))
        for candidate in raw_candidates
    ]
    displayed = candidates if args.all_ocr else [candidate for candidate in candidates if candidate.score >= args.threshold]

    print(f"Runeberg-URL: {source_url}")
    print(f"OCR-bild: {tif_url}")
    print(f"Runeberg-token: {len(runeberg_tokens)}")
    print(f"Tesseract-token totalt: {len(observations)}")
    print(f"Detekterad spaltmarginal: vänster={margins.get(1,0):.0f}px, höger={margins.get(2,0):.0f}px")
    print(f"Visade {'radstarter' if args.all_ocr else 'grundordskandidater'}: {len(displayed)}")
    for candidate in displayed[:40]:
        item = candidate.item
        print(f"  spalt={candidate.column} y={item.top:4d} x={item.left:4d} score={candidate.score:.2f}: {item.text!r}")

    if args.html is not None or args.open:
        output = Path(args.html or f"page{args.page:04d}-review.html").resolve()
        output.write_text(
            _review_html(
                args.page, source_url, image_response.content, raw_text, runeberg_lines,
                observations, displayed, pairs, all_ocr=args.all_ocr,
                threshold=args.threshold, margins=margins,
            ),
            encoding="utf-8",
        )
        print(f"HTML-rapport: {output}")
        if args.open:
            webbrowser.open(output.as_uri())


if __name__ == "__main__":
    main()
