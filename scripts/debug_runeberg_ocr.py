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
class LayoutCandidate:
    item: object
    column: int
    margin: float
    indent: float
    ink_ratio: float
    score: float


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


def _normalize_homonym_observations(observations: list) -> list:
    """Join SAOL homonym numbers with their headword.

    Tesseract may return the printed superscript as either one token (``1a``)
    or as a small raised digit followed by a separate ``a`` token. Both forms
    become ``¹a`` so homonyms remain distinct dictionary entries.
    """
    normalized = []
    consumed: set[int] = set()

    # First normalize combined OCR tokens such as 1a, 2a and 3a.
    observations = [
        replace(
            item,
            text=_homonym_text(match.group(1), match.group(2)),
            ocr_tesseract=_homonym_text(match.group(1), match.group(2)),
        )
        if (match := COMBINED_HOMONYM_RE.match(item.text.strip()))
        else item
        for item in observations
    ]

    median_height = statistics.median((item.height for item in observations), default=1.0)
    lines = _observation_line_indices(observations)
    for indices in lines:
        ordered = sorted(indices, key=lambda index: observations[index].left)
        for position, index in enumerate(ordered):
            if index in consumed:
                continue
            marker = observations[index]
            marker_text = marker.text.strip()
            if marker_text not in "123456789" or position + 1 >= len(ordered):
                continue

            word_index = ordered[position + 1]
            if word_index in consumed:
                continue
            word = observations[word_index]
            if not WORD_RE.match(word.text.strip()):
                continue

            gap = word.left - (marker.left + marker.width)
            marker_bottom = marker.top + marker.height
            word_middle = word.top + word.height * 0.55
            is_small = marker.height <= max(median_height * 0.9, word.height * 0.85)
            is_raised = marker_bottom <= word.top + word.height * 0.85 or marker.top < word.top
            is_close = -2 <= gap <= max(8.0, word.height * 0.8)
            if not (is_small and is_raised and is_close and marker.top < word_middle):
                continue

            left = min(marker.left, word.left)
            top = min(marker.top, word.top)
            right = max(marker.left + marker.width, word.left + word.width)
            bottom = max(marker.top + marker.height, word.top + word.height)
            text = _homonym_text(marker_text, word.text)
            total_area = max(1, marker.width * marker.height + word.width * word.height)
            ink_density = (
                marker.ink_density * marker.width * marker.height
                + word.ink_density * word.width * word.height
            ) / total_area
            merged = replace(
                word,
                text=text,
                left=left,
                top=top,
                width=right - left,
                height=bottom - top,
                confidence=min(marker.confidence, word.confidence),
                ink_density=ink_density,
                ocr_tesseract=text,
            )
            normalized.append(merged)
            consumed.add(index)
            consumed.add(word_index)

    normalized.extend(item for index, item in enumerate(observations) if index not in consumed)
    return sorted(normalized, key=lambda item: (item.top, item.left))


def _layout_candidates(observations: list, image_width: int, image_height: int) -> tuple[list[LayoutCandidate], dict[int, float]]:
    lines = _observation_line_indices(observations)
    if not lines:
        return [], {1: 0.0, 2: image_width / 2}

    page_left = min(item.left for item in observations)
    page_right = max(item.left + item.width for item in observations)
    split = page_left + (page_right - page_left) / 2
    median_height = statistics.median(item.height for item in observations)
    indent_tolerance = max(5.0, median_height * 0.9)

    line_starts: dict[int, list] = {1: [], 2: []}
    for indices in lines:
        if not indices:
            continue
        first = min((observations[index] for index in indices), key=lambda item: item.left)
        center_y = first.top + first.height / 2
        if center_y < image_height * 0.07 or center_y > image_height * 0.93:
            continue
        if not WORD_RE.search(first.text):
            continue
        column = 1 if first.left < split else 2
        line_starts[column].append(first)

    margins: dict[int, float] = {}
    candidates: list[LayoutCandidate] = []
    for column in (1, 2):
        starts = line_starts[column]
        if not starts:
            margins[column] = page_left if column == 1 else split
            continue

        lefts = [float(item.left) for item in starts]
        low_cut = _quantile(lefts, 0.30)
        low_cluster = [value for value in lefts if value <= low_cut + indent_tolerance]
        margin = statistics.median(low_cluster or lefts)
        margins[column] = margin

        inks = [item.ink_density for item in starts if item.ink_density > 0]
        ordinary_ink = statistics.median(inks) if inks else 1.0
        bold_reference = max(_quantile(inks, 0.72) if inks else ordinary_ink, ordinary_ink, 1e-6)

        for item in starts:
            indent = max(0.0, item.left - margin)
            margin_score = max(0.0, 1.0 - indent / (indent_tolerance * 2.2))
            ink_ratio = item.ink_density / bold_reference
            bold_score = max(0.0, min(1.0, (ink_ratio - 0.72) / 0.38))
            height_score = max(0.0, min(1.0, item.height / max(median_height, 1.0) - 0.72))
            confidence_score = max(0.0, min(1.0, item.confidence / 100.0))
            score = 0.58 * margin_score + 0.27 * bold_score + 0.10 * height_score + 0.05 * confidence_score
            if indent <= indent_tolerance * 2.2:
                candidates.append(LayoutCandidate(item, column, margin, indent, ink_ratio, score))

    return candidates, margins


def _review_html(
    page: int,
    source_url: str,
    image_content: bytes,
    raw_text: str,
    runeberg_lines: list[list[str]],
    observations: list,
    displayed: list[LayoutCandidate],
    pairs: list[tuple[int, int, float]],
    *,
    all_ocr: bool,
    threshold: float,
    margins: dict[int, float],
) -> str:
    displayed_items = [candidate.item for candidate in displayed]
    conflicts = [item for item in displayed_items if item.ocr_conflict]
    accepted = [item for item in displayed_items if item.ocr_runeberg and not item.ocr_conflict]
    pair_scores = [score for _, _, score in pairs]
    mean_similarity = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0

    rows = []
    for candidate in sorted(displayed, key=lambda value: (value.column, value.item.top, value.item.left)):
        item = candidate.item
        if item.ocr_conflict:
            status, css_class = "Konflikt", "conflict"
        elif item.ocr_runeberg:
            status, css_class = "Accepterad", "accepted"
        else:
            status, css_class = "Endast Tesseract", "tesseract-only"
        rows.append(
            '<tr class="%s" tabindex="0" role="button" data-left="%d" data-top="%d" data-width="%d" data-height="%d" onclick="focusObservation(this)" onkeydown="activateRow(event,this)">'
            '<td>%d</td><td>%s</td><td><strong>%s</strong></td><td>%s</td><td>%s</td><td>%.0f%%</td><td>%.1f</td><td>%.2f</td><td>%.1f</td></tr>'
            % (
                css_class, item.left, item.top, item.width, item.height,
                candidate.column, _escape(status), _escape(item.text),
                _escape(item.ocr_tesseract), _escape(item.ocr_runeberg),
                candidate.score * 100, candidate.indent, candidate.ink_ratio, item.confidence,
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
    mode = "alla OCR-radstarter" if all_ocr else f"artikelstarter ≥ {threshold:.0%}"
    heading = "Alla radstarter i två spalter" if all_ocr else "Troliga grundord / artikelstarter"
    return f"""<!doctype html>
<html lang="sv"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grundordsgranskning – sida {page}</title>
<style>
:root{{color-scheme:light dark;font-family:system-ui,sans-serif}}body{{margin:0;background:#f3f4f6;color:#111827}}header{{position:sticky;top:0;z-index:20;padding:14px 20px;background:#111827;color:white}}header h1{{margin:0 0 8px;font-size:1.25rem}}.summary{{display:flex;flex-wrap:wrap;gap:8px}}.badge{{padding:4px 9px;border-radius:999px;background:#374151;font-size:.88rem}}.badge.good{{background:#166534}}.badge.bad{{background:#991b1b}}.note{{margin-top:10px;font-size:.9rem;opacity:.9}}main{{max-width:1800px;margin:auto;padding:18px}}.grid{{display:grid;grid-template-columns:minmax(360px,48%) minmax(520px,52%);gap:18px;align-items:start}}.panel{{background:white;border:1px solid #d1d5db;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px #0002}}.panel h2{{margin:0;padding:12px 14px;font-size:1rem;background:#e5e7eb}}.image-toolbar{{display:flex;align-items:center;gap:7px;padding:8px 12px;border-bottom:1px solid #d1d5db}}.image-toolbar button{{padding:4px 10px;cursor:pointer}}.image-toolbar .hint{{margin-left:auto;font-size:.82rem;opacity:.72}}.image-wrap{{height:calc(100vh - 225px);min-height:420px;overflow:auto;padding:12px;background:#d1d5db}}.scan-stage{{position:relative;width:{image_width}px;height:{image_height}px;transform-origin:top left}}.scan-stage img{{display:block;width:{image_width}px;height:{image_height}px;background:white;user-select:none}}.marker{{position:absolute;display:none;box-sizing:border-box;border:4px solid #dc2626;background:#ef444433;border-radius:3px;pointer-events:none;z-index:5;box-shadow:0 0 0 2px white,0 0 14px #0009}}.table-wrap{{height:calc(100vh - 165px);min-height:480px;overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:.86rem}}th{{position:sticky;top:0;z-index:2;background:#e5e7eb;text-align:left}}th,td{{padding:6px 8px;border-bottom:1px solid #e5e7eb;vertical-align:top}}tbody tr{{cursor:pointer}}tbody tr:hover,tbody tr.selected{{outline:3px solid #2563eb;outline-offset:-3px}}tr.accepted{{background:#dcfce7}}tr.conflict{{background:#fee2e2}}tr.tesseract-only{{background:#fff}}details{{margin-top:18px}}summary{{cursor:pointer;font-weight:700;padding:10px 0}}pre{{white-space:pre-wrap;background:white;border:1px solid #d1d5db;border-radius:8px;padding:12px}}a{{color:inherit}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}.image-wrap{{height:62vh}}.table-wrap{{height:auto;max-height:62vh}}}}@media(prefers-color-scheme:dark){{body{{background:#111827;color:#f9fafb}}.panel,pre{{background:#1f2937;border-color:#4b5563}}.panel h2,th{{background:#374151}}th,td,.image-toolbar{{border-color:#4b5563}}.image-wrap{{background:#111827}}tr.accepted{{background:#14532d}}tr.conflict{{background:#7f1d1d}}tr.tesseract-only{{background:#1f2937}}}}
</style></head><body>
<header><h1>Grundordsgranskning – sida {page}</h1><div class="summary"><span class="badge">Läge: {_escape(mode)}</span><span class="badge">Spalt 1-marginal: {margins.get(1,0):.0f}px</span><span class="badge">Spalt 2-marginal: {margins.get(2,0):.0f}px</span><span class="badge">Kandidater: {len(displayed)}</span><span class="badge good">OCR-eniga: {len(accepted)}</span><span class="badge bad">OCR-konflikter: {len(conflicts)}</span><span class="badge">Radsimilaritet: {mean_similarity:.3f}</span></div><div class="note">Homonymnummer bevaras som upphöjda siffror: exempelvis ¹a, ²a och ³a är tre separata uppslagsord.</div></header>
<main><div class="grid"><section class="panel"><h2>Faksimil – <a href="{_escape(source_url)}">öppna hos Runeberg</a></h2><div class="image-toolbar"><button onclick="changeZoom(-.25)">−</button><button onclick="fitImage()">Anpassa</button><button onclick="changeZoom(.25)">+</button><span id="zoomLabel">100 %</span><span class="hint">Klicka på ett grundord för att hitta det</span></div><div class="image-wrap" id="imageWrap"><div class="scan-stage" id="scanStage"><img id="scanImage" src="{image_data}" alt="Skannad SAOL-sida {page}"><div class="marker" id="marker"></div></div></div></section><section class="panel"><h2>{heading}</h2><div class="table-wrap"><table><thead><tr><th>Spalt</th><th>Status</th><th>Grundord</th><th>Tesseract</th><th>Runeberg</th><th>Layout</th><th>Indrag</th><th>Svärta</th><th>OCR</th></tr></thead><tbody>{''.join(rows) or '<tr><td colspan="9">Inga artikelstarter hittades över vald tröskel.</td></tr>'}</tbody></table></div></section></div><details><summary>Debug: matchade OCR-rader ({len(pairs)})</summary><div class="panel table-wrap"><table><thead><tr><th>T-rad</th><th>R-rad</th><th>Likhet</th><th>Tesseract</th><th>Runeberg</th></tr></thead><tbody>{''.join(line_rows)}</tbody></table></div></details><details><summary>Debug: Runebergs råa OCR-text</summary><pre>{_escape(raw_text)}</pre></details></main>
<script>const naturalWidth={image_width},naturalHeight={image_height},imageWrap=document.getElementById('imageWrap'),scanStage=document.getElementById('scanStage'),scanImage=document.getElementById('scanImage'),marker=document.getElementById('marker'),zoomLabel=document.getElementById('zoomLabel');let zoom=1,selectedRow=null;function setZoom(v,k=true){{const o=zoom,cx=(imageWrap.scrollLeft+imageWrap.clientWidth/2)/o,cy=(imageWrap.scrollTop+imageWrap.clientHeight/2)/o;zoom=Math.max(.15,Math.min(3,v));scanStage.style.width=`${{naturalWidth*zoom}}px`;scanStage.style.height=`${{naturalHeight*zoom}}px`;scanImage.style.width=`${{naturalWidth*zoom}}px`;scanImage.style.height=`${{naturalHeight*zoom}}px`;zoomLabel.textContent=`${{Math.round(zoom*100)}} %`;if(selectedRow)positionMarker(selectedRow);if(k){{imageWrap.scrollLeft=cx*zoom-imageWrap.clientWidth/2;imageWrap.scrollTop=cy*zoom-imageWrap.clientHeight/2}}}}function fitImage(){{setZoom(Math.min(1,Math.max(100,imageWrap.clientWidth-24)/naturalWidth),false);imageWrap.scrollTo(0,0)}}function changeZoom(d){{setZoom(zoom+d)}}function positionMarker(r){{const l=+r.dataset.left,t=+r.dataset.top,w=+r.dataset.width,h=+r.dataset.height,m=Math.max(4,h*.25);marker.style.left=`${{(l-m)*zoom}}px`;marker.style.top=`${{(t-m)*zoom}}px`;marker.style.width=`${{(w+2*m)*zoom}}px`;marker.style.height=`${{(h+2*m)*zoom}}px`;marker.style.display='block'}}function focusObservation(r){{if(selectedRow)selectedRow.classList.remove('selected');selectedRow=r;r.classList.add('selected');if(zoom<.8)setZoom(1,false);positionMarker(r);const l=+r.dataset.left*zoom,t=+r.dataset.top*zoom,w=+r.dataset.width*zoom,h=+r.dataset.height*zoom;imageWrap.scrollTo({{left:Math.max(0,l+w/2-imageWrap.clientWidth/2),top:Math.max(0,t+h/2-imageWrap.clientHeight/2),behavior:'smooth'}})}}function activateRow(e,r){{if(e.key==='Enter'||e.key===' '){{e.preventDefault();focusObservation(r)}}}}window.addEventListener('load',fitImage);window.addEventListener('resize',()=>{{if(!selectedRow)fitImage()}});</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Granska SAOL-grundord utifrån sidans tvåspaltslayout.")
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
    observations = _normalize_homonym_observations(extract_observations(image_response.content))
    corrected = reconcile_contextual_observations(observations, source_response.text)
    observation_lines = _observation_line_indices(observations)
    tesseract_lines = [_normalized_observation_line(observations, line) for line in observation_lines]
    runeberg_normalized = [[token.casefold() for token in line] for line in runeberg_lines]
    pairs = _align_lines(tesseract_lines, runeberg_normalized)

    with Image.open(io.BytesIO(image_response.content)) as source_image:
        image_width, image_height = source_image.size
    raw_candidates, margins = _layout_candidates(observations, image_width, image_height)
    corrected_by_position = {(item.left, item.top, item.width, item.height): item for item in corrected}
    candidates = [
        LayoutCandidate(
            corrected_by_position.get((c.item.left, c.item.top, c.item.width, c.item.height), c.item),
            c.column, c.margin, c.indent, c.ink_ratio, c.score,
        )
        for c in raw_candidates
    ]
    displayed = candidates if args.all_ocr else [candidate for candidate in candidates if candidate.score >= args.threshold]

    conflicts = [candidate.item for candidate in displayed if candidate.item.ocr_conflict]
    print(f"Runeberg-URL: {source_url}")
    print(f"OCR-bild: {tif_url}")
    print(f"Runeberg-token: {len(runeberg_tokens)}")
    print(f"Tesseract-token efter homonymnormalisering: {len(observations)}")
    print(f"Detekterad spaltmarginal: vänster={margins.get(1,0):.0f}px, höger={margins.get(2,0):.0f}px")
    print(f"Visade {'radstarter' if args.all_ocr else 'grundordskandidater'}: {len(displayed)}")
    print(f"OCR-konflikter bland visade: {len(conflicts)}")
    for candidate in displayed[:40]:
        item = candidate.item
        print(f"  spalt={candidate.column} y={item.top:4d} x={item.left:4d} score={candidate.score:.2f} indrag={candidate.indent:.1f}: {item.text!r}")

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
