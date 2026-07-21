from __future__ import annotations

import argparse
import base64
import html
import io
import webbrowser
from pathlib import Path

import httpx
from PIL import Image

from app.classifier import HeadwordModel
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


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _image_data_url(content: bytes) -> tuple[str, int, int]:
    """Convert the Runeberg TIFF to a browser-friendly embedded PNG."""
    with Image.open(io.BytesIO(content)) as source:
        image = source.convert("RGB")
        width, height = image.size
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}", width, height


def _review_html(
    page: int,
    source_url: str,
    image_content: bytes,
    raw_text: str,
    runeberg_lines: list[list[str]],
    observations: list,
    displayed: list[tuple[object, float | None]],
    pairs: list[tuple[int, int, float]],
    *,
    all_ocr: bool,
    threshold: float,
    fallback_reason: str | None,
) -> str:
    displayed_items = [item for item, _ in displayed]
    runeberg_values = [item for item in displayed_items if item.ocr_runeberg]
    conflicts = [item for item in displayed_items if item.ocr_conflict]
    accepted = [item for item in runeberg_values if not item.ocr_conflict]
    pair_scores = [score for _, _, score in pairs]
    mean_similarity = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0

    rows: list[str] = []
    for item, probability in sorted(displayed, key=lambda value: (value[0].top, value[0].left)):
        if item.ocr_conflict:
            status = "Konflikt"
            css_class = "conflict"
        elif item.ocr_runeberg:
            status = "Accepterad"
            css_class = "accepted"
        else:
            status = "Endast Tesseract"
            css_class = "tesseract-only"
        probability_text = "–" if probability is None else f"{probability * 100:.1f}%"
        rows.append(
            '<tr class="%s" tabindex="0" role="button" '
            'data-left="%d" data-top="%d" data-width="%d" data-height="%d" '
            'onclick="focusObservation(this)" onkeydown="activateRow(event, this)">'
            '<td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
            '<td>%s</td><td>%d</td><td>%d</td><td>%.1f</td></tr>'
            % (
                css_class,
                item.left,
                item.top,
                item.width,
                item.height,
                _escape(status),
                _escape(item.text),
                _escape(item.ocr_tesseract),
                _escape(item.ocr_runeberg),
                probability_text,
                item.left,
                item.top,
                item.confidence,
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
    mode = "all OCR" if all_ocr else f"uppslagsord ≥ {threshold:.0%}"
    heading = "Alla OCR-observationer" if all_ocr else "Uppslagsordskandidater"
    empty_message = '<tr><td colspan="8">Inga observationer att visa.</td></tr>'
    warning = (
        f'<div class="warning">{_escape(fallback_reason)}</div>'
        if fallback_reason
        else ""
    )

    return f"""<!doctype html>
<html lang="sv">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR-granskning – sida {page}</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ margin: 0; background: #f3f4f6; color: #111827; }}
header {{ position: sticky; top: 0; z-index: 20; padding: 14px 20px; background: #111827; color: white; }}
header h1 {{ margin: 0 0 8px; font-size: 1.25rem; }}
.summary {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.badge {{ padding: 4px 9px; border-radius: 999px; background: #374151; font-size: .88rem; }}
.badge.good {{ background: #166534; }} .badge.bad {{ background: #991b1b; }}
.warning {{ margin-top: 10px; padding: 9px 12px; border-radius: 6px; background: #92400e; color: white; }}
main {{ max-width: 1800px; margin: auto; padding: 18px; }}
.grid {{ display: grid; grid-template-columns: minmax(360px, 48%) minmax(520px, 52%); gap: 18px; align-items: start; }}
.panel {{ background: white; border: 1px solid #d1d5db; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px #0002; }}
.panel h2 {{ margin: 0; padding: 12px 14px; font-size: 1rem; background: #e5e7eb; }}
.image-toolbar {{ display: flex; align-items: center; gap: 7px; padding: 8px 12px; border-bottom: 1px solid #d1d5db; }}
.image-toolbar button {{ padding: 4px 10px; cursor: pointer; }}
.image-toolbar .hint {{ margin-left: auto; font-size: .82rem; opacity: .72; }}
.image-wrap {{ height: calc(100vh - 205px); min-height: 420px; overflow: auto; padding: 12px; background: #d1d5db; }}
.scan-stage {{ position: relative; width: {image_width}px; height: {image_height}px; transform-origin: top left; }}
.scan-stage img {{ display: block; width: {image_width}px; height: {image_height}px; background: white; user-select: none; }}
.marker {{ position: absolute; display: none; box-sizing: border-box; border: 4px solid #dc2626; background: #ef444433; border-radius: 3px; pointer-events: none; z-index: 5; box-shadow: 0 0 0 2px white, 0 0 14px #0009; }}
.marker::after {{ content: ""; position: absolute; inset: -10px; border: 2px dashed #dc2626; }}
.table-wrap {{ height: calc(100vh - 145px); min-height: 480px; overflow: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: .86rem; }}
th {{ position: sticky; top: 0; z-index: 2; background: #e5e7eb; text-align: left; }}
th, td {{ padding: 6px 8px; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
tbody tr {{ cursor: pointer; }}
tbody tr:hover, tbody tr.selected {{ outline: 3px solid #2563eb; outline-offset: -3px; }}
tr.accepted {{ background: #dcfce7; }} tr.conflict {{ background: #fee2e2; font-weight: 650; }} tr.tesseract-only {{ background: #fff; }}
details {{ margin-top: 18px; }} summary {{ cursor: pointer; font-weight: 700; padding: 10px 0; }}
pre {{ white-space: pre-wrap; background: white; border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; }}
a {{ color: inherit; }}
@media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} .image-wrap {{ height: 62vh; }} .table-wrap {{ height: auto; max-height: 62vh; }} }}
@media (prefers-color-scheme: dark) {{
 body {{ background: #111827; color: #f9fafb; }} .panel, pre {{ background: #1f2937; border-color: #4b5563; }}
 .panel h2, th {{ background: #374151; }} th, td, .image-toolbar {{ border-color: #4b5563; }}
 .image-wrap {{ background: #111827; }}
 tr.accepted {{ background: #14532d; }} tr.conflict {{ background: #7f1d1d; }} tr.tesseract-only {{ background: #1f2937; }}
}}
</style>
</head>
<body>
<header>
  <h1>OCR-granskning – sida {page}</h1>
  <div class="summary">
    <span class="badge">Läge: {_escape(mode)}</span>
    <span class="badge">Visade: {len(displayed)}</span>
    <span class="badge">Runeberg-kopplade: {len(runeberg_values)}</span>
    <span class="badge good">Accepterade: {len(accepted)}</span>
    <span class="badge bad">Konflikter: {len(conflicts)}</span>
    <span class="badge">Radsimilaritet: {mean_similarity:.3f}</span>
  </div>
  {warning}
</header>
<main>
  <div class="grid">
    <section class="panel">
      <h2>Faksimil – <a href="{_escape(source_url)}">öppna hos Runeberg</a></h2>
      <div class="image-toolbar">
        <button type="button" onclick="changeZoom(-0.25)">−</button>
        <button type="button" onclick="fitImage()">Anpassa</button>
        <button type="button" onclick="changeZoom(0.25)">+</button>
        <span id="zoomLabel">100 %</span>
        <span class="hint">Klicka på en rad för att hitta ordet</span>
      </div>
      <div class="image-wrap" id="imageWrap"><div class="scan-stage" id="scanStage">
        <img id="scanImage" src="{image_data}" alt="Skannad SAOL-sida {page}"><div class="marker" id="marker"></div>
      </div></div>
    </section>
    <section class="panel">
      <h2>{heading}</h2>
      <div class="table-wrap"><table>
        <thead><tr><th>Status</th><th>Vald text</th><th>Tesseract</th><th>Runeberg</th><th>Grundord</th><th>x</th><th>y</th><th>OCR-säkerhet</th></tr></thead>
        <tbody>{''.join(rows) or empty_message}</tbody>
      </table></div>
    </section>
  </div>
  <details><summary>Debug: matchade OCR-rader ({len(pairs)})</summary><div class="panel table-wrap"><table>
    <thead><tr><th>T-rad</th><th>R-rad</th><th>Likhet</th><th>Tesseract</th><th>Runeberg</th></tr></thead><tbody>{''.join(line_rows)}</tbody>
  </table></div></details>
  <details><summary>Debug: Runebergs råa OCR-text</summary><pre>{_escape(raw_text)}</pre></details>
</main>
<script>
const naturalWidth = {image_width};
const naturalHeight = {image_height};
const imageWrap = document.getElementById('imageWrap');
const scanStage = document.getElementById('scanStage');
const marker = document.getElementById('marker');
const zoomLabel = document.getElementById('zoomLabel');
let zoom = 1;
let selectedRow = null;
function setZoom(value, keepCenter = true) {{
  const oldZoom = zoom;
  const centerX = (imageWrap.scrollLeft + imageWrap.clientWidth / 2) / oldZoom;
  const centerY = (imageWrap.scrollTop + imageWrap.clientHeight / 2) / oldZoom;
  zoom = Math.max(0.15, Math.min(3, value));
  scanStage.style.width = `${{naturalWidth * zoom}}px`;
  scanStage.style.height = `${{naturalHeight * zoom}}px`;
  document.getElementById('scanImage').style.width = `${{naturalWidth * zoom}}px`;
  document.getElementById('scanImage').style.height = `${{naturalHeight * zoom}}px`;
  zoomLabel.textContent = `${{Math.round(zoom * 100)}} %`;
  if (selectedRow) positionMarker(selectedRow);
  if (keepCenter) {{ imageWrap.scrollLeft = centerX * zoom - imageWrap.clientWidth / 2; imageWrap.scrollTop = centerY * zoom - imageWrap.clientHeight / 2; }}
}}
function fitImage() {{ const available = Math.max(100, imageWrap.clientWidth - 24); setZoom(Math.min(1, available / naturalWidth), false); imageWrap.scrollTo(0, 0); }}
function changeZoom(delta) {{ setZoom(zoom + delta); }}
function positionMarker(row) {{
  const left = Number(row.dataset.left), top = Number(row.dataset.top), width = Number(row.dataset.width), height = Number(row.dataset.height);
  const margin = Math.max(4, height * 0.25);
  marker.style.left = `${{(left - margin) * zoom}}px`; marker.style.top = `${{(top - margin) * zoom}}px`;
  marker.style.width = `${{(width + margin * 2) * zoom}}px`; marker.style.height = `${{(height + margin * 2) * zoom}}px`; marker.style.display = 'block';
}}
function focusObservation(row) {{
  if (selectedRow) selectedRow.classList.remove('selected'); selectedRow = row; row.classList.add('selected');
  if (zoom < 0.8) setZoom(1, false); positionMarker(row);
  const left = Number(row.dataset.left) * zoom, top = Number(row.dataset.top) * zoom;
  const width = Number(row.dataset.width) * zoom, height = Number(row.dataset.height) * zoom;
  imageWrap.scrollTo({{left: Math.max(0, left + width / 2 - imageWrap.clientWidth / 2), top: Math.max(0, top + height / 2 - imageWrap.clientHeight / 2), behavior: 'smooth'}});
}}
function activateRow(event, row) {{ if (event.key === 'Enter' || event.key === ' ') {{ event.preventDefault(); focusObservation(row); }} }}
window.addEventListener('load', () => {{ fitImage(); const firstConflict = document.querySelector('tr.conflict'); if (firstConflict) firstConflict.scrollIntoView({{block: 'nearest'}}); }});
window.addEventListener('resize', () => {{ if (!selectedRow) fitImage(); }});
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Granska uppslagsord i Tesseract/Runeberg-jämförelsen för en SAOL-sida.")
    parser.add_argument("page", nargs="?", type=int, default=19)
    parser.add_argument("--html", nargs="?", const="", metavar="FIL", help="Skapa en HTML-granskningssida; standardnamn är pageNNNN-review.html.")
    parser.add_argument("--open", action="store_true", help="Öppna HTML-rapporten i webbläsaren.")
    parser.add_argument("--all-ocr", action="store_true", help="Visa all OCR i stället för bara uppslagsordskandidater.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Lägsta sannolikhet för uppslagsord (standard: 0.5).")
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold måste ligga mellan 0 och 1")

    model = HeadwordModel.load()
    fallback_reason: str | None = None
    effective_all_ocr = args.all_ocr
    if model is None and not effective_all_ocr:
        fallback_reason = "Ingen tränad HeadwordModel hittades. Visar all OCR i stället."
        print(f"Varning: {fallback_reason}")
        effective_all_ocr = True

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
    observation_lines = _observation_line_indices(observations)
    tesseract_lines = [_normalized_observation_line(observations, line) for line in observation_lines]
    runeberg_normalized = [[token.casefold() for token in line] for line in runeberg_lines]
    pairs = _align_lines(tesseract_lines, runeberg_normalized)
    corrected = reconcile_contextual_observations(observations, source_response.text)

    probabilities: list[float | None] = [
        model.probability(item) if model is not None else None
        for item in observations
    ]
    displayed = [
        (item, probability)
        for item, probability in zip(corrected, probabilities)
        if effective_all_ocr or (probability is not None and probability >= args.threshold)
    ]
    displayed_items = [item for item, _ in displayed]
    conflicts = [item for item in displayed_items if item.ocr_conflict]
    runeberg_values = [item for item in displayed_items if item.ocr_runeberg]
    accepted = [item for item in runeberg_values if not item.ocr_conflict]

    print(f"Runeberg-URL: {source_url}")
    print(f"OCR-bild: {tif_url}")
    print(f"Runeberg-token: {len(runeberg_tokens)}")
    print(f"Tesseract-token totalt: {len(observations)}")
    print(f"Visade {'OCR-token' if effective_all_ocr else 'uppslagsordskandidater'}: {len(displayed)}")
    print(f"Runeberg-kopplade bland visade: {len(runeberg_values)}")
    print(f"Automatiskt accepterade bland visade: {len(accepted)}")
    print(f"Konflikter bland visade: {len(conflicts)}")
    for item in conflicts[:30]:
        print(f"  y={item.top:4d} x={item.left:4d}: Tesseract={item.ocr_tesseract!r}, Runeberg={item.ocr_runeberg!r}, text={item.text!r}")

    if args.html is not None or args.open:
        output = Path(args.html or f"page{args.page:04d}-review.html").resolve()
        output.write_text(
            _review_html(
                args.page,
                source_url,
                image_response.content,
                raw_text,
                runeberg_lines,
                observations,
                displayed,
                pairs,
                all_ocr=effective_all_ocr,
                threshold=args.threshold,
                fallback_reason=fallback_reason,
            ),
            encoding="utf-8",
        )
        print(f"HTML-rapport: {output}")
        if args.open:
            webbrowser.open(output.as_uri())


if __name__ == "__main__":
    main()
