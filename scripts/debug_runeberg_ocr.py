from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import webbrowser
from pathlib import Path

import httpx
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


def _image_data_url(content: bytes, source_url: str) -> str:
    mime_type = mimetypes.guess_type(source_url)[0] or "image/tiff"
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _review_html(
    page: int,
    source_url: str,
    image_url: str,
    image_content: bytes,
    raw_text: str,
    runeberg_lines: list[list[str]],
    observations: list,
    corrected: list,
    pairs: list[tuple[int, int, float]],
) -> str:
    runeberg_values = [item for item in corrected if item.ocr_runeberg]
    conflicts = [item for item in corrected if item.ocr_conflict]
    accepted = [item for item in runeberg_values if not item.ocr_conflict]
    pair_scores = [score for _, _, score in pairs]
    mean_similarity = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0

    rows: list[str] = []
    for item in sorted(corrected, key=lambda value: (value.top, value.left)):
        if item.ocr_conflict:
            status = "Konflikt"
            css_class = "conflict"
        elif item.ocr_runeberg:
            status = "Accepterad"
            css_class = "accepted"
        else:
            status = "Endast Tesseract"
            css_class = "tesseract-only"
        rows.append(
            "<tr class=\"%s\">"
            "<td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
            "<td>%d</td><td>%d</td><td>%.1f</td>"
            "</tr>"
            % (
                css_class,
                _escape(status),
                _escape(item.text),
                _escape(item.ocr_tesseract),
                _escape(item.ocr_runeberg),
                item.left,
                item.top,
                item.confidence,
            )
        )

    line_rows: list[str] = []
    observation_lines = _observation_line_indices(observations)
    tesseract_lines = [_normalized_observation_line(observations, line) for line in observation_lines]
    for left, right, score in pairs:
        line_rows.append(
            "<tr><td>%d</td><td>%d</td><td>%.3f</td><td>%s</td><td>%s</td></tr>"
            % (
                left,
                right,
                score,
                _escape(" ".join(tesseract_lines[left])),
                _escape(" ".join(runeberg_lines[right])),
            )
        )

    image_data = _image_data_url(image_content, image_url)
    return f"""<!doctype html>
<html lang=\"sv\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>OCR-granskning – sida {page}</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ margin: 0; background: #f3f4f6; color: #111827; }}
header {{ position: sticky; top: 0; z-index: 2; padding: 14px 20px; background: #111827; color: white; }}
header h1 {{ margin: 0 0 8px; font-size: 1.25rem; }}
.summary {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.badge {{ padding: 4px 9px; border-radius: 999px; background: #374151; font-size: .88rem; }}
.badge.good {{ background: #166534; }} .badge.bad {{ background: #991b1b; }}
main {{ max-width: 1600px; margin: auto; padding: 18px; }}
.grid {{ display: grid; grid-template-columns: minmax(320px, 42%) minmax(520px, 58%); gap: 18px; align-items: start; }}
.panel {{ background: white; border: 1px solid #d1d5db; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px #0002; }}
.panel h2 {{ margin: 0; padding: 12px 14px; font-size: 1rem; background: #e5e7eb; }}
.image-wrap {{ max-height: calc(100vh - 145px); overflow: auto; padding: 12px; text-align: center; }}
.image-wrap img {{ max-width: 100%; height: auto; background: white; }}
.table-wrap {{ max-height: calc(100vh - 145px); overflow: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: .86rem; }}
th {{ position: sticky; top: 0; background: #e5e7eb; text-align: left; }}
th, td {{ padding: 6px 8px; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
tr.accepted {{ background: #dcfce7; }} tr.conflict {{ background: #fee2e2; font-weight: 650; }} tr.tesseract-only {{ background: #fff; }}
details {{ margin-top: 18px; }} summary {{ cursor: pointer; font-weight: 700; padding: 10px 0; }}
pre {{ white-space: pre-wrap; background: white; border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; }}
a {{ color: inherit; }}
@media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} .image-wrap, .table-wrap {{ max-height: none; }} }}
@media (prefers-color-scheme: dark) {{
 body {{ background: #111827; color: #f9fafb; }} .panel, pre {{ background: #1f2937; border-color: #4b5563; }}
 .panel h2, th {{ background: #374151; }} th, td {{ border-color: #4b5563; }}
 tr.accepted {{ background: #14532d; }} tr.conflict {{ background: #7f1d1d; }} tr.tesseract-only {{ background: #1f2937; }}
}}
</style>
</head>
<body>
<header>
  <h1>OCR-granskning – sida {page}</h1>
  <div class=\"summary\">
    <span class=\"badge\">Tesseract: {len(observations)}</span>
    <span class=\"badge\">Runeberg-kopplade: {len(runeberg_values)}</span>
    <span class=\"badge good\">Accepterade: {len(accepted)}</span>
    <span class=\"badge bad\">Konflikter: {len(conflicts)}</span>
    <span class=\"badge\">Radsimilaritet: {mean_similarity:.3f}</span>
  </div>
</header>
<main>
  <div class=\"grid\">
    <section class=\"panel\">
      <h2>Faksimil – <a href=\"{_escape(source_url)}\">öppna hos Runeberg</a></h2>
      <div class=\"image-wrap\"><img src=\"{image_data}\" alt=\"Skannad SAOL-sida {page}\"></div>
    </section>
    <section class=\"panel\">
      <h2>Sammanfogade observationer</h2>
      <div class=\"table-wrap\"><table>
        <thead><tr><th>Status</th><th>Vald text</th><th>Tesseract</th><th>Runeberg</th><th>x</th><th>y</th><th>Säkerhet</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table></div>
    </section>
  </div>
  <details><summary>Matchade OCR-rader ({len(pairs)})</summary>
    <div class=\"panel table-wrap\"><table>
      <thead><tr><th>T-rad</th><th>R-rad</th><th>Likhet</th><th>Tesseract</th><th>Runeberg</th></tr></thead>
      <tbody>{''.join(line_rows)}</tbody>
    </table></div>
  </details>
  <details><summary>Runebergs råa OCR-text</summary><pre>{_escape(raw_text)}</pre></details>
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostisera Tesseract/Runeberg-jämförelsen för en SAOL-sida.")
    parser.add_argument("page", nargs="?", type=int, default=19)
    parser.add_argument("--html", nargs="?", const="", metavar="FIL", help="Skapa en HTML-granskningssida; standardnamn är pageNNNN-review.html.")
    parser.add_argument("--open", action="store_true", help="Öppna HTML-rapporten i webbläsaren.")
    args = parser.parse_args()

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

    print(f"Runeberg-URL: {source_url}")
    print(f"OCR-bild: {tif_url}")
    print(f"HTML-tecken: {len(source_response.text)}")
    print(f"Extraherad OCR-text: {len(raw_text)} tecken")
    print(f"Runeberg-token: {len(runeberg_tokens)}")
    print(f"Tesseract-token: {len(observations)}")
    print(f"Runeberg-rader: {len(runeberg_lines)}")
    print(f"Tesseract-rader: {len(observation_lines)}")
    print(f"Matchade rader: {len(pairs)}")
    if pairs:
        similarities = [score for _, _, score in pairs]
        print(f"Radsimilaritet: min={min(similarities):.3f}, medel={sum(similarities)/len(similarities):.3f}, max={max(similarities):.3f}")
        print("\nExempel på matchade rader:")
        for left, right, score in pairs[:8]:
            print(f"  {left:2d} ↔ {right:2d} ({score:.3f})")
            print(f"    T: {' '.join(tesseract_lines[left])}")
            print(f"    R: {' '.join(runeberg_lines[right])}")

    corrected = reconcile_contextual_observations(observations, source_response.text)
    conflicts = [item for item in corrected if item.ocr_conflict]
    runeberg_values = [item for item in corrected if item.ocr_runeberg]
    accepted = [item for item in runeberg_values if not item.ocr_conflict]
    print(f"\nObservationer med Runeberg-värde: {len(runeberg_values)}")
    print(f"Automatiskt accepterade: {len(accepted)}")
    print(f"Konflikter: {len(conflicts)}")
    for item in conflicts[:30]:
        print(
            f"  y={item.top:4d} x={item.left:4d}: "
            f"Tesseract={item.ocr_tesseract!r}, Runeberg={item.ocr_runeberg!r}, text={item.text!r}"
        )

    if args.html is not None or args.open:
        output = Path(args.html or f"page{args.page:04d}-review.html").resolve()
        output.write_text(
            _review_html(
                args.page,
                source_url,
                tif_url,
                image_response.content,
                raw_text,
                runeberg_lines,
                observations,
                corrected,
                pairs,
            ),
            encoding="utf-8",
        )
        print(f"HTML-rapport: {output}")
        if args.open:
            webbrowser.open(output.as_uri())


if __name__ == "__main__":
    main()
