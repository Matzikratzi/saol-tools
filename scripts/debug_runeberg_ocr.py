from __future__ import annotations

import argparse

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostisera Tesseract/Runeberg-jämförelsen för en SAOL-sida.")
    parser.add_argument("page", nargs="?", type=int, default=19)
    args = parser.parse_args()
    source_url, image_url = page_urls(args.page)
    headers = {"User-Agent": "saol-tools/debug"}
    source_response = httpx.get(source_url, timeout=60.0, follow_redirects=True, headers=headers)
    source_response.raise_for_status()
    image_response = httpx.get(ocr_image_url(image_url), timeout=60.0, follow_redirects=True, headers=headers)
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
    print(f"OCR-bild: {ocr_image_url(image_url)}")
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


if __name__ == "__main__":
    main()
