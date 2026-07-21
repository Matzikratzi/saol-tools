from __future__ import annotations

import argparse
from difflib import SequenceMatcher

import httpx

from app.runeberg import (
    _printed_order_indices,
    _runeberg_ocr_text,
    _runeberg_ocr_tokens,
    _word_letters,
    extract_observations,
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
    image_response = httpx.get(image_url, timeout=60.0, follow_redirects=True, headers=headers)
    image_response.raise_for_status()

    raw_text = _runeberg_ocr_text(source_response.text)
    runeberg_tokens = _runeberg_ocr_tokens(source_response.text)
    observations = extract_observations(image_response.content)
    order = _printed_order_indices(observations)
    tesseract_tokens = [_word_letters(observations[index].text) or f"__unreadable_{position}" for position, index in enumerate(order)]
    runeberg_normalized = [_word_letters(token) for token in runeberg_tokens]

    print(f"Runeberg-URL: {source_url}")
    print(f"HTML-tecken: {len(source_response.text)}")
    print(f"Extraherad OCR-text: {len(raw_text)} tecken")
    print(f"Runeberg-token: {len(runeberg_tokens)}")
    print(f"Tesseract-token: {len(observations)}")
    print(f"Runeberg innehåller abnorm: {'abnorm' in runeberg_normalized}")
    print(f"Första Runeberg-token: {runeberg_tokens[:20]}")

    matcher = SequenceMatcher(None, tesseract_tokens, runeberg_normalized, autojunk=False)
    print("\nBlock kring 'abnorm':")
    abnorm_index = runeberg_normalized.index("abnorm") if "abnorm" in runeberg_normalized else None
    if abnorm_index is None:
        print("  'abnorm' hittades inte i Runebergs tokenlista.")
    else:
        for opcode in matcher.get_opcodes():
            tag, left_start, left_end, right_start, right_end = opcode
            if right_start <= abnorm_index < right_end:
                print(f"  opcode={opcode}")
                print(f"  Tesseract: {tesseract_tokens[max(0, left_start-5):left_end+5]}")
                print(f"  Runeberg:  {runeberg_normalized[max(0, right_start-5):right_end+5]}")
                break

    corrected = reconcile_contextual_observations(observations, runeberg_tokens)
    conflicts = [item for item in corrected if item.ocr_conflict]
    runeberg_values = [item for item in corrected if item.ocr_runeberg]
    print(f"\nObservationer med Runeberg-värde: {len(runeberg_values)}")
    print(f"Konflikter: {len(conflicts)}")
    for item in conflicts[:30]:
        print(
            f"  y={item.top:4d} x={item.left:4d}: "
            f"Tesseract={item.ocr_tesseract!r}, Runeberg={item.ocr_runeberg!r}, text={item.text!r}"
        )


if __name__ == "__main__":
    main()
