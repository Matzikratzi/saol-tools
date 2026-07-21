from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "app" / "runeberg.py"


def replace_once(source: str, old: str, new: str, description: str) -> str:
    if new in source:
        print(f"Redan ändrat: {description}")
        return source
    if old not in source:
        raise SystemExit(f"Hittade inte kodstycket för: {description}")
    print(f"Ändrar: {description}")
    return source.replace(old, new, 1)


def main() -> None:
    source = TARGET.read_text(encoding="utf-8")

    source = replace_once(
        source,
        '''    image_response = httpx.get(image_url, timeout=60.0, follow_redirects=True, headers=headers)
    image_response.raise_for_status()
    observations = extract_observations(image_response.content)
''',
        '''    # Use Runeberg's full-resolution TIFF for OCR, while keeping the
    # browser-friendly PNG URL in ImportedPage for display in the UI.
    ocr_image_url = image_url.replace(".3.png", ".1.tif")
    image_response = httpx.get(ocr_image_url, timeout=60.0, follow_redirects=True, headers=headers)
    image_response.raise_for_status()
    observations = extract_observations(image_response.content)
''',
        "fullupplöst TIFF som Tesseract-underlag",
    )

    source = replace_once(
        source,
        '''        if left_count != right_count:
            continue
        for offset in range(left_count):
''',
        '''        # A larger replacement block means that one OCR stream has
        # inserted, removed or split tokens. Pairing tokens by offset inside
        # such a block creates false conflicts far from the real error.
        # Only accept an unambiguous one-token-to-one-token replacement here.
        if left_count != 1 or right_count != 1:
            continue
        for offset in range(left_count):
''',
        "konservativ en-till-en-alignering",
    )

    source = replace_once(
        source,
        '''    actual = _word_letters(tesseract_token)
    if actual and actual[0] != expected[0] and len(actual) >= 3:
        return
''',
        '''    actual = _word_letters(tesseract_token)
    # Very short OCR fragments are poor anchors. Do not turn punctuation,
    # numbers or one/two-letter fragments into unrelated dictionary words.
    if not actual or len(actual) < 3:
        return
    if actual[0] != expected[0]:
        return
''',
        "avvisa korta och uppenbart felparade token",
    )

    TARGET.write_text(source, encoding="utf-8")
    print("Uppdaterade app/runeberg.py")


if __name__ == "__main__":
    main()
