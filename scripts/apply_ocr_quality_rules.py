from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNEberg = ROOT / "app" / "runeberg.py"
DEBUG = ROOT / "scripts" / "debug_runeberg_ocr.py"

NEW_FUNCTION = r'''def _only_leading_hyphen_differs(left: str, right: str) -> bool:
    """Return true when the normalized words differ only by one leading hyphen."""
    return left.lstrip("-") == right.lstrip("-") and left.startswith("-") != right.startswith("-")


def _runeberg_l_as_i_error(actual: str, expected: str) -> bool:
    """Detect Runeberg's recurring OCR confusion where lowercase l becomes i.

    Examples seen in the source are ``till`` -> ``tiii`` and ``til`` -> ``tiii``.
    We only reject the Runeberg value when replacing i with l can make it equal
    to, or one edit away from, the Tesseract value.
    """
    if "i" not in expected or "l" not in actual:
        return False
    candidate = expected.replace("i", "l")
    return candidate == actual or _edit_distance_at_most_one(candidate, actual)


def _apply_contextual_replacement(
    corrected: list[WordObservation],
    observation_index: int,
    runeberg_token: str,
    expected: str,
) -> None:
    """Merge one aligned token pair without letting weak Runeberg OCR win.

    Minor differences and an added leading hyphen are accepted automatically.
    A recurring Runeberg l/i confusion is ignored. Other plausible disagreements
    preserve Tesseract as display text while retaining both OCR values and a
    conflict flag for manual review.
    """
    original = corrected[observation_index]
    tesseract_token = original.ocr_tesseract or original.text
    actual = _word_letters(tesseract_token)
    if len(actual) < 3 or len(expected) < 3 or actual == expected:
        return

    if _runeberg_l_as_i_error(actual, expected):
        return

    minor = _edit_distance_at_most_one(actual, expected)
    hyphen_variant = _only_leading_hyphen_differs(actual, expected)
    similarity = SequenceMatcher(None, actual, expected, autojunk=False).ratio()
    plausible = minor or hyphen_variant or actual[0] == expected[0] or similarity >= 0.58
    if not plausible:
        return

    accepted = minor or hyphen_variant
    corrected[observation_index] = replace(
        original,
        text=runeberg_token if accepted else tesseract_token,
        ocr_tesseract=tesseract_token,
        ocr_runeberg=runeberg_token,
        ocr_conflict=not accepted,
    )
'''


def replace_function(source: str) -> str:
    pattern = re.compile(
        r"def _apply_contextual_replacement\(.*?\n(?=def reconcile_contextual_observations\()",
        flags=re.DOTALL,
    )
    match = pattern.search(source)
    if not match:
        raise SystemExit("Hittade inte _apply_contextual_replacement i app/runeberg.py")
    return source[:match.start()] + NEW_FUNCTION + "\n\n" + source[match.end():]


def update_debug(source: str) -> str:
    old = '''    conflicts = [item for item in corrected if item.ocr_conflict]\n    runeberg_values = [item for item in corrected if item.ocr_runeberg]\n    print(f"\\nObservationer med Runeberg-värde: {len(runeberg_values)}")\n    print(f"Konflikter: {len(conflicts)}")\n'''
    new = '''    conflicts = [item for item in corrected if item.ocr_conflict]\n    runeberg_values = [item for item in corrected if item.ocr_runeberg]\n    accepted = [item for item in runeberg_values if not item.ocr_conflict]\n    print(f"\\nObservationer med Runeberg-värde: {len(runeberg_values)}")\n    print(f"Automatiskt accepterade: {len(accepted)}")\n    print(f"Konflikter: {len(conflicts)}")\n'''
    if old not in source:
        print("Varning: kunde inte utöka debugräkningen; OCR-reglerna är ändå installerade")
        return source
    return source.replace(old, new, 1)


def main() -> None:
    source = RUNEberg.read_text(encoding="utf-8")
    if "def _runeberg_l_as_i_error(" in source:
        print("OCR-kvalitetsreglerna finns redan i app/runeberg.py")
    else:
        RUNEberg.write_text(replace_function(source), encoding="utf-8")
        print("Uppdaterade app/runeberg.py med konservativa OCR-kvalitetsregler")

    debug_source = DEBUG.read_text(encoding="utf-8")
    if "Automatiskt accepterade:" not in debug_source:
        DEBUG.write_text(update_debug(debug_source), encoding="utf-8")
        print("Uppdaterade debugutskriften")


if __name__ == "__main__":
    main()
