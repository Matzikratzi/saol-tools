from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNEberg = ROOT / "app" / "runeberg.py"
DEBUG = ROOT / "scripts" / "debug_runeberg_ocr.py"

NEW_FUNCTION = r'''def _only_leading_hyphen_differs(left: str, right: str) -> bool:
    """Return true when normalized words differ only by one leading hyphen."""
    return left.lstrip("-") == right.lstrip("-") and left.startswith("-") != right.startswith("-")


def _runeberg_l_as_i_error(actual: str, expected: str) -> bool:
    """Detect Runeberg's recurring OCR confusion where lowercase l becomes i.

    The source repeatedly contains ``tiii`` where the printed word is ``til`` or
    ``till``. Keep the Tesseract value and do not create a conflict for these
    known variants. The conservative fallback only accepts an i-to-l rewrite
    when it becomes at most one edit away from Tesseract.
    """
    if expected == "tiii" and actual in {"til", "till"}:
        return True
    if "i" not in expected or "l" not in actual:
        return False
    candidate = expected.replace("i", "l")
    return candidate == actual or _edit_distance_at_most_one(candidate, actual)


def _stem_boundary_variant(tesseract_token: str, runeberg_token: str, actual: str, expected: str) -> bool:
    """Accept a close Runeberg form when it contains an explicit stem boundary.

    Stem markers are removed by ``_word_letters`` before comparison. Requiring
    a marker in Runeberg plus a high normalized similarity keeps this rule from
    accepting unrelated ordinary OCR disagreements.
    """
    if not any(mark in runeberg_token for mark in STEM_BOUNDARY_MARKS):
        return False
    if not actual or not expected:
        return False
    similarity = SequenceMatcher(None, actual.lstrip("-"), expected.lstrip("-"), autojunk=False).ratio()
    return similarity >= 0.80


def _leading_compound_i_variant(actual: str, expected: str) -> bool:
    """Recognize Runeberg's leading ``-i`` compound notation, e.g. hop/-ihop."""
    return expected.startswith("-i") and expected[2:] == actual.lstrip("-")


def _strong_length_mismatch(left: str, right: str) -> bool:
    """Return true when one aligned token is more than twice as long as the other."""
    shorter = min(len(left), len(right))
    longer = max(len(left), len(right))
    return shorter > 0 and longer > 2 * shorter


def _apply_contextual_replacement(
    corrected: list[WordObservation],
    observation_index: int,
    runeberg_token: str,
    expected: str,
) -> None:
    """Merge one aligned token pair without letting weak Runeberg OCR win.

    Small differences, explicit stem-boundary forms and leading ``-i`` compound
    notation are accepted automatically. Runeberg's recurring l/i confusion is
    ignored. Strong length mismatches and other plausible disagreements retain
    Tesseract as display text while preserving both OCR readings as a conflict.
    """
    original = corrected[observation_index]
    tesseract_token = original.ocr_tesseract or original.text
    actual = _word_letters(tesseract_token)
    expected = _word_letters(expected)
    if len(actual) < 3 or len(expected) < 3 or actual == expected:
        return

    if _runeberg_l_as_i_error(actual, expected):
        return

    minor = _edit_distance_at_most_one(actual, expected)
    hyphen_variant = _only_leading_hyphen_differs(actual, expected)
    stem_variant = _stem_boundary_variant(tesseract_token, runeberg_token, actual, expected)
    compound_i_variant = _leading_compound_i_variant(actual, expected)
    accepted = minor or hyphen_variant or stem_variant or compound_i_variant

    if accepted:
        corrected[observation_index] = replace(
            original,
            text=runeberg_token,
            ocr_tesseract=tesseract_token,
            ocr_runeberg=runeberg_token,
            ocr_conflict=False,
        )
        return

    similarity = SequenceMatcher(None, actual, expected, autojunk=False).ratio()
    plausible = _strong_length_mismatch(actual, expected) or actual[0] == expected[0] or similarity >= 0.58
    if not plausible:
        return

    corrected[observation_index] = replace(
        original,
        text=tesseract_token,
        ocr_tesseract=tesseract_token,
        ocr_runeberg=runeberg_token,
        ocr_conflict=True,
    )
'''


def replace_function(source: str) -> str:
    pattern = re.compile(
        r"def _only_leading_hyphen_differs\(.*?\n(?=def reconcile_contextual_observations\()",
        flags=re.DOTALL,
    )
    match = pattern.search(source)
    if not match:
        pattern = re.compile(
            r"def _apply_contextual_replacement\(.*?\n(?=def reconcile_contextual_observations\()",
            flags=re.DOTALL,
        )
        match = pattern.search(source)
    if not match:
        raise SystemExit("Hittade inte OCR-kvalitetsfunktionerna i app/runeberg.py")
    return source[:match.start()] + NEW_FUNCTION + "\n\n" + source[match.end():]


def update_debug(source: str) -> str:
    old = '''    conflicts = [item for item in corrected if item.ocr_conflict]\n    runeberg_values = [item for item in corrected if item.ocr_runeberg]\n    print(f"\\nObservationer med Runeberg-värde: {len(runeberg_values)}")\n    print(f"Konflikter: {len(conflicts)}")\n'''
    new = '''    conflicts = [item for item in corrected if item.ocr_conflict]\n    runeberg_values = [item for item in corrected if item.ocr_runeberg]\n    accepted = [item for item in runeberg_values if not item.ocr_conflict]\n    print(f"\\nObservationer med Runeberg-värde: {len(runeberg_values)}")\n    print(f"Automatiskt accepterade: {len(accepted)}")\n    print(f"Konflikter: {len(conflicts)}")\n'''
    if old not in source:
        return source
    return source.replace(old, new, 1)


def main() -> None:
    source = RUNEberg.read_text(encoding="utf-8")
    RUNEberg.write_text(replace_function(source), encoding="utf-8")
    print("Uppdaterade app/runeberg.py med de fyra förfinade OCR-reglerna")

    debug_source = DEBUG.read_text(encoding="utf-8")
    updated_debug = update_debug(debug_source)
    if updated_debug != debug_source:
        DEBUG.write_text(updated_debug, encoding="utf-8")
        print("Uppdaterade debugutskriften")


if __name__ == "__main__":
    main()
