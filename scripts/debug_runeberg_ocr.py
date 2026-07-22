from __future__ import annotations

"""Run the rebuilt OCR classifier with the body starting at the header rule.

The rebuilt implementation is kept at the preceding known commit.  This small
loader makes the intended body boundary explicit: every OCR line whose centre
is below the detected horizontal rule is part of the dictionary body.  There
is no extra margin and no search for a later whitespace gap.
"""

import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "a06254a88ccf051cfb265c46bdf2760d197e56f7"
SOURCE_PATH = "scripts/debug_runeberg_ocr.py"


def _source() -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "show",
            f"{SOURCE_COMMIT}:{SOURCE_PATH}",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    source = result.stdout
    old = (
        "    body_top = _BODY_TOP_Y if _BODY_TOP_Y is not None else image_height * 0.03\n"
        "    body_top += max(1.0, median_height * 0.10)\n"
    )
    new = (
        "    # Dictionary text starts immediately below the detected rule.\n"
        "    body_top = _BODY_TOP_Y if _BODY_TOP_Y is not None else image_height * 0.03\n"
    )
    if old not in source:
        raise RuntimeError("Kunde inte sätta textstart direkt under sidhuvudsstrecket")
    return source.replace(old, new, 1)


exec(compile(_source(), str(Path(__file__).resolve()), "exec"), globals(), globals())
