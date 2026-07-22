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
    source = source.replace(old, new, 1)

    old_guides = "def _guides_html(x_models: dict, image_width: int) -> str:\n    result: list[str] = []\n"
    new_guides = (
        "def _guides_html(x_models: dict, image_width: int) -> str:\n"
        "    result: list[str] = [\n"
        "        '<div class=\"x-guide x-guide-column-split\" data-x=\"%.3f\" ' \n"
        "        'style=\"--guide-color:#dc2626\"></div>' % (image_width / 2)\n"
        "    ]\n"
    )
    if old_guides not in source:
        raise RuntimeError("Kunde inte lägga till kolumngränsen i HTML-rapporten")
    source = source.replace(old_guides, new_guides, 1)

    old_css = ".x-guide-continuation { border-left-style:dashed; }\n.marker { z-index:40; }"
    new_css = (
        ".x-guide-continuation { border-left-style:dashed; }\n"
        ".x-guide-column-split { border-left-width:2px; border-left-style:solid; opacity:1; }\n"
        ".marker { z-index:40; }"
    )
    if old_css not in source:
        raise RuntimeError("Kunde inte formatera kolumngränsen i HTML-rapporten")
    return source.replace(old_css, new_css, 1)


exec(compile(_source(), str(Path(__file__).resolve()), "exec"), globals(), globals())
