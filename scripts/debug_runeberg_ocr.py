from __future__ import annotations

"""Run the rebuilt OCR classifier with geometry derived from the header rule.

The implementation is materialised from the preceding rebuilt commit and then
patched so dictionary text starts immediately below the rule. The midpoint of
the detected rule is used as the boundary between the columns. For this
experiment, the left column first shows a guide just before its leftmost OCR
character and then its two leftmost recurring line-start levels, without
assigning A/F/H meanings to them yet. Their midpoint T is used as the article
start threshold in the left column.
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

    old_globals = "_BODY_TOP_Y: float | None = None\n"
    new_globals = (
        "_BODY_TOP_Y: float | None = None\n"
        "_DESKEW_RULE_Y: float | None = None\n"
        "_COLUMN_SPLIT_X: float | None = None\n"
        "_LEFT_A_X: float | None = None\n"
        "_LEFT_LEVELS: list[float] = []\n"
        "_LEFT_THRESHOLD_X: float | None = None\n"
    )
    if old_globals not in source:
        raise RuntimeError("Kunde inte lägga till geometrins globala positioner")
    source = source.replace(old_globals, new_globals, 1)

    old_deskew_start = (
        "    global _BODY_TOP_Y\n"
        "\n"
        "    detected = _header_rule(content)\n"
    )
    new_deskew_start = (
        "    global _BODY_TOP_Y, _DESKEW_RULE_Y\n"
        "    _DESKEW_RULE_Y = None\n"
        "\n"
        "    detected = _header_rule(content)\n"
    )
    if old_deskew_start not in source:
        raise RuntimeError("Kunde inte spara upprätningslinjens y-position")
    source = source.replace(old_deskew_start, new_deskew_start, 1)

    old_rule_result = "    angle, rule_y = detected\n"
    new_rule_result = (
        "    angle, rule_y = detected\n"
        "    _DESKEW_RULE_Y = rule_y\n"
    )
    if old_rule_result not in source:
        raise RuntimeError("Kunde inte registrera vald upprätningslinje")
    source = source.replace(old_rule_result, new_rule_result, 1)

    old_header_start = (
        "def _header_rule(content: bytes) -> tuple[float, float] | None:\n"
        "    \"\"\"Return the separating rule's angle in degrees and centre y.\"\"\"\n"
    )
    new_header_start = (
        "def _header_rule(content: bytes) -> tuple[float, float] | None:\n"
        "    \"\"\"Return the separating rule's angle in degrees and centre y.\"\"\"\n"
        "    global _COLUMN_SPLIT_X\n"
        "    _COLUMN_SPLIT_X = None\n"
    )
    if old_header_start not in source:
        raise RuntimeError("Kunde inte förbereda mätning av streckets mittpunkt")
    source = source.replace(old_header_start, new_header_start, 1)

    old_peak = (
        "        span, dark_count, peak_y = max(row_scores)\n"
        "        if span < width * 0.60 or dark_count < width * 0.16:\n"
        "            return None\n"
    )
    new_peak = (
        "        span, dark_count, peak_y = max(row_scores)\n"
        "        if span < width * 0.60 or dark_count < width * 0.16:\n"
        "            return None\n"
        "\n"
        "        # Use the detected rule's actual endpoints, not the image edges.\n"
        "        peak_dark_x = [x for x in range(x0, x1) if pixels[x, peak_y] < 160]\n"
        "        if peak_dark_x:\n"
        "            _COLUMN_SPLIT_X = (peak_dark_x[0] + peak_dark_x[-1]) / 2\n"
    )
    if old_peak not in source:
        raise RuntimeError("Kunde inte mäta streckets vänster- och högerände")
    source = source.replace(old_peak, new_peak, 1)

    old_point_pick = (
        "            if ys:\n"
        "                points.append((float(x), float(statistics.median(ys))))\n"
    )
    new_point_pick = (
        "            if ys:\n"
        "                # Follow the dark pixel nearest the peak row. Taking the\n"
        "                # median of every dark pixel in the band lets nearby text\n"
        "                # influence the fitted angle.\n"
        "                rule_y = min(ys, key=lambda value: abs(value - peak_y))\n"
        "                points.append((float(x), float(rule_y)))\n"
    )
    if old_point_pick not in source:
        raise RuntimeError("Kunde inte isolera sidhuvudsstrecket från närliggande text")
    source = source.replace(old_point_pick, new_point_pick, 1)

    old_rule_y = (
        "        centre_y = slope * (width / 2) + intercept\n"
        "        return angle, centre_y\n"
    )
    new_rule_y = (
        "        # The fitted line supplies only the angle. Anchor y directly\n"
        "        # to the dark scan row that identified the physical rule; the\n"
        "        # regression can otherwise be pulled down by nearby print.\n"
        "        return angle, float(peak_y)\n"
    )
    if old_rule_y not in source:
        raise RuntimeError("Kunde inte fästa y-positionen vid den svarta upprätningslinjen")
    source = source.replace(old_rule_y, new_rule_y, 1)

    # Keep Pillow's original sign convention from the rebuilt implementation.
    # Changing -angle to +angle doubles the skew instead of removing it.

    old_redetection = (
        "    after = _header_rule(result)\n"
        "    _BODY_TOP_Y = after[1] if after is not None else rule_y\n"
        "    return result, angle\n"
    )
    new_redetection = (
        "    # Keep the exact rule selected for deskewing as the body boundary.\n"
        "    # A second detection pass can mistake a lower horizontal structure\n"
        "    # for the header rule and move the article start too far down.\n"
        "    _BODY_TOP_Y = rule_y\n"
        "    return result, angle\n"
    )
    if old_redetection not in source:
        raise RuntimeError("Kunde inte låsa artikelstarten till upprätningslinjen")
    source = source.replace(old_redetection, new_redetection, 1)

    old_body = (
        "    body_top = _BODY_TOP_Y if _BODY_TOP_Y is not None else image_height * 0.03\n"
        "    body_top += max(1.0, median_height * 0.10)\n"
    )
    new_body = (
        "    # Dictionary text starts immediately below the detected rule.\n"
        "    body_top = _BODY_TOP_Y if _BODY_TOP_Y is not None else image_height * 0.03\n"
    )
    if old_body not in source:
        raise RuntimeError("Kunde inte sätta textstart direkt under sidhuvudsstrecket")
    source = source.replace(old_body, new_body, 1)

    old_split = "    split = image_width / 2\n"
    new_split = (
        "    # Split the columns at the midpoint of the detected header rule.\n"
        "    split = _COLUMN_SPLIT_X if _COLUMN_SPLIT_X is not None else image_width / 2\n"
    )
    if old_split not in source:
        raise RuntimeError("Kunde inte använda streckets mittpunkt som kolumngräns")
    source = source.replace(old_split, new_split, 1)

    old_result_start = (
        "    result = []\n"
        "    models = {}\n"
    )
    new_result_start = (
        "    global _LEFT_A_X, _LEFT_LEVELS, _LEFT_THRESHOLD_X\n"
        "    # Experimental A: two pixels immediately left of the leftmost OCR\n"
        "    # character in the left column. No clustering and no F/H influence.\n"
        "    _LEFT_A_X = (\n"
        "        min(line.left for line, _x, _word, _marker, _text in raw[1]) - 2.0\n"
        "        if raw[1]\n"
        "        else None\n"
        "    )\n"
        "    _LEFT_LEVELS = []\n"
        "    _LEFT_THRESHOLD_X = None\n"
        "\n"
        "    result = []\n"
        "    models = {}\n"
    )
    if old_result_start not in source:
        raise RuntimeError("Kunde inte mäta vänsterspaltens vänstraste tecken")
    source = source.replace(old_result_start, new_result_start, 1)

    old_positions = (
        "        article_x, continuation_x = _split_two_positions(lexical_x, median_height)\n"
        "        boundary_x = (article_x + continuation_x) / 2\n"
    )
    new_positions = (
        "        article_x, continuation_x = _split_two_positions(lexical_x, median_height)\n"
        "        if column == 1:\n"
        "            # N1 is the guide just before the leftmost OCR character.\n"
        "            # Measure raw row starts after it so separate homonym digits\n"
        "            # remain visible, and keep the next two clusters neutral.\n"
        "            raw_starts = [\n"
        "                float(line.raw_start_x) for line, _x, _word, _marker, text in prepared\n"
        "                if LETTER_RE.search(text)\n"
        "            ]\n"
        "            tolerance = max(2.5, median_height * 0.20)\n"
        "            clusters = []\n"
        "            for value in sorted(raw_starts):\n"
        "                if clusters and value - clusters[-1][-1] <= tolerance:\n"
        "                    clusters[-1].append(value)\n"
        "                else:\n"
        "                    clusters.append([value])\n"
        "            minimum_count = max(2, round(len(raw_starts) * 0.04))\n"
        "            recurring = [\n"
        "                float(statistics.median(cluster)) for cluster in clusters\n"
        "                if len(cluster) >= minimum_count\n"
        "            ]\n"
        "            _LEFT_LEVELS = ([_LEFT_A_X] if _LEFT_A_X is not None else []) + recurring[:2]\n"
        "            if len(recurring) >= 2:\n"
        "                article_x = recurring[0]\n"
        "                continuation_x = recurring[1]\n"
        "                _LEFT_THRESHOLD_X = (article_x + continuation_x) / 2\n"
        "        boundary_x = (article_x + continuation_x) / 2\n"
    )
    if old_positions not in source:
        raise RuntimeError("Kunde inte använda vänstraste tecknet som A-position")
    source = source.replace(old_positions, new_positions, 1)

    old_article_decision = (
        "            is_article = bool(LETTER_RE.search(word_text)) and (\n"
        "                clearly_at_article or (ambiguous_left and bold_score >= 0.45)\n"
        "            )\n"
    )
    new_article_decision = (
        "            if column == 1 and _LEFT_THRESHOLD_X is not None:\n"
        "                # In the left column T is the sole article threshold.\n"
        "                is_article = bool(LETTER_RE.search(word_text)) and line.raw_start_x < _LEFT_THRESHOLD_X\n"
        "            else:\n"
        "                is_article = bool(LETTER_RE.search(word_text)) and (\n"
        "                    clearly_at_article or (ambiguous_left and bold_score >= 0.45)\n"
        "                )\n"
    )
    if old_article_decision not in source:
        raise RuntimeError("Kunde inte använda T som artikeltröskel")
    source = source.replace(old_article_decision, new_article_decision, 1)

    old_guides = "def _guides_html(x_models: dict, image_width: int) -> str:\n    result: list[str] = []\n"
    new_guides = (
        "def _guides_html(x_models: dict, image_width: int, image_height: int) -> str:\n"
        "    split_x = _COLUMN_SPLIT_X if _COLUMN_SPLIT_X is not None else image_width / 2\n"
        "    result: list[str] = [\n"
        "        '<div class=\"x-guide x-guide-column-split\" data-x=\"%.3f\" ' \n"
        "        'style=\"--guide-color:#2563eb\"></div>' % split_x\n"
        "    ]\n"
        "    level_colors = ('#7c3aed', '#16a34a', '#ea580c')\n"
        "    for index, position in enumerate(_LEFT_LEVELS, start=1):\n"
        "        x = max(0.0, min(float(image_width), float(position)))\n"
        "        delta = '' if index == 1 else ' · Δ=%.1f' % (position - _LEFT_LEVELS[index - 2])\n"
        "        result.append(\n"
        "            '<div class=\"x-guide x-guide-level\" data-x=\"%.3f\" ' \n"
        "            'style=\"--guide-color:%s\"><span class=\"x-guide-label\">N%d · x=%.1f%s</span></div>'\n"
        "            % (x, level_colors[index - 1], index, x, delta)\n"
        "        )\n"
        "    if _LEFT_THRESHOLD_X is not None:\n"
        "        x = max(0.0, min(float(image_width), float(_LEFT_THRESHOLD_X)))\n"
        "        result.append(\n"
        "            '<div class=\"x-guide x-guide-threshold\" data-x=\"%.3f\" ' \n"
        "            'style=\"--guide-color:#dc2626\"><span class=\"x-guide-label\">T · x=%.1f</span></div>'\n"
        "            % (x, x)\n"
        "        )\n"
        "    if _BODY_TOP_Y is not None and image_height > 0:\n"
        "        y = max(0.0, min(float(image_height), float(_BODY_TOP_Y)))\n"
        "        top_percent = 100.0 * y / image_height\n"
        "        result.append(\n"
        "            '<div class=\"y-guide y-guide-body-start\" style=\"--guide-y:%.6f%%\">'\n"
        "            '<span class=\"y-guide-label\">Artikelkontroll börjar här · y=%.1f</span></div>'\n"
        "            % (top_percent, y)\n"
        "        )\n"
        "    if _DESKEW_RULE_Y is not None and image_height > 0:\n"
        "        y = max(0.0, min(float(image_height), float(_DESKEW_RULE_Y)))\n"
        "        top_percent = 100.0 * y / image_height\n"
        "        result.append(\n"
        "            '<div class=\"y-guide y-guide-deskew\" style=\"--guide-y:%.6f%%\">'\n"
        "            '<span class=\"y-guide-label y-guide-label-deskew\">Upprätningslinje · y=%.1f</span></div>'\n"
        "            % (top_percent, y)\n"
        "        )\n"
    )
    if old_guides not in source:
        raise RuntimeError("Kunde inte lägga till kolumngränsen i HTML-rapporten")
    source = source.replace(old_guides, new_guides, 1)

    old_guide_loop = (
        "        for kind, color in definitions:\n"
        "            position = positions[kind]\n"
    )
    new_guide_loop = (
        "        for kind, color in definitions:\n"
        "            # The left column uses neutral N1/N2/N3 measurement guides.\n"
        "            if column == 1:\n"
        "                continue\n"
        "            position = positions[kind]\n"
    )
    if old_guide_loop not in source:
        raise RuntimeError("Kunde inte ersätta vänsterspaltens semantiska hjälplinjer")
    source = source.replace(old_guide_loop, new_guide_loop, 1)

    old_css = ".x-guide-continuation { border-left-style:dashed; }\n.marker { z-index:40; }"
    new_css = (
        ".x-guide-continuation { border-left-style:dashed; }\n"
        ".x-guide-column-split { border-left-width:4px; border-left-style:solid; opacity:1; }\n"
        ".x-guide-level { border-left-width:2px; }\n"
        ".x-guide-threshold { border-left-width:2px; border-left-style:dashed; }\n"
        ".x-guide-label { position:absolute; top:8px; left:4px; padding:2px 4px; "
        "border-radius:3px; background:var(--guide-color); color:white; "
        "font:700 11px/1.1 system-ui,sans-serif; white-space:nowrap; }\n"
        ".y-guide { position:absolute; left:0; right:0; top:var(--guide-y); height:0; "
        "z-index:35; border-top:3px solid #0891b2; pointer-events:none; }\n"
        ".y-guide-deskew { z-index:36; border-top:2px dashed #dc2626; }\n"
        ".y-guide-label { position:absolute; top:4px; left:8px; padding:2px 5px; "
        "border-radius:3px; background:#0891b2; color:white; "
        "font:700 11px/1.1 system-ui,sans-serif; white-space:nowrap; }\n"
        ".y-guide-label-deskew { left:220px; background:#dc2626; }\n"
        ".x-guide-article { border-left-width:2px; }\n"
        ".marker { z-index:40; }"
    )
    if old_css not in source:
        raise RuntimeError("Kunde inte formatera hjälplinjerna i HTML-rapporten")
    source = source.replace(old_css, new_css, 1)

    old_image_size = "            image_width = image.width\n"
    new_image_size = (
        "            image_width = image.width\n"
        "            image_height = image.height\n"
    )
    if old_image_size not in source:
        raise RuntimeError("Kunde inte läsa bildhöjden för den vågräta hjälplinjen")
    source = source.replace(old_image_size, new_image_size, 1)

    old_guides_call = "        guides = _guides_html(x_models, image_width)\n"
    new_guides_call = "        guides = _guides_html(x_models, image_width, image_height)\n"
    if old_guides_call not in source:
        raise RuntimeError("Kunde inte skicka bildhöjden till hjälplinjerna")
    return source.replace(old_guides_call, new_guides_call, 1)


exec(compile(_source(), str(Path(__file__).resolve()), "exec"), globals(), globals())
