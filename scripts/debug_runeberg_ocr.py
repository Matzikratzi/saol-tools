from __future__ import annotations

"""Run the rebuilt OCR classifier with geometry derived from the header rule.

The implementation is materialised from the preceding rebuilt commit and then
patched so dictionary text starts immediately below the rule. The midpoint of
the detected rule is used as the boundary between the columns. For this
experiment, the left column's A guide is placed just to the left of the
leftmost printed OCR character; F and H are hidden there until A is settled.
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

    old_imports = "import sys\n"
    new_imports = (
        "import sys\n"
        "import subprocess\n"
        "import webbrowser\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "from urllib.parse import urlparse\n"
    )
    if old_imports not in source:
        raise RuntimeError("Kunde inte lägga till webbläsarstegringens standardbibliotek")
    source = source.replace(old_imports, new_imports, 1)

    old_globals = "_BODY_TOP_Y: float | None = None\n"
    new_globals = (
        "_BODY_TOP_Y: float | None = None\n"
        "_COLUMN_SPLIT_X: float | None = None\n"
        "_LEFT_A_X: float | None = None\n"
        "_LEFT_LEVELS: list[float] = []\n"
        "_LEFT_LEVEL_COUNTS: list[int | None] = []\n"
        "_LEFT_IGNORED_HEADINGS: list[str] = []\n"
        "_LEFT_PREFIX_PAIRS: list[tuple[float, float]] = []\n"
        "_HAF_LEVELS: tuple[float, float, float] | None = None\n"
        "_LEFT_THRESHOLD_X: float | None = None\n"
    )
    if old_globals not in source:
        raise RuntimeError("Kunde inte lägga till geometrins globala positioner")
    source = source.replace(old_globals, new_globals, 1)

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

    rule_deskew_marker = "def _rule_deskew(module, content: bytes, observations: list) -> tuple[bytes, float]:\n"
    horizontal_y_helper = '''def _horizontal_rule_y(content: bytes) -> float | None:
    """Find the physical horizontal rule after deskewing, without changing angle."""
    with Image.open(io.BytesIO(content)) as source:
        gray = source.convert("L")
        width, height = gray.size
        pixels = gray.load()
        x0 = max(0, round(width * 0.04))
        x1 = min(width, round(width * 0.96))
        y0 = max(0, round(height * 0.025))
        y1 = min(height, round(height * 0.24))
        candidates = []
        for y in range(y0, y1):
            dark_x = [x for x in range(x0, x1) if pixels[x, y] < 160]
            if not dark_x:
                continue
            run_start = dark_x[0]
            previous = dark_x[0]
            best_span = 0
            for x in dark_x[1:]:
                if x - previous > 3:
                    best_span = max(best_span, previous - run_start)
                    run_start = x
                previous = x
            best_span = max(best_span, previous - run_start)
            candidates.append((best_span, -y))
        if not candidates:
            return None
        span, negative_y = max(candidates)
        if span < width * 0.35:
            return None
        return float(-negative_y)


'''
    if rule_deskew_marker not in source:
        raise RuntimeError("Kunde inte lägga till separat y-mätning efter upprätning")
    source = source.replace(rule_deskew_marker, horizontal_y_helper + rule_deskew_marker, 1)

    old_unrotated_y = (
        "    if abs(angle) < 0.03:\n"
        "        _BODY_TOP_Y = rule_y\n"
        "        return content, 0.0\n"
    )
    new_unrotated_y = (
        "    if abs(angle) < 0.03:\n"
        "        horizontal_y = _horizontal_rule_y(content)\n"
        "        _BODY_TOP_Y = horizontal_y if horizontal_y is not None else rule_y\n"
        "        return content, 0.0\n"
    )
    if old_unrotated_y not in source:
        raise RuntimeError("Kunde inte mäta startlinjen på en redan vågrät bild")
    source = source.replace(old_unrotated_y, new_unrotated_y, 1)

    old_after_y = (
        "    after = _header_rule(result)\n"
        "    _BODY_TOP_Y = after[1] if after is not None else rule_y\n"
        "    return result, angle\n"
    )
    new_after_y = (
        "    horizontal_y = _horizontal_rule_y(result)\n"
        "    _BODY_TOP_Y = horizontal_y if horizontal_y is not None else rule_y\n"
        "    return result, angle\n"
    )
    if old_after_y not in source:
        raise RuntimeError("Kunde inte mäta startlinjen efter upprätning")
    source = source.replace(old_after_y, new_after_y, 1)

    # Keep Pillow's original sign convention from the rebuilt implementation.
    # Changing -angle to +angle doubles the skew instead of removing it.

    old_body = (
        "    body_top = _BODY_TOP_Y if _BODY_TOP_Y is not None else image_height * 0.03\n"
        "    body_top += max(1.0, median_height * 0.10)\n"
    )
    new_body = (
        "    # Start three image pixels below the deskewed header rule.\n"
        "    body_top = (_BODY_TOP_Y + 3.0) if _BODY_TOP_Y is not None else image_height * 0.03\n"
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
        "    global _LEFT_A_X, _LEFT_LEVELS, _LEFT_LEVEL_COUNTS, _LEFT_IGNORED_HEADINGS, _LEFT_PREFIX_PAIRS, _HAF_LEVELS, _LEFT_THRESHOLD_X\n"
        "    # Experimental A: two pixels immediately left of the leftmost OCR\n"
        "    # character in the left column. No clustering and no F/H influence.\n"
        "    _LEFT_A_X = (\n"
        "        min(line.left for line, _x, _word, _marker, _text in raw[1]) - 2.0\n"
        "        if raw[1]\n"
        "        else None\n"
        "    )\n"
        "    raw_starts = []\n"
        "    _LEFT_IGNORED_HEADINGS = []\n"
        "    _LEFT_PREFIX_PAIRS = []\n"
        "    chapter_limit = body_top + image_height * 0.15\n"
        "    for line, word_x, word, marker_x, text in raw[1]:\n"
        "        letters = re.sub(r'[^A-Za-zÅÄÖåäö]', '', text)\n"
        "        is_chapter_heading = (\n"
        "            1 <= len(letters) <= 3\n"
        "            and letters[0].isupper()\n"
        "            and word.height >= median_height * 1.60\n"
        "            and line.top <= chapter_limit\n"
        "        )\n"
        "        if is_chapter_heading:\n"
        "            _LEFT_IGNORED_HEADINGS.append(text.strip())\n"
        "            continue\n"
        "        raw_starts.append(float(line.raw_start_x))\n"
        "        if marker_x is not None:\n"
        "            _LEFT_PREFIX_PAIRS.append((float(marker_x), float(word_x)))\n"
        "    raw_starts.sort()\n"
        "    tolerance = max(2.5, median_height * 0.20)\n"
        "    clusters = []\n"
        "    for value in raw_starts:\n"
        "        if clusters and value - clusters[-1][-1] <= tolerance:\n"
        "            clusters[-1].append(value)\n"
        "        else:\n"
        "            clusters.append([value])\n"
        "    minimum_count = max(2, round(len(raw_starts) * 0.04))\n"
        "    recurring = [\n"
        "        (float(statistics.median(cluster)), len(cluster)) for cluster in clusters\n"
        "        if len(cluster) >= minimum_count\n"
        "    ]\n"
        "    _LEFT_LEVELS = ([_LEFT_A_X] if _LEFT_A_X is not None else []) + [value for value, _count in recurring[:3]]\n"
        "    _LEFT_LEVEL_COUNTS = ([None] if _LEFT_A_X is not None else []) + [count for _value, count in recurring[:3]]\n"
        "    matches = []\n"
        "    for first in range(len(_LEFT_LEVELS) - 2):\n"
        "        for second in range(first + 1, len(_LEFT_LEVELS) - 1):\n"
        "            for third in range(second + 1, len(_LEFT_LEVELS)):\n"
        "                h, a, f = (_LEFT_LEVELS[first], _LEFT_LEVELS[second], _LEFT_LEVELS[third])\n"
        "                delta_ha, delta_af = a - h, f - a\n"
        "                if abs(delta_ha - 23.0) <= 8.0 and abs(delta_af - 40.0) <= 10.0:\n"
        "                    matches.append((abs(delta_ha - 23.0) + abs(delta_af - 40.0), h, a, f))\n"
        "    _HAF_LEVELS = tuple(min(matches)[1:]) if matches else None\n"
        "    _LEFT_THRESHOLD_X = (\n"
        "        (_HAF_LEVELS[1] + _HAF_LEVELS[2]) / 2\n"
        "        if _HAF_LEVELS is not None\n"
        "        else None\n"
        "    )\n"
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
        "        if column == 1 and _HAF_LEVELS is not None:\n"
        "            _h_x, article_x, continuation_x = _HAF_LEVELS\n"
        "        elif column == 1 and _LEFT_A_X is not None:\n"
        "            article_x = _LEFT_A_X\n"
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
        "                # Count the line as an article when any part of its first\n"
        "                # lexical OCR box reaches left of T. Ignore unrelated raw\n"
        "                # OCR fragments that precede that lexical box.\n"
        "                is_article = bool(LETTER_RE.search(word_text)) and word.left < _LEFT_THRESHOLD_X\n"
        "            else:\n"
        "                is_article = bool(LETTER_RE.search(word_text)) and (\n"
        "                    clearly_at_article or (ambiguous_left and bold_score >= 0.45)\n"
        "                )\n"
    )
    if old_article_decision not in source:
        raise RuntimeError("Kunde inte använda T som gräns för fortsättningstext")
    source = source.replace(old_article_decision, new_article_decision, 1)

    old_guides = "def _guides_html(x_models: dict, image_width: int) -> str:\n    result: list[str] = []\n"
    new_guides = (
        "def _guides_html(x_models: dict, image_width: int, image_height: int) -> str:\n"
        "    split_x = _COLUMN_SPLIT_X if _COLUMN_SPLIT_X is not None else image_width / 2\n"
        "    result: list[str] = [\n"
        "        '<div class=\"x-guide x-guide-column-split\" data-x=\"%.3f\" ' \n"
        "        'style=\"--guide-color:#2563eb\"></div>' % split_x\n"
        "    ]\n"
        "    if _BODY_TOP_Y is not None and image_height > 0:\n"
        "        start_y = max(0.0, min(float(image_height), _BODY_TOP_Y + 3.0))\n"
        "        result.append(\n"
        "            '<div class=\"y-guide y-guide-body-start\" style=\"top:%.6f%%\"></div>'\n"
        "            % (100.0 * start_y / image_height)\n"
        "        )\n"
        "    if _HAF_LEVELS is not None:\n"
        "        for index, (label, color, position) in enumerate(zip(('H', 'A', 'F'), ('#7c3aed', '#16a34a', '#ea580c'), _HAF_LEVELS)):\n"
        "            x = max(0.0, min(float(image_width), float(position)))\n"
        "            result.append(\n"
        "                '<div class=\"x-guide x-guide-level\" data-x=\"%.3f\" ' \n"
        "                'style=\"--guide-color:%s\"><span style=\"top:%dpx\">%s</span></div>'\n"
        "                % (x, color, 8 + index * 22, label)\n"
        "            )\n"
        "    if _LEFT_THRESHOLD_X is not None:\n"
        "        x = max(0.0, min(float(image_width), float(_LEFT_THRESHOLD_X)))\n"
        "        result.append(\n"
        "            '<div class=\"x-guide x-guide-threshold\" data-x=\"%.3f\" ' \n"
        "            'style=\"--guide-color:#dc2626\"><span style=\"top:74px\">T</span></div>'\n"
        "            % x\n"
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
        "            # The left column uses neutral N1/N2/N3 guides.\n"
        "            if column == 1:\n"
        "                continue\n"
        "            position = positions[kind]\n"
    )
    if old_guide_loop not in source:
        raise RuntimeError("Kunde inte ersätta vänsterspaltens semantiska linjer")
    source = source.replace(old_guide_loop, new_guide_loop, 1)

    old_report_image = "        with Image.open(io.BytesIO(image_content)) as image:\n"
    new_report_image = (
        "        previous_page = max(1, page - 1)\n"
        "        navigation = (\n"
        "            f'<a class=\"page-step\" href=\"/page/{previous_page}\">← Sida {previous_page}</a>'\n"
        "            f'<a class=\"page-step\" href=\"/page/{page + 1}\">Sida {page + 1} →</a>'\n"
        "        )\n"
        "        report = report.replace(\n"
        "            '<span id=\"zoomLabel\">100 %</span>',\n"
        "            '<span id=\"zoomLabel\">100 %</span>' + navigation,\n"
        "            1,\n"
        "        )\n"
        "        with Image.open(io.BytesIO(image_content)) as image:\n"
    )
    if old_report_image not in source:
        raise RuntimeError("Kunde inte lägga till sidknappar i rapporten")
    source = source.replace(old_report_image, new_report_image, 1)

    old_css = ".x-guide-continuation { border-left-style:dashed; }\n.marker { z-index:40; }"
    new_css = (
        ".x-guide-continuation { border-left-style:dashed; }\n"
        ".x-guide-column-split { border-left-width:4px; border-left-style:solid; opacity:1; }\n"
        ".x-guide-article { border-left-width:2px; }\n"
        ".x-guide-level { border-left-width:2px; }\n"
        ".x-guide-threshold { border-left-width:2px; border-left-style:dashed; }\n"
        ".x-guide-threshold span { position:absolute; left:4px; padding:2px 4px; "
        "border-radius:3px; background:var(--guide-color); color:white; "
        "font:700 11px/1.1 system-ui,sans-serif; white-space:nowrap; }\n"
        ".x-guide-level span { position:absolute; left:4px; padding:2px 4px; "
        "border-radius:3px; background:var(--guide-color); color:white; "
        "font:700 11px/1.1 system-ui,sans-serif; white-space:nowrap; }\n"
        ".page-step { display:inline-block; margin-left:8px; padding:4px 8px; "
        "border-radius:4px; background:#2563eb; color:white; text-decoration:none; "
        "font-weight:700; }\n"
        ".y-guide { position:absolute; left:0; right:0; height:0; z-index:35; "
        "border-top:3px solid #0891b2; pointer-events:none; }\n"
        ".marker { z-index:40; }"
    )
    if old_css not in source:
        raise RuntimeError("Kunde inte formatera hjälplinjerna i HTML-rapporten")
    source = source.replace(old_css, new_css, 1)

    main_marker = "def main() -> None:\n"
    server_helper = '''def _serve_reviews(initial_page: int, open_browser: bool) -> None:
    cache = REPOSITORY_ROOT / ".page-review-cache"
    cache.mkdir(exist_ok=True)

    class ReviewHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send_response(302)
                self.send_header("Location", f"/page/{initial_page}")
                self.end_headers()
                return
            match = re.fullmatch(r"/page/(\\d+)", path)
            if match is None:
                self.send_error(404)
                return
            page = max(1, int(match.group(1)))
            output = cache / f"page{page:04d}-review.html"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                str(page),
                "--html",
                str(output),
            ]
            try:
                subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
                content = output.read_bytes()
            except Exception as exc:
                self.send_error(500, str(exc))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format, *args):
            return

    url = f"http://127.0.0.1:8001/page/{initial_page}"
    server = HTTPServer(("127.0.0.1", 8001), ReviewHandler)
    print(f"Sidgranskning: {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


'''
    if main_marker not in source:
        raise RuntimeError("Kunde inte lägga till lokalt webbläge")
    source = source.replace(main_marker, server_helper + main_marker, 1)

    old_main_start = (
        "def main() -> None:\n"
        "    module = _load_base_module()\n"
    )
    new_main_start = (
        "def main() -> None:\n"
        "    if '--serve' in sys.argv:\n"
        "        page = next((int(value) for value in sys.argv[1:] if value.isdigit()), 19)\n"
        "        _serve_reviews(page, '--open' in sys.argv)\n"
        "        return\n"
        "    module = _load_base_module()\n"
    )
    if old_main_start not in source:
        raise RuntimeError("Kunde inte aktivera webbläget")
    source = source.replace(old_main_start, new_main_start, 1)

    old_console_summary = "    module.main()\n"
    new_console_summary = (
        "    module.main()\n"
        "    measured = (_LEFT_LEVELS + [None, None, None, None])[:4]\n"
        "    counts = (_LEFT_LEVEL_COUNTS + [None, None, None, None])[:4]\n"
        "    n1, n2, n3, n4 = measured\n"
        "    d12 = n2 - n1 if n1 is not None and n2 is not None else None\n"
        "    d23 = n3 - n2 if n2 is not None and n3 is not None else None\n"
        "    d34 = n4 - n3 if n3 is not None and n4 is not None else None\n"
        "    start_y = _BODY_TOP_Y + 3.0 if _BODY_TOP_Y is not None else None\n"
        "    value = lambda number: '–' if number is None else f'{number:.1f}'\n"
        "    count = lambda number: '–' if number is None else str(number)\n"
        "    headings = ','.join(_LEFT_IGNORED_HEADINGS) or '–'\n"
        "    prefix_pairs = ','.join(f'{left:.1f}→{word:.1f}' for left, word in _LEFT_PREFIX_PAIRS) or '–'\n"
        "    haf = '–' if _HAF_LEVELS is None else '/'.join(f'{value:.1f}' for value in _HAF_LEVELS)\n"
        "    threshold = value(_LEFT_THRESHOLD_X)\n"
        "    print(\n"
        "        f'MÄTVÄRDEN start_y={value(start_y)} '\n"
        "        f'N1={value(n1)} N2={value(n2)}(rader={count(counts[1])}) '\n"
        "        f'N3={value(n3)}(rader={count(counts[2])}) '\n"
        "        f'N4={value(n4)}(rader={count(counts[3])}) '\n"
        "        f'Δ12={value(d12)} Δ23={value(d23)} Δ34={value(d34)} '\n"
        "        f'rubriker={headings} prefixpar={prefix_pairs} H/A/F={haf} T={threshold}'\n"
        "    )\n"
    )
    if old_console_summary not in source:
        raise RuntimeError("Kunde inte lägga till kopieringsvänliga mätvärden")
    source = source.replace(old_console_summary, new_console_summary, 1)

    old_image_size = "            image_width = image.width\n"
    new_image_size = (
        "            image_width = image.width\n"
        "            image_height = image.height\n"
    )
    if old_image_size not in source:
        raise RuntimeError("Kunde inte läsa bildhöjden för startlinjen")
    source = source.replace(old_image_size, new_image_size, 1)

    old_guides_call = "        guides = _guides_html(x_models, image_width)\n"
    new_guides_call = "        guides = _guides_html(x_models, image_width, image_height)\n"
    if old_guides_call not in source:
        raise RuntimeError("Kunde inte skicka bildhöjden till startlinjen")
    return source.replace(old_guides_call, new_guides_call, 1)


exec(compile(_source(), str(Path(__file__).resolve()), "exec"), globals(), globals())
