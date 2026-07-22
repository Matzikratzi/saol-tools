from __future__ import annotations

import importlib.util
import io
import statistics
import sys
from dataclasses import replace
from pathlib import Path

from PIL import Image


BASE = Path(__file__).with_name("debug_runeberg_ocr_base.py")
AMBIGUOUS_PREFIXES = set("123456789Iil|oO°.")


def _load_base_module():
    spec = importlib.util.spec_from_file_location("debug_runeberg_ocr_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Kunde inte läsa {BASE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prefix_geometry(module, line, median_height: float):
    """Return (letter_x, word_object, stripped_text, is_prefix_candidate)."""
    items = line.items
    first = items[0]
    token = first.text.strip()

    # OCR has kept the index as a separate token, e.g. "o" + "a".
    if len(items) >= 2:
        second = items[1]
        second_text = second.text.strip()
        gap = second.left - (first.left + first.width)
        small_prefix = first.height <= max(median_height * 0.95, second.height * 0.95)
        short_prefix = len(token) <= 2 and token and token[0] in AMBIGUOUS_PREFIXES
        close_enough = -3 <= gap <= max(12.0, median_height * 1.25)
        if short_prefix and small_prefix and close_enough and module.WORD_RE.match(second_text):
            return float(second.left), second, second_text, True

    # OCR has joined index and word, e.g. "oa" instead of "2a".
    if len(token) >= 2 and token[0] in AMBIGUOUS_PREFIXES and module.WORD_RE.match(token[1:]):
        character_width = max(first.width / max(2, len(token)), first.height * 0.20)
        return float(first.left + character_width), first, token[1:], True

    return float(line.letter_start_x), line.first, line.first.text.strip(), line.has_homonym_marker


def _estimate_article_positions(module, geometry, column_lines, median_height: float):
    """Estimate A/F without allowing homonym-prefix positions to pull A left."""
    normal_x = [
        letter_x
        for line, (letter_x, _word, stripped, candidate) in zip(column_lines, geometry)
        if module.WORD_RE.search(stripped) and not candidate
    ]
    all_letter_x = [
        letter_x
        for line, (letter_x, _word, stripped, _candidate) in zip(column_lines, geometry)
        if module.WORD_RE.search(stripped)
    ]

    # Prefer ordinary rows. If the page fragment contains only homonym rows,
    # fall back to their corrected word starts, never their raw prefix starts.
    samples = normal_x if len(normal_x) >= 2 else all_letter_x
    article_x, continuation_x = module._kmeans_1d_two(samples)
    minimum_separation = max(3.0, median_height * 0.35)
    if continuation_x - article_x < minimum_separation:
        article_x = statistics.median(samples) if samples else column_lines[0].left
        continuation_x = article_x + max(8.0, median_height * 0.8)
    return article_x, continuation_x


def _geometry_build_lines(module, original_build_lines, observations, image_width, image_height):
    original_lines, _ = original_build_lines(observations, image_width, image_height)
    if not original_lines:
        return original_lines, {
            1: module.ColumnXModel(None, 0.0, 0.0, 0.0),
            2: module.ColumnXModel(None, 0.0, 0.0, 0.0),
        }

    heights = [item.height for line in original_lines for item in line.items]
    median_height = statistics.median(heights) if heights else 1.0
    result = []
    models = {}

    for column in (1, 2):
        column_lines = sorted(
            (line for line in original_lines if line.column == column),
            key=lambda line: (line.top, line.left),
        )
        if not column_lines:
            models[column] = module.ColumnXModel(None, 0.0, 0.0, 0.0)
            continue

        geometry = [_prefix_geometry(module, line, median_height) for line in column_lines]
        article_x, continuation_x = _estimate_article_positions(
            module, geometry, column_lines, median_height
        )
        boundary_x = (article_x + continuation_x) / 2

        prefix_gap = max(3.0, (continuation_x - article_x) * 0.28, median_height * 0.22)
        accepted_prefix_x = []
        prepared = []

        for line, (letter_x, word_object, stripped, candidate) in zip(column_lines, geometry):
            raw_gap = letter_x - line.raw_start_x
            in_article_zone = letter_x <= boundary_x
            geometric_homonym = candidate and in_article_zone and raw_gap >= prefix_gap

            headword_object = word_object
            if geometric_homonym and stripped:
                headword_object = replace(
                    word_object,
                    text=stripped,
                    ocr_tesseract=stripped,
                )
                accepted_prefix_x.append(line.raw_start_x)

            prepared.append((line, letter_x, headword_object, geometric_homonym))

        # H exists only when there is actual homonym evidence on this column.
        homonym_x = statistics.median(accepted_prefix_x) if accepted_prefix_x else None
        models[column] = module.ColumnXModel(
            homonym_x,
            article_x,
            continuation_x,
            boundary_x,
        )

        lexical_objects = [
            headword
            for line, _letter_x, headword, _homonym in prepared
            if module.WORD_RE.search(headword.text) and headword.ink_density > 0
        ]
        inks = [item.ink_density for item in lexical_objects]
        ordinary_ink = statistics.median(inks) if inks else 1.0
        bold_reference = max(
            sorted(inks)[min(len(inks) - 1, round((len(inks) - 1) * 0.75))]
            if inks
            else ordinary_ink,
            ordinary_ink,
            1e-6,
        )

        for line, letter_x, headword, geometric_homonym in prepared:
            ink_ratio = headword.ink_density / bold_reference
            bold_score = max(0.0, min(1.0, (ink_ratio - 0.70) / 0.34))
            x_class = "article" if letter_x <= boundary_x else "continuation"
            if geometric_homonym:
                x_class = "homonym+article"
            result.append(
                replace(
                    line,
                    first=headword,
                    letter_start_x=letter_x,
                    has_homonym_marker=geometric_homonym,
                    x_class=x_class,
                    bold_score=bold_score,
                )
            )

    return sorted(result, key=lambda line: (line.column, line.top, line.left)), models


def _geometry_group_articles(module, lines, threshold: float):
    """Group articles using the midpoint as the sole start-zone boundary."""
    articles = []
    for column in (1, 2):
        current = []
        current_score = 0.0
        for line in (line for line in lines if line.column == column):
            lexical = bool(module.WORD_RE.search(line.first.text))
            at_article_x = line.x_class.endswith("article")
            is_start = lexical and at_article_x

            if is_start:
                if current:
                    first = current[0]
                    articles.append(
                        module.Article(
                            column,
                            first.first.text,
                            first.first.text,
                            tuple(current),
                            current_score,
                        )
                    )
                current = [line]
                current_score = line.bold_score
            elif current:
                current.append(line)

        if current:
            first = current[0]
            articles.append(
                module.Article(
                    column,
                    first.first.text,
                    first.first.text,
                    tuple(current),
                    current_score,
                )
            )
    return articles


def _guides_html(x_models: dict, image_width: int) -> str:
    result: list[str] = []
    definitions = (
        ("homonym", "H", "Homonymindex", "#7c3aed", 4),
        ("article", "A", "Artikelstart", "#16a34a", 24),
        ("continuation", "F", "Fortsättningsrad", "#ea580c", 44),
    )
    for column in (1, 2):
        model = x_models[column]
        positions = {
            "homonym": model.homonym_x,
            "article": model.article_x,
            "continuation": model.continuation_x,
        }
        for kind, short, _label, color, label_top in definitions:
            if positions[kind] is None:
                continue
            x = max(0.0, min(float(image_width), float(positions[kind])))
            result.append(
                '<div class="x-guide x-guide-%s" data-x="%.3f" style="--guide-color:%s">'
                '<span class="x-guide-label" style="top:%dpx">%s%d · x=%.1f</span>'
                '</div>'
                % (kind, x, color, label_top, short, column, x)
            )
    return "".join(result)


def main() -> None:
    module = _load_base_module()
    original_build_lines = module._build_lines
    original_review_html = module._review_html

    module._build_lines = lambda observations, image_width, image_height: _geometry_build_lines(
        module,
        original_build_lines,
        observations,
        image_width,
        image_height,
    )
    module._group_articles = lambda lines, threshold: _geometry_group_articles(
        module,
        lines,
        threshold,
    )

    def review_html_with_guides(
        page,
        source_url,
        image_content,
        raw_text,
        runeberg_lines,
        observations,
        articles,
        pairs,
        x_models,
        threshold,
        skew_degrees,
    ):
        report = original_review_html(
            page,
            source_url,
            image_content,
            raw_text,
            runeberg_lines,
            observations,
            articles,
            pairs,
            x_models,
            threshold,
            skew_degrees,
        )
        with Image.open(io.BytesIO(image_content)) as image:
            image_width = image.width

        css = """
<style>
.x-guide {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 0;
    z-index: 30;
    border-left: 1px solid var(--guide-color);
    pointer-events: none;
    opacity: .95;
}
.x-guide-homonym { border-left-style: dotted; }
.x-guide-article { border-left-style: solid; }
.x-guide-continuation { border-left-style: dashed; }
.x-guide-label {
    position: absolute;
    left: 4px;
    padding: 2px 4px;
    border-radius: 3px;
    background: var(--guide-color);
    color: white;
    font: 700 11px/1.1 system-ui, sans-serif;
    white-space: nowrap;
}
.marker { z-index: 40; }
.guide-legend { margin-left: 14px; font-weight: 700; }
.guide-legend .h { color: #7c3aed; }
.guide-legend .a { color: #16a34a; }
.guide-legend .f { color: #ea580c; }
</style>
"""
        legend = (
            '<span class="guide-legend">'
            '<span class="h">H = homonym</span> · '
            '<span class="a">A = artikelstart</span> · '
            '<span class="f">F = fortsättning</span>'
            '</span>'
        )
        guides = _guides_html(x_models, image_width)
        report = report.replace("</head>", css + "</head>", 1)
        report = report.replace(
            '<span id="zoomLabel">100 %</span>',
            '<span id="zoomLabel">100 %</span>' + legend,
            1,
        )
        report = report.replace(
            '<div class="marker" id="marker"></div>',
            guides + '<div class="marker" id="marker"></div>',
            1,
        )
        report = report.replace(
            "const rows=[...document.querySelectorAll('#articleRows tr[data-index]')];let zoom=1,selected=-1;",
            "const rows=[...document.querySelectorAll('#articleRows tr[data-index]')],guides=[...document.querySelectorAll('.x-guide')];let zoom=1,selected=-1;",
            1,
        )
        report = report.replace(
            "function setZoom(v){zoom=Math.max(.15,Math.min(3,v));scanStage.style.width=`${naturalWidth*zoom}px`;scanStage.style.height=`${naturalHeight*zoom}px`;scanImage.style.width=`${naturalWidth*zoom}px`;scanImage.style.height=`${naturalHeight*zoom}px`;document.getElementById('zoomLabel').textContent=`${Math.round(zoom*100)} %`;if(selected>=0)positionMarker(rows[selected])}",
            "function setZoom(v){zoom=Math.max(.15,Math.min(3,v));scanStage.style.width=`${naturalWidth*zoom}px`;scanStage.style.height=`${naturalHeight*zoom}px`;scanImage.style.width=`${naturalWidth*zoom}px`;scanImage.style.height=`${naturalHeight*zoom}px`;guides.forEach(g=>g.style.left=`${(+g.dataset.x)*zoom}px`);document.getElementById('zoomLabel').textContent=`${Math.round(zoom*100)} %`;if(selected>=0)positionMarker(rows[selected])}",
            1,
        )
        return report

    module._review_html = review_html_with_guides
    module.main()


if __name__ == "__main__":
    main()
