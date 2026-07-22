from __future__ import annotations

import importlib.util
import io
import statistics
import sys
from dataclasses import replace
from pathlib import Path

from PIL import Image


BASE = Path(__file__).with_name("debug_runeberg_ocr_base.py")
AMBIGUOUS_PREFIXES = set("123456789Iil|oO°.'`,:")


def _load_base_module():
    spec = importlib.util.spec_from_file_location("debug_runeberg_ocr_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Kunde inte läsa {BASE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prefix_geometry(module, line, median_height: float):
    """Return (word_x, word_object, stripped_text, geometric_candidate).

    Candidate classification is deliberately geometry-first. OCR does not need
    to have recognised the raised glyph as a digit.
    """
    items = line.items
    first = items[0]
    token = first.text.strip()

    # Separate raised glyph followed by the actual headword.
    if len(items) >= 2:
        second = items[1]
        second_text = second.text.strip()
        if module.WORD_RE.match(second_text):
            gap = second.left - (first.left + first.width)
            small = first.height <= max(median_height * 0.88, second.height * 0.86)
            raised = (
                first.top < second.top
                or first.top + first.height <= second.top + second.height * 0.82
            )
            narrow = first.width <= max(second.height * 0.90, median_height * 0.85)
            close = -3.0 <= gap <= max(12.0, second.height * 0.95)
            isolated = len(token) <= 2
            if small and raised and narrow and close and isolated:
                return float(second.left), second, second_text, True

    # OCR may have joined index and word. This remains a weaker candidate;
    # the first-pass A position must confirm it later.
    if len(token) >= 2 and token[0] in AMBIGUOUS_PREFIXES:
        rest = token[1:]
        if module.WORD_RE.match(rest):
            character_width = max(first.width / max(2, len(token)), first.height * 0.20)
            return float(first.left + character_width), first, rest, True

    return float(line.letter_start_x), line.first, line.first.text.strip(), False


def _anchored_median(values: list[float], anchor: float, radius: float) -> float:
    nearby = [value for value in values if abs(value - anchor) <= radius]
    return statistics.median(nearby) if nearby else anchor


def _geometry_build_lines(module, original_build_lines, observations, image_width, image_height):
    # Pass 1: retain the base implementation's A/F estimate as stable anchors.
    original_lines, preliminary_models = original_build_lines(
        observations, image_width, image_height
    )
    if not original_lines:
        return original_lines, preliminary_models

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

        preliminary = preliminary_models[column]
        preliminary_a = float(preliminary.article_x)
        preliminary_f = float(preliminary.continuation_x)
        separation = max(median_height * 0.7, preliminary_f - preliminary_a)
        article_radius = max(median_height * 1.15, separation * 0.48)
        continuation_radius = max(median_height * 1.30, separation * 0.55)
        minimum_prefix_gap = max(2.0, median_height * 0.12)
        maximum_prefix_gap = max(median_height * 1.35, separation * 0.75)

        geometry = [_prefix_geometry(module, line, median_height) for line in column_lines]
        prepared = []
        article_samples: list[float] = []
        continuation_samples: list[float] = []
        accepted_prefix_x: list[float] = []

        # Pass 2a: identify raised-prefix rows relative to preliminary A,
        # and collect only samples close to the first-pass anchors.
        for line, (word_x, word_object, stripped, candidate) in zip(column_lines, geometry):
            raw_gap = word_x - line.raw_start_x
            near_article = abs(word_x - preliminary_a) <= article_radius
            geometric_homonym = (
                candidate
                and near_article
                and minimum_prefix_gap <= raw_gap <= maximum_prefix_gap
            )

            headword_object = word_object
            if geometric_homonym and stripped:
                headword_object = replace(
                    word_object,
                    text=stripped,
                    ocr_tesseract=stripped,
                )
                accepted_prefix_x.append(float(line.raw_start_x))

            lexical = bool(module.WORD_RE.search(headword_object.text))
            if lexical:
                if abs(word_x - preliminary_a) <= article_radius:
                    article_samples.append(word_x)
                elif abs(word_x - preliminary_f) <= continuation_radius:
                    continuation_samples.append(word_x)

            prepared.append((line, word_x, headword_object, geometric_homonym))

        # Pass 2b: refine around the anchors. Outliers cannot move either line.
        article_x = _anchored_median(article_samples, preliminary_a, article_radius)
        continuation_x = _anchored_median(
            continuation_samples, preliminary_f, continuation_radius
        )
        if continuation_x <= article_x + max(3.0, median_height * 0.35):
            continuation_x = preliminary_f
        boundary_x = (article_x + continuation_x) / 2

        # H is unusual: require at least one strong geometric candidate, but do
        # not require OCR to call the raised glyph a digit.
        homonym_x = statistics.median(accepted_prefix_x) if accepted_prefix_x else None
        models[column] = module.ColumnXModel(
            homonym_x,
            article_x,
            continuation_x,
            boundary_x,
        )

        lexical_objects = [
            headword
            for _line, _word_x, headword, _homonym in prepared
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

        for line, word_x, headword, geometric_homonym in prepared:
            ink_ratio = headword.ink_density / bold_reference
            bold_score = max(0.0, min(1.0, (ink_ratio - 0.70) / 0.34))
            x_class = "article" if word_x <= boundary_x else "continuation"
            if geometric_homonym:
                x_class = "homonym+article"
            result.append(
                replace(
                    line,
                    first=headword,
                    letter_start_x=word_x,
                    has_homonym_marker=geometric_homonym,
                    x_class=x_class,
                    bold_score=bold_score,
                )
            )

    return sorted(result, key=lambda line: (line.column, line.top, line.left)), models


def _geometry_group_articles(module, lines, threshold: float):
    """Group articles from the final A/F classification."""
    articles = []
    for column in (1, 2):
        current = []
        current_score = 0.0
        for line in (line for line in lines if line.column == column):
            lexical = bool(module.WORD_RE.search(line.first.text))
            is_start = lexical and line.x_class.endswith("article")
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
        ("homonym", "H", "#7c3aed", 4),
        ("article", "A", "#16a34a", 24),
        ("continuation", "F", "#ea580c", 44),
    )
    for column in (1, 2):
        model = x_models[column]
        positions = {
            "homonym": model.homonym_x,
            "article": model.article_x,
            "continuation": model.continuation_x,
        }
        for kind, short, color, label_top in definitions:
            if positions[kind] is None:
                continue
            x = max(0.0, min(float(image_width), float(positions[kind])))
            result.append(
                '<div class="x-guide x-guide-%s" data-x="%.3f" style="--guide-color:%s">'
                '<span class="x-guide-label" style="top:%dpx">%s%d · x=%.1f</span>'
                "</div>"
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
.x-guide { position:absolute; top:0; bottom:0; width:0; z-index:30;
  border-left:1px solid var(--guide-color); pointer-events:none; opacity:.95; }
.x-guide-homonym { border-left-style:dotted; }
.x-guide-article { border-left-style:solid; }
.x-guide-continuation { border-left-style:dashed; }
.x-guide-label { position:absolute; left:4px; padding:2px 4px; border-radius:3px;
  background:var(--guide-color); color:white; font:700 11px/1.1 system-ui,sans-serif;
  white-space:nowrap; }
.marker { z-index:40; }
.guide-legend { margin-left:14px; font-weight:700; }
.guide-legend .h { color:#7c3aed; }
.guide-legend .a { color:#16a34a; }
.guide-legend .f { color:#ea580c; }
</style>
"""
        legend = (
            '<span class="guide-legend">'
            '<span class="h">H = homonym</span> · '
            '<span class="a">A = artikelstart</span> · '
            '<span class="f">F = fortsättning</span>'
            "</span>"
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
