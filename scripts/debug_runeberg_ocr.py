from __future__ import annotations

import importlib.util
import io
import math
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
    """Return (word_x, word_object, stripped_text, geometric_candidate)."""
    items = line.items
    first = items[0]
    token = first.text.strip()

    if len(items) >= 2:
        second = items[1]
        second_text = second.text.strip()
        if module.WORD_RE.match(second_text):
            gap = second.left - (first.left + first.width)
            small = first.height <= max(median_height * 0.92, second.height * 0.90)
            raised = (
                first.top < second.top
                or first.top + first.height <= second.top + second.height * 0.86
            )
            narrow = first.width <= max(second.height, median_height * 0.95)
            close = -3.0 <= gap <= max(14.0, second.height * 1.10)
            isolated = 0 < len(token) <= 2
            if small and raised and narrow and close and isolated:
                return float(second.left), second, second_text, True

    if len(token) >= 2 and token[0] in AMBIGUOUS_PREFIXES:
        rest = token[1:]
        if module.WORD_RE.match(rest):
            character_width = max(first.width / max(2, len(token)), first.height * 0.20)
            return float(first.left + character_width), first, rest, True

    return float(line.letter_start_x), line.first, line.first.text.strip(), False


def _remove_outliers(values: list[float], median_height: float) -> list[float]:
    if len(values) < 6:
        return values[:]
    centre = statistics.median(values)
    mad = statistics.median(abs(value - centre) for value in values)
    if mad <= 0:
        radius = max(10.0, median_height * 4.0)
    else:
        radius = max(median_height * 1.5, 4.5 * 1.4826 * mad)
    filtered = [value for value in values if abs(value - centre) <= radius]
    return filtered if len(filtered) >= 4 else values[:]


def _largest_gap_centres(
    values: list[float],
    median_height: float,
    fallback_a: float,
    fallback_f: float,
) -> tuple[float, float]:
    clean = sorted(_remove_outliers(values, median_height))
    if len(clean) < 4:
        return fallback_a, fallback_f

    minimum_side = max(2, round(len(clean) * 0.06))
    candidates: list[tuple[float, int]] = []
    for index in range(minimum_side - 1, len(clean) - minimum_side):
        gap = clean[index + 1] - clean[index]
        if gap >= max(2.5, median_height * 0.22):
            candidates.append((gap, index))

    if not candidates:
        return fallback_a, fallback_f

    _gap, split_index = max(candidates, key=lambda item: item[0])
    article_x = float(statistics.median(clean[: split_index + 1]))
    continuation_x = float(statistics.median(clean[split_index + 1 :]))
    if continuation_x - article_x < max(3.0, median_height * 0.35):
        return fallback_a, fallback_f
    return article_x, continuation_x


def _filter_h_positions(
    values: list[float], median_height: float, article_x: float
) -> list[float]:
    margin = max(1.5, median_height * 0.10)
    values = [value for value in values if value < article_x - margin]
    if len(values) <= 2:
        return values
    centre = statistics.median(values)
    radius = max(4.0, median_height * 0.8)
    nearby = [value for value in values if abs(value - centre) <= radius]
    return nearby if nearby else values


def _geometry_build_lines(module, original_build_lines, observations, image_width, image_height):
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
        geometry = [_prefix_geometry(module, line, median_height) for line in column_lines]
        lexical_x = [
            word_x
            for word_x, _word, stripped, _candidate in geometry
            if module.WORD_RE.search(stripped)
        ]
        article_x, continuation_x = _largest_gap_centres(
            lexical_x,
            median_height,
            float(preliminary.article_x),
            float(preliminary.continuation_x),
        )
        boundary_x = (article_x + continuation_x) / 2

        minimum_prefix_gap = max(1.5, median_height * 0.10)
        maximum_prefix_gap = max(
            median_height * 1.8,
            (continuation_x - article_x) * 0.85,
        )
        article_tolerance = max(
            median_height * 0.9,
            (continuation_x - article_x) * 0.30,
        )
        prepared = []
        accepted_prefix_x: list[float] = []

        for line, (word_x, word_object, stripped, candidate) in zip(column_lines, geometry):
            raw_x = float(line.raw_start_x)
            raw_gap = word_x - raw_x
            geometric_homonym = (
                candidate
                and raw_x < article_x - minimum_prefix_gap
                and abs(word_x - article_x) <= article_tolerance
                and minimum_prefix_gap <= raw_gap <= maximum_prefix_gap
            )

            headword_object = word_object
            if geometric_homonym and stripped:
                headword_object = replace(
                    word_object,
                    text=stripped,
                    ocr_tesseract=stripped,
                )
                accepted_prefix_x.append(raw_x)

            prepared.append((line, word_x, headword_object, geometric_homonym))

        h_samples = _filter_h_positions(accepted_prefix_x, median_height, article_x)
        homonym_x = float(statistics.median(h_samples)) if h_samples else None
        if homonym_x is not None and homonym_x >= article_x:
            homonym_x = None

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


def _linear_fit(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if len(points) < 3:
        return None
    mean_x = statistics.fmean(x for x, _ in points)
    mean_y = statistics.fmean(y for _, y in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator <= 1e-9:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    return slope, mean_y - slope * mean_x


def _header_rule_skew_degrees(content: bytes) -> float | None:
    """Measure the long rule directly below the page number."""
    with Image.open(io.BytesIO(content)) as source:
        gray = source.convert("L")
        width, height = gray.size
        pixels = gray.load()

        # The page number and running head are above this band; body text starts
        # below it. This deliberately targets the separating rule, not text.
        x0 = max(0, round(width * 0.04))
        x1 = min(width, round(width * 0.96))
        y0 = max(0, round(height * 0.09))
        y1 = min(height, round(height * 0.23))
        if x1 - x0 < 100 or y1 - y0 < 20:
            return None

        # Find the row with the greatest horizontal dark coverage. The rule spans
        # most of the page, unlike the page number and running head.
        row_scores: list[tuple[int, int, int]] = []
        for y in range(y0, y1):
            dark_x = [x for x in range(x0, x1) if pixels[x, y] < 160]
            if not dark_x:
                continue
            span = dark_x[-1] - dark_x[0]
            row_scores.append((span, len(dark_x), y))
        if not row_scores:
            return None
        span, dark_count, peak_y = max(row_scores)
        if span < width * 0.60 or dark_count < width * 0.18:
            return None

        # For each sampled x, take the median dark y close to the detected rule.
        # A generous vertical band still follows a visibly skewed rule.
        half_band = max(8, round(height * 0.018))
        step = max(1, width // 1400)
        points: list[tuple[float, float]] = []
        for x in range(x0, x1, step):
            ys = [
                y
                for y in range(max(y0, peak_y - half_band), min(y1, peak_y + half_band + 1))
                if pixels[x, y] < 160
            ]
            if ys:
                points.append((float(x), float(statistics.median(ys))))

        if len(points) < 50 or points[-1][0] - points[0][0] < width * 0.55:
            return None

        fit = _linear_fit(points)
        if fit is None:
            return None
        slope, intercept = fit

        # Remove page-number/running-head fragments and dust that happen to fall
        # inside the band, then fit the rule once more.
        residuals = [abs(y - (slope * x + intercept)) for x, y in points]
        median_residual = statistics.median(residuals)
        mad = statistics.median(abs(value - median_residual) for value in residuals)
        tolerance = max(1.5, median_residual + 4.0 * 1.4826 * mad)
        clean = [
            point
            for point, residual in zip(points, residuals)
            if residual <= tolerance
        ]
        if len(clean) < 40 or clean[-1][0] - clean[0][0] < width * 0.55:
            return None

        fit = _linear_fit(clean)
        if fit is None:
            return None
        slope, _intercept = fit
        angle = math.degrees(math.atan(slope))
        if abs(angle) > 7.0:
            return None
        return angle


def _text_skew_degrees(module, observations: list) -> float | None:
    slopes: list[float] = []
    for indices in module._observation_line_indices(observations):
        items = sorted((observations[index] for index in indices), key=lambda item: item.left)
        if len(items) < 3:
            continue
        span = (items[-1].left + items[-1].width) - items[0].left
        median_height = statistics.median(item.height for item in items)
        if span < max(80.0, median_height * 5):
            continue
        points = [
            (item.left + item.width / 2, item.top + item.height / 2)
            for item in items
        ]
        slope = module._linear_slope(points)
        if slope is not None and abs(slope) < math.tan(math.radians(7.0)):
            slopes.append(slope)
    if not slopes:
        return None
    return math.degrees(math.atan(statistics.median(slopes)))


def _rule_deskew(module, content: bytes, observations: list) -> tuple[bytes, float]:
    skew_degrees = _header_rule_skew_degrees(content)
    if skew_degrees is None:
        skew_degrees = _text_skew_degrees(module, observations)
    if skew_degrees is None:
        return content, 0.0

    skew_degrees = max(-6.0, min(6.0, skew_degrees))
    if abs(skew_degrees) < 0.03:
        return content, 0.0

    with Image.open(io.BytesIO(content)) as source:
        image = source.convert("RGB")
        deskewed = image.rotate(
            -skew_degrees,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor="white",
        )
        output = io.BytesIO()
        deskewed.save(output, format="PNG", optimize=True)
    return output.getvalue(), skew_degrees


def _guides_html(x_models: dict, image_width: int) -> str:
    result: list[str] = []
    definitions = (
        ("homonym", "#7c3aed"),
        ("article", "#16a34a"),
        ("continuation", "#ea580c"),
    )
    for column in (1, 2):
        model = x_models[column]
        positions = {
            "homonym": model.homonym_x,
            "article": model.article_x,
            "continuation": model.continuation_x,
        }
        for kind, color in definitions:
            position = positions[kind]
            if position is None:
                continue
            x = max(0.0, min(float(image_width), float(position)))
            result.append(
                '<div class="x-guide x-guide-%s" data-x="%.3f" '
                'style="--guide-color:%s"></div>' % (kind, x, color)
            )
    return "".join(result)


def main() -> None:
    module = _load_base_module()
    original_build_lines = module._build_lines
    original_review_html = module._review_html

    module._deskew_image = lambda content, observations: _rule_deskew(
        module, content, observations
    )
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
.marker { z-index:40; }
</style>
"""
        guides = _guides_html(x_models, image_width)
        report = report.replace("</head>", css + "</head>", 1)
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
