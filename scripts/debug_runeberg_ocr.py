from __future__ import annotations

import importlib.util
import io
import math
import re
import statistics
import sys
from dataclasses import replace
from pathlib import Path

from PIL import Image


BASE = Path(__file__).with_name("debug_runeberg_ocr_base.py")
LETTER_RE = re.compile(r"[A-Za-zÅÄÖåäö]")
MERGED_HOMONYM_RE = re.compile(r"^([1-9])([A-Za-zÅÄÖåäö].*)$")
_BODY_TOP_Y: float | None = None


def _load_base_module():
    spec = importlib.util.spec_from_file_location("debug_runeberg_ocr_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Kunde inte läsa {BASE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def _header_rule(content: bytes) -> tuple[float, float] | None:
    """Return the separating rule's angle in degrees and centre y."""
    with Image.open(io.BytesIO(content)) as source:
        gray = source.convert("L")
        width, height = gray.size
        pixels = gray.load()

        x0 = max(0, round(width * 0.04))
        x1 = min(width, round(width * 0.96))
        y0 = max(0, round(height * 0.025))
        y1 = min(height, round(height * 0.24))
        if x1 - x0 < 100 or y1 - y0 < 20:
            return None

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
        if span < width * 0.60 or dark_count < width * 0.16:
            return None

        half_band = max(8, round(height * 0.018))
        step = max(1, width // 1400)
        points: list[tuple[float, float]] = []
        for x in range(x0, x1, step):
            ys = [
                y
                for y in range(
                    max(y0, peak_y - half_band),
                    min(y1, peak_y + half_band + 1),
                )
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
        residuals = [abs(y - (slope * x + intercept)) for x, y in points]
        median_residual = statistics.median(residuals)
        mad = statistics.median(abs(value - median_residual) for value in residuals)
        tolerance = max(1.5, median_residual + 4.0 * 1.4826 * mad)
        clean = [
            point
            for point, residual in zip(points, residuals)
            if residual <= tolerance
        ]
        if len(clean) < 40:
            return None

        fit = _linear_fit(clean)
        if fit is None:
            return None
        slope, intercept = fit
        angle = math.degrees(math.atan(slope))
        if abs(angle) > 7.0:
            return None
        centre_y = slope * (width / 2) + intercept
        return angle, centre_y


def _rule_deskew(module, content: bytes, observations: list) -> tuple[bytes, float]:
    del module, observations
    global _BODY_TOP_Y

    detected = _header_rule(content)
    if detected is None:
        _BODY_TOP_Y = None
        return content, 0.0

    angle, rule_y = detected
    angle = max(-6.0, min(6.0, angle))
    if abs(angle) < 0.03:
        _BODY_TOP_Y = rule_y
        return content, 0.0

    with Image.open(io.BytesIO(content)) as source:
        image = source.convert("RGB")
        deskewed = image.rotate(
            -angle,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor="white",
        )
        output = io.BytesIO()
        deskewed.save(output, format="PNG", optimize=True)
        result = output.getvalue()

    after = _header_rule(result)
    _BODY_TOP_Y = after[1] if after is not None else rule_y
    return result, angle


def _split_two_positions(
    values: list[float], median_height: float
) -> tuple[float, float]:
    values = sorted(values)
    if not values:
        return 0.0, max(8.0, median_height * 0.8)
    if len(values) < 4:
        left = float(min(values))
        return left, left + max(8.0, median_height * 0.8)

    trim = max(0, round(len(values) * 0.02))
    clean = values[trim : len(values) - trim] if trim and len(values) > 2 * trim else values
    minimum_side = max(2, round(len(clean) * 0.08))
    candidates: list[tuple[float, int]] = []
    for index in range(minimum_side - 1, len(clean) - minimum_side):
        gap = clean[index + 1] - clean[index]
        if gap >= max(2.5, median_height * 0.20):
            candidates.append((gap, index))

    if not candidates:
        left = float(statistics.median(clean))
        return left, left + max(8.0, median_height * 0.8)

    _gap, index = max(candidates)
    article_x = float(statistics.median(clean[: index + 1]))
    continuation_x = float(statistics.median(clean[index + 1 :]))
    return article_x, continuation_x


def _word_geometry(module, items: tuple, median_height: float):
    """Return word x, word object and possible homonym-marker x."""
    first = items[0]
    first_text = first.text.strip()

    if len(items) >= 2:
        second = items[1]
        second_text = second.text.strip()
        marker_like = bool(re.fullmatch(r"[1-9Iil|]", first_text))
        raised = (
            first.top < second.top
            or first.top + first.height <= second.top + second.height * 0.90
        )
        small = first.height <= max(median_height * 0.95, second.height * 0.95)
        close = second.left - (first.left + first.width) <= max(16.0, median_height)
        if marker_like and raised and small and close and LETTER_RE.search(second_text):
            return float(second.left), second, float(first.left), second_text

    merged = MERGED_HOMONYM_RE.match(first_text)
    if merged:
        word_text = merged.group(2)
        character_width = max(first.width / max(2, len(first_text)), first.height * 0.20)
        word_x = float(first.left + character_width)
        word = replace(first, text=word_text, ocr_tesseract=word_text)
        return word_x, word, float(first.left), word_text

    for item in items[:3]:
        text = item.text.strip()
        if LETTER_RE.search(text):
            if item is first:
                merged_word = module._merge_line_headword(items, median_height)
                return float(item.left), merged_word, None, merged_word.text.strip()
            return float(item.left), item, None, text

    merged_word = module._merge_line_headword(items, median_height)
    return float(first.left), merged_word, None, merged_word.text.strip()


def _bold_scores(prepared: list[tuple]) -> dict[int, float]:
    densities = [word.ink_density for _line, _x, word, _h, _text in prepared if word.ink_density > 0]
    if not densities:
        return {id(word): 0.0 for _line, _x, word, _h, _text in prepared}
    ordinary = statistics.median(densities)
    ordered = sorted(densities)
    high = ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.80))]
    reference = max(ordinary, high, 1e-6)
    return {
        id(word): max(0.0, min(1.0, (word.ink_density / reference - 0.68) / 0.34))
        for _line, _x, word, _h, _text in prepared
    }


def _build_lines(module, observations, image_width: int, image_height: int):
    if not observations:
        empty = module.ColumnXModel(None, 0.0, 0.0, 0.0)
        return [], {1: empty, 2: empty}

    heights = [item.height for item in observations]
    median_height = statistics.median(heights) if heights else 1.0
    body_top = _BODY_TOP_Y if _BODY_TOP_Y is not None else image_height * 0.03
    body_top += max(1.0, median_height * 0.10)
    split = image_width / 2

    raw: dict[int, list] = {1: [], 2: []}
    for indices in module._observation_line_indices(observations):
        items = tuple(sorted((observations[index] for index in indices), key=lambda item: item.left))
        if not items:
            continue
        top = min(item.top for item in items)
        bottom = max(item.top + item.height for item in items)
        centre_y = (top + bottom) / 2
        if centre_y <= body_top or centre_y >= image_height * 0.94:
            continue

        left = min(item.left for item in items)
        right = max(item.left + item.width for item in items)
        column = 1 if (left + right) / 2 < split else 2
        word_x, word, marker_x, word_text = _word_geometry(module, items, median_height)
        text = " ".join(item.text for item in items)
        line = module.PrintedLine(
            column=column,
            items=items,
            text=text,
            first=word,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            raw_start_x=float(left),
            letter_start_x=word_x,
            has_homonym_marker=marker_x is not None,
        )
        raw[column].append((line, word_x, word, marker_x, word_text))

    result = []
    models = {}
    for column in (1, 2):
        prepared = sorted(raw[column], key=lambda item: (item[0].top, item[0].left))
        if not prepared:
            fallback = 0.0 if column == 1 else split
            models[column] = module.ColumnXModel(None, fallback, fallback, fallback)
            continue

        lexical_x = [x for _line, x, _word, _marker, text in prepared if LETTER_RE.search(text)]
        article_x, continuation_x = _split_two_positions(lexical_x, median_height)
        boundary_x = (article_x + continuation_x) / 2
        marker_xs = [marker for _line, _x, _word, marker, _text in prepared if marker is not None]
        homonym_x = float(statistics.median(marker_xs)) if marker_xs else None
        if homonym_x is not None and homonym_x >= article_x:
            homonym_x = None
        models[column] = module.ColumnXModel(homonym_x, article_x, continuation_x, boundary_x)

        scores = _bold_scores(prepared)
        separation = max(continuation_x - article_x, median_height * 0.5)
        for line, word_x, word, marker_x, word_text in prepared:
            bold_score = scores[id(word)]
            distance_a = abs(word_x - article_x)
            distance_f = abs(word_x - continuation_x)
            clearly_at_article = distance_a <= min(distance_f, separation * 0.38)
            ambiguous_left = word_x <= boundary_x and distance_a <= separation * 0.55
            is_article = bool(LETTER_RE.search(word_text)) and (
                clearly_at_article or (ambiguous_left and bold_score >= 0.45)
            )
            if is_article and marker_x is not None:
                x_class = "homonym+article"
            elif is_article:
                x_class = "article"
            else:
                x_class = "continuation"
            result.append(
                replace(
                    line,
                    first=word,
                    letter_start_x=word_x,
                    has_homonym_marker=marker_x is not None and is_article,
                    x_class=x_class,
                    bold_score=bold_score,
                )
            )

    return sorted(result, key=lambda line: (line.column, line.top, line.left)), models


def _group_articles(module, lines, threshold: float):
    del threshold
    articles = []
    for column in (1, 2):
        current = []
        current_score = 0.0
        for line in (line for line in lines if line.column == column):
            if line.x_class.endswith("article"):
                if current:
                    first = current[0]
                    articles.append(
                        module.Article(column, first.first.text, first.first.text, tuple(current), current_score)
                    )
                current = [line]
                current_score = line.bold_score
            elif current:
                current.append(line)
        if current:
            first = current[0]
            articles.append(
                module.Article(column, first.first.text, first.first.text, tuple(current), current_score)
            )
    return articles


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
    original_review_html = module._review_html

    module._deskew_image = lambda content, observations: _rule_deskew(module, content, observations)
    module._build_lines = lambda observations, image_width, image_height: _build_lines(
        module, observations, image_width, image_height
    )
    module._group_articles = lambda lines, threshold: _group_articles(module, lines, threshold)

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
