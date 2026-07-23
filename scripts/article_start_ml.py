from __future__ import annotations

"""Train and compare article-start classifiers against manually transcribed pages.

The OCR/deskew geometry is shared with ``debug_runeberg_ocr.py``.  Evaluation
uses leave-one-page-out validation so that no row from the test page occurs in
the corresponding training set.

Usage:
    PYTHONPATH=. python3 scripts/article_start_ml.py
    PYTHONPATH=. python3 scripts/article_start_ml.py --pages 19 20 21 22
"""

import argparse
import difflib
import html
import io
import json
import math
import re
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from PIL import Image

from scripts import debug_runeberg_ocr as debug


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRUTH = ROOT / "data" / "facit_sidor.txt"
DEFAULT_CACHE = ROOT / ".article-start-ml-cache"
T_FRACTIONS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
FEATURE_NAMES = [
    "column",
    "y_fraction",
    "letter_x_fraction",
    "raw_x_fraction",
    "width_fraction",
    "height_in_lines",
    "item_count",
    "character_count",
    "bold_score",
    "homonym_marker",
    "distance_a",
    "distance_f",
    "distance_t",
    "gap_before",
    "gap_after",
    "mean_confidence",
    "chapter_heading",
    "left_ink",
]


def normalize_word(value: str) -> str:
    """Normalize facit and OCR without treating SAOL's lodstreck as letters."""
    value = value.casefold().replace("|", "")
    return "".join(character for character in value if character.isalnum())


def read_ground_truth(path: Path) -> dict[int, dict[int, list[str]]]:
    result: dict[int, dict[int, list[str]]] = {}
    page: int | None = None
    column: int | None = None
    page_pattern = re.compile(r"^sida\s*:?\s*(\d+)\s*:?\s*$", re.IGNORECASE)
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        page_match = page_pattern.match(line)
        if page_match:
            page = int(page_match.group(1))
            result.setdefault(page, {1: [], 2: []})
            column = None
            continue
        if line.casefold() == "vänster:":
            column = 1
            continue
        if line.casefold() == "höger:":
            column = 2
            continue
        if page is None or column is None:
            raise ValueError(f"{path}:{number}: ord utanför sida/spalt: {raw_line!r}")
        word = re.sub(r"\s*[\[(]homonym\s+\d+[\])]\s*$", "", line, flags=re.IGNORECASE)
        if not normalize_word(word):
            raise ValueError(f"{path}:{number}: tomt normaliserat ord")
        result[page][column].append(word)
    if not result:
        raise ValueError(f"Inget facit hittades i {path}")
    return result


def _configure_debug_module():
    module = debug._load_base_module()

    def deskew(content, observations):
        result, angle = debug._rule_deskew(module, content, observations)
        debug._DESKEWED_CONTENT = result
        return result, angle

    module._deskew_image = deskew
    module._build_lines = lambda observations, width, height: debug._build_lines(
        module, observations, width, height
    )
    return module


def _mean_confidence(items: Iterable[object]) -> float:
    values = []
    for item in items:
        value = getattr(item, "confidence", getattr(item, "conf", None))
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            values.append(number / 100.0 if number > 1.0 else number)
    return statistics.fmean(values) if values else 0.0


def _threshold_for_column(column: int) -> float | None:
    return debug._LEFT_THRESHOLD_X if column == 1 else debug._RIGHT_THRESHOLD_X


def _haf_for_column(column: int) -> tuple[float, float, float] | None:
    return debug._HAF_LEVELS if column == 1 else debug._RIGHT_HAF_LEVELS


def _is_chapter_heading(text: str, height: float, median_height: float) -> bool:
    letters = "".join(character for character in text.strip() if character.isalpha())
    return (
        len(letters) == 2
        and letters[0].isupper()
        and letters[1].islower()
        and letters[0].casefold() == letters[1].casefold()
        and height >= median_height * 1.60
    )


def _slanted_geometry(ordered, median_height: float, fallback):
    """Estimate parallel A/F margins as x = intercept + slope*y."""
    if not ordered:
        return 0.0, 0.0, fallback
    centres = [(float(line.top + line.bottom) / 2, float(line.raw_start_x)) for line in ordered]
    slopes = []
    minimum_dy = median_height * 5
    for index, (first_y, first_x) in enumerate(centres):
        for second_y, second_x in centres[index + 1 :]:
            dy = second_y - first_y
            if dy < minimum_dy:
                continue
            slope = (second_x - first_x) / dy
            if abs(slope) <= 0.08:
                slopes.append(slope)
    slope = statistics.median(slopes) if slopes else 0.0
    anchor_y = statistics.median(y for y, _x in centres)
    adjusted = sorted(x - slope * (y - anchor_y) for y, x in centres)
    tolerance = max(2.5, median_height * 0.20)
    clusters = []
    for value in adjusted:
        if clusters and value - clusters[-1][-1] <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    minimum_count = max(2, round(len(adjusted) * 0.04))
    levels = [
        (float(statistics.median(cluster)), len(cluster))
        for cluster in clusters
        if len(cluster) >= minimum_count
    ]
    pairs = [
        (abs((f - a) - 40.0), -(a_count + f_count), a, f)
        for index, (a, a_count) in enumerate(levels)
        for f, f_count in levels[index + 1 :]
        if 30.0 <= f - a <= 50.0
    ]
    if pairs:
        _error, _count, article_x, continuation_x = min(pairs)
        haf = (article_x - 23.0, article_x, continuation_x)
    else:
        haf = fallback
    return slope, anchor_y, haf


def _rows_from_lines(
    lines, models, width: int, height: int, page: int, gray: Image.Image
) -> list[dict]:
    rows: list[dict] = []
    split = debug._COLUMN_SPLIT_X if debug._COLUMN_SPLIT_X is not None else width / 2
    for column in (1, 2):
        ordered = sorted((line for line in lines if line.column == column), key=lambda row: row.top)
        heights = [max(1.0, float(line.bottom - line.top)) for line in ordered]
        median_height = statistics.median(heights) if heights else 1.0
        model = models[column]
        slope, anchor_y, haf = _slanted_geometry(
            ordered, median_height, _haf_for_column(column)
        )
        article_x = haf[1] if haf else float(model.article_x)
        continuation_x = haf[2] if haf else float(model.continuation_x)
        threshold_x = (article_x + continuation_x) / 2
        column_left = 0.0 if column == 1 else float(split)
        column_width = float(split) if column == 1 else max(1.0, width - float(split))
        for index, line in enumerate(ordered):
            previous = ordered[index - 1] if index else None
            following = ordered[index + 1] if index + 1 < len(ordered) else None
            text_items = sorted(line.items, key=lambda item: item.left)
            first_tokens = [normalize_word(item.text) for item in text_items if normalize_word(item.text)]
            match_text = " ".join(first_tokens[:5])
            line_y = float(line.top + line.bottom) / 2
            correction = slope * (line_y - anchor_y)
            letter_x = float(line.letter_start_x) - correction
            raw_x = float(line.raw_start_x) - correction
            line_height = max(1.0, float(line.bottom - line.top))
            chapter_heading = _is_chapter_heading(
                line.text, line_height, median_height
            )
            # En tryckt bokstav kan ligga precis över den beräknade T-linjen
            # genom avrundning och kvarvarande snedhet. Tillåt ungefär en pixel.
            threshold_tolerance = max(1.0, median_height * 0.03)
            ocr_reaches_left = any(
                normalize_word(item.text)
                and float(item.left) - correction <= threshold_x + threshold_tolerance
                for item in line.items
            )
            left_ink = 0
            pixel_reaches_left = False
            if (
                not chapter_heading
                and haf is not None
                and line_height <= median_height * 1.60
            ):
                x0 = max(0, round(haf[0] + correction - median_height * 0.30))
                x1 = min(
                    gray.width,
                    round(threshold_x + correction + threshold_tolerance),
                )
                y0 = max(0, round(line.top))
                y1 = min(gray.height, round(line.bottom))
                pixels = gray.load()
                left_ink = sum(
                    1
                    for y in range(y0, y1)
                    for x in range(x0, x1)
                    if pixels[x, y] < 160
                )
                pixel_reaches_left = left_ink >= max(
                    6, round(median_height * 0.50)
                )
            # Tesseracts ordboxar kan innehålla luft och nå över T trots att
            # inget tryckt tecken gör det (t.ex. raden "som" under amfibie).
            # Pixlarna är därför huvudsignalen. Ett mycket smalt undantag för
            # en bokstavsstart precis efter T hanterar avrundningsfallet
            # "abstinens" utan att acceptera boxar som börjar på fel sida.
            near_threshold_ocr = (
                0.0 <= letter_x - threshold_x <= threshold_tolerance
            )
            baseline = not chapter_heading and (
                pixel_reaches_left or near_threshold_ocr
            )
            t_candidates = {}
            if not chapter_heading and haf is not None and line_height <= median_height * 1.60:
                pixels = gray.load()
                sweep_x0 = max(
                    0, round(haf[0] + correction - median_height * 0.30)
                )
                sweep_y0 = max(0, round(line.top))
                sweep_y1 = min(gray.height, round(line.bottom))
                for fraction in T_FRACTIONS:
                    candidate_t = article_x + fraction * (
                        continuation_x - article_x
                    )
                    candidate_x1 = min(
                        gray.width,
                        round(candidate_t + correction + threshold_tolerance),
                    )
                    candidate_ink = sum(
                        1
                        for y in range(sweep_y0, sweep_y1)
                        for x in range(sweep_x0, candidate_x1)
                        if pixels[x, y] < 160
                    )
                    candidate_pixel = candidate_ink >= max(
                        6, round(median_height * 0.50)
                    )
                    candidate_near_ocr = (
                        0.0 <= letter_x - candidate_t <= threshold_tolerance
                    )
                    t_candidates[f"{fraction:.2f}"] = bool(
                        candidate_pixel or candidate_near_ocr
                    )
            rows.append(
                {
                    "page": page,
                    "column": column,
                    "top": float(line.top),
                    "bottom": float(line.bottom),
                    "left": float(line.left),
                    "right": float(line.right),
                    "text": line.text,
                    "match_text": match_text,
                    "baseline": bool(baseline),
                    "chapter_heading": chapter_heading,
                    "ocr_reaches_left": ocr_reaches_left,
                    "pixel_reaches_left": pixel_reaches_left,
                    "left_ink": left_ink,
                    "t_candidates": t_candidates,
                    "margin_slope": slope,
                    "geometry": [float(value) for value in haf] if haf else None,
                    "features": [
                        float(column - 1),
                        float(line.top) / max(1.0, height),
                        (letter_x - column_left) / column_width,
                        (raw_x - column_left) / column_width,
                        max(1.0, float(line.right - line.left)) / column_width,
                        max(1.0, float(line.bottom - line.top)) / median_height,
                        float(len(line.items)),
                        float(len(line.text)),
                        float(line.bold_score),
                        float(bool(line.has_homonym_marker)),
                        (letter_x - article_x) / median_height,
                        (letter_x - continuation_x) / median_height,
                        (letter_x - threshold_x) / median_height,
                        (
                            (float(line.top) - float(previous.bottom)) / median_height
                            if previous is not None
                            else 2.0
                        ),
                        (
                            (float(following.top) - float(line.bottom)) / median_height
                            if following is not None
                            else 2.0
                        ),
                        _mean_confidence(line.items),
                        float(chapter_heading),
                        left_ink / max(1.0, median_height * median_height),
                    ],
                }
            )
    return rows


def extract_page(page: int, cache_dir: Path, refresh: bool = False) -> list[dict]:
    cache_file = cache_dir / f"page-{page:04d}-columns-v12.json"
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text(encoding="utf-8"))

    module = _configure_debug_module()
    source_url, image_url = module.page_urls(page)
    headers = {"User-Agent": "saol-tools/article-start-ml"}
    source_response = module.httpx.get(
        source_url, timeout=60.0, follow_redirects=True, headers=headers
    )
    source_response.raise_for_status()
    image_response = module.httpx.get(
        module.ocr_image_url(image_url),
        timeout=60.0,
        follow_redirects=True,
        headers=headers,
    )
    image_response.raise_for_status()
    initial = module.extract_observations(image_response.content)
    deskewed, angle = module._deskew_image(image_response.content, initial)
    # Tesseract --psm 6 occasionally joins both columns and can even leak TSV
    # fields into a token. OCR each deskewed column independently instead.
    observations = []
    with Image.open(io.BytesIO(deskewed)) as image:
        width, height = image.size
        gray = image.convert("L")
        split = round(
            debug._COLUMN_SPLIT_X
            if debug._COLUMN_SPLIT_X is not None
            else width / 2
        )
        for left, right in ((0, split), (split, width)):
            crop = image.crop((left, 0, right, height))
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG")
            observations.extend(
                replace(item, left=item.left + left)
                for item in module.extract_observations(buffer.getvalue())
            )
    # The review UI deliberately starts below the detected physical rule. For
    # training data, losing a real row is worse than retaining harmless header
    # debris. Never let a mistaken rule detection suppress the top 4%+ of a page.
    safe_rule_y = height * 0.04
    debug._BODY_TOP_Y = min(
        debug._BODY_TOP_Y if debug._BODY_TOP_Y is not None else safe_rule_y,
        safe_rule_y,
    )
    # Context reconciliation assumes full-width Tesseract lines. Applying it
    # after column OCR shifts words between unrelated rows and may copy raw TSV
    # payload into tokens, so the ML track deliberately uses clean OCR boxes.
    lines, models = module._build_lines(observations, width, height)
    rows = _rows_from_lines(lines, models, width, height, page, gray)
    for row in rows:
        row["deskew_degrees"] = float(angle)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def _match_score(expected: str, row: dict) -> float:
    wanted = normalize_word(expected)
    tokens = row["match_text"].split()
    # Ett facitord hör hemma i början av den tryckta raden. Att jämföra mot
    # varje senare OCR-token gjorde t.ex. fortsättningsordet "abstraktion" till
    # en falsk match för uppslagsordet "abstrakt".
    candidates = [tokens[0]] if tokens else []
    for length in range(2, min(5, len(tokens)) + 1):
        candidates.append("".join(tokens[:length]))
    if not candidates:
        return 0.0
    best = 0.0
    for candidate in candidates:
        if candidate == wanted:
            return 1.0
        ratio = difflib.SequenceMatcher(None, wanted, candidate).ratio()
        if len(wanted) >= 4 and candidate.startswith(wanted):
            ratio = max(ratio, 0.92)
        best = max(best, ratio)
    return best


def _alignment_score(expected: str, row: dict) -> float:
    """Combine text matching with independent evidence at the printed margin."""
    score = _match_score(expected, row)
    if row.get("pixel_reaches_left"):
        score += 0.75
    elif row.get("ocr_reaches_left"):
        score += 0.15
    return score


def align_truth(words: list[str], rows: list[dict]) -> list[tuple[int, str, float]]:
    """Monotonically align every facit word to one OCR row."""
    expected_count, row_count = len(words), len(rows)
    negative = -1e9
    scores = [[negative] * (row_count + 1) for _ in range(expected_count + 1)]
    choice = [[""] * (row_count + 1) for _ in range(expected_count + 1)]
    for column in range(row_count + 1):
        scores[0][column] = 0.0
    for expected_index in range(1, expected_count + 1):
        for row_index in range(1, row_count + 1):
            skip = scores[expected_index][row_index - 1]
            match = (
                scores[expected_index - 1][row_index - 1]
                + _alignment_score(words[expected_index - 1], rows[row_index - 1])
            )
            if match > skip:
                scores[expected_index][row_index] = match
                choice[expected_index][row_index] = "match"
            else:
                scores[expected_index][row_index] = skip
                choice[expected_index][row_index] = "skip"
    if scores[expected_count][row_count] <= negative / 2:
        raise ValueError("Facit innehåller fler ord än det finns OCR-rader")
    result = []
    expected_index, row_index = expected_count, row_count
    while expected_index:
        if row_index <= 0:
            raise ValueError("Facit kunde inte linjeras mot OCR-raderna")
        if choice[expected_index][row_index] == "match":
            word = words[expected_index - 1]
            result.append((row_index - 1, word, _match_score(word, rows[row_index - 1])))
            expected_index -= 1
            row_index -= 1
        else:
            row_index -= 1
    return list(reversed(result))


def label_rows(
    all_rows: list[dict],
    truth: dict[int, dict[int, list[str]]],
    minimum_score: float = 0.72,
) -> tuple[list[dict], list[dict]]:
    diagnostics = []
    for row in all_rows:
        row["label"] = 0
        row["facit_word"] = ""
        row["match_score"] = 0.0
        row["usable"] = True
    for page, columns in truth.items():
        for column, words in columns.items():
            selected = [
                row for row in all_rows if row["page"] == page and row["column"] == column
            ]
            matches = align_truth(words, selected)
            weak_indices = [
                index for index, (_row, _word, score) in enumerate(matches)
                if score < minimum_score
            ]
            mean_score = statistics.fmean(score for _row, _word, score in matches)
            weak_have_left_ink = all(
                selected[matches[index][0]].get("pixel_reaches_left", False)
                for index in weak_indices
            )
            accepted = not weak_indices or (
                mean_score >= 0.90
                and (
                    (
                        len(weak_indices) == 1
                        and 0 < weak_indices[0] < len(matches) - 1
                    )
                    or weak_have_left_ink
                )
            )
            if not accepted:
                for row in selected:
                    row["usable"] = False
            for index, word, score in matches:
                if accepted:
                    selected[index]["label"] = 1
                selected[index]["facit_word"] = word
                selected[index]["match_score"] = score
                diagnostics.append(
                    {
                        "page": page,
                        "column": column,
                        "word": word,
                        "ocr": selected[index]["text"],
                        "score": score,
                        "accepted": accepted,
                    }
                )
    return all_rows, diagnostics


def _metrics(labels, predictions) -> dict[str, float]:
    tp = sum(1 for label, pred in zip(labels, predictions) if label and pred)
    fp = sum(1 for label, pred in zip(labels, predictions) if not label and pred)
    fn = sum(1 for label, pred in zip(labels, predictions) if label and not pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def compare_t_positions(rows: list[dict]) -> dict[str, dict[str, float]]:
    usable = [row for row in rows if row.get("usable", True)]
    labels = [bool(row["label"]) for row in usable]
    return {
        fraction: _metrics(
            labels,
            [
                bool(row.get("t_candidates", {}).get(fraction, row["baseline"]))
                for row in usable
            ],
        )
        for fraction in (f"{value:.2f}" for value in T_FRACTIONS)
    }


def compare_models(rows: list[dict]) -> tuple[dict[str, dict], list[dict]]:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = [row for row in rows if row.get("usable", True)]
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    features = np.asarray([row["features"] for row in rows], dtype=float)
    pages = sorted({row["page"] for row in rows})
    if len(pages) < 2:
        raise ValueError("För få fullt matchade facitsidor för sidvis korsvalidering")
    predictions = {
        "T-regel": np.asarray([row["baseline"] for row in rows], dtype=bool),
        "Logistisk regression": np.zeros(len(rows), dtype=bool),
        "Gradient boosting": np.zeros(len(rows), dtype=bool),
    }
    probabilities = {
        "Logistisk regression": np.zeros(len(rows), dtype=float),
        "Gradient boosting": np.zeros(len(rows), dtype=float),
    }
    for page in pages:
        train = np.asarray([row["page"] != page for row in rows])
        test = ~train
        positive = max(1, int(labels[train].sum()))
        negative = max(1, int(train.sum() - positive))
        sample_weight = np.where(labels[train] == 1, negative / positive, 1.0)
        logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=1),
        )
        boosting = HistGradientBoostingClassifier(
            max_iter=160,
            max_depth=3,
            min_samples_leaf=5,
            learning_rate=0.06,
            l2_regularization=1.0,
            random_state=1,
        )
        logistic.fit(features[train], labels[train])
        boosting.fit(features[train], labels[train], sample_weight=sample_weight)
        for name, model in (
            ("Logistisk regression", logistic),
            ("Gradient boosting", boosting),
        ):
            probability = model.predict_proba(features[test])[:, 1]
            probabilities[name][test] = probability
            predictions[name][test] = probability >= 0.5
    results = {
        name: _metrics(labels.tolist(), values.tolist())
        for name, values in predictions.items()
    }
    details = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["predictions"] = {
            name: bool(values[index]) for name, values in predictions.items()
        }
        item["probabilities"] = {
            name: float(values[index]) for name, values in probabilities.items()
        }
        details.append(item)
    return results, details


def _report_html(results: dict[str, dict], rows: list[dict], alignment: list[dict]) -> str:
    metric_rows = "".join(
        "<tr><th>%s</th><td>%.3f</td><td>%.3f</td><td>%.3f</td>"
        "<td>%d</td><td>%d</td><td>%d</td></tr>"
        % (
            html.escape(name),
            values["precision"],
            values["recall"],
            values["f1"],
            values["tp"],
            values["fp"],
            values["fn"],
        )
        for name, values in results.items()
    )
    error_rows = []
    for row in rows:
        disagreements = [
            name
            for name, prediction in row["predictions"].items()
            if prediction != bool(row["label"])
        ]
        if not disagreements:
            continue
        probability = row["probabilities"]
        error_rows.append(
            "<tr><td>%d</td><td>%d</td><td>%.0f</td><td>%s</td><td>%s</td>"
            "<td>%s</td><td>%.3f</td><td>%.3f</td></tr>"
            % (
                row["page"],
                row["column"],
                row["top"],
                html.escape(row["facit_word"] or "—"),
                html.escape(row["text"]),
                html.escape(", ".join(disagreements)),
                probability.get("Logistisk regression", math.nan),
                probability.get("Gradient boosting", math.nan),
            )
        )
    weak = [item for item in alignment if not item["accepted"]]
    weak_rows = "".join(
        "<tr><td>%d</td><td>%d</td><td>%s</td><td>%s</td><td>%.3f</td></tr>"
        % (
            item["page"],
            item["column"],
            html.escape(item["word"]),
            html.escape(item["ocr"]),
            item["score"],
        )
        for item in weak
    )
    return f"""<!doctype html>
<html lang="sv"><head><meta charset="utf-8"><title>ML – artikelstarter</title>
<style>body{{font:15px system-ui;margin:24px;max-width:1500px}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}}thead th{{background:#eee;position:sticky;top:0}}code{{white-space:pre-wrap}}</style></head>
<body><h1>Artikelstarter – sidvis korsvalidering</h1>
<p>Varje testresultat kommer från en modell tränad på de övriga sidorna.</p>
<h2>Resultat</h2><table><thead><tr><th>Metod</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP</th><th>FP</th><th>FN</th></tr></thead><tbody>{metric_rows}</tbody></table>
<h2>Felklassade rader</h2><table><thead><tr><th>Sida</th><th>Spalt</th><th>y</th><th>Facit</th><th>OCR-rad</th><th>Fel metod</th><th>P log</th><th>P boost</th></tr></thead><tbody>{''.join(error_rows)}</tbody></table>
<h2>Osäkra facit–OCR-matchningar</h2><table><thead><tr><th>Sida</th><th>Spalt</th><th>Facit</th><th>OCR-rad</th><th>Likhet</th></tr></thead><tbody>{weak_rows or '<tr><td colspan="5">Inga under 0,72.</td></tr>'}</tbody></table>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--pages", nargs="*", type=int)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--features-json", type=Path, default=Path("article-start-features.json"))
    parser.add_argument("--report", type=Path, default=Path("article-start-ml-report.html"))
    args = parser.parse_args()

    truth = read_ground_truth(args.ground_truth)
    pages = args.pages or sorted(truth)
    missing = sorted(set(pages) - set(truth))
    if missing:
        parser.error(f"facit saknas för sidor: {missing}")
    selected_truth = {page: truth[page] for page in pages}
    rows = []
    for page in pages:
        print(f"OCR sida {page} ...", flush=True)
        rows.extend(extract_page(page, args.cache_dir, args.refresh))
    rows, alignment = label_rows(rows, selected_truth)
    weak = [item for item in alignment if not item["accepted"]]
    usable = [row for row in rows if row["usable"]]
    print(f"Rader: {len(rows)}, användbara efter facitmatchning: {len(usable)}")
    print(f"Artikelstarter i användbart facit: {sum(row['label'] for row in usable)}")
    print(f"Osäkra facit–OCR-matchningar (<0,72): {len(weak)}")
    t_positions = compare_t_positions(rows)
    print("T-svep (0=A, 1=F):")
    for fraction, values in t_positions.items():
        print(
            f"  T={fraction}: precision={values['precision']:.3f} "
            f"recall={values['recall']:.3f} F1={values['f1']:.3f} "
            f"FP={values['fp']} FN={values['fn']}"
        )
    results, details = compare_models(rows)
    for name, values in results.items():
        print(
            f"{name}: precision={values['precision']:.3f} "
            f"recall={values['recall']:.3f} F1={values['f1']:.3f} "
            f"FP={values['fp']} FN={values['fn']}"
        )
    args.features_json.write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "rows": details,
                "all_rows": rows,
                "alignment": alignment,
                "t_positions": t_positions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    args.report.write_text(_report_html(results, details, alignment), encoding="utf-8")
    print(f"Data: {args.features_json.resolve()}")
    print(f"Rapport: {args.report.resolve()}")


if __name__ == "__main__":
    main()
