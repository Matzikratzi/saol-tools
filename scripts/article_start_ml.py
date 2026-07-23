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
from pathlib import Path
from typing import Iterable

from PIL import Image

from scripts import debug_runeberg_ocr as debug


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRUTH = ROOT / "data" / "facit_sidor.txt"
DEFAULT_CACHE = ROOT / ".article-start-ml-cache"
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


def _rows_from_lines(lines, models, width: int, height: int, page: int) -> list[dict]:
    rows: list[dict] = []
    split = debug._COLUMN_SPLIT_X if debug._COLUMN_SPLIT_X is not None else width / 2
    for column in (1, 2):
        ordered = sorted((line for line in lines if line.column == column), key=lambda row: row.top)
        heights = [max(1.0, float(line.bottom - line.top)) for line in ordered]
        median_height = statistics.median(heights) if heights else 1.0
        model = models[column]
        haf = _haf_for_column(column)
        article_x = haf[1] if haf else float(model.article_x)
        continuation_x = haf[2] if haf else float(model.continuation_x)
        threshold_x = _threshold_for_column(column)
        if threshold_x is None:
            threshold_x = (article_x + continuation_x) / 2
        column_left = 0.0 if column == 1 else float(split)
        column_width = float(split) if column == 1 else max(1.0, width - float(split))
        for index, line in enumerate(ordered):
            previous = ordered[index - 1] if index else None
            following = ordered[index + 1] if index + 1 < len(ordered) else None
            text_items = sorted(line.items, key=lambda item: item.left)
            first_tokens = [normalize_word(item.text) for item in text_items if normalize_word(item.text)]
            match_text = " ".join(first_tokens[:5])
            baseline = any(
                normalize_word(item.text) and float(item.left) < threshold_x
                for item in line.items
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
                    "features": [
                        float(column - 1),
                        float(line.top) / max(1.0, height),
                        (float(line.letter_start_x) - column_left) / column_width,
                        (float(line.raw_start_x) - column_left) / column_width,
                        max(1.0, float(line.right - line.left)) / column_width,
                        max(1.0, float(line.bottom - line.top)) / median_height,
                        float(len(line.items)),
                        float(len(line.text)),
                        float(line.bold_score),
                        float(bool(line.has_homonym_marker)),
                        (float(line.letter_start_x) - article_x) / median_height,
                        (float(line.letter_start_x) - continuation_x) / median_height,
                        (float(line.letter_start_x) - threshold_x) / median_height,
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
                    ],
                }
            )
    return rows


def extract_page(page: int, cache_dir: Path, refresh: bool = False) -> list[dict]:
    cache_file = cache_dir / f"page-{page:04d}.json"
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
    observations = (
        module.extract_observations(deskewed)
        if deskewed is not image_response.content
        else initial
    )
    corrected = module.reconcile_contextual_observations(observations, source_response.text)
    with Image.open(io.BytesIO(deskewed)) as image:
        width, height = image.size
    lines, models = module._build_lines(corrected, width, height)
    rows = _rows_from_lines(lines, models, width, height, page)
    for row in rows:
        row["deskew_degrees"] = float(angle)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def _match_score(expected: str, row: dict) -> float:
    wanted = normalize_word(expected)
    tokens = row["match_text"].split()
    candidates = tokens[:]
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
                + _match_score(words[expected_index - 1], rows[row_index - 1])
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
    all_rows: list[dict], truth: dict[int, dict[int, list[str]]]
) -> tuple[list[dict], list[dict]]:
    diagnostics = []
    for row in all_rows:
        row["label"] = 0
        row["facit_word"] = ""
        row["match_score"] = 0.0
    for page, columns in truth.items():
        for column, words in columns.items():
            selected = [
                row for row in all_rows if row["page"] == page and row["column"] == column
            ]
            for index, word, score in align_truth(words, selected):
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


def compare_models(rows: list[dict]) -> tuple[dict[str, dict], list[dict]]:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray([row["label"] for row in rows], dtype=int)
    features = np.asarray([row["features"] for row in rows], dtype=float)
    pages = sorted({row["page"] for row in rows})
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
    weak = [item for item in alignment if item["score"] < 0.72]
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
    weak = [item for item in alignment if item["score"] < 0.72]
    print(f"Rader: {len(rows)}, artikelstarter i facit: {sum(row['label'] for row in rows)}")
    print(f"Osäkra facit–OCR-matchningar (<0,72): {len(weak)}")
    results, details = compare_models(rows)
    for name, values in results.items():
        print(
            f"{name}: precision={values['precision']:.3f} "
            f"recall={values['recall']:.3f} F1={values['f1']:.3f} "
            f"FP={values['fp']} FN={values['fn']}"
        )
    args.features_json.write_text(
        json.dumps(
            {"feature_names": FEATURE_NAMES, "rows": details, "alignment": alignment},
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
