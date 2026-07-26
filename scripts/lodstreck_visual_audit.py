from __future__ import annotations

"""Measure how well printed glyph geometry identifies SAOL stem boundaries.

This is an audit only: it never changes headwords or lemma extraction.

Usage:
    PYTHONPATH=. python3 scripts/lodstreck_visual_audit.py
"""

import argparse
import json
import math
import statistics
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from scripts.article_start_ml import DEFAULT_CACHE


ROOT = Path(__file__).resolve().parents[1]
NEGATIVE_GLYPHS = frozenset("ilj1")


@dataclass(frozen=True)
class Component:
    left: int
    top: int
    right: int
    bottom: int
    area: int
    has_dot: bool = False

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2


def otsu_threshold(image: Image.Image) -> int:
    histogram = image.convert("L").histogram()
    count = sum(histogram)
    weighted_sum = sum(index * amount for index, amount in enumerate(histogram))
    background_weight = 0
    background_sum = 0
    best_variance = -1.0
    best = 160
    for level, amount in enumerate(histogram):
        background_weight += amount
        if not background_weight:
            continue
        foreground_weight = count - background_weight
        if not foreground_weight:
            break
        background_sum += level * amount
        background_mean = background_sum / background_weight
        foreground_mean = (
            weighted_sum - background_sum
        ) / foreground_weight
        variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )
        if variance > best_variance:
            best_variance = variance
            best = level
    return min(best, 220)


def connected_components(image: Image.Image) -> list[Component]:
    gray = image.convert("L")
    threshold = otsu_threshold(gray)
    width, height = gray.size
    pixels = gray.load()
    ink = {
        (x, y)
        for y in range(height)
        for x in range(width)
        if pixels[x, y] <= threshold
    }
    result: list[Component] = []
    while ink:
        start = ink.pop()
        queue = deque([start])
        points = [start]
        while queue:
            x, y = queue.popleft()
            for neighbour in (
                (x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
                (x - 1, y),                     (x + 1, y),
                (x - 1, y + 1), (x, y + 1), (x + 1, y + 1),
            ):
                if neighbour in ink:
                    ink.remove(neighbour)
                    queue.append(neighbour)
                    points.append(neighbour)
        if len(points) < 3:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        result.append(
            Component(
                min(xs), min(ys), max(xs) + 1, max(ys) + 1, len(points)
            )
        )
    return sorted(result, key=lambda component: component.left)


def glyph_components(image: Image.Image) -> list[Component]:
    components = connected_components(image)
    if not components:
        return []
    substantial = [
        component
        for component in components
        if component.height >= image.height * 0.28
        and component.area >= max(5, image.height // 3)
    ]
    small = [component for component in components if component not in substantial]
    result = []
    for component in substantial:
        has_dot = any(
            dot.bottom <= component.top + max(2, image.height * 0.12)
            and dot.center_x >= component.left - 1
            and dot.center_x <= component.right + 1
            and dot.area >= 2
            for dot in small
        )
        result.append(
            Component(
                component.left,
                component.top,
                component.right,
                component.bottom,
                component.area,
                has_dot,
            )
        )
    return result


def visual_scores(components: list[Component]) -> list[float]:
    if len(components) < 2:
        return [float("-inf")] * len(components)
    median_width = max(1.0, statistics.median(c.width for c in components))
    median_height = max(1.0, statistics.median(c.height for c in components))
    median_bottom = statistics.median(c.bottom for c in components)
    scores = []
    for component in components:
        thinness = (median_width - component.width) / median_width
        depth = (component.bottom - median_bottom) / median_height
        dot_penalty = 1.25 if component.has_dot else 0.0
        scores.append(1.35 * thinness + 2.4 * depth - dot_penalty)
    return scores


def normalized_characters(value: str) -> str:
    return "".join(
        character.casefold()
        for character in value
        if character.isalnum()
    )


def boundary_fraction(raw: str, structured: str) -> float | None:
    if "|" not in structured:
        return None
    prefix = normalized_characters(structured.split("|", 1)[0])
    plain = normalized_characters(raw.replace("|", ""))
    if not plain:
        return None
    if prefix and plain.startswith(prefix):
        boundary = len(prefix)
    else:
        boundary = min(len(prefix), len(plain))
    return max(0.0, min(1.0, boundary / len(plain)))


def nearest_component(
    components: list[Component], fraction: float, width: int
) -> int | None:
    if not components:
        return None
    expected_x = fraction * width
    index = min(
        range(len(components)),
        key=lambda candidate: abs(components[candidate].center_x - expected_x),
    )
    median_width = statistics.median(c.width for c in components)
    if abs(components[index].center_x - expected_x) > max(5.0, 1.8 * median_width):
        return None
    return index


def source_box(item: dict) -> tuple[int, int, int, int] | None:
    required = ("source_left", "source_top", "source_right", "source_bottom")
    if not all(key in item for key in required):
        return None
    left, top, right, bottom = (
        int(round(float(item[key]))) for key in required
    )
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def structured_form(item: dict) -> str:
    candidates = (
        item.get("runeberg_stem_headword", ""),
        item.get("stem_headword", ""),
        item.get("raw", ""),
        item.get("stem_lemma", ""),
    )
    return next((value for value in candidates if "|" in value), "")


def instances(heads: dict, lemmas: dict) -> list[dict]:
    result = []
    seen = set()
    for item in heads.get("headwords", []) + lemmas.get("candidates", []):
        structured = structured_form(item)
        box = source_box(item)
        if not structured or box is None:
            continue
        key = (int(item["page"]), *box, structured)
        if key in seen:
            continue
        seen.add(key)
        result.append({**item, "_structured": structured, "_box": box})
    return result


def negative_fractions(raw: str) -> list[tuple[str, float]]:
    plain = normalized_characters(raw.replace("|", ""))
    return [
        (character, (index + 0.5) / len(plain))
        for index, character in enumerate(plain)
        if character in NEGATIVE_GLYPHS
    ] if plain else []


def evaluate(
    heads: dict, lemmas: dict, cache_dir: Path
) -> tuple[list[dict], list[dict], list[str]]:
    positives = []
    negatives = []
    skipped = []
    page_images: dict[int, Image.Image] = {}
    try:
        for item in instances(heads, lemmas):
            page = int(item["page"])
            path = cache_dir / f"page-{page:04d}-deskewed.png"
            if not path.exists():
                skipped.append(f"sida {page}: bild saknas ({path})")
                continue
            if page not in page_images:
                page_images[page] = Image.open(path).convert("L")
            left, top, right, bottom = item["_box"]
            padding = 3
            crop = page_images[page].crop(
                (
                    max(0, left - padding),
                    max(0, top - padding),
                    min(page_images[page].width, right + padding),
                    min(page_images[page].height, bottom + padding),
                )
            )
            components = glyph_components(crop)
            scores = visual_scores(components)
            fraction = boundary_fraction(
                str(item.get("raw_headword") or item.get("raw") or ""),
                item["_structured"],
            )
            if fraction is None:
                continue
            positive_index = nearest_component(components, fraction, crop.width)
            if positive_index is None:
                skipped.append(
                    f"sida {page} artikel {item['article_number']}: "
                    f"kunde inte lokalisera {item['_structured']}"
                )
                continue
            positives.append(
                {
                    "page": page,
                    "article": int(item["article_number"]),
                    "word": item.get("headword") or item.get("lemma"),
                    "structured": item["_structured"],
                    "score": scores[positive_index],
                }
            )
            used_negative_components = set()
            raw = str(item.get("raw_headword") or item.get("raw") or "")
            for glyph, negative_fraction in negative_fractions(raw):
                index = nearest_component(
                    components, negative_fraction, crop.width
                )
                if index is None or index == positive_index:
                    continue
                key = (page, item["_box"], index)
                if key in used_negative_components:
                    continue
                used_negative_components.add(key)
                negatives.append(
                    {
                        "page": page,
                        "article": int(item["article_number"]),
                        "word": item.get("headword") or item.get("lemma"),
                        "glyph": glyph,
                        "score": scores[index],
                    }
                )
    finally:
        for image in page_images.values():
            image.close()
    return positives, negatives, sorted(set(skipped))


def percentage(part: int, whole: int) -> str:
    return "—" if not whole else f"{100 * part / whole:.1f} %"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headwords", type=Path, default=Path("headword-review.json")
    )
    parser.add_argument(
        "--lemmas", type=Path, default=Path("lemma-review.json")
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--json", type=Path, default=Path("lodstreck-visual-audit.json")
    )
    args = parser.parse_args()
    heads = json.loads(args.headwords.read_text(encoding="utf-8"))
    lemmas = json.loads(args.lemmas.read_text(encoding="utf-8"))
    positives, negatives, skipped = evaluate(heads, lemmas, args.cache_dir)
    thresholds = []
    for threshold in (-0.25, 0.0, 0.25, 0.5, 0.75, 1.0):
        found = sum(item["score"] >= threshold for item in positives)
        false = sum(item["score"] >= threshold for item in negatives)
        thresholds.append(
            {
                "threshold": threshold,
                "found": found,
                "known": len(positives),
                "recall": found / len(positives) if positives else None,
                "false_positives": false,
                "negative_examples": len(negatives),
                "false_positive_rate": (
                    false / len(negatives) if negatives else None
                ),
            }
        )
    output = {
        "known_boundaries": positives,
        "negative_glyphs": negatives,
        "thresholds": thresholds,
        "skipped": skipped,
    }
    args.json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Kända lodstreck som kunde mätas: {len(positives)}")
    print(f"Kontrolltecken (i/l/j/1): {len(negatives)}")
    print("\nTRÖSKEL   HITTADE   ANDEL      FALSKA   FALSK ANDEL")
    for row in thresholds:
        print(
            f"{row['threshold']:>7.2f}"
            f"{row['found']:>10}/{row['known']:<5}"
            f"{percentage(row['found'], row['known']):>10}"
            f"{row['false_positives']:>10}/{row['negative_examples']:<5}"
            f"{percentage(row['false_positives'], row['negative_examples']):>12}"
        )
    if skipped:
        print(f"\nÖverhoppade mätningar: {len(skipped)}")
        for message in skipped[:10]:
            print(f"  {message}")
        if len(skipped) > 10:
            print(f"  … och {len(skipped) - 10} till")
    print(f"\nDetaljer: {args.json.resolve()}")


if __name__ == "__main__":
    main()
