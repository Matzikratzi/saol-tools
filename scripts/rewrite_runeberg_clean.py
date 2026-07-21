from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNEberg = ROOT / "app" / "runeberg.py"
DEBUG = ROOT / "scripts" / "debug_runeberg_ocr.py"

RUNEberg_SOURCE = r'''from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from PIL import Image, ImageOps

from .classifier import WordObservation

BASE_URL = "https://runeberg.org/saol/11-6"
STEM_BOUNDARY_MARKS = "|¦│ǀ"
STEM_BOUNDARY_TRANSLATION = str.maketrans("", "", STEM_BOUNDARY_MARKS)
TOKEN_STRIP = ".,:;!?()[]{}<>\"“”"


@dataclass(frozen=True)
class ImportedPage:
    page_number: int
    source_url: str
    image_url: str
    observations: list[WordObservation]


def page_id(page_number: int) -> str:
    if page_number < 1 or page_number > 9999:
        raise ValueError("Sidnumret måste vara mellan 1 och 9999")
    return f"{page_number:04d}"


def page_urls(page_number: int) -> tuple[str, str]:
    identifier = page_id(page_number)
    return (
        f"{BASE_URL}/{identifier}.html",
        f"https://runeberg.org/img/saol/11-6/{identifier}.3.png",
    )


def ocr_image_url(image_url: str) -> str:
    return image_url.replace(".3.png", ".1.tif")


def _normalize_line_text(text: str) -> str:
    return re.sub(r"[^a-zåäö]+", " ", text.casefold()).strip()


def _instruction_marker_count(text: str) -> int:
    normalized = _normalize_line_text(text)
    words = set(normalized.split())
    hits = 0
    hits += bool(words & {"här", "har"})
    hits += "nedan" in words
    hits += any(word.startswith(("maskintolk", "misstolk")) for word in words)
    hits += "texten" in words or "text" in words
    hits += "från" in words or "fran" in words
    hits += any(word.startswith("faksimil") for word in words)
    hits += any(word.startswith("korrekturl") for word in words)
    hits += "sidan" in words
    hits += "ovan" in words
    hits += sum(token in words for token in ("below", "raw", "ocr", "scanned", "image", "proofread", "page"))
    hits += int("project runeberg" in normalized)
    return hits


def is_runeberg_instruction_line(text: str) -> bool:
    normalized = _normalize_line_text(text)
    words = set(normalized.split())
    swedish_hits = 0
    swedish_hits += bool(words & {"här", "har"})
    swedish_hits += "nedan" in words
    swedish_hits += any(word.startswith(("maskintolk", "misstolk")) for word in words)
    swedish_hits += "texten" in words or "text" in words
    swedish_hits += "från" in words or "fran" in words
    swedish_hits += any(word.startswith("faksimil") for word in words)
    swedish_hits += any(word.startswith("korrekturl") for word in words)
    swedish_hits += "sidan" in words
    swedish_hits += "ovan" in words
    english_hits = sum(token in words for token in ("below", "raw", "ocr", "text", "scanned", "image", "proofread", "page"))
    return swedish_hits >= 3 or english_hits >= 4 or "project runeberg" in normalized


def instruction_line_keys(
    ordered_lines: list[tuple[tuple[str, str, str, str], int, str]],
) -> set[tuple[str, str, str, str]]:
    excluded: set[tuple[str, str, str, str]] = set()
    lines = sorted(ordered_lines, key=lambda item: item[1])
    cluster: list[tuple[tuple[str, str, str, str], int, str]] = []

    def flush() -> None:
        if not cluster:
            return
        combined = " ".join(text for _, _, text in cluster)
        if is_runeberg_instruction_line(combined):
            excluded.update(key for key, _, _ in cluster)
        cluster.clear()

    for line in lines:
        if _instruction_marker_count(line[2]) > 0:
            cluster.append(line)
        else:
            flush()
    flush()
    return excluded


def _run_tesseract_tsv(image_path: Path) -> str:
    executable = shutil.which("tesseract")
    if executable is None:
        raise RuntimeError("Tesseract saknas. Installera med: brew install tesseract tesseract-lang")
    process = subprocess.run(
        [executable, str(image_path), "stdout", "-l", "swe", "--psm", "6", "tsv"],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or "okänt Tesseract-fel"
        raise RuntimeError(f"Tesseract misslyckades: {detail}")
    return process.stdout


def _ink_density(gray: Image.Image, left: int, top: int, width: int, height: int) -> float:
    margin = 1
    box = (
        max(0, left - margin),
        max(0, top - margin),
        min(gray.width, left + width + margin),
        min(gray.height, top + height + margin),
    )
    crop = gray.crop(box)
    if crop.width == 0 or crop.height == 0:
        return 0.0
    pixels = list(crop.getdata())
    return sum((255 - value) / 255.0 for value in pixels) / len(pixels)


def _is_printed_page_number(text: str, top: int, height: int, image_height: int) -> bool:
    token = text.strip().strip(".,:;()[]")
    if not token.isdigit() or len(token) > 4:
        return False
    center_y = top + height / 2
    return center_y < image_height * 0.10 or center_y > image_height * 0.90


def _runeberg_fragment(html: str) -> str:
    start_match = re.search(r"<!--\s*mode=[^>]*-->", html, flags=re.IGNORECASE)
    if not start_match:
        return html
    fragment = html[start_match.end():]
    end_match = re.search(r"<!--\s*(?:NEWIMAGE\d*|####)\s*-->", fragment, flags=re.IGNORECASE)
    if end_match:
        fragment = fragment[:end_match.start()]
    return fragment


def _runeberg_ocr_text(html: str) -> str:
    fragment = _runeberg_fragment(html)
    soup = BeautifulSoup(fragment, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text("", strip=False)
    if fragment == html:
        raw_marker = "Below is the raw OCR text"
        start = text.find(raw_marker)
        if start >= 0:
            text = text[start + len(raw_marker):]
        footer = text.rfind("Project Runeberg")
        if footer >= 0:
            text = text[:footer]
    return text.strip()


def _word_letters(text: str) -> str:
    without_boundary = text.translate(STEM_BOUNDARY_TRANSLATION).casefold()
    return re.sub(r"[^a-zåäöàáé-]+", "", without_boundary)


def _tokenize_line(raw_line: str) -> list[str]:
    result: list[str] = []
    for raw in raw_line.split():
        token = raw.strip(TOKEN_STRIP)
        if _word_letters(token):
            result.append(token)
    return result


def _runeberg_ocr_lines(html: str) -> list[list[str]]:
    return [tokens for raw in _runeberg_ocr_text(html).splitlines() if (tokens := _tokenize_line(raw))]


def _runeberg_ocr_tokens(html: str) -> list[str]:
    return [token for line in _runeberg_ocr_lines(html) for token in line]


def _stem_marked_tokens(html: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for token in _runeberg_ocr_tokens(html):
        if not any(mark in token for mark in STEM_BOUNDARY_MARKS) or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    i = j = differences = 0
    while i < len(shorter) and j < len(longer):
        if shorter[i] == longer[j]:
            i += 1
            j += 1
        else:
            differences += 1
            j += 1
            if differences > 1:
                return False
    return True


def _printed_order_indices(observations: list[WordObservation]) -> list[int]:
    return [index for line in _observation_line_indices(observations) for index in line]


def _observation_line_indices(observations: list[WordObservation]) -> list[list[int]]:
    if not observations:
        return []
    page_left = min(item.left for item in observations)
    page_right = max(item.left + item.width for item in observations)
    split = page_left + (page_right - page_left) / 2
    median_height = sorted(item.height for item in observations)[len(observations) // 2]
    tolerance = max(3, int(median_height * 0.55))
    result: list[list[int]] = []
    for predicate in (
        lambda item: item.left < split,
        lambda item: item.left >= split,
    ):
        column = [index for index, item in enumerate(observations) if predicate(item)]
        ordered = sorted(column, key=lambda index: (observations[index].top, observations[index].left))
        groups: list[list[int]] = []
        centers: list[float] = []
        for index in ordered:
            item = observations[index]
            center = item.top + item.height / 2
            if groups and abs(center - centers[-1]) <= tolerance:
                groups[-1].append(index)
                centers[-1] = sum(observations[i].top + observations[i].height / 2 for i in groups[-1]) / len(groups[-1])
            else:
                groups.append([index])
                centers.append(center)
        result.extend(sorted(group, key=lambda index: observations[index].left) for group in groups)
    return result


def _normalized_observation_line(observations: list[WordObservation], indices: list[int]) -> list[str]:
    return [token for index in indices if (token := _word_letters(observations[index].text))]


def _line_similarity(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    left_text = " ".join(left)
    right_text = " ".join(right)
    character_score = SequenceMatcher(None, left_text, right_text, autojunk=False).ratio()
    token_score = SequenceMatcher(None, left, right, autojunk=False).ratio()
    return 0.65 * character_score + 0.35 * token_score


def _align_lines(
    tesseract_lines: list[list[str]],
    runeberg_lines: list[list[str]],
) -> list[tuple[int, int, float]]:
    rows = len(tesseract_lines)
    columns = len(runeberg_lines)
    gap_penalty = -0.32
    minimum_pair_score = 0.28
    scores = [[float("-inf")] * (columns + 1) for _ in range(rows + 1)]
    moves = [[""] * (columns + 1) for _ in range(rows + 1)]
    scores[0][0] = 0.0
    for i in range(1, rows + 1):
        scores[i][0] = scores[i - 1][0] + gap_penalty
        moves[i][0] = "up"
    for j in range(1, columns + 1):
        scores[0][j] = scores[0][j - 1] + gap_penalty
        moves[0][j] = "left"
    similarities = [
        [_line_similarity(tesseract_lines[i], runeberg_lines[j]) for j in range(columns)]
        for i in range(rows)
    ]
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            similarity = similarities[i - 1][j - 1]
            diagonal = scores[i - 1][j - 1] + (2.2 * similarity - 0.72)
            up = scores[i - 1][j] + gap_penalty
            left = scores[i][j - 1] + gap_penalty
            best = max(diagonal, up, left)
            scores[i][j] = best
            moves[i][j] = "diag" if best == diagonal else ("up" if best == up else "left")
    pairs: list[tuple[int, int, float]] = []
    i, j = rows, columns
    while i > 0 or j > 0:
        move = moves[i][j]
        if move == "diag":
            similarity = similarities[i - 1][j - 1]
            if similarity >= minimum_pair_score:
                pairs.append((i - 1, j - 1, similarity))
            i -= 1
            j -= 1
        elif move == "up":
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _apply_contextual_replacement(
    corrected: list[WordObservation],
    observation_index: int,
    runeberg_token: str,
    expected: str,
) -> None:
    original = corrected[observation_index]
    tesseract_token = original.ocr_tesseract or original.text
    actual = _word_letters(tesseract_token)
    if len(actual) < 3 or len(expected) < 3:
        return
    minor = _edit_distance_at_most_one(actual, expected)
    plausible = minor or actual[0] == expected[0] or SequenceMatcher(None, actual, expected, autojunk=False).ratio() >= 0.58
    if not plausible:
        return
    corrected[observation_index] = replace(
        original,
        text=runeberg_token,
        ocr_tesseract=tesseract_token,
        ocr_runeberg=runeberg_token,
        ocr_conflict=not minor and actual != expected,
    )


def reconcile_contextual_observations(
    observations: list[WordObservation],
    runeberg: str | list[str] | list[list[str]],
) -> list[WordObservation]:
    if isinstance(runeberg, str):
        runeberg_lines = _runeberg_ocr_lines(runeberg)
    elif runeberg and isinstance(runeberg[0], list):
        runeberg_lines = runeberg  # type: ignore[assignment]
    else:
        # Compatibility fallback for older callers. A flat token stream is one line
        # and is deliberately conservative.
        runeberg_lines = [runeberg] if runeberg else []  # type: ignore[list-item]
    observation_lines = _observation_line_indices(observations)
    if not observation_lines or not runeberg_lines:
        return observations
    tesseract_lines = [_normalized_observation_line(observations, line) for line in observation_lines]
    runeberg_normalized = [[_word_letters(token) for token in line if _word_letters(token)] for line in runeberg_lines]
    corrected = list(observations)
    for observation_line_index, runeberg_line_index, _ in _align_lines(tesseract_lines, runeberg_normalized):
        observation_indices = observation_lines[observation_line_index]
        left_tokens = [_word_letters(observations[index].text) for index in observation_indices]
        right_tokens = runeberg_normalized[runeberg_line_index]
        right_original = runeberg_lines[runeberg_line_index]
        matcher = SequenceMatcher(None, left_tokens, right_tokens, autojunk=False)
        for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
            if tag != "replace" or left_end - left_start != 1 or right_end - right_start != 1:
                continue
            _apply_contextual_replacement(
                corrected,
                observation_indices[left_start],
                right_original[right_start],
                right_tokens[right_start],
            )
    return corrected


def reconcile_stem_marked_observations(
    observations: list[WordObservation],
    runeberg_tokens: list[str],
) -> list[WordObservation]:
    corrected = list(observations)
    used_observations: set[int] = set()
    for runeberg_token in runeberg_tokens:
        expected = _word_letters(runeberg_token)
        if len(expected) < 4:
            continue
        matches = []
        for index, observation in enumerate(corrected):
            if index in used_observations:
                continue
            actual = _word_letters(observation.text)
            if not actual or actual[:2] != expected[:2]:
                continue
            if _edit_distance_at_most_one(actual, expected):
                matches.append(index)
        if len(matches) != 1:
            continue
        index = matches[0]
        corrected[index] = replace(
            corrected[index],
            text=runeberg_token,
            ocr_runeberg=corrected[index].ocr_runeberg or runeberg_token,
        )
        used_observations.add(index)
    return corrected


def extract_observations(image_bytes: bytes) -> list[WordObservation]:
    with tempfile.TemporaryDirectory(prefix="saol-tools-") as directory:
        image_path = Path(directory) / "page.tif"
        image_path.write_bytes(image_bytes)
        tsv = _run_tesseract_tsv(image_path)
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    gray = ImageOps.autocontrast(image)
    rows = list(csv.DictReader(io.StringIO(tsv), delimiter="\t"))
    parsed_rows = []
    line_text: dict[tuple[str, str, str, str], list[str]] = {}
    line_top: dict[tuple[str, str, str, str], int] = {}
    line_first: dict[tuple[str, str, str, str], int] = {}
    heights = []
    for row in rows:
        text = (row.get("text") or "").strip()
        if row.get("level") != "5" or not text:
            continue
        try:
            left = int(row["left"])
            top = int(row["top"])
            width = int(row["width"])
            height = int(row["height"])
            confidence = float(row["conf"])
        except (ValueError, KeyError):
            continue
        if confidence < 15 or width < 2 or height < 4 or _is_printed_page_number(text, top, height, gray.height):
            continue
        key = (row.get("page_num", ""), row.get("block_num", ""), row.get("par_num", ""), row.get("line_num", ""))
        line_text.setdefault(key, []).append(text)
        line_top[key] = min(top, line_top.get(key, top))
        line_first[key] = min(left, line_first.get(key, left))
        parsed_rows.append((text, left, top, width, height, confidence, key))
        heights.append(height)
    if not heights:
        return []
    ordered_lines = [(key, line_top[key], " ".join(tokens)) for key, tokens in line_text.items()]
    excluded_lines = instruction_line_keys(ordered_lines)
    usable_rows = [row for row in parsed_rows if row[-1] not in excluded_lines]
    if not usable_rows:
        return []
    usable_heights = sorted(row[4] for row in usable_rows)
    median_height = usable_heights[len(usable_heights) // 2]
    page_width = max(gray.width, 1)
    observations = []
    for text, left, top, width, height, confidence, key in usable_rows:
        observations.append(WordObservation(
            text=text,
            left=left,
            top=top,
            width=width,
            height=height,
            confidence=confidence,
            ink_density=_ink_density(gray, left, top, width, height),
            line_left=max(0.0, min(1.0, (left - line_first[key]) / page_width)),
            relative_height=height / max(median_height, 1),
            ocr_tesseract=text,
        ))
    return observations


def fetch_page(page_number: int) -> ImportedPage:
    source_url, image_url = page_urls(page_number)
    headers = {"User-Agent": "saol-tools/0.5"}
    image_response = httpx.get(ocr_image_url(image_url), timeout=60.0, follow_redirects=True, headers=headers)
    image_response.raise_for_status()
    observations = extract_observations(image_response.content)
    if not observations:
        raise ValueError("Inga OCR-ord hittades på sidan")
    try:
        source_response = httpx.get(source_url, timeout=60.0, follow_redirects=True, headers=headers)
        source_response.raise_for_status()
        observations = reconcile_contextual_observations(observations, source_response.text)
        observations = reconcile_stem_marked_observations(observations, _stem_marked_tokens(source_response.text))
    except httpx.HTTPError:
        pass
    return ImportedPage(page_number, source_url, image_url, observations)
'''

DEBUG_SOURCE = r'''from __future__ import annotations

import argparse

import httpx

from app.runeberg import (
    _align_lines,
    _normalized_observation_line,
    _observation_line_indices,
    _runeberg_ocr_lines,
    _runeberg_ocr_text,
    _runeberg_ocr_tokens,
    extract_observations,
    ocr_image_url,
    page_urls,
    reconcile_contextual_observations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostisera Tesseract/Runeberg-jämförelsen för en SAOL-sida.")
    parser.add_argument("page", nargs="?", type=int, default=19)
    args = parser.parse_args()
    source_url, image_url = page_urls(args.page)
    headers = {"User-Agent": "saol-tools/debug"}
    source_response = httpx.get(source_url, timeout=60.0, follow_redirects=True, headers=headers)
    source_response.raise_for_status()
    image_response = httpx.get(ocr_image_url(image_url), timeout=60.0, follow_redirects=True, headers=headers)
    image_response.raise_for_status()
    raw_text = _runeberg_ocr_text(source_response.text)
    runeberg_tokens = _runeberg_ocr_tokens(source_response.text)
    runeberg_lines = _runeberg_ocr_lines(source_response.text)
    observations = extract_observations(image_response.content)
    observation_lines = _observation_line_indices(observations)
    tesseract_lines = [_normalized_observation_line(observations, line) for line in observation_lines]
    runeberg_normalized = [[token.casefold() for token in line] for line in runeberg_lines]
    pairs = _align_lines(tesseract_lines, runeberg_normalized)
    print(f"Runeberg-URL: {source_url}")
    print(f"OCR-bild: {ocr_image_url(image_url)}")
    print(f"HTML-tecken: {len(source_response.text)}")
    print(f"Extraherad OCR-text: {len(raw_text)} tecken")
    print(f"Runeberg-token: {len(runeberg_tokens)}")
    print(f"Tesseract-token: {len(observations)}")
    print(f"Runeberg-rader: {len(runeberg_lines)}")
    print(f"Tesseract-rader: {len(observation_lines)}")
    print(f"Matchade rader: {len(pairs)}")
    if pairs:
        similarities = [score for _, _, score in pairs]
        print(f"Radsimilaritet: min={min(similarities):.3f}, medel={sum(similarities)/len(similarities):.3f}, max={max(similarities):.3f}")
        print("\nExempel på matchade rader:")
        for left, right, score in pairs[:8]:
            print(f"  {left:2d} ↔ {right:2d} ({score:.3f})")
            print(f"    T: {' '.join(tesseract_lines[left])}")
            print(f"    R: {' '.join(runeberg_lines[right])}")
    corrected = reconcile_contextual_observations(observations, source_response.text)
    conflicts = [item for item in corrected if item.ocr_conflict]
    runeberg_values = [item for item in corrected if item.ocr_runeberg]
    print(f"\nObservationer med Runeberg-värde: {len(runeberg_values)}")
    print(f"Konflikter: {len(conflicts)}")
    for item in conflicts[:30]:
        print(
            f"  y={item.top:4d} x={item.left:4d}: "
            f"Tesseract={item.ocr_tesseract!r}, Runeberg={item.ocr_runeberg!r}, text={item.text!r}"
        )


if __name__ == "__main__":
    main()
'''


def main() -> None:
    RUNEberg.write_text(RUNEberg_SOURCE, encoding="utf-8")
    DEBUG.write_text(DEBUG_SOURCE, encoding="utf-8")
    print("Skrev om app/runeberg.py från grunden")
    print("Uppdaterade scripts/debug_runeberg_ocr.py")


if __name__ == "__main__":
    main()
