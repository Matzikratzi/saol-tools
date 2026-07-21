from __future__ import annotations

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


def instruction_line_keys(ordered_lines: list[tuple[tuple[str, str, str, str], int, str]]) -> set[tuple[str, str, str, str]]:
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
    box = (max(0, left - margin), max(0, top - margin), min(gray.width, left + width + margin), min(gray.height, top + height + margin))
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


def _runeberg_ocr_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    pre_blocks = [element.get_text("\n") for element in soup.find_all("pre")]
    if pre_blocks:
        return max(pre_blocks, key=len)
    text = soup.get_text("\n")
    markers = ("This page has never been proofread.", "Denna sida har aldrig korrekturlästs.")
    starts = [text.find(marker) + len(marker) for marker in markers if marker in text]
    if starts:
        text = text[max(starts):]
    footer = text.find("Project Runeberg")
    if footer >= 0:
        text = text[:footer]
    return text


def _runeberg_ocr_tokens(html: str) -> list[str]:
    result: list[str] = []
    for raw in _runeberg_ocr_text(html).split():
        token = raw.strip(".,:;!?()[]{}<>\"“”")
        if _word_letters(token):
            result.append(token)
    return result


def _stem_marked_tokens(html: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for token in _runeberg_ocr_tokens(html):
        if not any(mark in token for mark in STEM_BOUNDARY_MARKS) or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def _word_letters(text: str) -> str:
    without_boundary = text.translate(STEM_BOUNDARY_TRANSLATION).casefold()
    return re.sub(r"[^a-zåäöàáé-]+", "", without_boundary)


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    index_short = index_long = differences = 0
    while index_short < len(shorter) and index_long < len(longer):
        if shorter[index_short] == longer[index_long]:
            index_short += 1
            index_long += 1
        else:
            differences += 1
            index_long += 1
            if differences > 1:
                return False
    return True


def _printed_order_indices(observations: list[WordObservation]) -> list[int]:
    if not observations:
        return []
    page_left = min(item.left for item in observations)
    page_right = max(item.left + item.width for item in observations)
    split = page_left + (page_right - page_left) / 2
    left = [index for index, item in enumerate(observations) if item.left < split]
    right = [index for index, item in enumerate(observations) if item.left >= split]
    key = lambda index: (observations[index].top, observations[index].left)
    return sorted(left, key=key) + sorted(right, key=key)


def reconcile_contextual_observations(observations: list[WordObservation], runeberg_tokens: list[str]) -> list[WordObservation]:
    """Align both OCR streams and retain genuine one-token disagreements."""
    order = _printed_order_indices(observations)
    if not order or not runeberg_tokens:
        return observations
    tesseract_tokens = []
    for sequence_index, observation_index in enumerate(order):
        normalized = _word_letters(observations[observation_index].text)
        tesseract_tokens.append(normalized or f"__unreadable_{sequence_index}")
    runeberg_normalized = [_word_letters(token) for token in runeberg_tokens]
    corrected = list(observations)
    matcher = SequenceMatcher(None, tesseract_tokens, runeberg_normalized, autojunk=False)
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag != "replace" or left_end - left_start != 1 or right_end - right_start != 1:
            continue
        runeberg_token = runeberg_tokens[right_start]
        expected = runeberg_normalized[right_start]
        if len(expected) < 3:
            continue
        observation_index = order[left_start]
        original = corrected[observation_index]
        tesseract_token = original.ocr_tesseract or original.text
        actual = _word_letters(tesseract_token)
        if actual and actual[0] != expected[0] and len(actual) >= 3:
            continue
        minor = bool(actual) and _edit_distance_at_most_one(actual, expected)
        corrected[observation_index] = replace(
            original,
            text=runeberg_token,
            ocr_tesseract=tesseract_token,
            ocr_runeberg=runeberg_token,
            ocr_conflict=not minor and actual != expected,
        )
    return corrected


def reconcile_stem_marked_observations(observations: list[WordObservation], runeberg_tokens: list[str]) -> list[WordObservation]:
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
        corrected[index] = replace(corrected[index], text=runeberg_token, ocr_runeberg=corrected[index].ocr_runeberg or runeberg_token)
        used_observations.add(index)
    return corrected


def extract_observations(image_bytes: bytes) -> list[WordObservation]:
    with tempfile.TemporaryDirectory(prefix="saol-tools-") as directory:
        image_path = Path(directory) / "page.png"
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
            left, top, width, height = int(row["left"]), int(row["top"]), int(row["width"]), int(row["height"])
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
    headers = {"User-Agent": "saol-tools/0.4"}
    image_response = httpx.get(image_url, timeout=60.0, follow_redirects=True, headers=headers)
    image_response.raise_for_status()
    observations = extract_observations(image_response.content)
    if not observations:
        raise ValueError("Inga OCR-ord hittades på sidan")
    try:
        source_response = httpx.get(source_url, timeout=60.0, follow_redirects=True, headers=headers)
        source_response.raise_for_status()
        runeberg_tokens = _runeberg_ocr_tokens(source_response.text)
        observations = reconcile_contextual_observations(observations, runeberg_tokens)
        observations = reconcile_stem_marked_observations(observations, _stem_marked_tokens(source_response.text))
    except httpx.HTTPError:
        pass
    return ImportedPage(page_number, source_url, image_url, observations)
