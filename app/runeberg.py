from __future__ import annotations

import csv
import io
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image, ImageOps

from .classifier import WordObservation

BASE_URL = "https://runeberg.org/saol/11-6"


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


def extract_observations(image_bytes: bytes) -> list[WordObservation]:
    with tempfile.TemporaryDirectory(prefix="saol-tools-") as directory:
        image_path = Path(directory) / "page.png"
        image_path.write_bytes(image_bytes)
        tsv = _run_tesseract_tsv(image_path)

    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    gray = ImageOps.autocontrast(image)
    rows = list(csv.DictReader(io.StringIO(tsv), delimiter="\t"))
    word_rows = []
    heights = []
    line_first: dict[tuple[str, str, str, str], int] = {}
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
        if confidence < 15 or width < 2 or height < 4:
            continue
        key = (row.get("block_num", ""), row.get("par_num", ""), row.get("line_num", ""), row.get("page_num", ""))
        line_first[key] = min(left, line_first.get(key, left))
        word_rows.append((text, left, top, width, height, confidence, key))
        heights.append(height)

    if not heights:
        return []
    median_height = sorted(heights)[len(heights) // 2]
    page_width = max(gray.width, 1)
    observations = []
    for text, left, top, width, height, confidence, key in word_rows:
        observations.append(
            WordObservation(
                text=text,
                left=left,
                top=top,
                width=width,
                height=height,
                confidence=confidence,
                ink_density=_ink_density(gray, left, top, width, height),
                line_left=max(0.0, min(1.0, (left - line_first[key]) / page_width)),
                relative_height=height / max(median_height, 1),
            )
        )
    return observations


def fetch_page(page_number: int) -> ImportedPage:
    source_url, image_url = page_urls(page_number)
    response = httpx.get(
        image_url,
        timeout=60.0,
        follow_redirects=True,
        headers={"User-Agent": "saol-tools/0.4"},
    )
    response.raise_for_status()
    observations = extract_observations(response.content)
    if not observations:
        raise ValueError("Inga OCR-ord hittades på sidan")
    return ImportedPage(page_number, source_url, image_url, observations)
