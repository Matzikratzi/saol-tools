from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

BASE_URL = "https://runeberg.org/saol/11-6"


@dataclass(frozen=True)
class ImportedPage:
    page_number: int
    source_url: str
    image_url: str
    ocr_text: str


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


def _clean_headword(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(".,;:!?()[]{}«»\"“”")


def extract_bold_headwords(hocr: str) -> list[str]:
    """Return consecutive bold word groups from Tesseract hOCR.

    SAOL 11 states that every item printed in semi-bold is a headword. Extra
    bold only marks the first headword of a paragraph and has no additional
    lexical meaning. Tesseract represents detected bold words with <strong>.
    """
    soup = BeautifulSoup(hocr, "html.parser")
    result: list[str] = []
    seen: set[str] = set()

    for line in soup.select(".ocr_line"):
        group: list[str] = []

        def flush() -> None:
            if not group:
                return
            word = _clean_headword(" ".join(group))
            group.clear()
            key = word.casefold()
            if word and key not in seen:
                seen.add(key)
                result.append(word)

        for node in line.select(".ocrx_word"):
            text = _clean_headword(node.get_text(" ", strip=True))
            is_bold = node.find("strong") is not None or node.find("b") is not None
            if is_bold and text:
                group.append(text)
            else:
                flush()
        flush()

    return result


def _run_tesseract_hocr(image: bytes) -> str:
    executable = shutil.which("tesseract")
    if executable is None:
        raise RuntimeError(
            "Tesseract saknas. Installera med: brew install tesseract tesseract-lang"
        )

    with tempfile.TemporaryDirectory(prefix="saol-tools-") as directory:
        image_path = Path(directory) / "page.png"
        image_path.write_bytes(image)
        command = [
            executable,
            str(image_path),
            "stdout",
            "-l",
            "swe",
            "--psm",
            "6",
            "hocr",
        ]
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    if process.returncode != 0:
        detail = process.stderr.strip() or "okänt Tesseract-fel"
        if "Failed loading language 'swe'" in detail:
            raise RuntimeError(
                "Svenska Tesseract-data saknas. Installera med: brew install tesseract-lang"
            )
        raise RuntimeError(f"Tesseract misslyckades: {detail}")
    return process.stdout


def fetch_page(page_number: int) -> ImportedPage:
    source_url, image_url = page_urls(page_number)
    response = httpx.get(
        image_url,
        timeout=60.0,
        follow_redirects=True,
        headers={"User-Agent": "saol-tools/0.3"},
    )
    response.raise_for_status()

    hocr = _run_tesseract_hocr(response.content)
    headwords = extract_bold_headwords(hocr)
    if not headwords:
        raise ValueError(
            "Tesseract hittade inga fetstilta uppslagsord. Kontrollera sidan manuellt."
        )

    # The temporary hOCR is deliberately discarded. Only the extracted
    # candidates continue through the existing parser and into the database.
    return ImportedPage(page_number, source_url, image_url, "\n".join(headwords))
