from __future__ import annotations

import re
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup, NavigableString

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
    return (f"{BASE_URL}/{identifier}.html", f"https://runeberg.org/img/saol/11-6/{identifier}.3.png")


def _clean_lines(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _strip_ocr_prelude(text: str) -> str:
    """Remove Runeberg's bilingual instructions and proofreading status."""
    cleaned = _clean_lines(text)
    lines = cleaned.splitlines()

    # The actual OCR always follows Runeberg's proofreading status line.
    # Use the last matching line in case the bilingual text is split.
    status_index: int | None = None
    for index, line in enumerate(lines):
        normalized = re.sub(r"\s+", " ", line).strip()
        if re.search(
            r"This page has .*proofread|Denna sida har .*korrekturlästs",
            normalized,
            re.I,
        ):
            status_index = index

    if status_index is not None:
        return _clean_lines("\n".join(lines[status_index + 1 :]))

    # Fallback for pages where Runeberg omits or changes the status line.
    prelude_patterns = (
        r"Below is the raw OCR text from the above scanned image",
        r"Do you see an error\?",
        r"Proofread the page now!",
        r"Här nedan syns maskintolkade texten från faksimilbilden ovan",
        r"Ser du något fel\?",
        r"Korrekturläs sidan nu!",
    )
    filtered = []
    for line in lines:
        normalized = re.sub(r"\s+", " ", line).strip()
        if any(re.search(pattern, normalized, re.I) for pattern in prelude_patterns):
            continue
        filtered.append(line)
    return _clean_lines("\n".join(filtered))


def extract_ocr(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for selector in ("textarea", "pre"):
        candidate = soup.select_one(selector)
        if candidate and len(candidate.get_text("\n", strip=False).strip()) > 5:
            result = _strip_ocr_prelude(candidate.get_text("\n", strip=False))
            if len(result) >= 5:
                return result

    body = soup.body or soup
    start_found = False
    marker_window = ""
    pieces: list[str] = []

    # Runeberg sometimes splits the marker sentence across text nodes because
    # the proofreading link sits in the middle of the sentence. Search a
    # rolling window instead of requiring the complete sentence in one node.
    for node in body.descendants:
        if not isinstance(node, NavigableString):
            continue
        parent = node.parent
        if parent and parent.name in {"script", "style", "nav"}:
            continue

        text = str(node)
        if not start_found:
            marker_window = (marker_window + " " + text)[-1000:]
            normalized = re.sub(r"\s+", " ", marker_window)
            if re.search(
                r"Below is the raw OCR text from the above scanned image|"
                r"Här nedan syns maskintolkade texten från faksimilbilden ovan",
                normalized,
                re.I,
            ):
                start_found = True
            continue

        if "<< prev. page" in text or "Project Runeberg," in text:
            break
        pieces.append(text)

    if not start_found:
        raise ValueError("Kunde inte hitta OCR-avsnittet på Runeberg-sidan")

    result = _strip_ocr_prelude(_clean_lines("\n".join(pieces)))
    if len(result) < 5:
        raise ValueError("OCR-avsnittet hittades men verkar vara tomt")
    return result


def fetch_page(page_number: int) -> ImportedPage:
    source_url, image_url = page_urls(page_number)
    response = httpx.get(
        source_url,
        timeout=30.0,
        follow_redirects=True,
        headers={"User-Agent": "saol-tools/0.2"},
    )
    response.raise_for_status()
    return ImportedPage(page_number, source_url, image_url, extract_ocr(response.text))
