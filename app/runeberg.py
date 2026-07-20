from __future__ import annotations

import re
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

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


def extract_ocr(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for selector in ("textarea", "pre"):
        candidate = soup.select_one(selector)
        if candidate and len(candidate.get_text("\n", strip=False).strip()) > 5:
            return _clean_lines(candidate.get_text("\n", strip=False))

    marker = soup.find(string=re.compile(r"Below is the raw OCR text", re.I))
    if marker is None:
        marker = soup.find(string=re.compile(r"Här nedan syns maskintolkade texten", re.I))
    if marker is None:
        raise ValueError("Kunde inte hitta OCR-avsnittet på Runeberg-sidan")

    body = soup.body or soup
    start_found = False
    pieces: list[str] = []
    for node in body.descendants:
        if node is marker:
            start_found = True
            continue
        if not start_found or not isinstance(node, NavigableString):
            continue
        text = str(node)
        if "<< prev. page" in text or "Project Runeberg," in text:
            break
        if re.search(r"This page has .*proofread|Denna sida har .*korrekturlästs", text, re.I):
            continue
        parent = node.parent
        if parent and parent.name in {"script", "style", "nav"}:
            continue
        pieces.append(text)

    result = _clean_lines("\n".join(pieces))
    result = re.sub(r"^(?:Proofread the page now!|Korrekturläs sidan nu!)\s*", "", result, flags=re.I)
    if len(result) < 5:
        raise ValueError("OCR-avsnittet hittades men verkar vara tomt")
    return result


def fetch_page(page_number: int) -> ImportedPage:
    source_url, image_url = page_urls(page_number)
    response = httpx.get(source_url, timeout=30.0, follow_redirects=True, headers={"User-Agent": "saol-tools/0.2"})
    response.raise_for_status()
    return ImportedPage(page_number, source_url, image_url, extract_ocr(response.text))
