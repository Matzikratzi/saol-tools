from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "app" / "runeberg.py"

ORIGINAL = '''def _runeberg_ocr_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    pre_blocks = [element.get_text("\\n") for element in soup.find_all("pre")]
    if pre_blocks:
        return max(pre_blocks, key=len)
    text = soup.get_text("\\n")
    markers = ("This page has never been proofread.", "Denna sida har aldrig korrekturlästs.")
    starts = [text.find(marker) + len(marker) for marker in markers if marker in text]
    if starts:
        text = text[max(starts):]
    footer = text.find("Project Runeberg")
    if footer >= 0:
        text = text[:footer]
    return text
'''

PREVIOUS_FIX = '''def _runeberg_ocr_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\\n")

    markers = (
        "This page has never been proofread.",
        "Denna sida har aldrig korrekturlästs.",
    )
    starts = [text.find(marker) + len(marker) for marker in markers if marker in text]
    if starts:
        text = text[max(starts):]
    else:
        pre_blocks = [element.get_text("\\n") for element in soup.find_all("pre")]
        if pre_blocks:
            return max(pre_blocks, key=len)

    endings = (
        "<< prev. page << föreg. sida <<",
        "Project Runeberg",
    )
    stops = [text.find(marker) for marker in endings if text.find(marker) >= 0]
    if stops:
        text = text[:min(stops)]
    return text.strip()
'''

FIXED = '''def _runeberg_ocr_text(html: str) -> str:
    # Runeberg places the OCR directly in the HTML after a mode comment.
    # Parsing all visible text first is unreliable because markup splits phrases
    # such as ``has <b>never</b> been proofread`` into separate text nodes.
    start_match = re.search(r"<!--\\s*mode=[^>]*-->", html, flags=re.IGNORECASE)
    if start_match:
        fragment = html[start_match.end():]
        end_match = re.search(
            r"<!--\\s*(?:NEWIMAGE\\d*|####)\\s*-->",
            fragment,
            flags=re.IGNORECASE,
        )
        if end_match:
            fragment = fragment[:end_match.start()]
        return BeautifulSoup(fragment, "html.parser").get_text("\\n").strip()

    # Fallback for Runeberg pages with a different template.
    soup = BeautifulSoup(html, "html.parser")
    pre_blocks = [element.get_text("\\n") for element in soup.find_all("pre")]
    if pre_blocks:
        return max(pre_blocks, key=len).strip()

    text = soup.get_text("\\n")
    raw_marker = "Below is the raw OCR text"
    start = text.find(raw_marker)
    if start >= 0:
        text = text[start + len(raw_marker):]
    footer = text.rfind("Project Runeberg")
    if footer >= 0:
        text = text[:footer]
    return text.strip()
'''


def main() -> None:
    source = TARGET.read_text(encoding="utf-8")
    if FIXED in source:
        print("Fixen finns redan i app/runeberg.py")
        return

    for old in (PREVIOUS_FIX, ORIGINAL):
        if old in source:
            TARGET.write_text(source.replace(old, FIXED), encoding="utf-8")
            print("Uppdaterade app/runeberg.py")
            return

    raise SystemExit("Hittade inte någon känd version av Runeberg-parsern; filen har ändrats.")


if __name__ == "__main__":
    main()
