from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "app" / "runeberg.py"

OLD = '''def _runeberg_ocr_text(html: str) -> str:
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

NEW = '''def _runeberg_ocr_text(html: str) -> str:
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


def main() -> None:
    source = TARGET.read_text(encoding="utf-8")
    if NEW in source:
        print("Fixen finns redan i app/runeberg.py")
        return
    if OLD not in source:
        raise SystemExit("Hittade inte den förväntade gamla parsern; filen har ändrats.")
    TARGET.write_text(source.replace(OLD, NEW), encoding="utf-8")
    print("Uppdaterade app/runeberg.py")


if __name__ == "__main__":
    main()
