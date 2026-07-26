from __future__ import annotations

"""Compatibility loader for the complete OCR debug implementation.

The complete tool currently lives on ``agent/complete-saol-tool`` while main
contains the experimental geometry wrapper.  This loader materializes that
known-good implementation from the local Git object database and executes it
in this module, so the normal ``git pull`` + run workflow works again.
"""

import io
import statistics
import subprocess
import sys
import tarfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REF = "origin/agent/complete-saol-tool"
CACHE_ROOT = REPOSITORY_ROOT / ".debug-runeberg-ocr-runtime"
BASE_PATH = CACHE_ROOT / "scripts" / "debug_runeberg_ocr_base.py"
RUNEBERG_PATH = CACHE_ROOT / "app" / "runeberg.py"
MARKER_PREFIXES = set("123456789Iil|oO°.'`,:")


def _git(*args: str, capture: bool = False) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *args],
        check=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _ref_exists() -> bool:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "--verify", "--quiet", SOURCE_REF],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _materialize_runtime() -> None:
    if BASE_PATH.exists():
        return

    if not _ref_exists():
        _git("fetch", "origin", "agent/complete-saol-tool:refs/remotes/origin/agent/complete-saol-tool")

    archive = _git(
        "archive",
        "--format=tar",
        SOURCE_REF,
        "app",
        "scripts/debug_runeberg_ocr_base.py",
        capture=True,
    ).stdout

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        tar.extractall(CACHE_ROOT)


def _patch_runtime_tsv_parser() -> None:
    """Do not let OCR quote characters turn Tesseract TSV into multiline CSV."""
    source = RUNEBERG_PATH.read_text(encoding="utf-8")
    old = 'csv.DictReader(io.StringIO(tsv), delimiter="\\t")'
    new = 'csv.DictReader(io.StringIO(tsv), delimiter="\\t", quoting=csv.QUOTE_NONE)'
    if new in source:
        return
    if old not in source:
        raise RuntimeError("Kunde inte säkra Tesseracts TSV-parser")
    RUNEBERG_PATH.write_text(source.replace(old, new, 1), encoding="utf-8")


def _body_top_y(observations: list, image_height: int, median_height: float) -> float:
    """Place the body cutoff immediately above the first row below the header rule."""
    centres: list[float] = []
    for indices in _observation_line_indices(observations):
        items = [observations[index] for index in indices]
        if not items:
            continue
        top = min(item.top for item in items)
        bottom = max(item.top + item.height for item in items)
        centre = (top + bottom) / 2
        if centre <= image_height * 0.25:
            centres.append(float(centre))

    centres.sort()
    if len(centres) >= 2:
        for index in range(len(centres) - 1):
            gap = centres[index + 1] - centres[index]
            if gap >= max(median_height * 1.4, image_height * 0.008):
                return centres[index + 1] - max(1.0, median_height * 0.55)

    return image_height * 0.03


def _install_robust_prefix_geometry() -> None:
    """Make article detection independent of Tesseract's token splitting.

    The geometry wrapper used to inspect mainly the first OCR token.  A normal
    headword could therefore be missed when punctuation, dust or a homonym
    number occupied that token.  Scan the first few boxes for the first actual
    word and detect a homonym marker separately, including merged forms such as
    ``2abstrakt``.
    """
    this_path = Path(__file__).resolve()
    for parent in tuple(sys.modules.values()):
        if parent is None or not hasattr(parent, "_prefix_geometry"):
            continue
        parent_base = getattr(parent, "BASE", None)
        if parent_base is None or Path(parent_base).resolve() != this_path:
            continue

        old_prefix_geometry = parent._prefix_geometry

        def robust_prefix_geometry(module, line, median_height: float, _old=old_prefix_geometry):
            items = tuple(sorted(line.items, key=lambda item: item.left))
            for index, item in enumerate(items[:5]):
                token = item.text.strip()
                if not token:
                    continue

                word = token.lstrip("123456789Iil|oO°.'`,:")
                prefix_length = len(token) - len(word)
                if prefix_length and module.WORD_RE.match(word):
                    character_width = max(
                        item.width / max(2, len(token)),
                        item.height * 0.20,
                    )
                    return (
                        float(item.left + character_width * prefix_length),
                        item,
                        word,
                        True,
                    )

                if not module.WORD_RE.match(token):
                    continue

                if index > 0:
                    marker = items[index - 1]
                    marker_text = marker.text.strip()
                    gap = item.left - (marker.left + marker.width)
                    markerish = (
                        0 < len(marker_text) <= 2
                        and all(character in MARKER_PREFIXES for character in marker_text)
                    )
                    small = marker.height <= max(median_height * 1.15, item.height * 1.10)
                    narrow = marker.width <= max(item.height * 1.10, median_height)
                    close = -3.0 <= gap <= max(16.0, item.height * 1.25)
                    if markerish and small and narrow and close:
                        return float(item.left), item, token, True

                return float(item.left), item, token, False

            return _old(module, line, median_height)

        parent._prefix_geometry = robust_prefix_geometry
        return


_materialize_runtime()
_patch_runtime_tsv_parser()

if str(CACHE_ROOT) not in sys.path:
    sys.path.insert(0, str(CACHE_ROOT))

source = BASE_PATH.read_text(encoding="utf-8")
old_filter = "if center_y < image_height * 0.07 or center_y > image_height * 0.93:"
new_filter = (
    "if center_y < _body_top_y(observations, image_height, median_height) "
    "or center_y > image_height * 0.93:"
)
if old_filter not in source:
    raise RuntimeError("Kunde inte ersätta den fasta sidhuvudsgränsen i OCR-basmodulen")
source = source.replace(old_filter, new_filter, 1)
exec(compile(source, str(BASE_PATH), "exec"), globals(), globals())
_install_robust_prefix_geometry()
