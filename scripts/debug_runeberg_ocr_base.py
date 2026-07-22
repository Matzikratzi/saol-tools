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


def _body_top_y(observations: list, image_height: int, median_height: float) -> float:
    """Keep the first OCR row immediately below the header separator.

    The separator itself is normally not returned as OCR text.  Its position is
    therefore represented by the first substantial vertical gap near the top of
    the page.  Scan gaps from top to bottom instead of selecting the largest one:
    a later, unusually large dictionary line gap must never remove the first
    headword.  The cutoff is placed just above the first row below that gap.
    """
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

    # Multiple OCR engines can produce almost identical rows.  Collapse those
    # before looking for the whitespace directly below the header rule.
    centres.sort()
    distinct: list[float] = []
    merge_distance = max(1.0, median_height * 0.35)
    for centre in centres:
        if not distinct or centre - distinct[-1] > merge_distance:
            distinct.append(centre)
        else:
            distinct[-1] = (distinct[-1] + centre) / 2

    minimum_gap = max(median_height * 1.4, image_height * 0.008)
    for upper, lower in zip(distinct, distinct[1:]):
        if lower - upper >= minimum_gap:
            # The lower item is the first printed row below the separator.  Put
            # the cutoff above its bounding-box centre so that the row survives.
            return max(0.0, lower - median_height * 0.75)

    # Conservative fallback: retain substantially more than the old 7 % rule.
    return image_height * 0.03


_materialize_runtime()

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
