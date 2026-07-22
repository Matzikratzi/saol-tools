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
    """Infer the body start from the large gap below the running head.

    The old implementation discarded the top seven percent unconditionally,
    which can remove the first dictionary row.  OCR line centres let us place
    the cutoff in the whitespace between the running head and the body instead.
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

    centres.sort()
    if len(centres) >= 2:
        gaps = [
            (centres[index + 1] - centres[index], index)
            for index in range(len(centres) - 1)
        ]
        gap, index = max(gaps)
        if gap >= max(median_height * 1.4, image_height * 0.008):
            return (centres[index] + centres[index + 1]) / 2

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
