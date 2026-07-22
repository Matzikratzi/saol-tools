from __future__ import annotations

"""Compatibility loader for the complete OCR debug implementation.

The complete tool currently lives on ``agent/complete-saol-tool`` while main
contains the experimental geometry wrapper.  This loader materializes that
known-good implementation from the local Git object database and executes it
in this module, so the normal ``git pull`` + run workflow works again.
"""

import io
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


_materialize_runtime()

if str(CACHE_ROOT) not in sys.path:
    sys.path.insert(0, str(CACHE_ROOT))

source = BASE_PATH.read_text(encoding="utf-8")
exec(compile(source, str(BASE_PATH), "exec"), globals(), globals())
