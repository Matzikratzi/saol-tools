from __future__ import annotations

import json
from pathlib import Path

from .database import connect
from .parser import normalize_forms, normalize_word

EXPORT_DIR = Path(__file__).resolve().parents[1] / "exports"


def swedish_sort_key(value: str) -> str:
    return normalize_word(value).translate(str.maketrans({"å": "{", "ä": "|", "ö": "}"}))


def export_approved() -> tuple[Path, int]:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT w.word, w.forms_json
            FROM words w
            JOIN pages p ON p.page_number = w.page_number
            WHERE p.status = 'approved'
            """
        ).fetchall()

    allowed: set[str] = set()
    for row in rows:
        headword = normalize_word(row["word"])
        if headword:
            allowed.add(headword)
        try:
            forms = json.loads(row["forms_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            forms = []
        allowed.update(normalize_forms(forms))

    words = sorted(allowed, key=swedish_sort_key)
    output = EXPORT_DIR / "ordlista.txt"
    output.write_text("\n".join(words) + ("\n" if words else ""), encoding="utf-8")
    return output, len(words)
