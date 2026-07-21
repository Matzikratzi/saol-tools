from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .classifier import HeadwordModel, normalize_token, train_model
from .database import connect, init_db
from .exporter import export_approved
from .parser import normalize_forms, split_headword_marker, suspicious_word
from .runeberg import fetch_page

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
TRAINING_PAGES = range(19, 29)

app = FastAPI(title="SAOL-tools", version="0.7.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ImportRequest(BaseModel):
    page_number: int = Field(ge=1, le=9999)
    force: bool = False


class ArticleInput(BaseModel):
    word: str = Field(min_length=1, max_length=100)
    sense_number: int | None = Field(default=None, ge=1, le=99)
    word_class: str = Field(default="", max_length=50)
    inflection_raw: str = Field(default="", max_length=500)
    forms: list[str] = Field(default_factory=list)
    bbox_left: int | None = None
    bbox_top: int | None = None
    bbox_width: int | None = None
    bbox_height: int | None = None


class SaveRequest(BaseModel):
    words: list[ArticleInput]
    status: str = "started"
    verified_by: str = ""


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


def page_payload(connection, page_number: int):
    page = connection.execute("SELECT * FROM pages WHERE page_number=?", (page_number,)).fetchone()
    if page is None:
        return None
    rows = connection.execute(
        """
        SELECT id, word, sort_order, suspicious, sense_number, word_class,
               inflection_raw, forms_json,
               bbox_left, bbox_top, bbox_width, bbox_height
        FROM words WHERE page_number=? ORDER BY sort_order
        """,
        (page_number,),
    ).fetchall()
    result = dict(page)
    result["words"] = []
    for row in rows:
        item = dict(row)
        try:
            item["forms"] = json.loads(item.pop("forms_json") or "[]")
        except (TypeError, json.JSONDecodeError):
            item["forms"] = []
        result["words"].append(item)
    return result


def dictionary_body_observations(observations):
    """Remove a genuine running header from a full OCR page.

    Sparse inputs do not contain enough line-spacing evidence to distinguish a
    running header from ordinary dictionary rows. In those cases every OCR box
    is preserved. Real SAOL pages contain hundreds of observations, so requiring
    a modest amount of evidence does not weaken header removal there.
    """
    if len(observations) < 12:
        return observations
    ordered = sorted(observations, key=lambda item: (item.top, item.left))
    tops = sorted({item.top for item in ordered})
    if len(tops) < 8:
        return observations
    min_top = tops[0]
    max_bottom = max(item.top + item.height for item in ordered)
    vertical_span = max(1, max_bottom - min_top)
    median_height = sorted(item.height for item in ordered)[len(ordered) // 2]
    early_limit = min_top + vertical_span * 0.30
    gaps = [
        (next_top - top, next_top)
        for top, next_top in zip(tops, tops[1:])
        if next_top <= early_limit
    ]
    if not gaps:
        return observations
    gap, body_top = max(gaps)
    if gap < median_height * 1.5:
        return observations
    return [item for item in observations if item.top >= body_top]


def _median(values):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _split_printed_columns(observations):
    """Split OCR observations at the widest central horizontal whitespace."""
    if len(observations) < 4:
        return [sorted(observations, key=lambda item: (item.top, item.left))]

    centres = sorted(item.left + item.width / 2 for item in observations)
    low = centres[0] + (centres[-1] - centres[0]) * 0.20
    high = centres[0] + (centres[-1] - centres[0]) * 0.80
    gaps = [
        (right - left, (left + right) / 2)
        for left, right in zip(centres, centres[1:])
        if low <= (left + right) / 2 <= high
    ]
    if not gaps:
        return [sorted(observations, key=lambda item: (item.top, item.left))]

    gap, split = max(gaps)
    median_width = _median([item.width for item in observations])
    if gap < max(24.0, median_width * 1.8):
        return [sorted(observations, key=lambda item: (item.top, item.left))]

    left_column = [item for item in observations if item.left + item.width / 2 < split]
    right_column = [item for item in observations if item.left + item.width / 2 >= split]
    return [
        sorted(left_column, key=lambda item: (item.top, item.left)),
        sorted(right_column, key=lambda item: (item.top, item.left)),
    ]


def _group_printed_rows(column):
    """Group OCR boxes that overlap the same printed baseline."""
    if not column:
        return []
    median_height = max(1.0, _median([item.height for item in column]))
    tolerance = median_height * 0.55
    rows = []
    for item in sorted(column, key=lambda value: (value.top + value.height / 2, value.left)):
        centre = item.top + item.height / 2
        best = None
        best_distance = None
        for row in rows[-3:]:
            distance = abs(centre - row["centre"])
            if distance <= tolerance and (best_distance is None or distance < best_distance):
                best = row
                best_distance = distance
        if best is None:
            rows.append({"items": [item], "centre": centre})
        else:
            best["items"].append(item)
            best["centre"] = sum(
                value.top + value.height / 2 for value in best["items"]
            ) / len(best["items"])

    result = []
    for row in sorted(rows, key=lambda value: value["centre"]):
        result.append(sorted(row["items"], key=lambda value: value.left))
    return result


def _sense_only(text: str) -> int | None:
    token = text.strip().strip(".,:;()[]")
    translated = token.translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789"))
    if translated.isdigit() and 1 <= int(translated) <= 99:
        return int(translated)
    return None


def _union_box(items):
    left = min(item.left for item in items)
    top = min(item.top for item in items)
    right = max(item.left + item.width for item in items)
    bottom = max(item.top + item.height for item in items)
    return left, top, right - left, bottom - top


def _row_headword(row):
    """Return the possible article start from the beginning of a printed row."""
    if not row:
        return None
    first = row[0]
    separate_sense = _sense_only(first.text)
    if separate_sense is not None and len(row) >= 2:
        second = row[1]
        horizontal_gap = second.left - (first.left + first.width)
        vertical_overlap = min(first.top + first.height, second.top + second.height) - max(
            first.top, second.top
        )
        if horizontal_gap <= max(first.height, second.height) * 1.6 and vertical_overlap >= -first.height:
            _, word = split_headword_marker(second.text)
            if word:
                left, top, width, height = _union_box([first, second])
                return separate_sense, word, second, (left, top, width, height), True

    sense_number, word = split_headword_marker(first.text)
    if not word:
        return None
    return sense_number, word, first, _union_box([first]), sense_number is not None


def _looks_like_inflection_fragment(word: str) -> bool:
    """Reject common compact endings, not all hyphenated dictionary words."""
    normalized = word.casefold().strip()
    if normalized.startswith("-"):
        return True
    return bool(re.fullmatch(r"[a-zåäö]-[a-zåäö]{1,3}", normalized))


def _column_margin(rows, median_height):
    first_lefts = sorted(row[0].left for row in rows if row)
    if not first_lefts:
        return 0.0
    sample_count = max(1, len(first_lefts) // 3)
    return _median(first_lefts[:sample_count])


def observations_to_candidates(observations):
    """Find article starts in printed reading order: left column, then right."""
    observations = dictionary_body_observations(observations)
    model = HeadwordModel.load()
    result = []
    seen = set()

    for column_number, column in enumerate(_split_printed_columns(observations)):
        if not column:
            continue
        rows = _group_printed_rows(column)
        median_height = max(1.0, _median([item.height for item in column]))
        margin = _column_margin(rows, median_height)
        indent_limit = margin + median_height * 1.35

        densities = sorted(item.ink_density for item in column)
        density_limit = densities[int(len(densities) * 0.62)] if densities else 0.0

        for row in rows:
            candidate = _row_headword(row)
            if candidate is None:
                continue
            sense_number, word, source, box, explicit_sense = candidate
            if source.left > indent_limit and not explicit_sense:
                continue
            if _looks_like_inflection_fragment(word) and not explicit_sense:
                continue

            probability = model.probability(source) if model is not None else 0.5
            typographic_support = (
                probability >= 0.50 if model is not None else source.ink_density >= density_limit
            )
            key = (word.casefold(), sense_number)
            if key in seen:
                continue
            seen.add(key)
            left, top, width, height = box
            result.append(
                {
                    "word": word,
                    "sense_number": sense_number,
                    "word_class": "",
                    "inflection_raw": "",
                    "forms": [],
                    "suspicious": (
                        not explicit_sense
                        and (not typographic_support or suspicious_word(word))
                    ),
                    "bbox_left": left,
                    "bbox_top": top,
                    "bbox_width": width,
                    "bbox_height": height,
                    "column": column_number,
                }
            )

    return result


@app.get("/api/pages")
def list_pages():
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT p.page_number, p.status, p.verified_by, p.updated_at, COUNT(w.id) AS word_count
            FROM pages p LEFT JOIN words w ON w.page_number=p.page_number
            GROUP BY p.page_number ORDER BY p.page_number
            """
        ).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/model")
def model_status():
    model = HeadwordModel.load()
    if model is None:
        return {"trained": False, "training_pages": list(TRAINING_PAGES)}
    return {
        "trained": True,
        "samples": model.samples,
        "positive_samples": model.positive_samples,
        "training_pages": list(TRAINING_PAGES),
    }


@app.post("/api/model/train")
def train_headword_model():
    samples = []
    missing = []
    with connect() as connection:
        labels_by_page = {}
        for page_number in TRAINING_PAGES:
            page = connection.execute(
                "SELECT status FROM pages WHERE page_number=?", (page_number,)
            ).fetchone()
            if page is None or page["status"] != "approved":
                missing.append(page_number)
                continue
            rows = connection.execute(
                "SELECT word FROM words WHERE page_number=?", (page_number,)
            ).fetchall()
            labels_by_page[page_number] = {
                token
                for row in rows
                for token in (normalize_token(part) for part in row["word"].split())
                if token
            }

    if missing:
        raise HTTPException(
            status_code=409,
            detail=f"Godkänn först träningssidorna: {', '.join(map(str, missing))}",
        )

    try:
        for page_number in TRAINING_PAGES:
            imported = fetch_page(page_number)
            labels = labels_by_page[page_number]
            for observation in dictionary_body_observations(imported.observations):
                _, headword = split_headword_marker(observation.text)
                token = normalize_token(headword)
                if token:
                    samples.append((observation, int(token in labels)))
        model = train_model(samples)
        model.save()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Kunde inte träna modellen: {exc}") from exc

    return {
        "trained": True,
        "samples": model.samples,
        "positive_samples": model.positive_samples,
    }


@app.post("/api/pages/import")
def import_page(request: ImportRequest):
    try:
        imported = fetch_page(request.page_number)
        candidates = observations_to_candidates(imported.observations)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Kunde inte importera sidan: {exc}") from exc

    with connect() as connection:
        existing = connection.execute(
            "SELECT status FROM pages WHERE page_number=?", (request.page_number,)
        ).fetchone()
        if existing is not None and request.force and existing["status"] == "approved":
            raise HTTPException(status_code=409, detail="En godkänd sida kan inte importeras om")
        if existing is None:
            connection.execute(
                "INSERT INTO pages(page_number, source_url, image_url) VALUES (?, ?, ?)",
                (imported.page_number, imported.source_url, imported.image_url),
            )
        elif request.force:
            connection.execute(
                "UPDATE pages SET source_url=?, image_url=?, status='started', verified_by='', updated_at=CURRENT_TIMESTAMP WHERE page_number=?",
                (imported.source_url, imported.image_url, imported.page_number),
            )
            connection.execute("DELETE FROM words WHERE page_number=?", (request.page_number,))
        if existing is None or request.force:
            connection.executemany(
                """
                INSERT INTO words(
                    page_number, word, sort_order, suspicious, sense_number,
                    word_class, inflection_raw, forms_json,
                    bbox_left, bbox_top, bbox_width, bbox_height
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        imported.page_number,
                        candidate["word"],
                        index,
                        int(candidate["suspicious"]),
                        candidate["sense_number"],
                        "", "", "[]",
                        candidate["bbox_left"], candidate["bbox_top"],
                        candidate["bbox_width"], candidate["bbox_height"],
                    )
                    for index, candidate in enumerate(candidates)
                ],
            )
        result = page_payload(connection, request.page_number)
    return result


@app.get("/api/pages/{page_number}")
def get_page(page_number: int):
    with connect() as connection:
        result = page_payload(connection, page_number)
    if result is None:
        raise HTTPException(status_code=404, detail="Sidan är inte importerad")
    return result


@app.put("/api/pages/{page_number}")
def save_page(page_number: int, request: SaveRequest):
    if request.status not in {"started", "approved"}:
        raise HTTPException(status_code=400, detail="Ogiltig status")
    normalized = []
    seen = set()
    for item in request.words:
        detected_sense, word = split_headword_marker(item.word)
        sense_number = item.sense_number if item.sense_number is not None else detected_sense
        key = (word.casefold(), sense_number)
        if not word or key in seen:
            continue
        seen.add(key)
        forms = normalize_forms(item.forms)
        normalized.append((word, sense_number, forms, item))

    with connect() as connection:
        if connection.execute("SELECT 1 FROM pages WHERE page_number=?", (page_number,)).fetchone() is None:
            raise HTTPException(status_code=404, detail="Sidan är inte importerad")
        connection.execute("DELETE FROM words WHERE page_number=?", (page_number,))
        connection.executemany(
            """
            INSERT INTO words(
                page_number, word, sort_order, suspicious, sense_number,
                word_class, inflection_raw, forms_json,
                bbox_left, bbox_top, bbox_width, bbox_height
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    page_number, word, index, int(suspicious_word(word)),
                    sense_number, item.word_class.strip(), item.inflection_raw.strip(),
                    json.dumps(forms, ensure_ascii=False),
                    item.bbox_left, item.bbox_top, item.bbox_width, item.bbox_height,
                )
                for index, (word, sense_number, forms, item) in enumerate(normalized)
            ],
        )
        connection.execute(
            "UPDATE pages SET status=?, verified_by=?, updated_at=CURRENT_TIMESTAMP WHERE page_number=?",
            (request.status, request.verified_by.strip(), page_number),
        )
        result = page_payload(connection, page_number)
    return result


@app.post("/api/export")
def export_words():
    path, count = export_approved()
    return {"filename": path.name, "word_count": count, "download_url": "/api/export/download"}


@app.get("/api/export/download")
def download_export():
    path, _ = export_approved()
    return FileResponse(path, filename=path.name, media_type="text/plain; charset=utf-8")