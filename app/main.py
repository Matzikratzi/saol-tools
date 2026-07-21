from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .classifier import HeadwordModel, normalize_token, train_model
from .database import connect, init_db
from .exporter import export_approved
from .parser import normalize_word, suspicious_word
from .runeberg import fetch_page

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
TRAINING_PAGES = range(19, 29)

app = FastAPI(title="SAOL-tools", version="0.4.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ImportRequest(BaseModel):
    page_number: int = Field(ge=1, le=9999)
    force: bool = False


class WordInput(BaseModel):
    word: str = Field(min_length=1, max_length=100)


class SaveRequest(BaseModel):
    words: list[WordInput]
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
    words = connection.execute(
        "SELECT id, word, sort_order, suspicious FROM words WHERE page_number=? ORDER BY sort_order",
        (page_number,),
    ).fetchall()
    result = dict(page)
    result["words"] = [dict(row) for row in words]
    return result


def observations_to_candidates(observations):
    model = HeadwordModel.load()
    scored = []
    if model is None:
        densities = sorted(item.ink_density for item in observations)
        density_limit = densities[int(len(densities) * 0.62)] if densities else 0.0
        for item in observations:
            if item.line_left < 0.003 and item.ink_density >= density_limit:
                scored.append((item.top, item.left, item.text, 0.5))
    else:
        for item in observations:
            probability = model.probability(item)
            if probability >= 0.50:
                scored.append((item.top, item.left, item.text, probability))

    result = []
    seen = set()
    for _, _, text, probability in sorted(scored):
        word = normalize_word(text)
        key = word.casefold()
        if len(word) < 2 or key in seen:
            continue
        seen.add(key)
        result.append((word, probability < 0.72 or suspicious_word(word)))
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
            for observation in imported.observations:
                token = normalize_token(observation.text)
                if len(token) >= 2:
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
                "INSERT INTO words(page_number, word, sort_order, suspicious) VALUES (?, ?, ?, ?)",
                [(imported.page_number, word, index, int(flag)) for index, (word, flag) in enumerate(candidates)],
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
        word = normalize_word(item.word)
        if not word or word in seen:
            continue
        seen.add(word)
        normalized.append(word)
    with connect() as connection:
        if connection.execute("SELECT 1 FROM pages WHERE page_number=?", (page_number,)).fetchone() is None:
            raise HTTPException(status_code=404, detail="Sidan är inte importerad")
        connection.execute("DELETE FROM words WHERE page_number=?", (page_number,))
        connection.executemany(
            "INSERT INTO words(page_number, word, sort_order, suspicious) VALUES (?, ?, ?, ?)",
            [(page_number, word, index, int(suspicious_word(word))) for index, word in enumerate(normalized)],
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
