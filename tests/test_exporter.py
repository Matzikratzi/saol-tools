from pathlib import Path

from app import database, exporter


def test_export_contains_only_approved_pages(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(exporter, "EXPORT_DIR", tmp_path / "exports")
    database.init_db()
    with database.connect() as connection:
        connection.execute("INSERT INTO pages VALUES (1, 'u', 'i', 'approved', '', CURRENT_TIMESTAMP)")
        connection.execute("INSERT INTO pages VALUES (2, 'u', 'i', 'started', '', CURRENT_TIMESTAMP)")
        connection.execute("INSERT INTO words(page_number, word, sort_order) VALUES (1, 'Öga', 0)")
        connection.execute("INSERT INTO words(page_number, word, sort_order) VALUES (1, 'apa', 1)")
        connection.execute("INSERT INTO words(page_number, word, sort_order) VALUES (2, 'hemlig', 0)")
    path, count = exporter.export_approved()
    assert count == 2
    assert path.read_text(encoding="utf-8") == "apa\nöga\n"
