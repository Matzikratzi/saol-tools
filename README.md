# SAOL-tools

Ett fristående webbaserat verktyg för att skapa en ren ordlista från den OCR-tolkade skanningen av SAOL 11 hos Projekt Runeberg.

Verktyget sparar bara ord, sidreferenser, status och korrläsarens namn. Den fullständiga OCR-texten lagras inte.

## Start på macOS

```bash
cd ~/projs/saol-tools
git switch agent/complete-saol-tool
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

Öppna `http://127.0.0.1:8001`.

Du kan även dubbelklicka på `run.command` i Finder efter att ha gjort den körbar:

```bash
chmod +x run.command
```

## Arbetsflöde

1. Ange ett Runeberg-sidnummer och välj **Öppna/importera**.
2. Kontrollera orden mot den skannade sidan.
3. Rätta, ta bort eller lägg till ord.
4. Spara som utkast eller välj **Godkänn sida**.
5. Exportera till `exports/ordlista.txt`.

## Tester

```bash
pytest
```
