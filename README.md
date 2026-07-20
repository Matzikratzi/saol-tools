# SAOL-tools

Ett fristående webbaserat granskningsverktyg för att skapa en ren ordlista från den skannade SAOL 11 hos Projekt Runeberg.

Verktyget använder själva sidbilden som källa. Lokal Tesseract-OCR körs i hOCR-läge för att identifiera ord som är tryckta i halvfet eller extra fet stil. Den tillfälliga OCR- och layoutinformationen lagras inte. Databasen innehåller bara ord, sidreferenser, status och korrläsarens namn.

## Tolkningsprincip

Enligt vägledningen på bokens sidor 8–13 gäller bland annat:

- alla ord i halvfet stil är uppslagsord, även sammansättningar
- extra fet stil markerar bara det första uppslagsordet i ett stycke
- flerordiga uttryck kan vara uppslagsord
- lodstreck, bindestreck, parenteser och andra specialtecken påverkar hur sammansättningar och varianter ska rekonstrueras

Automatiken föreslår därför ord utifrån identifierad fetstil. Sammansättningsfragment och flerordiga uttryck markeras som misstänkta för manuell kontroll mot bilden.

## Installation på macOS

Installera först Tesseract och svenska språkdata:

```bash
brew install tesseract tesseract-lang
```

Installera och starta sedan applikationen:

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
2. Kontrollera alla föreslagna uppslagsord mot fetstilen i den skannade sidan.
3. Rekonstruera sammansättningar och varianter enligt reglerna på sidorna 8–13.
4. Rätta, ta bort eller lägg till ord.
5. Spara som utkast eller välj **Godkänn sida**.
6. Exportera till `exports/ordlista.txt`.

## Tester

```bash
pytest
```
