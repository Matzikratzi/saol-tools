# SAOL-tools

Ett fristående webbaserat verktyg för att skapa en ren ordlista från den OCR-tolkade skanningen av SAOL 11 hos Projekt Runeberg.

Verktyget är **inte** en digital utgåva av SAOL. Runebergs OCR används endast i minnet för att föreslå uppslagsord. Därefter sparas bara ord, sidreferenser, granskningsstatus och anteckningar.

## Funktioner

- visar skannad sida och föreslagna uppslagsord sida vid sida,
- markerar misstänkta OCR-fel och brott mot alfabetisk ordning,
- låter granskaren rätta, lägga till, förkasta eller ta bort kandidater,
- har två steg: förstagranskning och slutgodkännande,
- exporterar bara godkända ord från slutgodkända sidor,
- exporterar till TXT, JSON och CSV,
- lagrar aldrig den fullständiga OCR-texten.

## Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Öppna `http://127.0.0.1:8000`.

## Arbetsflöde

1. Ange ett Runeberg-sidnummer och välj **Öppna/importera**.
2. Kontrollera kandidaterna mot den skannade sidan.
3. Rätta, förkasta, ta bort eller lägg till ord.
4. Spara utkast eller välj **Förstagranska**.
5. En andra kontroll kan välja **Slutgodkänn**.
6. Exportera ordlistan som TXT, JSON eller CSV.

## Data

SQLite-databasen skapas i `data/saol-tools.db`. Den är ignorerad av Git. Exporter hamnar i `exports/` och är också ignorerade.

## Tester

```bash
pytest
```
