# ML-pilot för artikelstarter

Piloten jämför den geometriska T-regeln med logistisk regression och gradient
boosting. Den återanvänder samma TIFF, deskew och Tesseract-tolkning som
`scripts/debug_runeberg_ocr.py`.

## Köra

```bash
python3 -m pip install -r requirements-ml.txt
PYTHONPATH=. python3 scripts/article_start_ml.py
```

Första körningen OCR-läser sidorna 19–22 och kan därför ta tid. Siddata sparas
i `.article-start-ml-cache`; senare körningar återanvänder cachen. Använd
`--refresh` för att göra om OCR.

Körningen skapar:

- `article-start-features.json` med egenskaper, facit och modellbeslut,
- `article-start-ml-report.html` med mått och samtliga felklassningar.

Varje sida utvärderas med en modell som tränats på de övriga tre sidorna.
Rapportens resultat är alltså inte träningsresultat.

## Facit

`data/facit_sidor.txt` innehåller rader på den avsedda A-nivån. Det är
indenteringen i den tryckta SAOL-sidan som är facit, även när ett lexikografiskt
huvudord verkar ha fått avvikande indrag. Lodstreck behöver inte skrivas.

Fler sidor kan läggas till med formatet:

```text
sida 23:

vänster:
förstaordet
andraordet

höger:
tredjeordet
```

## Avgränsning

Piloten avgör bara vilka tryckta rader som börjar en artikel på A-nivån.
Fortsatt plan för full artikeltext och samtliga böjningsformer finns i
`docs/ordlista-plan.md`.
