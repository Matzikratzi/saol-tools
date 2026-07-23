# Plan: fullständig spelordlista från SAOL 11

Målet är en reproducerbar ordlista som innehåller samtliga godtagna uppslagsord
och deras böjningsformer, inte bara artikelrubrikerna.

Arbetet delas upp eftersom varje steg behöver kunna mätas separat:

1. **Artikelstarter**
   - hitta rätt tryckt rad och indenteringsnivå,
   - utvärdera mot `data/facit_sidor.txt`,
   - redovisa precision, recall och osäkra matchningar.
2. **Artikeltext**
   - knyt fortsättningsrader till rätt artikel,
   - hantera spalt- och sidbrytningar,
   - bevara originalets lodstreck, homonymnummer och böjningsnotation.
3. **Böjningsanalys**
   - maskinläs instruktionerna före sida 19,
   - implementera och testa notationens regler,
   - skilj uttryckliga former från former som genereras av en regel.
4. **Spelord**
   - generera alla böjningsformer,
   - normalisera typografiska markörer men bevara å, ä och ö,
   - ta bort dubletter och kontrollera alfabetisk ordning.
5. **Kvalitetskontroll**
   - kör flera OCR-tolkningar där de är oense,
   - flagga låg säkerhet, ovanliga tecken och brutna böjningsmönster,
   - exportera både ordlistan och en separat granskningslista.

Planerade slutprodukter:

- `saol11-spelord.txt`: ett normaliserat ord per rad,
- `saol11-artiklar.json`: källform, böjningsanalys, sida och säkerhet,
- `saol11-osakra.csv`: fall som behöver granskas,
- kod och tester för att återskapa samtliga filer.

ML-piloten klassificerar endast artikelstarter. Den ska inte ensam avgöra
böjningsformer eller korrigera avvikande indentering i originalet.
