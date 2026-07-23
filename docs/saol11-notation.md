# SAOL 11: notation som parsern använder

Källor är avsnittet **Vägledning till ordlistan** på faksimilsidorna 8–13:

- https://runeberg.org/saol/11-6/0008.html
- https://runeberg.org/saol/11-6/0009.html
- https://runeberg.org/saol/11-6/0010.html
- https://runeberg.org/saol/11-6/0012.html
- https://runeberg.org/saol/11-6/0013.html

## Grundregler

- Substantiv anger bestämd singular och obestämd, ibland även bestämd, plural.
- Adjektiv anger neutrum, ibland plural, samt komparativ när den används.
- Regelbundna verb anger främst imperfekt; oregelbundna verb ger flera former.
- `=` betyder att böjningsformen är identisk med uppslagsformen.
- `|` avskiljer den del av uppslagsordet som ett följande `-` ersätter.
  Exempel: `amp|el -elt -la -lare` ger `ampelt`, `ampla`, `amplare`.
- `~` ersätter hela närmast föregående uppslagsord, inte bara delen före `|`.
- Hakparenteser anger växelformer. `-[e]n` ger både `-n` och `-en`.
- Rund parentes kring slutdel anger att ordet främst förekommer med denna del.

## Avsiktlig försiktighet

SAOL utelämnar böjningsuppgifter i vissa regelbundna fall, särskilt för
sammansättningar och ofta för substantiv på `-het`, `-ing` och `-ning`.
Den första parsern i `scripts/inflection_review.py` genererar därför bara
former som stöds direkt av den tryckta notationen. Implicit regelbunden
böjning byggs som ett separat och mätbart lager.
