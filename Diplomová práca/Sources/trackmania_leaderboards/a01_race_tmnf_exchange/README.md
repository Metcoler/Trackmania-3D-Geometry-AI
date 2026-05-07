# A01-Race leaderboard data

Tento balík obsahuje verejné replay/leaderboard dáta pre mapu `A01-Race`
v TrackMania Nations Forever Exchange.

- Trať: `A01-Race`, `TrackId=2233`
- Stránka trate: https://tmnf.exchange/trackshow/2233
- API dokumentácia: https://api2.mania.exchange/Method/Index/45
- Endpoint: `https://tmnf.exchange/api/replays`
- Dátum prístupu: 2026-05-07

Skript `generate_a01_human_times.py` sťahuje replaye cez stránkovanie
parametrom `after`. API dokumentácia upozorňuje, že výsledky obsahujú aj
prekonané replaye toho istého jazdca a aj staršie verzie trate. Preto sa
pre obrázok používajú iba záznamy, ktoré spĺňajú:

- `TrackAt` je najnovšia dostupná verzia trate v stiahnutých dátach,
- `Validated = true`,
- `ReplayRespawns = 0`,
- `Position != null`,
- použije sa horných `1000` záznamov podľa leaderboard pozície.

Tieto dáta nie sú reprezentatívnou vzorkou všetkých hráčov Trackmanie.
Sú to leaderboard záznamy, teda výkonnostná vzorka hráčov, ktorí nahrali
replay na TMNF-X. V práci ich používame iba ako kontext k tomu, že aj
rozdiely v desatinách sekundy môžu v Trackmanii znamenať veľký posun v
poradí.
