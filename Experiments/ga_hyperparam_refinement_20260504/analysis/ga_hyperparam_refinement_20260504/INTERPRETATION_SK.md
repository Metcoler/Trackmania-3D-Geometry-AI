# Interpretácia GA Hyperparameter Refinement Sweepu

## Rozsah analýzy

Táto analýza používa iba nové refinement runy:

- `Experiments/runs_ga_hyperparam/pc1_selection_refinement_seed_2026050401`
- `Experiments/runs_ga_hyperparam/pc2_mutation_refinement_seed_2026050401`

Staršie gridy so širším a horším rozsahom nie sú miešané do záverov. Cieľom je čisto odpovedať, čo ukázal tento jemnejší rozsah.

Kontrola integrity:

- Dokončené runy: `54/54`
- Selection refinement: `24` runov
- Mutation refinement: `30` runov
- Elite cache: `0` cached evaluations vo všetkých runoch
- Spoločný baseline: fixed FPS `100`, AABB-clearance lidar, binary gas/brake, max time `30`, reward `(finished, progress, -time, -crashes)`

## Selection Pressure

Najsilnejší výsledok je jednoznačne:

`population=48`, `parent_count=14`, `elite_count=1`

Kľúčové metriky:

- Prvý finish: generácia `105`
- Najlepší čas: `17.70 s`
- Last50 finish rate: `40.83 %`
- Last50 mean dense progress: `55.90`
- Last50 crash rate: `58.79 %`
- Last50 timeout rate: `0.38 %`
- Last50 penalized mean time: `26.23 s`

Druhý najsilnejší kandidát bol `parent_count=10`, `elite_count=1`, ale mal nižší late finish rate (`31.75 %`) a vyšší crash rate (`67.00 %`). Konfigurácia `parent_count=12`, `elite_count=2` našla prvý finish najskôr, už v generácii `86`, ale v závere bola menej stabilná než `14/1`.

Interpretácia:

- Pre túto úlohu sa javí ako dôležité mať relatívne úzky elite set. Jeden elitný jedinec zachová smer učenia, ale nezamrazí populáciu príliš skoro.
- `parent_count=14` dáva dobrý kompromis medzi evolučným tlakom a diverzitou.
- Príliš veľa rodičov alebo príliš veľa elít často rozriedi selekčný tlak alebo udržiava priveľa slabších stratégií.
- Najlepší selection výsledok leží na hrane `elite_count=1`, takže do finálneho dôkazu treba tento záver potvrdiť aspoň druhým seedom.

## Mutation Probability A Sigma

Najlepší vnútorný stabilný bod:

`mutation_prob=0.10`, `mutation_sigma=0.25`

Kľúčové metriky:

- Prvý finish: generácia `132`
- Najlepší čas: `18.80 s`
- Last50 finish rate: `27.63 %`
- Last50 mean dense progress: `53.18`
- Last50 crash rate: `69.50 %`
- Last50 penalized mean time: `28.13 s`

Najlepší kompromis podľa pomocného skóre:

`mutation_prob=0.05`, `mutation_sigma=0.325`

Kľúčové metriky:

- Prvý finish: generácia `154`
- Najlepší čas: `18.70 s`
- Last50 finish rate: `27.38 %`
- Last50 mean dense progress: `57.44`
- Last50 crash rate: `72.38 %`
- Last50 timeout rate: `0.25 %`
- Last50 penalized mean time: `28.05 s`

Interpretácia:

- `sigma=0.20` je v tomto refinement rozsahu príliš slabá: prakticky nevytvára užitočnú exploráciu.
- Dobrá oblasť vyzerá skôr ako menej časté, ale výraznejšie mutácie.
- `prob=0.10, sigma=0.25` je bezpečnejší default, lebo je vnútorný bod gridu a našiel finish skôr.
- `prob=0.05, sigma=0.325` je zaujímavý kandidát, ale leží na hrane gridu. To znamená, že nemusíme mať ešte zachytené optimum; možno by stálo za to testovať aj `sigma=0.35` alebo veľmi blízke hodnoty.
- Vyššie pravdepodobnosti `0.125` a `0.15` často pôsobia deštruktívne, najmä pri vyššej sigme.

## Praktický Verdikt

Ako nový rozumný baseline pre ďalšie TM2D a live TM experimenty by som použil:

`population=48`, `parent_count=14`, `elite_count=1`, `mutation_prob=0.10`, `mutation_sigma=0.25`

Toto je konzervatívny kandidát: kombinuje najlepší selection pressure s najstabilnejším vnútorným mutation bodom.

Ako riskantnejší/exploračný kandidát:

`population=48`, `parent_count=14`, `elite_count=1`, `mutation_prob=0.05`, `mutation_sigma=0.325`

Tento kandidát môže byť dobrý, ale potrebuje potvrdenie, pretože mutation optimum vyšlo na hrane testovaného rozsahu.

Pre mutation decay dáva zmysel začať viac exploračne a končiť jemnejšie, napríklad:

- štart: `mutation_prob=0.10`, `mutation_sigma=0.30`
- minimum: `mutation_prob=0.05`, `mutation_sigma=0.25`

Toto rešpektuje zistenie, že slabá sigma `0.20` nestačí a že príliš vysoká pravdepodobnosť mutácie rozbíja dobré riešenia.

## Ďalší Experiment

Pred diplomovkovým záverom by som nepovažoval tento single-seed screening za finálny dôkaz. Najlepší ďalší krok je malý kombinovaný potvrdzovací sweep s novým seedom:

Selection kandidáti:

- `parent_count=14`, `elite_count=1`
- `parent_count=10`, `elite_count=1`
- `parent_count=12`, `elite_count=2`

Mutation kandidáti:

- `mutation_prob=0.10`, `mutation_sigma=0.25`
- `mutation_prob=0.075`, `mutation_sigma=0.30`
- `mutation_prob=0.05`, `mutation_sigma=0.325`
- voliteľne `mutation_prob=0.05`, `mutation_sigma=0.35`

To je 9 až 12 behov podľa toho, či pridáme `0.05/0.35`. Tento experiment by už testoval kombinácie naraz, nie oddelene selection a mutation pri starých fixných hodnotách.

## Thesis Poznámka

Do práce by som tento experiment prezentoval ako screening hyperparametrov, nie ako absolútne finálne optimum. Najdôležitejší poznatok je kvalitatívny: pre daný genóm a AABB-lidar prostredie funguje lepšie mierne silnejší selekčný tlak s veľmi malou elitou a mutácia typu “menej častá, ale nie príliš slabá”.
