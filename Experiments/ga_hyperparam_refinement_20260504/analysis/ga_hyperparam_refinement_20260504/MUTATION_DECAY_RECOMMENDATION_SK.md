# Mutation Decay Recommendation

Tento doplnok odpoveda na prakticku otazku: ake hodnoty pouzit pre `mutation_prob`, `mutation_sigma` a ich decay. Vychadza iba z mutation refinement gridu z experimentu `ga_hyperparam_refinement_20260504`.

Vsetky porovnavane behy mali rovnaky baseline: `population=48`, `parent_count=16`, `elite_count=4`, fixed `100 Hz`, AABB-clearance lidar, binary gas/brake, no elite cache a ranking `(finished, progress, -time, -crashes)`.

## Odporucany decay pre 200 generacii

Pre bezny 200-generacny TM2D run odporucam:

```text
mutation_prob       = 0.10
mutation_prob_min   = 0.05
mutation_prob_decay = 0.9965229

mutation_sigma       = 0.30
mutation_sigma_min   = 0.25
mutation_sigma_decay = 0.9990842
```

Tieto hodnoty znamenaju, ze trening zacne v exploracnej oblasti a do generacie 200 plynulo prejde k jemnejsej stabilnej oblasti.

Priebeh vybranych generacii:

| generation | mutation_prob | mutation_sigma |
| ---: | ---: | ---: |
| 1 | 0.1000 | 0.3000 |
| 50 | 0.0843 | 0.2868 |
| 100 | 0.0708 | 0.2740 |
| 132 | 0.0634 | 0.2661 |
| 150 | 0.0595 | 0.2617 |
| 200 | 0.0500 | 0.2500 |

## Preco prave tieto hodnoty

### Start probability: `0.10`

`prob=0.10` mal najlepsi agregovany signal pre skore najdenie finishu:

- najlepsi first finish v tejto probability skupine: generacia `132`
- median first finish: generacia `135`
- najlepsi vnutorny bod gridu: `prob=0.10`, `sigma=0.25`

Vyssie hodnoty `0.125` a `0.15` sice obcas finish nasli, ale v late faze vyzerali destruktivnejsie: mali nizsi finish rate, vyssi crash profil a horsi compromise score. Preto `prob=0.10` berieme ako horne bezpecne exploration nastavenie, nie `0.15`.

### Minimum probability: `0.05`

`prob=0.05` sam o sebe nebol dobry vseobecny start, ale v kombinacii so silnejsou sigmou `0.325` dal najlepsi compromise score a najlepsi late progress:

- `prob=0.05`, `sigma=0.325`
- first finish: generacia `154`
- best finish time: `18.70 s`
- last50 finish rate: `27.38 %`
- last50 mean dense progress: `57.44`
- compromise score: `0.124`

Interpretacia: mala pravdepodobnost mutacie vie byt dobra v neskorsej faze, ked uz nechceme menit privela vah naraz. Nie je to vsak dokaz, ze `prob=0.05` je dobry from-scratch start.

### Start sigma: `0.30`

`sigma=0.30` mala najlepsi agregovany exploracny signal medzi samotnymi sigma skupinami:

- finish naslo `80 %` behov v tejto sigma skupine
- najlepsi first finish v skupine: generacia `138`
- najvyssi priemerny late finish rate medzi sigma skupinami

Konkretny bod `prob=0.10`, `sigma=0.30` bol uz crashovo agresivnejsi a menej stabilny nez `0.10/0.25`, preto ho nechceme ako konstantne nastavenie na cely trening. Ako start decay krivky je vsak uzitocny, lebo ma exploracnu silu.

### Minimum sigma: `0.25`

`sigma=0.25` obsahuje najlepsi stabilny vnutorny bod gridu:

- `prob=0.10`, `sigma=0.25`
- first finish: generacia `132`
- last50 finish rate: `27.63 %`
- last50 mean dense progress: `53.18`
- crash rate: `69.50 %`

Naopak `sigma=0.20` bola v tomto from-scratch experimente prilis slaba: prakticky nevytvarala uzitocnu exploraciu a mala najhorsi agregovany compromise. Preto decay minimum nenastavujeme na `0.20`, aj ked by to intuitivne znelo ako jemny fine-tune.

## Poznamka k fine-tune

Tento grid bol from-scratch experiment. Z neho nevieme ferovo tvrdit, ze `sigma=0.15` alebo `sigma=0.20` je dobry fine-tune, pretoze male sigmy nemali sancu opravit uz existujuce dobre riesenie; museli ho najskor same najst.

Ak chceme dokazat skutocny fine-tune rezim, treba samostatny continuation experiment:

1. zobrat silny checkpoint po najdeni finishera,
2. pokracovat s mensimi hodnotami `sigma=0.15-0.25`,
3. merat zlepsenie casu, finish stability a crash rate.

Dovtedy je najbezpecnejsie tvrdenie:

```text
sigma_min = 0.25 je dolna hranica podlozena from-scratch gridom.
mensie sigmy su hypoteza pre continuation fine-tune, nie potvrdeny default.
```

## Alternativny rare-large decay

Ak by sme chceli testovat variant inspirovany najlepsim edge bodom `0.05/0.325`, mozeme pouzit:

```text
mutation_prob       = 0.05
mutation_prob_min   = 0.05
mutation_prob_decay = 1.0

mutation_sigma       = 0.325
mutation_sigma_min   = 0.25
mutation_sigma_decay = 0.9986825
```

Toto nie je hlavny odporucany baseline, pretoze bod `0.05/0.325` lezi na hrane gridu a prvy finish nasiel neskor. Je to vsak zaujimavy kandidat pre diverzifikacny beh alebo pre druhy seed.

## 300-generacny variant

Ak trening pobezi 300 generacii a chceme rovnaky start aj minimum, decay treba spomalit:

```text
mutation_prob_decay  = 0.9976845
mutation_sigma_decay = 0.9993904
```

Tento variant dosiahne `prob=0.05` a `sigma=0.25` az na konci generacie 300.

## Subory

- `mutation_decay_recommendations.csv`
- `mutation_decay_recommendation.png`
