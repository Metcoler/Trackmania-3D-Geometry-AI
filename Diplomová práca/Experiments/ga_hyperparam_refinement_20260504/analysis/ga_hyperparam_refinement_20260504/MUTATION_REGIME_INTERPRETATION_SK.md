# Mutation Probability / Sigma Regime Interpretation

Tento doplnok interpretuje iba mutation refinement cast experimentu `ga_hyperparam_refinement_20260504`. Vsetky behy pouzivali rovnaky baseline: `population=48`, `parent_count=16`, `elite_count=4`, fixed `100 Hz`, AABB-clearance lidar, binary gas/brake, no elite cache a ranking `(finished, progress, -time, -crashes)`.

## Prakticky verdikt

| Ucel | Odporucanie | Preco |
| --- | --- | --- |
| Rychla exploracia | `prob=0.10`, `sigma=0.25-0.30` | `prob=0.10` mal najlepsi agregovany skory finish: najlepsi first finish v generacii `132` a median first finish `135`. Z konkretnych bodov `0.10/0.25` nasiel finish skoro a ostal stabilny; `0.10/0.275` bol rovnako skory, ale horsi v late stabilite; `0.10/0.30` nasiel finish v generacii `138`, ale uz mal vyrazne vyssi crash profil. |
| Bezpecny baseline | `prob=0.10`, `sigma=0.25` | Najlepsi vnutorny bod gridu: first finish `132`, last50 finish rate `27.63 %`, last50 progress `53.18`, crash rate `69.50 %`, compromise `0.106`. Nie je na okraji gridu, preto je obhajitelnejsi ako edge kandidat. |
| Rare-large / neskora diverzita | `prob=0.05`, `sigma=0.325` | Najvyssi compromise `0.124` a najlepsi late progress `57.44`, ale first finish az `154` a bod lezi na hrane gridu. Toto nie je klasicky jemny fine-tune; je to skor menej casta, ale vyrazna mutacia, ktora vie zachovat vacsiu diverzitu. |
| Jemny fine-tune | zatial nepreukazany tymto gridom | Slaba sigma `0.20` zlyhala takmer napriec pravdepodobnostami a vacsina nizkych sigma behov nenasla pouzitelny finish. Ak chceme tvrdit `sigma=0.15-0.20` ako fine-tune, treba osobitny continuation experiment zo silneho checkpointu, nie from-scratch grid. |

## Co vyplynulo podla samostatnej probability

- `prob=0.10` je najlepsi kandidat na zaklad: ma najlepsi agregovany first finish (`132` best, `135` median), najvyssi priemerny late finish rate v skupine a najlepsi vnutorny bod `0.10/0.25`.
- `prob=0.05` nie je dobry vseobecny start sam osebe, ale pri vysokej sigme `0.325` dal najlepsi kompromis a late progress. Preto ho berieme ako rare-large/diverzifikacny kandidat, nie ako automaticky fine-tune.
- `prob=0.125` a `prob=0.15` uz vyzeraju destruktivne: aj ked niektore behy finish nasli, late finish rate a compromise vyrazne klesli.

## Co vyplynulo podla samotnej sigmy

- `sigma=0.20` je v tejto ulohe prilis slaba pre from-scratch GA: max last50 finish rate bol iba `0.08 %` a priemerny compromise bol najhorsi.
- `sigma=0.25` obsahuje najlepsi stabilny baseline bod `0.10/0.25`, ale ako samotna sigma nie je univerzalne dobra, lebo nizke alebo vysoke probability ju vedia pokazit.
- `sigma=0.30` ma najlepsi agregovany signal pre sirsiu exploraciu: finish naslo `80 %` behov v tejto sigma skupine a priemerny late finish rate bol najvyssi, hoci top bod nebol taky dobry ako `0.10/0.25` alebo `0.05/0.325`.
- `sigma=0.325` vie byt velmi dobra pri malej probability, ale pri vyssej probability sa sprava destruktivne. Preto ju treba viazat na nizku `prob`, nie pouzivat plosne.

## Odporucane nastavenia pre dalsie experimenty

- Experimental baseline: `mutation_prob=0.10`, `mutation_sigma=0.25`.
- Exploration/decay start: `mutation_prob=0.10`, `mutation_sigma=0.30`.
- Decay minimum podla dat: `mutation_prob=0.05`, `mutation_sigma=0.25`; neist automaticky na `sigma=0.20`, lebo ta z from-scratch dat vyzerala prilis slaba.
- Rare-large alternativa: `mutation_prob=0.05`, `mutation_sigma=0.325`, ale opakovat druhym seedom alebo otestovat `0.05/0.35`, lebo optimum je na hrane gridu.

## Vygenerovane subory

- `mutation_probability_regime_summary.csv`
- `mutation_sigma_regime_summary.csv`
- `mutation_role_candidates.csv`
- `mutation_regime_role_map.png`
