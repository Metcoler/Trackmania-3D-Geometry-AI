# Experiment Reports Index

This file aggregates the relevant experiment reports used while writing the thesis.
It is a working source document, not the final thesis chapter.

Generated: `2026-05-05T23:10:08`

## Index

| # | Experiment | Status | Source | Note |
| ---: | --- | --- | --- | --- |
| 1 | [Lexicographic Reward Sweep With AABB Lidar](#lexicographic-reward-sweep-with-aabb-lidar) | thesis-grade | `Diplomová práca/Experiments/lex_reward_aabb_lidar_fixed100_20260503/REPORT.md` | Reward tuple selection; current best lexicographic tuple. |
| 2 | [Lexicographic Reward Sweep Detailed Analysis](#lexicographic-reward-sweep-detailed-analysis) | thesis-grade detail | `Diplomová práca/Experiments/lex_reward_aabb_lidar_fixed100_20260503/analysis/lex_sweep_aabb_lidar_fixed100_20260503/REPORT.md` | Detailed plots and metric interpretation for the reward sweep. |
| 3 | [GA Hyperparameter Refinement](#ga-hyperparameter-refinement) | thesis-grade | `Diplomová práca/Experiments/ga_hyperparam_refinement_20260504/REPORT.md` | Refined parent/elite and mutation baseline evidence. |
| 4 | [GA Hyperparameter Refinement Interpretation](#ga-hyperparameter-refinement-interpretation) | thesis-grade detail | `Diplomová práca/Experiments/ga_hyperparam_refinement_20260504/analysis/ga_hyperparam_refinement_20260504/INTERPRETATION_SK.md` | Slovak interpretation of parent/elite and mutation ranges. |
| 5 | [GA Mutation Grid](#ga-mutation-grid) | thesis-grade | `Diplomová práca/Experiments/ga_mutation_grid_20260504/REPORT.md` | Original mutation probability/sigma grid evidence. |
| 6 | [GA Mutation Grid Detailed Analysis](#ga-mutation-grid-detailed-analysis) | thesis-grade detail | `Diplomová práca/Experiments/ga_mutation_grid_20260504/analysis/ga_hyperparam_mutation_grid_20260504/REPORT.md` | Detailed mutation grid report. |
| 7 | [Architecture And Activation Ablation](#architecture-and-activation-ablation) | thesis-grade | `Diplomová práca/Experiments/ga_architecture_activation_ablation_20260504/REPORT.md` | Closed-loop evidence for relu,tanh and 32x16/48x24 tradeoff. |
| 8 | [AABB Vehicle Hitbox](#aabb-vehicle-hitbox) | thesis-grade | `Diplomová práca/Experiments/vehicle_hitbox_aabb_20260503/REPORT.md` | Justification for AABB-clearance lidar. |
| 9 | [AABB Vehicle Hitbox Detailed Report](#aabb-vehicle-hitbox-detailed-report) | thesis-grade detail | `Diplomová práca/Experiments/vehicle_hitbox_aabb_20260503/analysis/vehicle_hitbox_20260503_smoke/REPORT.md` | Detailed hitbox sanity checks. |
| 10 | [Supervised Map Specialists](#supervised-map-specialists) | thesis-grade visualization | `Diplomová práca/Experiments/supervised_map_specialists_20260505/REPORT.md` | Teacher/agent path plots and supervised specialist result package. |
| 11 | [Latest GA Training Improvements](#latest-ga-training-improvements) | thesis-grade and diagnostic mix | `Diplomová práca/Experiments/training_improvements_20260505/analysis/latest_training_results_20260505/REPORT.md` | Elite cache, decay, mirror, max-touch, MOO and live TM summary. |
| 12 | [RL Reward-Equivalent Sweep](#rl-reward-equivalent-sweep) | comparison / useful negative | `Diplomová práca/Experiments/rl_reward_equivalent_sweep_20260505/analysis/rl_reward_equivalent_sweep_20260505/DEEP_REPORT.md` | PPO vs SAC vs TD3 under reward-equivalent scalarization. |
| 13 | [GA Supervised-Seeded Hybrid](#ga-supervised-seeded-hybrid) | thesis-grade positive hybrid | `Diplomová práca/Experiments/ga_supervised_seeded_20260505/analysis/ga_supervised_seeded_20260505/DEEP_REPORT.md` | Behavior cloning initialization plus GA fine-tuning. |
| 14 | [Supervised Physics Tick Distribution](#supervised-physics-tick-distribution) | supporting diagnostic | `Experiments/analysis/supervised_physics_ticks_20260504/REPORT.md` | Timing evidence for physics tick delay and 100Hz/variable tick discussion. |

## Reading Notes

- `progress` means dense/continuous progress unless explicitly named `block_progress`.
- Reports from `Experiments/_discarded`, smoke tests, and temporary bounce/butterfly probes are intentionally excluded.
- Thesis packages are preferred over duplicate working reports when both exist.


---

## Lexicographic Reward Sweep With AABB Lidar

- Status: `thesis-grade`
- Source: `Diplomová práca/Experiments/lex_reward_aabb_lidar_fixed100_20260503/REPORT.md`
- Note: Reward tuple selection; current best lexicographic tuple.

# Lexicographic reward sweep with AABB-clearance lidar

## Status

- Experiment ID: `lex_reward_aabb_lidar_fixed100_20260503`
- Category: `thesis_grade`
- Status: `single_seed_strong_signal`
- Curated at: `2026-05-04T13:13:05`

## Thesis Relevance

Main reward-function comparison after switching to AABB-clearance lidar, fixed FPS 100, binary gas/brake, no elite cache.

## Interpretation

Current evidence favors (finished, progress, -time, -crashes) as the base lexicographic GA ranking tuple.

## Package Contents

- `runs/`: selected raw run logs copied from the working experiment folders.
- `analysis/`: summary tables, reports, and generated plots.
- `scripts/`: scripts needed to reproduce or analyze the experiment.
- `metadata.json`: source paths, keywords, and curation metadata.

## Notes

Large aggregate files such as `combined_individual_metrics.csv` are intentionally not copied into the thesis package. They remain local working artifacts if needed.


---

## Lexicographic Reward Sweep Detailed Analysis

- Status: `thesis-grade detail`
- Source: `Diplomová práca/Experiments/lex_reward_aabb_lidar_fixed100_20260503/analysis/lex_sweep_aabb_lidar_fixed100_20260503/REPORT.md`
- Note: Detailed plots and metric interpretation for the reward sweep.

# Lexicographic reward sweep analysis: AABB lidar, fixed FPS 100

Single-seed analysis for `Experiments/runs_ga/lex_sweep_aabb_lidar_fixed100_seed_2026050306`.

Common configuration: AI Training #5, TM2D AABB-clearance lidar, fixed FPS 100, binary gas/brake, no elite cache, dense progress, population 48, elite 4, parents 16, mutation prob 0.2, mutation sigma 0.2, max time 30s, 300 generations, 8 workers.

## Summary table

| Ranking key | First finish gen | Total finish individuals | Last50 finish/gen | Best finish time | Last50 mean dense progress | Final mean dense progress | Final finish count | Total crashes | Total timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `(finished, progress)` | 216 | 501 | 6.62 | 22.390 | 43.46 | 55.48 | 5 | 13650 | 249 |
| `(finished, progress, -time)` | 247 | 389 | 7.64 | 17.840 | 45.66 | 41.41 | 7 | 13874 | 137 |
| `(finished, progress, -time, -crashes)` | 137 | 2209 | 15.46 | 17.330 | 49.39 | 50.37 | 15 | 12090 | 101 |
| `(finished, progress, -crashes, -time)` | 136 | 2246 | 14.70 | 17.970 | 54.04 | 59.27 | 15 | 11935 | 219 |

## Ranking by practical usefulness

1. `(finished, progress, -time, -crashes)` - last50 finish/gen `15.46`, best time `17.330s`, last50 mean progress `49.39%`.
2. `(finished, progress, -crashes, -time)` - last50 finish/gen `14.70`, best time `17.970s`, last50 mean progress `54.04%`.
3. `(finished, progress, -time)` - last50 finish/gen `7.64`, best time `17.840s`, last50 mean progress `45.66%`.
4. `(finished, progress)` - last50 finish/gen `6.62`, best time `22.390s`, last50 mean progress `43.46%`.

## Generated graphs

- `progress_curves.png`
- `finish_count_moving_average.png`
- `best_finish_time_so_far.png`
- `final_population_progress_boxplot.png`
- `outcome_totals_and_last50.png`
- `finish_stability_vs_best_time.png`

## Interpretation notes

- This analysis is intentionally single-seed until the `_2` replication sweep is available. Treat the conclusion as a strong current signal, not yet a thesis-grade reproducibility claim.
- The best base reward should not be selected only by the single best time. A useful base should repeatedly produce finishers in late generations and preserve high mean dense progress.
- Crash-aware terms should be interpreted carefully: in a lexicographic tuple, `-crashes` after `-time` only affects policies tied on finish/progress/time; before `-time` it can prefer safer but slower behavior.


---

## GA Hyperparameter Refinement

- Status: `thesis-grade`
- Source: `Diplomová práca/Experiments/ga_hyperparam_refinement_20260504/REPORT.md`
- Note: Refined parent/elite and mutation baseline evidence.

# GA Hyperparameter Refinement Sweep 2026-05-04

## Status

Thesis-grade screening experiment. The sweep is complete and internally consistent:

- `54/54` runs completed.
- `24` selection-pressure refinement runs.
- `30` mutation probability/sigma refinement runs.
- `cached_evaluations = 0` in all runs.
- Shared baseline: fixed FPS `100`, AABB-clearance lidar, binary gas/brake, max time `30`, dense progress, ranking `(finished, progress, -time, -crashes)`.

## Main Result

The best practical configuration from this refinement is:

`population=48`, `parent_count=14`, `elite_count=1`, `mutation_prob=0.10`, `mutation_sigma=0.25`

The strongest selection-pressure run was:

`parent_count=14`, `elite_count=1`

Key metrics:

- First finish generation: `105`
- Best finish time: `17.70 s`
- Last50 finish rate: `40.83 %`
- Last50 mean dense progress: `55.90`
- Last50 crash rate: `58.79 %`
- Last50 penalized mean time: `26.23 s`

For mutation parameters, the safest interior candidate is:

`mutation_prob=0.10`, `mutation_sigma=0.25`

The strongest edge candidate is:

`mutation_prob=0.05`, `mutation_sigma=0.325`

This edge result is useful, but it should be repeated or refined before treating it as the final optimum.

## Interpretation

The experiment suggests that this GA benefits from moderate selection pressure and a very small elite set. A single elite preserves the best direction without freezing too much of the population. The mutation results point toward less frequent but stronger mutations; too weak mutation (`sigma=0.20`) often failed to explore, while high mutation probability became destructive.

This package should be used as the main hyperparameter screening evidence instead of the older coarse selection grid. The older ratio-based selection analysis was moved to soft-delete because the refined `parent_count x elite_count` grid is more directly interpretable.

## Package Contents

- `runs/`: raw run directories for the selection and mutation refinement grids.
- `analysis/ga_hyperparam_refinement_20260504/`: generated summary tables, heatmaps, and detailed reports.
- `scripts/`: launch scripts and analyzer used to reproduce the package.

The redundant `combined_individual_metrics.csv` was not kept in the package because it was very large and all per-run individual metrics remain available inside `runs/`.

## Recommended Next Step

Use the configuration below as the next baseline:

`population=48`, `parent_count=14`, `elite_count=1`, `mutation_prob=0.10`, `mutation_sigma=0.25`

For a confirmation run with another seed, compare it against:

- `parent_count=10`, `elite_count=1`, `mutation_prob=0.10`, `mutation_sigma=0.25`
- `parent_count=14`, `elite_count=1`, `mutation_prob=0.05`, `mutation_sigma=0.325`
- optionally `parent_count=14`, `elite_count=1`, `mutation_prob=0.05`, `mutation_sigma=0.35`


---

## GA Hyperparameter Refinement Interpretation

- Status: `thesis-grade detail`
- Source: `Diplomová práca/Experiments/ga_hyperparam_refinement_20260504/analysis/ga_hyperparam_refinement_20260504/INTERPRETATION_SK.md`
- Note: Slovak interpretation of parent/elite and mutation ranges.

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


---

## GA Mutation Grid

- Status: `thesis-grade`
- Source: `Diplomová práca/Experiments/ga_mutation_grid_20260504/REPORT.md`
- Note: Original mutation probability/sigma grid evidence.

# GA mutation probability and sigma grid

## Status

- Experiment ID: `ga_mutation_grid_20260504`
- Category: `thesis_grade`
- Status: `screening_single_seed`
- Curated at: `2026-05-04T13:13:09`

## Thesis Relevance

Mutation probability/sigma screening for the selected reward tuple and baseline selection settings.

## Interpretation

Best region suggests mutating fewer weights with medium-to-larger steps, around mutation_prob=0.10 and mutation_sigma=0.25-0.30.

## Package Contents

- `runs/`: selected raw run logs copied from the working experiment folders.
- `analysis/`: summary tables, reports, and generated plots.
- `scripts/`: scripts needed to reproduce or analyze the experiment.
- `metadata.json`: source paths, keywords, and curation metadata.

## Notes

Large aggregate files such as `combined_individual_metrics.csv` are intentionally not copied into the thesis package. They remain local working artifacts if needed.


---

## GA Mutation Grid Detailed Analysis

- Status: `thesis-grade detail`
- Source: `Diplomová práca/Experiments/ga_mutation_grid_20260504/analysis/ga_hyperparam_mutation_grid_20260504/REPORT.md`
- Note: Detailed mutation grid report.

# GA Hyperparameter Sweep Analysis

## Input Roots

- `Experiments\runs_ga_hyperparam\pc1_mutation_grid_seed_2026050311`

## Summary

This analysis compares GA hyperparameters for the fixed reward tuple `(finished, progress, -time, -crashes)`.
The environment baseline is fixed FPS 100, AABB-clearance lidar, binary gas/brake, max time 30, dense progress, and disabled elite cache.

Loaded runs: `25`.
Incomplete runs: `0`.
Runs with cached elite evaluations: `0`.

This is a screening experiment. The tables below identify promising regions; they are not a thesis-final proof until the best candidates are repeated with another seed.

## Best Overall Candidates

| grid | grid_x | grid_y | compromise_score | mutation_prob | mutation_sigma | parent_count | elite_count | first_finish_generation | best_finish_time | last50_finish_rate | last50_mean_dense_progress | last50_crash_rate | last50_timeout_rate | last50_penalized_mean_time | max_cached_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mutation | 0.100 | 0.300 | 0.208 | 0.100 | 0.300 | 16 | 4 | 114.000 | 19.120 | 0.328 | 53.259 | 0.645 | 0.027 | 27.861 | 0 |
| mutation | 0.100 | 0.250 | 0.243 | 0.100 | 0.250 | 16 | 4 | 124.000 | 18.900 | 0.328 | 58.163 | 0.664 | 0.009 | 27.800 | 0 |
| mutation | 0.150 | 0.200 | 0.200 | 0.150 | 0.200 | 16 | 4 | 137.000 | 21.270 | 0.302 | 56.835 | 0.660 | 0.038 | 28.393 | 0 |
| mutation | 0.150 | 0.250 | 0.118 | 0.150 | 0.250 | 16 | 4 | 105.000 | 17.000 | 0.300 | 50.571 | 0.684 | 0.016 | 27.309 | 0 |
| mutation | 0.300 | 0.150 | 0.034 | 0.300 | 0.150 | 16 | 4 | 104.000 | 20.640 | 0.267 | 48.808 | 0.718 | 0.015 | 28.469 | 0 |
| mutation | 0.250 | 0.300 | -0.258 | 0.250 | 0.300 | 16 | 4 | 102.000 | 18.960 | 0.167 | 37.736 | 0.792 | 0.041 | 28.747 | 0 |
| mutation | 0.150 | 0.300 | -0.225 | 0.150 | 0.300 | 16 | 4 | 125.000 | 19.710 | 0.167 | 41.812 | 0.803 | 0.030 | 28.885 | 0 |
| mutation | 0.200 | 0.300 | -0.212 | 0.200 | 0.300 | 16 | 4 | 103.000 | 20.010 | 0.161 | 41.061 | 0.766 | 0.073 | 28.940 | 0 |
| mutation | 0.150 | 0.150 | -0.218 | 0.150 | 0.150 | 16 | 4 | 167.000 | 22.780 | 0.125 | 49.854 | 0.830 | 0.045 | 29.565 | 0 |
| mutation | 0.250 | 0.250 | -0.437 | 0.250 | 0.250 | 16 | 4 | 138.000 | 19.790 | 0.120 | 31.227 | 0.866 | 0.013 | 29.130 | 0 |

## Best Compromise Candidates

Compromise score is `last50_finish_rate + last50_mean_dense_progress / 100 - last50_crash_rate - 0.25 * last50_timeout_rate`.
It is a screening helper, not a final fitness value.

| grid | grid_x | grid_y | compromise_score | mutation_prob | mutation_sigma | parent_count | elite_count | first_finish_generation | best_finish_time | last50_finish_rate | last50_mean_dense_progress | last50_crash_rate | last50_timeout_rate | last50_penalized_mean_time | max_cached_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mutation | 0.100 | 0.250 | 0.243 | 0.100 | 0.250 | 16 | 4 | 124.000 | 18.900 | 0.328 | 58.163 | 0.664 | 0.009 | 27.800 | 0 |
| mutation | 0.100 | 0.300 | 0.208 | 0.100 | 0.300 | 16 | 4 | 114.000 | 19.120 | 0.328 | 53.259 | 0.645 | 0.027 | 27.861 | 0 |
| mutation | 0.150 | 0.200 | 0.200 | 0.150 | 0.200 | 16 | 4 | 137.000 | 21.270 | 0.302 | 56.835 | 0.660 | 0.038 | 28.393 | 0 |
| mutation | 0.150 | 0.250 | 0.118 | 0.150 | 0.250 | 16 | 4 | 105.000 | 17.000 | 0.300 | 50.571 | 0.684 | 0.016 | 27.309 | 0 |
| mutation | 0.300 | 0.150 | 0.034 | 0.300 | 0.150 | 16 | 4 | 104.000 | 20.640 | 0.267 | 48.808 | 0.718 | 0.015 | 28.469 | 0 |
| mutation | 0.200 | 0.300 | -0.212 | 0.200 | 0.300 | 16 | 4 | 103.000 | 20.010 | 0.161 | 41.061 | 0.766 | 0.073 | 28.940 | 0 |
| mutation | 0.150 | 0.150 | -0.218 | 0.150 | 0.150 | 16 | 4 | 167.000 | 22.780 | 0.125 | 49.854 | 0.830 | 0.045 | 29.565 | 0 |
| mutation | 0.150 | 0.300 | -0.225 | 0.150 | 0.300 | 16 | 4 | 125.000 | 19.710 | 0.167 | 41.812 | 0.803 | 0.030 | 28.885 | 0 |
| mutation | 0.250 | 0.300 | -0.258 | 0.250 | 0.300 | 16 | 4 | 102.000 | 18.960 | 0.167 | 37.736 | 0.792 | 0.041 | 28.747 | 0 |
| mutation | 0.300 | 0.200 | -0.409 | 0.300 | 0.200 | 16 | 4 | 142.000 | 23.510 | 0.108 | 36.331 | 0.876 | 0.016 | 29.491 | 0 |

## Mutation Grid Candidates

| grid | grid_x | grid_y | compromise_score | mutation_prob | mutation_sigma | parent_count | elite_count | first_finish_generation | best_finish_time | last50_finish_rate | last50_mean_dense_progress | last50_crash_rate | last50_timeout_rate | last50_penalized_mean_time | max_cached_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mutation | 0.100 | 0.300 | 0.208 | 0.100 | 0.300 | 16 | 4 | 114.000 | 19.120 | 0.328 | 53.259 | 0.645 | 0.027 | 27.861 | 0 |
| mutation | 0.100 | 0.250 | 0.243 | 0.100 | 0.250 | 16 | 4 | 124.000 | 18.900 | 0.328 | 58.163 | 0.664 | 0.009 | 27.800 | 0 |
| mutation | 0.150 | 0.200 | 0.200 | 0.150 | 0.200 | 16 | 4 | 137.000 | 21.270 | 0.302 | 56.835 | 0.660 | 0.038 | 28.393 | 0 |
| mutation | 0.150 | 0.250 | 0.118 | 0.150 | 0.250 | 16 | 4 | 105.000 | 17.000 | 0.300 | 50.571 | 0.684 | 0.016 | 27.309 | 0 |
| mutation | 0.300 | 0.150 | 0.034 | 0.300 | 0.150 | 16 | 4 | 104.000 | 20.640 | 0.267 | 48.808 | 0.718 | 0.015 | 28.469 | 0 |
| mutation | 0.250 | 0.300 | -0.258 | 0.250 | 0.300 | 16 | 4 | 102.000 | 18.960 | 0.167 | 37.736 | 0.792 | 0.041 | 28.747 | 0 |
| mutation | 0.150 | 0.300 | -0.225 | 0.150 | 0.300 | 16 | 4 | 125.000 | 19.710 | 0.167 | 41.812 | 0.803 | 0.030 | 28.885 | 0 |
| mutation | 0.200 | 0.300 | -0.212 | 0.200 | 0.300 | 16 | 4 | 103.000 | 20.010 | 0.161 | 41.061 | 0.766 | 0.073 | 28.940 | 0 |

## Mutation Grid Compromise Candidates

| grid | grid_x | grid_y | compromise_score | mutation_prob | mutation_sigma | parent_count | elite_count | first_finish_generation | best_finish_time | last50_finish_rate | last50_mean_dense_progress | last50_crash_rate | last50_timeout_rate | last50_penalized_mean_time | max_cached_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mutation | 0.100 | 0.250 | 0.243 | 0.100 | 0.250 | 16 | 4 | 124.000 | 18.900 | 0.328 | 58.163 | 0.664 | 0.009 | 27.800 | 0 |
| mutation | 0.100 | 0.300 | 0.208 | 0.100 | 0.300 | 16 | 4 | 114.000 | 19.120 | 0.328 | 53.259 | 0.645 | 0.027 | 27.861 | 0 |
| mutation | 0.150 | 0.200 | 0.200 | 0.150 | 0.200 | 16 | 4 | 137.000 | 21.270 | 0.302 | 56.835 | 0.660 | 0.038 | 28.393 | 0 |
| mutation | 0.150 | 0.250 | 0.118 | 0.150 | 0.250 | 16 | 4 | 105.000 | 17.000 | 0.300 | 50.571 | 0.684 | 0.016 | 27.309 | 0 |
| mutation | 0.300 | 0.150 | 0.034 | 0.300 | 0.150 | 16 | 4 | 104.000 | 20.640 | 0.267 | 48.808 | 0.718 | 0.015 | 28.469 | 0 |
| mutation | 0.200 | 0.300 | -0.212 | 0.200 | 0.300 | 16 | 4 | 103.000 | 20.010 | 0.161 | 41.061 | 0.766 | 0.073 | 28.940 | 0 |
| mutation | 0.150 | 0.150 | -0.218 | 0.150 | 0.150 | 16 | 4 | 167.000 | 22.780 | 0.125 | 49.854 | 0.830 | 0.045 | 29.565 | 0 |
| mutation | 0.150 | 0.300 | -0.225 | 0.150 | 0.300 | 16 | 4 | 125.000 | 19.710 | 0.167 | 41.812 | 0.803 | 0.030 | 28.885 | 0 |

## Selection Pressure Grid Candidates

_No data._

## Selection Pressure Compromise Candidates

_No data._

## Edge Check

- `mutation` best by stability ranking is on grid edge (minimum x, maximum y): `prob_010_sigma_030`.
- `mutation` best by compromise score is on grid edge (minimum x): `prob_010_sigma_025`.

## Generated Plots

- `candidate_stability_vs_speed.png`
- `heatmap_mutation_best_finish_time.png`
- `heatmap_mutation_first_finish_generation.png`
- `heatmap_mutation_last50_crash_rate.png`
- `heatmap_mutation_last50_finish_rate.png`
- `heatmap_mutation_last50_mean_dense_progress.png`
- `heatmap_mutation_last50_penalized_mean_time.png`
- `heatmap_mutation_last50_timeout_rate.png`

## Reading Guide

- `first_finish_generation` measures how quickly a configuration discovers a complete lap.
- `last50_finish_rate` measures late training stability, not only one lucky finisher.
- `best_finish_time` measures speed, but should not override stability by itself.
- `last50_penalized_mean_time` treats unfinished individuals as `max_time`, so it combines finish quality and failure rate.
- If the best value lies on a grid edge, the next experiment should be a smaller refinement grid around that edge.


---

## Architecture And Activation Ablation

- Status: `thesis-grade`
- Source: `Diplomová práca/Experiments/ga_architecture_activation_ablation_20260504/REPORT.md`
- Note: Closed-loop evidence for relu,tanh and 32x16/48x24 tradeoff.

# Closed-loop architecture activation ablation

## Status

- Experiment ID: `ga_architecture_activation_ablation_20260504`
- Category: `thesis_grade`
- Status: `closed_loop_single_seed`
- Curated at: `2026-05-04T13:13:10`

## Thesis Relevance

Closed-loop GA comparison of 32x16/48x24 and relu,tanh versus relu,relu under the selected reward tuple.

## Interpretation

relu,tanh outperformed relu,relu in the closed-loop ablation; 48x24 is the stronger candidate, while 32x16 remains the cheaper experimental baseline.

## Package Contents

- `runs/`: selected raw run logs copied from the working experiment folders.
- `analysis/`: summary tables, reports, and generated plots.
- `scripts/`: scripts needed to reproduce or analyze the experiment.
- `metadata.json`: source paths, keywords, and curation metadata.

## Notes

Large aggregate files such as `combined_individual_metrics.csv` are intentionally not copied into the thesis package. They remain local working artifacts if needed.


---

## AABB Vehicle Hitbox

- Status: `thesis-grade`
- Source: `Diplomová práca/Experiments/vehicle_hitbox_aabb_20260503/REPORT.md`
- Note: Justification for AABB-clearance lidar.

# Vehicle AABB hitbox analysis

## Status

- Experiment ID: `vehicle_hitbox_aabb_20260503`
- Category: `thesis_grade`
- Status: `empirical_model`
- Curated at: `2026-05-04T13:13:09`

## Thesis Relevance

Justification for replacing a global raw lidar threshold with AABB-relative clearance lidar.

## Interpretation

The selected AABB is empirical, derived from mesh estimates and sanity-checked against near-contact supervised data.

## Package Contents

- `runs/`: selected raw run logs copied from the working experiment folders.
- `analysis/`: summary tables, reports, and generated plots.
- `scripts/`: scripts needed to reproduce or analyze the experiment.
- `metadata.json`: source paths, keywords, and curation metadata.

## Notes

Large aggregate files such as `combined_individual_metrics.csv` are intentionally not copied into the thesis package. They remain local working artifacts if needed.


---

## AABB Vehicle Hitbox Detailed Report

- Status: `thesis-grade detail`
- Source: `Diplomová práca/Experiments/vehicle_hitbox_aabb_20260503/analysis/vehicle_hitbox_20260503_smoke/REPORT.md`
- Note: Detailed hitbox sanity checks.

# Vehicle AABB Hitbox Analysis

## Inputs
- Attempt files: `1`
- Frames: `11770`
- Raw laser source: `raw_laser_distances` when available, otherwise reconstructed as `observations[:, :15] * 160.0`
- Mesh estimates:
```json
{
  "visual_main_body": {
    "size": [
      2.1432,
      3.7842,
      0.8822
    ],
    "center": [
      0.0004,
      -0.2737,
      0.4292
    ],
    "note": "MainBody lod1 visual mesh OBJ export."
  },
  "car_primitives": {
    "size": [
      2.1326,
      4.0792,
      1.1792
    ],
    "center": [
      0.0,
      0.1051,
      -0.5795
    ],
    "note": "ManiaPark primitive-style CPlugSolid estimate."
  }
}
```

## Selected Empirical AABB
```json
{
  "half_width": 1.1,
  "front_half_length": 2.15,
  "rear_half_length": 1.95,
  "half_height": 0.6,
  "source": "empirical_mesh_and_supervised_aabb_20260503"
}
```

## Key Checks
- Minimum supervised-minus-AABB margin: `0.216`
- Median supervised-minus-AABB margin: `0.326`
- Minimum q01 supervised-minus-AABB margin: `0.355`
- Median q01 supervised-minus-AABB margin: `0.467`
- Plot written: `True`

## Interpretation
The current AABB is intentionally empirical. It is based on the primitive mesh
footprint and checked against supervised near-contact lidar distances. The
supervised minima should not be treated as exact collision truth because the
latest near-contact run finished successfully; they are mainly a sanity check
that the selected AABB does not exceed observed close-pass distances.


---

## Supervised Map Specialists

- Status: `thesis-grade visualization`
- Source: `Diplomová práca/Experiments/supervised_map_specialists_20260505/REPORT.md`
- Note: Teacher/agent path plots and supervised specialist result package.

# Supervised Map Specialists 20260505

## Summary
This package contains the supervised map-specialist result used to visualize how a small map-specific imitation model behaves compared with human teacher trajectories.

Three supervised specialists were trained from newly collected v3d/surface datasets:

- `single_surface_flat`
- `multi_surface_flat`
- `single_surface_height`

Each model uses observation dimension `53`, hidden architecture `48,24`, hidden activations `relu,tanh`, target action mode, height features, surface features and mirror augmentation.

## Main Thesis Figure
The currently validated thesis-ready figure is:

`analysis/single_surface_flat/single_surface_flat_teacher_agent_paths.png`

It overlays:

- Track map background with start, finish, road and edge legend.
- Ten teacher trajectories from supervised data.
- The trained supervised agent rollout with the same global speed color scale.
- Crash/touch markers as black dots.

The shared speed gradient makes it possible to discuss where the agent differs from teacher driving, for example where it slows down, turns differently, or touches the wall.

## Result Status
For `single_surface_flat`, the deterministic TM2D replay finished the track:

- finished: `1`
- progress: `100.0`
- time: `43.39`
- crashes/touches: `32`
- left road surface: `0`

This is useful as a qualitative supervised-learning diagnostic figure, not as proof of a final autonomous racing policy. The graph is mainly evidence for the limitation of pure supervised imitation: the model can follow the track, but small trajectory deviations can accumulate into wall contacts.

## Included Files
- `analysis/`: generated figures, rollout metrics, trajectory file and training summaries.
- `runs/`: small supervised training outputs and best models for the three map specialists.
- `scripts/`: reproduction scripts used to train and analyze this result.

Raw supervised datasets are not duplicated here to avoid unnecessary package size. Their source paths are listed in `metadata.json`.


---

## Latest GA Training Improvements

- Status: `thesis-grade and diagnostic mix`
- Source: `Diplomová práca/Experiments/training_improvements_20260505/analysis/latest_training_results_20260505/REPORT.md`
- Note: Elite cache, decay, mirror, max-touch, MOO and live TM summary.

# Latest Training Results 20260505

This package summarizes the latest runs received from the second PC.

## Main takeaways

- Variable physics tick with elite cache is the strongest positive result: no-cache had 0 total finishes, cache had 572 total finishes and best time 18.71s.
- First-finish mutation decay is useful as a tradeoff experiment: best time 18.84s, but last50 finish rate 27.3% is not better than baseline stability.
- Max touches 3 is diagnostic/mid: progress is high, but the crash profile remains too noisy (100.0% last50 crash rate).
- MOO trackmania_racing is promising but not the new default: best time 18.55s, first finish generation 204, last50 finish rate 27.4%.
- Both-mirror holdout evaluation did not yet prove generalization: train total finishes 13, holdout top-1 finishes 0.

## Recommended thesis usage

- Include elite-cache vs no-cache under variable physics as a thesis-grade positive result.
- Include first-finish decay as an optimization idea with mixed results.
- Keep max-touches and mirror holdout as diagnostic evidence, not as final improvements.
- Mention MOO trackmania_racing as a promising revised MOO formulation, while lexicographic ranking remains the safer baseline.

## Generated files

- `base_vs_first_finish_decay_focus_progress.png`
- `generalization_holdout_train_vs_test.png`
- `live_tm_focus_progress.png`
- `live_tm_mutation_and_cache.png`
- `live_tm_outcomes.png`
- `moo_best_time_front.png`
- `moo_outcomes.png`
- `moo_progress.png`
- `summary_bars.png`
- `training_improvements_best_time.png`
- `training_improvements_focus_progress.png`
- `training_improvements_mutation_and_cache.png`
- `training_improvements_outcomes.png`
- `training_improvements_progress.png`
- `variable_tick_cache_focus_progress.png`
- `live_tm_trajectories/best_trajectory_speed.png`
- `live_tm_trajectories/trajectory_heatmap.png`
- `live_tm_trajectories/trajectory_overview.png`


---

## RL Reward-Equivalent Sweep

- Status: `comparison / useful negative`
- Source: `Diplomová práca/Experiments/rl_reward_equivalent_sweep_20260505/analysis/rl_reward_equivalent_sweep_20260505/DEEP_REPORT.md`
- Note: PPO vs SAC vs TD3 under reward-equivalent scalarization.

# Deep RL Sweep Analysis

Root: `Experiments/runs_rl/reward_equivalent_aabb_tick_sweep_seed_2026050508`

All runs used the same TM2D task, AABB-clearance lidar, strict `gas_brake_steer` action layout, `32,16` ReLU policy, scalar delta reward equivalent to `(finished, progress, -time, -crashes)`, and 1000 episodes.

## High-Level Summary

| run_short | algorithm | physics_tick_profile | episodes | max_progress | last100_mean_progress | finish_count | first_finish_episode | best_finish_time | crash_count | timeout_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppo_fixed100 | PPO | fixed100 | 1000 | 100.000 | 55.307 | 43 | 849 | 26.910 | 880 | 77 |
| ppo_supervised_v2d | PPO | supervised_v2d | 1000 | 100.000 | 51.479 | 33 | 759 | 25.040 | 964 | 3 |
| sac_fixed100 | SAC | fixed100 | 1000 | 7.088 | 2.473 | 0 | -1 |  | 1000 | 0 |
| sac_supervised_v2d | SAC | supervised_v2d | 1000 | 5.158 | 3.124 | 0 | -1 |  | 1000 | 0 |
| td3_fixed100 | TD3 | fixed100 | 1000 | 7.329 | 3.310 | 0 | -1 |  | 1000 | 0 |
| td3_supervised_v2d | TD3 | supervised_v2d | 1000 | 6.572 | 3.243 | 0 | -1 |  | 1000 | 0 |

## Progress Thresholds

Episode where each run first reached the given continuous progress threshold. `-1` means the threshold was never reached.

| run_short | algorithm | physics_tick_profile | first_ge_5 | first_ge_10 | first_ge_25 | first_ge_50 | first_ge_75 | first_ge_90 | first_ge_100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppo_fixed100 | PPO | fixed100 | 56 | 171 | 590 | 635 | 744 | 804 | 849 |
| ppo_supervised_v2d | PPO | supervised_v2d | 88 | 251 | 645 | 645 | 751 | 751 | 759 |
| sac_fixed100 | SAC | fixed100 | 6 | -1 | -1 | -1 | -1 | -1 | -1 |
| sac_supervised_v2d | SAC | supervised_v2d | 179 | -1 | -1 | -1 | -1 | -1 | -1 |
| td3_fixed100 | TD3 | fixed100 | 3 | -1 | -1 | -1 | -1 | -1 | -1 |
| td3_supervised_v2d | TD3 | supervised_v2d | 3 | -1 | -1 | -1 | -1 | -1 | -1 |

## Late Training Stability

| run_short | phase | mean_progress | p90_progress | max_progress | finish_count | finish_rate | crash_rate | timeout_rate | best_finish_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppo_fixed100 | ep751_1000 | 54.504 | 100.000 | 100.000 | 43 | 0.172 | 0.640 | 0.188 | 26.910 |
| ppo_fixed100 | last100 | 55.307 | 100.000 | 100.000 | 24 | 0.240 | 0.580 | 0.180 | 26.910 |
| ppo_supervised_v2d | ep751_1000 | 39.753 | 100.000 | 100.000 | 33 | 0.132 | 0.856 | 0.012 | 25.040 |
| ppo_supervised_v2d | last100 | 51.479 | 100.000 | 100.000 | 22 | 0.220 | 0.780 | 0.000 | 25.040 |
| sac_fixed100 | ep751_1000 | 2.357 | 2.747 | 3.282 | 0 | 0.000 | 1.000 | 0.000 |  |
| sac_fixed100 | last100 | 2.473 | 2.888 | 3.282 | 0 | 0.000 | 1.000 | 0.000 |  |
| sac_supervised_v2d | ep751_1000 | 2.855 | 3.589 | 4.803 | 0 | 0.000 | 1.000 | 0.000 |  |
| sac_supervised_v2d | last100 | 3.124 | 3.857 | 4.803 | 0 | 0.000 | 1.000 | 0.000 |  |
| td3_fixed100 | ep751_1000 | 3.374 | 3.491 | 3.684 | 0 | 0.000 | 1.000 | 0.000 |  |
| td3_fixed100 | last100 | 3.310 | 3.326 | 3.343 | 0 | 0.000 | 1.000 | 0.000 |  |
| td3_supervised_v2d | ep751_1000 | 3.243 | 3.254 | 3.277 | 0 | 0.000 | 1.000 | 0.000 |  |
| td3_supervised_v2d | last100 | 3.243 | 3.254 | 3.269 | 0 | 0.000 | 1.000 | 0.000 |  |

## Finish Statistics

| run_short | algorithm | physics_tick_profile | finish_count | first_finish_episode | last_finish_episode | best_finish_time | median_finish_time | mean_finish_time | finish_count_last100 | best_time_last100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppo_fixed100 | PPO | fixed100 | 43 | 849 | 973 | 26.910 | 28.740 | 28.613 | 24 | 26.910 |
| ppo_supervised_v2d | PPO | supervised_v2d | 33 | 759 | 999 | 25.040 | 27.330 | 27.318 | 22 | 25.040 |
| sac_fixed100 | SAC | fixed100 | 0 | -1 | -1 |  |  |  | 0 |  |
| sac_supervised_v2d | SAC | supervised_v2d | 0 | -1 | -1 |  |  |  | 0 |  |
| td3_fixed100 | TD3 | fixed100 | 0 | -1 | -1 |  |  |  | 0 |  |
| td3_supervised_v2d | TD3 | supervised_v2d | 0 | -1 | -1 |  |  |  | 0 |  |

## Interpretation

- PPO is the only algorithm that learned to finish in this screening. Both PPO variants reached 100% progress and produced finishers.
- PPO fixed100 was more stable overall: 43 finishes, 77 timeouts, 880 crash episodes, and 55.3% last-100 mean progress.
- PPO supervised_v2d found the best single finish time at 25.04s, but was less stable: 33 finishes and 964 crash episodes with only 3 timeouts.
- SAC and TD3 did not solve the task. Their maximum progress stayed below 8%, and every episode ended in crash. In this reward/action setup they are negative evidence rather than competitive baselines.
- Variable physics did not prevent PPO from learning, but in this single-seed run it reduced stability compared with fixed100. For SAC/TD3, variable ticks did not rescue learning.
- The result supports using PPO as the RL baseline in the thesis comparison, while keeping GA as the stronger practical method unless longer or better-shaped RL training is introduced.

## Generated Figures

- `01_cumulative_best_progress.png`
- `02_rolling_progress.png`
- `03_rolling_episode_reward.png`
- `04_summary_bars.png`
- `05_first_progress_thresholds.png`
- `06_phase_mean_progress_heatmap.png`
- `07_episode_outcomes.png`
- `08_best_finish_time_so_far.png`

## Best Time Plot

`08_best_finish_time_so_far.png` shows the cumulative best finish time over episodes. Only PPO produced finishers in this sweep, so SAC and TD3 have no best-time curve.


---

## GA Supervised-Seeded Hybrid

- Status: `thesis-grade positive hybrid`
- Source: `Diplomová práca/Experiments/ga_supervised_seeded_20260505/analysis/ga_supervised_seeded_20260505/DEEP_REPORT.md`
- Note: Behavior cloning initialization plus GA fine-tuning.

# GA Supervised-Seeded Deep Analysis 20260505

This analysis compares the random GA baseline with two behavior-cloning initialized populations:

- `exp00_base_fixed100`: random initial population.
- `exp06_supervised_seeded_fixed100`: one exact BC copy plus sparse mutation tiers.
- `exp06b_supervised_seeded_dense_fixed100`: one exact BC copy plus dense weight-noise tiers.

## Verdict

Dense-noise supervised seeding is a strong positive result. It found the first finisher at generation 116, slightly earlier than the random baseline at generation 121, and reached the best time in this comparison: 17.28s at generation 190. The random baseline remained a little more stable in the last 50 generations, but its best time was slower at 20.66s.

The original sparse supervised seeding is a negative result. Although generation 1 starts with higher mean progress (7.17% vs baseline 2.02%), the population stays too concentrated around the BC policy and only finds finishers very late, at generation 190. Its total finish count is only 19.

## Key Numbers

| Experiment | First finish gen | Total finishes | Best time | Last50 finish rate | Last50 mean progress | Last50 crash rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline random | 121 | 942 | 20.66s | 0.333 | 54.46% | 0.648 |
| Supervised sparse | 190 | 19 | 26.78s | 0.008 | 25.30% | 0.980 |
| Supervised dense | 116 | 1079 | 17.28s | 0.318 | 49.35% | 0.675 |

## Interpretation

- BC initialization alone is not enough. The BC policy starts above random at generation 1, but it is still far from a robust lap-completing solution.
- Sparse mutation around the BC model appears too conservative. It preserves the prior, but does not create enough behavioral diversity to escape local failure modes.
- Dense mutation is the useful version. Mutating every weight with several sigma tiers creates a population that keeps some BC structure while still exploring broadly enough for GA selection to improve it.
- For thesis use, present this as a hybrid method: behavior cloning initialization + evolutionary fine-tuning. It uses extra human demonstration data, so it should not be framed as a pure GA improvement.

## Recommended Thesis Status

- `exp06b_supervised_seeded_dense_fixed100`: thesis-grade positive hybrid experiment.
- `exp06_supervised_seeded_fixed100`: useful negative control showing why naive/sparse BC seeding is not enough.
- `exp00_base_fixed100`: comparison baseline from the same training-improvements sweep.

## Generated Files

- `summary.csv`
- `phase_summary.csv`
- `threshold_summary.csv`
- `01_progress_comparison.png`
- `02_finish_rate_rolling.png`
- `03_best_finish_time_so_far.png`
- `04_last50_outcome_rates.png`
- `05_initial_population_progress_distribution.png`
- `06_phase_mean_progress_heatmap.png`
- `07_training_improvements_best_time_context.png`


---

## Supervised Physics Tick Distribution

- Status: `supporting diagnostic`
- Source: `Experiments/analysis/supervised_physics_ticks_20260504/REPORT.md`
- Note: Timing evidence for physics tick delay and 100Hz/variable tick discussion.

# Supervised Physics Tick Analysis

- Attempt files: 12
- Valid attempt files: 12
- Valid frame deltas: 28486
- Recommended `--physics-tick-probs`: `1:0.938285,2:0.060381,3:0.000562,4:0.000772`
- Thesis plot: `physics_tick_distribution_thesis.png`

## Distribution

| ticks | physics Hz | probability | count |
| ---: | ---: | ---: | ---: |
| 1 | 100.00 | 0.9383 | 26728 |
| 2 | 50.00 | 0.0604 | 1720 |
| 3 | 33.33 | 0.0006 | 16 |
| 4 | 25.00 | 0.0008 | 22 |

Interpretation: the observation should encode physics-tick delay, not render FPS. `physics_delay_norm = 1 - 1 / tick_count` is zero at 100 Hz and grows when game physics updates are skipped.

Elite-cache motivation: fixed-100 Hz TM2D experiments are deterministic, but live Trackmania can miss physics updates. Re-evaluating an elite in a different tick sequence can therefore change the outcome even when the genotype did not change. This does not prove that elite cache is always better, but it explains why it is a meaningful training variant to test under variable physics timing.
