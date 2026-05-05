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
