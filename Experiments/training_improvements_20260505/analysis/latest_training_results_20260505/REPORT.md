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
