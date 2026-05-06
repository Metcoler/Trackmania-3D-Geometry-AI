# Top-2 lexicographic reward detail

This analysis adds per-population detail for the two strongest ranking tuples.
The shaded bands are within-population spread in a single seed, not cross-seed variance.

## Summary

| variant | ranking_key | max_time | first_finish_generation | total_finish_individuals | best_finish_time | best_finish_generation | last50_finish_per_generation | last50_mean_dense_progress | last50_mean_penalized_time | last50_best_finish_time_so_far |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| finished_progress_time_crashes | (finished, progress, -time, -crashes) | 30.000 | 137 | 2209 | 17.330 | 264 | 15.460 | 49.388 | 26.825 | 17.330 |
| finished_progress_crashes_time | (finished, progress, -crashes, -time) | 30.000 | 136 | 2246 | 17.970 | 276 | 14.700 | 54.040 | 27.572 | 17.970 |

## Generated plots

- `lex_top1_training_detail.png`
- `lex_top1_population_finish_time.png`
- `lex_top2_training_detail.png`
- `lex_top2_population_finish_time.png`

## Trajectory replay

The script attempted to replay the best policy from the selected top ranking.
- finished: `0`
- progress: `3.358328683035714`
- time: `1.370000000000001`
- copied to LaTeX: `False`
- note: rollout did not reproduce a usable final trajectory; keeping only diagnostic outputs
