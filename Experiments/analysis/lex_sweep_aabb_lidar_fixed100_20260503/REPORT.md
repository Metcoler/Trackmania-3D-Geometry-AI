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
