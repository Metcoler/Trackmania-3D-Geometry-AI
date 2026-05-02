# Lexicographic GA Reward Sweep Analysis

## Input Roots

- `Experiments\runs_ga\lex_sweep_seed_2026050201`
- `Experiments\runs_ga\lex_sweep_seed_2026050202`

## Summary

This report compares lexicographic GA ranking functions under the fixed fair-sweep setup:
fixed 60 FPS, laser collision threshold 2.0, binary gas/brake, v2d asphalt observation,
population 64, elite count 8, parent count 32, fixed mutation probability 0.2,
fixed mutation sigma 0.2, max episode time 30 seconds, 300 generations, and disabled elite cache.



## Per-Run Results

| variant | seed | completed | first_finish_generation | best_finish_time | final_finish_rate | final_crash_rate | final_timeout_rate | final_mean_dense_progress | max_cached_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| finished_progress | 2026050201 | 1 |  |  | 0.000 | 1.000 | 0.000 | 17.304 | 0 |
| finished_progress | 2026050202 | 1 |  |  | 0.000 | 1.000 | 0.000 | 9.099 | 0 |
| finished_progress_time | 2026050201 | 1 |  |  | 0.000 | 1.000 | 0.000 | 13.827 | 0 |
| finished_progress_time | 2026050202 | 1 |  |  | 0.000 | 1.000 | 0.000 | 13.188 | 0 |
| finished_progress_time_crashes | 2026050201 | 1 |  |  | 0.000 | 1.000 | 0.000 | 10.171 | 0 |
| finished_progress_time_crashes | 2026050202 | 1 |  |  | 0.000 | 1.000 | 0.000 | 13.146 | 0 |
| finished_progress_crashes_time | 2026050201 | 1 |  |  | 0.000 | 1.000 | 0.000 | 15.193 | 0 |
| finished_progress_crashes_time | 2026050202 | 1 |  |  | 0.000 | 1.000 | 0.000 | 18.398 | 0 |

## Cross-Seed Stability

| variant | ranking_key | runs | completed_runs | first_finish_mean | first_finish_std | best_finish_time_min | best_finish_time_mean | best_finish_time_std | final_finish_rate_mean | final_finish_rate_std | final_crash_rate_mean | final_timeout_rate_mean | final_mean_dense_progress_mean | max_cached_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| finished_progress | (finished, progress) | 2 | 2 |  |  |  |  |  | 0.000 | 0.000 | 1.000 | 0.000 | 13.202 | 0 |
| finished_progress_time | (finished, progress, -time) | 2 | 2 |  |  |  |  |  | 0.000 | 0.000 | 1.000 | 0.000 | 13.507 | 0 |
| finished_progress_time_crashes | (finished, progress, -time, -crashes) | 2 | 2 |  |  |  |  |  | 0.000 | 0.000 | 1.000 | 0.000 | 11.658 | 0 |
| finished_progress_crashes_time | (finished, progress, -crashes, -time) | 2 | 2 |  |  |  |  |  | 0.000 | 0.000 | 1.000 | 0.000 | 16.796 | 0 |

## Generated Plots

- `01_best_dense_progress.png`
- `02_finish_count.png`
- `03_best_finish_time.png`
- `04_finish_crash_rates.png`
- `05_final_population_scatter.png`

## Interpretation Notes

- `(finished, progress)` should show whether progress alone can reliably discover finish, but it has no incentive to become faster after a comparable progress value.
- `(finished, progress, -time)` adds speed pressure and is expected to be the strongest simple baseline, but may prefer faster risky endings when progress is equal.
- `(finished, progress, -time, -crashes)` tests whether crash safety matters when it is only a late tie-breaker after time.
- `(finished, progress, -crashes, -time)` tests the safer ordering where crash avoidance beats time among equally far policies.
- The key thesis signal is not only the best run, but cross-seed stability: first finish generation, final finish rate, crash/timeout rates, and whether the best finish time reproduces.
