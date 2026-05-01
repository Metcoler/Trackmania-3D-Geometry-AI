# GA lexicographic reward comparison

All runs use `AI Training #5`, population `48`, `300` generations, variable FPS, `corners` collision mode, hidden layers `[32, 16]`, activations `[relu, tanh]`, and `ranking_progress_source = dense_progress`.

## Compared ranking keys
- `finished_progress`: `(finished, progress)`
- `finished_progress_crashes_time`: `(finished, progress, -crashes, -time)`
- `finished_progress_time`: `(finished, progress, -time)`
- `finished_progress_time_crashes`: `(finished, progress, -time, -crashes)`

## Headline results
- Earliest finish: `finished_progress_time` at generation `114`.
- Fastest finish: `finished_progress_time` with `16.323s` at generation `295`.
- Highest one-generation finish count: `finished_progress_time` with `26` finished individuals.

## Interpretation

The strongest run in this batch is `finished_progress_time`, i.e. `(finished, progress, -time)`. It found the first finish much earlier than the other variants, reached the fastest observed finish time, and also produced the largest number of finished individuals in a single generation. In the last 50 generations it averaged about `17.28` finished individuals per generation, which suggests the solution was not just one lucky elite.

`finished_progress_crashes_time`, i.e. `(finished, progress, -crashes, -time)`, behaved like the safer alternative. It learned later and was slightly slower, but it reached a stable finish population with competitive times. This is useful evidence for the thesis discussion: moving crash safety before time changes the evolutionary pressure from aggressive speed toward cleaner but slower driving.

`finished_progress`, i.e. `(finished, progress)`, confirms the expected weakness of ignoring time. It eventually reaches finish, but the best finish time is much worse and the last generations contain many slow or unfinished individuals. This is a good baseline showing that progress alone is insufficient for a racing task.

`finished_progress_time_crashes`, i.e. `(finished, progress, -time, -crashes)`, did learn to finish, but it was the weakest of the time-aware variants in this run. Since crashes are only considered after time, the agent can prefer fast but less robust behavior. The final crash rate was the highest in this comparison.

All four runs used `ranking_progress_source = dense_progress`, so the `progress` symbol in the ranking key means continuous path progress, not discrete checkpoint/block progress.

## Last-50-Generation Stability

| run_key | mean_finish_count_last50 | median_finish_time_last50 | p10_finish_time_last50 |
| --- | --- | --- | --- |
| finished_progress | 5.98 | 40.742 | 30.039 |
| finished_progress_crashes_time | 11.08 | 17.427 | 16.730 |
| finished_progress_time | 17.28 | 16.548 | 16.343 |
| finished_progress_time_crashes | 10.76 | 20.813 | 18.163 |

## Summary table
| run_key | ranking_key | first_generation_with_any_finish | best_finish_time | best_finish_generation | max_finish_count | final_finish_count | final_crash_rate | cumulative_virtual_time_hours | cumulative_wall_minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| finished_progress | (finished, progress) | 163 | 22.727 | 264 | 10 | 10 | 0.688 | 39.897 | 123.449 |
| finished_progress_crashes_time | (finished, progress, -crashes, -time) | 176 | 16.577 | 297 | 19 | 15 | 0.688 | 30.931 | 105.248 |
| finished_progress_time | (finished, progress, -time) | 114 | 16.323 | 295 | 26 | 14 | 0.708 | 37.478 | 104.122 |
| finished_progress_time_crashes | (finished, progress, -time, -crashes) | 198 | 17.750 | 294 | 19 | 10 | 0.792 | 27.869 | 92.051 |

## Generated plots
- `01_best_dense_progress.png`
- `02_population_progress_distribution.png`
- `03_finish_count_and_best_finish_time.png`
- `04_outcome_rates.png`
- `05_final_population_scatter.png`
- `06_summary_bars.png`
- `07_virtual_time_efficiency.png`
