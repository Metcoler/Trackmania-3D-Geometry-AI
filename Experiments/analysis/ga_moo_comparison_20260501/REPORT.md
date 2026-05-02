# GA MOO Run Comparison

- Fastest training finish: `MOO | finished,progress,neg_time,neg_crashes,neg_distance` with `16.308s`.
- Most final-generation finishes: `MOO | finished,progress,neg_time,neg_crashes,neg_distance` with `7` / `48`.
- Best validation finish rate: `MOO | finished,progress,neg_time,neg_crashes,neg_distance` with `30.00%`.

## Summary

| run_label | objective_subset | first_generation_with_any_finish | best_finish_time | max_finish_count | final_finish_count | final_finish_rate | final_mean_dense_progress | final_front0_size | cumulative_virtual_time_hours | cumulative_wall_minutes | validation_finish_rate | validation_mean_dense_progress | validation_best_finish_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOO | finished,progress,neg_time | finished,progress,neg_time | -1 |  | 0 | 0 | 0.000 | 1.718 | 1 | 0.006 | 0.022 | 0.000 | 1.730 |  |
| MOO | finished,progress,neg_time,neg_crashes,neg_distance | finished,progress,neg_time,neg_crashes,neg_distance | -1 |  | 0 | 0 | 0.000 | 1.736 | 4 | 0.006 | 0.023 |  |  |  |
| MOO | finished,progress,neg_time,neg_crashes,neg_distance | finished,progress,neg_time,neg_crashes,neg_distance | -1 |  | 0 | 0 | 0.000 | 1.737 | 4 | 0.006 | 0.032 | 0.000 | 1.787 |  |
| MOO | finished,progress,neg_time,neg_crashes | finished,progress,neg_time,neg_crashes | 142 | 16.623 | 5 | 1 | 0.021 | 27.688 | 35 | 18.894 | 58.795 | 0.000 | 63.197 |  |
| MOO | finished,progress,neg_time | finished,progress,neg_time | -1 |  | 0 | 0 | 0.000 | 6.814 | 34 | 11.052 | 42.343 | 0.000 | 16.891 |  |
| MOO | finished,progress,neg_time,neg_crashes,neg_distance | finished,progress,neg_time,neg_crashes,neg_distance | 101 | 16.308 | 15 | 7 | 0.146 | 41.008 | 29 | 33.556 | 84.012 | 0.300 | 78.000 | 16.317 |

## Generated Graphs

- `01_best_dense_progress.png`
- `02_mean_dense_progress.png`
- `03_finish_count.png`
- `04_front0_size.png`
- `05_best_time_finished_only.png`
- `06_final_population_progress_time_scatter.png`
- `07_summary_bars.png`
