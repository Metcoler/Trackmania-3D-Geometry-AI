# GA vs RL vs GA MOO Comparison

This report compares local TM2D experiments on `AI Training #5` using the dense-progress racing setup.

- Fastest training finish: `GA MOO | finished,progress,neg_time,neg_crashes,neg_distance` with `16.308s`.
- Best validation finish rate: `RL | PPO | delta_finished_progress_time | gas_steer` with `100.0%`.

## Summary Table

| method | variant | first_success_unit | best_finish_time | final_finish_count | final_finish_rate | validation_finish_count | validation_finish_rate | validation_best_finish_time | validation_mean_dense_progress | wall_minutes | virtual_time_hours |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GA Lexicographic | (finished, progress) | 163.000 | 22.727 | 10.000 | 0.208 | 25.000 | 0.833 | 32.097 | 95.383 | 123.449 | 39.897 |
| GA Lexicographic | (finished, progress, -crashes, -time) | 176.000 | 16.577 | 15.000 | 0.312 | 27.000 | 0.900 | 16.586 | 97.693 | 105.248 | 30.931 |
| GA Lexicographic | (finished, progress, -time) | 114.000 | 16.323 | 14.000 | 0.292 | 19.000 | 0.633 | 16.294 | 79.626 | 104.122 | 37.478 |
| GA Lexicographic | (finished, progress, -time, -crashes) | 198.000 | 17.750 | 10.000 | 0.208 | 29.000 | 0.967 | 17.715 | 98.667 | 92.051 | 27.869 |
| GA MOO | finished,progress,neg_time,neg_crashes | 142.000 | 16.623 | 1.000 | 0.021 | 0.000 | 0.000 |  | 63.197 | 58.795 | 18.894 |
| GA MOO | finished,progress,neg_time | -1.000 |  | 0.000 | 0.000 | 0.000 | 0.000 |  | 16.891 | 42.343 | 11.052 |
| GA MOO | finished,progress,neg_time,neg_crashes,neg_distance | 101.000 | 16.308 | 7.000 | 0.146 | 9.000 | 0.300 | 16.317 | 78.000 | 84.012 | 33.556 |
| RL | PPO | delta_finished_progress_time | gas_steer | 2168.000 | 19.224 | 446.000 | 0.137 | 30.000 | 1.000 | 18.063 | 100.000 |  |  |
| RL | SAC | delta_finished_progress_time | gas_steer | -1.000 |  | 0.000 | 0.000 | 0.000 | 0.000 |  | 2.742 |  |  |
| RL | TD3 | delta_finished_progress_time | gas_steer | 469.000 | 21.268 | 1.000 | 0.000 | 0.000 | 0.000 |  | 48.904 |  |  |
| RL | PPO | delta_finished_progress_time | gas_brake_steer | -1.000 |  | 0.000 | 0.000 |  |  |  |  |  |  |

## Interpretation

- `GA Lexicographic` with `(finished, progress, -time)` remains the fastest simple tuple during training, but the 30-episode validation shows that adding crash awareness improves reproducibility.
- `GA Lexicographic` with `(finished, progress, -time, -crashes)` and `(finished, progress, -crashes, -time)` validated more reliably, trading a little lap-time pressure for safer policies.
- `GA MOO` is competitive only when the objective set includes `neg_distance`. Without distance, it can discover progress but validates poorly, which points to unstable/risky trajectories.
- `PPO` with `gas_steer` is the only RL setup that solved the map robustly in deterministic validation, but it is slower and more sample-expensive than GA.
- `SAC` did not leave the early local minimum in this setup. `TD3` found a finish once during training but did not validate as a stable policy.
- `PPO` with `gas_brake_steer` was a negative control in this batch: adding brake enlarged the action problem and the short run collapsed near the start.

## Generated Graphs

- `01_best_finish_time.png`
- `02_first_success_unit.png`
- `03_finish_rate.png`
- `04_validation_dense_progress.png`
