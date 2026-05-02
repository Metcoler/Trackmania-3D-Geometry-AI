# RL Run Comparison

- Best max dense progress: `PPO | delta_finished_progress_time | gas_steer` with 100.00%.
- Best finish time: `PPO | delta_finished_progress_time | gas_steer` with 19.859s.

## Summary

| run_label | run_short | episodes | max_dense_progress | max_progress | last100_mean_dense | last100_mean_reward | finish_count | crash_count | timeout_count | first_finish_episode | best_finish_time | best_dense_episode | best_dense_time | source_csv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | delta_finished_progress_time | gas_steer | 20260501_171735_tm2d_ppo_AI_Training__5_delta_finished_progress_time_gas_steer | 907 | 100.000 | 100.000 | 59.139 | 0.912 | 31 | 875 | 1 | 851 | 19.859 | 851 | 21.531 | Experiments\runs_rl\ai_training5_batch_20260501_171723_20min\20260501_171735_tm2d_ppo_AI_Training__5_delta_finished_progress_time_gas_steer\episode_metrics.csv |
| SAC | delta_finished_progress_time | gas_brake_steer | 20260501_171735_tm2d_sac_AI_Training__5_delta_finished_progress_time_gas_brake_steer | 981 | 1.946 | 0.000 | 1.752 | 0.018 | 0 | 981 | 0 | -1 | nan | 668 | 5.008 | Experiments\runs_rl\ai_training5_batch_20260501_171723_20min\20260501_171735_tm2d_sac_AI_Training__5_delta_finished_progress_time_gas_brake_steer\episode_metrics.csv |
| SAC | delta_finished_progress_time | gas_steer | 20260501_171735_tm2d_sac_AI_Training__5_delta_finished_progress_time_gas_steer | 294 | 8.558 | 5.714 | 3.563 | 0.036 | 0 | 214 | 80 | -1 | nan | 64 | 45.012 | Experiments\runs_rl\ai_training5_batch_20260501_171723_20min\20260501_171735_tm2d_sac_AI_Training__5_delta_finished_progress_time_gas_steer\episode_metrics.csv |
| SAC | terminal_finished_progress_time | gas_steer | 20260501_171735_tm2d_sac_AI_Training__5_terminal_finished_progress_time_gas_steer | 265 | 8.284 | 5.714 | 3.539 | 0.036 | 0 | 174 | 91 | -1 | nan | 44 | 45.008 | Experiments\runs_rl\ai_training5_batch_20260501_171723_20min\20260501_171735_tm2d_sac_AI_Training__5_terminal_finished_progress_time_gas_steer\episode_metrics.csv |

## Generated Graphs

- `01_cumulative_best_dense_progress.png`
- `02_rolling_dense_progress.png`
- `03_rolling_episode_reward.png`
- `04_summary_bars.png`
