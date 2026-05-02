# RL Run Comparison

- Best max dense progress: `PPO | delta_finished_progress_time | gas_steer` with 100.00%.
- Best finish time: `PPO | delta_finished_progress_time | gas_steer` with 19.224s.

## Summary

| run_label | run_short | episodes | max_dense_progress | max_progress | last100_mean_dense | last100_mean_reward | finish_count | crash_count | timeout_count | first_finish_episode | best_finish_time | best_dense_episode | best_dense_time | source_csv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | delta_finished_progress_time | gas_steer | 20260501_182025_tm2d_ppo_AI_Training__5_delta_finished_progress_time_gas_steer | 3244 | 100.000 | 100.000 | 81.138 | 1.559 | 446 | 2532 | 266 | 2168 | 19.224 | 2168 | 34.537 | C:\Users\sampu\Desktop\Trackmania-BC\Experiments\runs_rl\ai_training5_ppo_sac_td3_20260501_182013_2h\20260501_182025_tm2d_ppo_AI_Training__5_delta_finished_progress_time_gas_steer\episode_metrics.csv |
| SAC | delta_finished_progress_time | gas_steer | 20260501_182025_tm2d_sac_AI_Training__5_delta_finished_progress_time_gas_steer | 1913 | 7.963 | 5.714 | 3.589 | 0.036 | 0 | 1277 | 636 | -1 | nan | 977 | 45.009 | C:\Users\sampu\Desktop\Trackmania-BC\Experiments\runs_rl\ai_training5_ppo_sac_td3_20260501_182013_2h\20260501_182025_tm2d_sac_AI_Training__5_delta_finished_progress_time_gas_steer\episode_metrics.csv |
| TD3 | delta_finished_progress_time | gas_steer | 20260501_182025_tm2d_td3_AI_Training__5_delta_finished_progress_time_gas_steer | 8421 | 100.000 | 100.000 | 38.827 | 0.397 | 1 | 8365 | 55 | 469 | 21.268 | 469 | 21.268 | C:\Users\sampu\Desktop\Trackmania-BC\Experiments\runs_rl\ai_training5_ppo_sac_td3_20260501_182013_2h\20260501_182025_tm2d_td3_AI_Training__5_delta_finished_progress_time_gas_steer\episode_metrics.csv |

## Generated Graphs

- `01_cumulative_best_dense_progress.png`
- `02_rolling_dense_progress.png`
- `03_rolling_episode_reward.png`
- `04_summary_bars.png`
