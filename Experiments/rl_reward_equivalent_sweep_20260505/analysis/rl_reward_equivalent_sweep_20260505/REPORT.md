# RL Run Comparison

Comparison of PPO, SAC and TD3 from Stable-Baselines3 over the same local TM2D environment.

- Best max progress: `PPO | fixed100 | delta_finished_progress_time_crashes | gas_brake_steer` with 100.00%.
- Best finish time: `PPO | supervised_v2d | delta_finished_progress_time_crashes | gas_brake_steer` with 25.040s.

## Summary

| run_label | run_short | algorithm | physics_tick_profile | episodes | max_progress | max_block_progress | last100_mean_progress | last100_mean_reward | finish_count | crash_count | timeout_count | first_finish_episode | best_finish_time | best_progress_episode | best_progress_time | source_csv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | fixed100 | delta_finished_progress_time_crashes | gas_brake_steer | ppo_fixed100 | PPO | fixed100 | 1000 | 100.000 | 100.000 | 55.307 | 289.205 | 43 | 880 | 77 | 849 | 26.910 | 849 | 29.740 | Experiments\runs_rl\reward_equivalent_aabb_tick_sweep_seed_2026050508\ppo_fixed100\episode_metrics.csv |
| PPO | supervised_v2d | delta_finished_progress_time_crashes | gas_brake_steer | ppo_supervised_v2d | PPO | supervised_v2d | 1000 | 100.000 | 100.000 | 51.479 | 265.880 | 33 | 964 | 3 | 759 | 25.040 | 759 | 28.450 | Experiments\runs_rl\reward_equivalent_aabb_tick_sweep_seed_2026050508\ppo_supervised_v2d\episode_metrics.csv |
| SAC | fixed100 | delta_finished_progress_time_crashes | gas_brake_steer | sac_fixed100 | SAC | fixed100 | 1000 | 7.088 | 5.714 | 2.473 | -0.233 | 0 | 1000 | 0 | -1 | nan | 57 | 14.190 | Experiments\runs_rl\reward_equivalent_aabb_tick_sweep_seed_2026050508\sac_fixed100\episode_metrics.csv |
| SAC | supervised_v2d | delta_finished_progress_time_crashes | gas_brake_steer | sac_supervised_v2d | SAC | supervised_v2d | 1000 | 5.158 | 2.857 | 3.124 | 0.089 | 0 | 1000 | 0 | -1 | nan | 179 | 5.060 | Experiments\runs_rl\reward_equivalent_aabb_tick_sweep_seed_2026050508\sac_supervised_v2d\episode_metrics.csv |
| TD3 | fixed100 | delta_finished_progress_time_crashes | gas_brake_steer | td3_fixed100 | TD3 | fixed100 | 1000 | 7.329 | 5.714 | 3.310 | 1.851 | 0 | 1000 | 0 | -1 | nan | 11 | 6.340 | Experiments\runs_rl\reward_equivalent_aabb_tick_sweep_seed_2026050508\td3_fixed100\episode_metrics.csv |
| TD3 | supervised_v2d | delta_finished_progress_time_crashes | gas_brake_steer | td3_supervised_v2d | TD3 | supervised_v2d | 1000 | 6.572 | 5.714 | 3.243 | 1.801 | 0 | 1000 | 0 | -1 | nan | 53 | 3.420 | Experiments\runs_rl\reward_equivalent_aabb_tick_sweep_seed_2026050508\td3_supervised_v2d\episode_metrics.csv |

## Generated Graphs

- `01_cumulative_best_progress.png`
- `02_rolling_progress.png`
- `03_rolling_episode_reward.png`
- `04_summary_bars.png`
