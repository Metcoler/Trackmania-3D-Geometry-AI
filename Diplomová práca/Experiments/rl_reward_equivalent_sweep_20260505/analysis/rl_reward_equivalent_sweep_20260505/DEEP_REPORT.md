# Deep RL Sweep Analysis

Root: `Experiments/runs_rl/reward_equivalent_aabb_tick_sweep_seed_2026050508`

All runs used the same TM2D task, AABB-clearance lidar, strict `gas_brake_steer` action layout, `32,16` ReLU policy, scalar delta reward equivalent to `(finished, progress, -time, -crashes)`, and 1000 episodes.

## High-Level Summary

| run_short | algorithm | physics_tick_profile | episodes | max_progress | last100_mean_progress | finish_count | first_finish_episode | best_finish_time | crash_count | timeout_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppo_fixed100 | PPO | fixed100 | 1000 | 100.000 | 55.307 | 43 | 849 | 26.910 | 880 | 77 |
| ppo_supervised_v2d | PPO | supervised_v2d | 1000 | 100.000 | 51.479 | 33 | 759 | 25.040 | 964 | 3 |
| sac_fixed100 | SAC | fixed100 | 1000 | 7.088 | 2.473 | 0 | -1 |  | 1000 | 0 |
| sac_supervised_v2d | SAC | supervised_v2d | 1000 | 5.158 | 3.124 | 0 | -1 |  | 1000 | 0 |
| td3_fixed100 | TD3 | fixed100 | 1000 | 7.329 | 3.310 | 0 | -1 |  | 1000 | 0 |
| td3_supervised_v2d | TD3 | supervised_v2d | 1000 | 6.572 | 3.243 | 0 | -1 |  | 1000 | 0 |

## Progress Thresholds

Episode where each run first reached the given continuous progress threshold. `-1` means the threshold was never reached.

| run_short | algorithm | physics_tick_profile | first_ge_5 | first_ge_10 | first_ge_25 | first_ge_50 | first_ge_75 | first_ge_90 | first_ge_100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppo_fixed100 | PPO | fixed100 | 56 | 171 | 590 | 635 | 744 | 804 | 849 |
| ppo_supervised_v2d | PPO | supervised_v2d | 88 | 251 | 645 | 645 | 751 | 751 | 759 |
| sac_fixed100 | SAC | fixed100 | 6 | -1 | -1 | -1 | -1 | -1 | -1 |
| sac_supervised_v2d | SAC | supervised_v2d | 179 | -1 | -1 | -1 | -1 | -1 | -1 |
| td3_fixed100 | TD3 | fixed100 | 3 | -1 | -1 | -1 | -1 | -1 | -1 |
| td3_supervised_v2d | TD3 | supervised_v2d | 3 | -1 | -1 | -1 | -1 | -1 | -1 |

## Late Training Stability

| run_short | phase | mean_progress | p90_progress | max_progress | finish_count | finish_rate | crash_rate | timeout_rate | best_finish_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppo_fixed100 | ep751_1000 | 54.504 | 100.000 | 100.000 | 43 | 0.172 | 0.640 | 0.188 | 26.910 |
| ppo_fixed100 | last100 | 55.307 | 100.000 | 100.000 | 24 | 0.240 | 0.580 | 0.180 | 26.910 |
| ppo_supervised_v2d | ep751_1000 | 39.753 | 100.000 | 100.000 | 33 | 0.132 | 0.856 | 0.012 | 25.040 |
| ppo_supervised_v2d | last100 | 51.479 | 100.000 | 100.000 | 22 | 0.220 | 0.780 | 0.000 | 25.040 |
| sac_fixed100 | ep751_1000 | 2.357 | 2.747 | 3.282 | 0 | 0.000 | 1.000 | 0.000 |  |
| sac_fixed100 | last100 | 2.473 | 2.888 | 3.282 | 0 | 0.000 | 1.000 | 0.000 |  |
| sac_supervised_v2d | ep751_1000 | 2.855 | 3.589 | 4.803 | 0 | 0.000 | 1.000 | 0.000 |  |
| sac_supervised_v2d | last100 | 3.124 | 3.857 | 4.803 | 0 | 0.000 | 1.000 | 0.000 |  |
| td3_fixed100 | ep751_1000 | 3.374 | 3.491 | 3.684 | 0 | 0.000 | 1.000 | 0.000 |  |
| td3_fixed100 | last100 | 3.310 | 3.326 | 3.343 | 0 | 0.000 | 1.000 | 0.000 |  |
| td3_supervised_v2d | ep751_1000 | 3.243 | 3.254 | 3.277 | 0 | 0.000 | 1.000 | 0.000 |  |
| td3_supervised_v2d | last100 | 3.243 | 3.254 | 3.269 | 0 | 0.000 | 1.000 | 0.000 |  |

## Finish Statistics

| run_short | algorithm | physics_tick_profile | finish_count | first_finish_episode | last_finish_episode | best_finish_time | median_finish_time | mean_finish_time | finish_count_last100 | best_time_last100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppo_fixed100 | PPO | fixed100 | 43 | 849 | 973 | 26.910 | 28.740 | 28.613 | 24 | 26.910 |
| ppo_supervised_v2d | PPO | supervised_v2d | 33 | 759 | 999 | 25.040 | 27.330 | 27.318 | 22 | 25.040 |
| sac_fixed100 | SAC | fixed100 | 0 | -1 | -1 |  |  |  | 0 |  |
| sac_supervised_v2d | SAC | supervised_v2d | 0 | -1 | -1 |  |  |  | 0 |  |
| td3_fixed100 | TD3 | fixed100 | 0 | -1 | -1 |  |  |  | 0 |  |
| td3_supervised_v2d | TD3 | supervised_v2d | 0 | -1 | -1 |  |  |  | 0 |  |

## Interpretation

- PPO is the only algorithm that learned to finish in this screening. Both PPO variants reached 100% progress and produced finishers.
- PPO fixed100 was more stable overall: 43 finishes, 77 timeouts, 880 crash episodes, and 55.3% last-100 mean progress.
- PPO supervised_v2d found the best single finish time at 25.04s, but was less stable: 33 finishes and 964 crash episodes with only 3 timeouts.
- SAC and TD3 did not solve the task. Their maximum progress stayed below 8%, and every episode ended in crash. In this reward/action setup they are negative evidence rather than competitive baselines.
- Variable physics did not prevent PPO from learning, but in this single-seed run it reduced stability compared with fixed100. For SAC/TD3, variable ticks did not rescue learning.
- The result supports using PPO as the RL baseline in the thesis comparison, while keeping GA as the stronger practical method unless longer or better-shaped RL training is introduced.

## Generated Figures

- `01_cumulative_best_progress.png`
- `02_rolling_progress.png`
- `03_rolling_episode_reward.png`
- `04_summary_bars.png`
- `05_first_progress_thresholds.png`
- `06_phase_mean_progress_heatmap.png`
- `07_episode_outcomes.png`
- `08_best_finish_time_so_far.png`

## Best Time Plot

`08_best_finish_time_so_far.png` shows the cumulative best finish time over episodes. Only PPO produced finishers in this sweep, so SAC and TD3 have no best-time curve.
