# RL reward-equivalent PPO/SAC/TD3 sweep

## Status

- Experiment ID: `rl_reward_equivalent_sweep_20260505`
- Category: `comparison_branch`
- Status: `single_seed_screening`
- Curated at: `2026-05-05T23:09:45`

## Thesis Relevance

Stable-Baselines3 PPO, SAC, and TD3 comparison using the same TM2D environment, AABB lidar, strict gas/brake/steer actions, and a scalarized equivalent of the GA tuple.

## Interpretation

Only PPO learned to finish in this screening. SAC and TD3 were useful negative comparisons under the tested setup; GA remains the stronger practical baseline.

## Package Contents

- `runs/`: selected raw run logs copied from the working experiment folders.
- `analysis/`: summary tables, reports, and generated plots.
- `scripts/`: scripts needed to reproduce or analyze the experiment.
- `metadata.json`: source paths, keywords, and curation metadata.

## Notes

Large aggregate files such as `combined_individual_metrics.csv` are intentionally not copied into the thesis package. They remain local working artifacts if needed.
