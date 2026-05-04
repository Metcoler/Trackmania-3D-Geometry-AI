# Lexicographic reward sweep with AABB-clearance lidar

## Status

- Experiment ID: `lex_reward_aabb_lidar_fixed100_20260503`
- Category: `thesis_grade`
- Status: `single_seed_strong_signal`
- Curated at: `2026-05-04T13:13:05`

## Thesis Relevance

Main reward-function comparison after switching to AABB-clearance lidar, fixed FPS 100, binary gas/brake, no elite cache.

## Interpretation

Current evidence favors (finished, progress, -time, -crashes) as the base lexicographic GA ranking tuple.

## Package Contents

- `runs/`: selected raw run logs copied from the working experiment folders.
- `analysis/`: summary tables, reports, and generated plots.
- `scripts/`: scripts needed to reproduce or analyze the experiment.
- `metadata.json`: source paths, keywords, and curation metadata.

## Notes

Large aggregate files such as `combined_individual_metrics.csv` are intentionally not copied into the thesis package. They remain local working artifacts if needed.
