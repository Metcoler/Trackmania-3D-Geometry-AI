# Vehicle AABB hitbox analysis

## Status

- Experiment ID: `vehicle_hitbox_aabb_20260503`
- Category: `thesis_grade`
- Status: `empirical_model`
- Curated at: `2026-05-04T13:13:09`

## Thesis Relevance

Justification for replacing a global raw lidar threshold with AABB-relative clearance lidar.

## Interpretation

The selected AABB is empirical, derived from mesh estimates and sanity-checked against near-contact supervised data.

## Package Contents

- `runs/`: selected raw run logs copied from the working experiment folders.
- `analysis/`: summary tables, reports, and generated plots.
- `scripts/`: scripts needed to reproduce or analyze the experiment.
- `metadata.json`: source paths, keywords, and curation metadata.

## Notes

Large aggregate files such as `combined_individual_metrics.csv` are intentionally not copied into the thesis package. They remain local working artifacts if needed.
