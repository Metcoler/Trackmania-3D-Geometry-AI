# Supervised-seeded GA initialization

## Status

- Experiment ID: `ga_supervised_seeded_20260505`
- Category: `thesis_grade_hybrid`
- Status: `single_seed_positive_with_negative_control`
- Curated at: `2026-05-05T23:09:46`

## Thesis Relevance

Hybrid behavior-cloning initialization plus GA fine-tuning experiment, compared against the random initial population baseline.

## Interpretation

Dense weight-noise seeding produced a strong positive result and the best time in the comparison. Sparse supervised seeding is a useful negative control showing that naive BC seeding is too conservative.

## Package Contents

- `runs/`: selected raw run logs copied from the working experiment folders.
- `analysis/`: summary tables, reports, and generated plots.
- `scripts/`: scripts needed to reproduce or analyze the experiment.
- `metadata.json`: source paths, keywords, and curation metadata.

## Notes

Large aggregate files such as `combined_individual_metrics.csv` are intentionally not copied into the thesis package. They remain local working artifacts if needed.
