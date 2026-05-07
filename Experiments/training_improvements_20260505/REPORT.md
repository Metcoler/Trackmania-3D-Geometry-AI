# TM2D training improvements comparison

## Status

- Experiment ID: `training_improvements_20260505`
- Category: `thesis_grade_and_diagnostic`
- Status: `mixed_single_seed_results`
- Curated at: `2026-05-05T23:09:45`

## Thesis Relevance

Comparison of selected GA training improvements after the reward and hyperparameter sweeps: mutation decay, mirror evaluation, multi-touch, variable tick, elite cache, MOO variants, and live TM overnight run.

## Interpretation

Variable physics tick with elite cache is the strongest positive result. First-finish decay is a useful tradeoff; mirror holdout and max-touches remain diagnostic rather than final improvements.

## Package Contents

- `runs/`: selected raw run logs copied from the working experiment folders.
- `analysis/`: summary tables, reports, and generated plots.
- `scripts/`: scripts needed to reproduce or analyze the experiment.
- `metadata.json`: source paths, keywords, and curation metadata.

## Notes

Large aggregate files such as `combined_individual_metrics.csv` are intentionally not copied into the thesis package. They remain local working artifacts if needed.
