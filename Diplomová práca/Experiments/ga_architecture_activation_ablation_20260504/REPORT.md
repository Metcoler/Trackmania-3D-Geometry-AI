# Closed-loop architecture activation ablation

## Status

- Experiment ID: `ga_architecture_activation_ablation_20260504`
- Category: `thesis_grade`
- Status: `closed_loop_single_seed`
- Curated at: `2026-05-04T13:13:10`

## Thesis Relevance

Closed-loop GA comparison of 32x16/48x24 and relu,tanh versus relu,relu under the selected reward tuple.

## Interpretation

relu,tanh outperformed relu,relu in the closed-loop ablation; 48x24 is the stronger candidate, while 32x16 remains the cheaper experimental baseline.

## Package Contents

- `runs/`: selected raw run logs copied from the working experiment folders.
- `analysis/`: summary tables, reports, and generated plots.
- `scripts/`: scripts needed to reproduce or analyze the experiment.
- `metadata.json`: source paths, keywords, and curation metadata.

## Notes

Large aggregate files such as `combined_individual_metrics.csv` are intentionally not copied into the thesis package. They remain local working artifacts if needed.
