# GA mutation probability and sigma grid

## Status

- Experiment ID: `ga_mutation_grid_20260504`
- Category: `thesis_grade`
- Status: `screening_single_seed`
- Curated at: `2026-05-04T13:13:09`

## Thesis Relevance

Mutation probability/sigma screening for the selected reward tuple and baseline selection settings.

## Interpretation

Best region suggests mutating fewer weights with medium-to-larger steps, around mutation_prob=0.10 and mutation_sigma=0.25-0.30.

## Package Contents

- `runs/`: selected raw run logs copied from the working experiment folders.
- `analysis/`: summary tables, reports, and generated plots.
- `scripts/`: scripts needed to reproduce or analyze the experiment.
- `metadata.json`: source paths, keywords, and curation metadata.

## Notes

Large aggregate files such as `combined_individual_metrics.csv` are intentionally not copied into the thesis package. They remain local working artifacts if needed.
