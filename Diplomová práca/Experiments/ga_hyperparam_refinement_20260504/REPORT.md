# GA Hyperparameter Refinement Sweep 2026-05-04

## Status

Thesis-grade screening experiment. The sweep is complete and internally consistent:

- `54/54` runs completed.
- `24` selection-pressure refinement runs.
- `30` mutation probability/sigma refinement runs.
- `cached_evaluations = 0` in all runs.
- Shared baseline: fixed FPS `100`, AABB-clearance lidar, binary gas/brake, max time `30`, dense progress, ranking `(finished, progress, -time, -crashes)`.

## Main Result

The best practical configuration from this refinement is:

`population=48`, `parent_count=14`, `elite_count=1`, `mutation_prob=0.10`, `mutation_sigma=0.25`

The strongest selection-pressure run was:

`parent_count=14`, `elite_count=1`

Key metrics:

- First finish generation: `105`
- Best finish time: `17.70 s`
- Last50 finish rate: `40.83 %`
- Last50 mean dense progress: `55.90`
- Last50 crash rate: `58.79 %`
- Last50 penalized mean time: `26.23 s`

For mutation parameters, the safest interior candidate is:

`mutation_prob=0.10`, `mutation_sigma=0.25`

The strongest edge candidate is:

`mutation_prob=0.05`, `mutation_sigma=0.325`

This edge result is useful, but it should be repeated or refined before treating it as the final optimum.

## Interpretation

The experiment suggests that this GA benefits from moderate selection pressure and a very small elite set. A single elite preserves the best direction without freezing too much of the population. The mutation results point toward less frequent but stronger mutations; too weak mutation (`sigma=0.20`) often failed to explore, while high mutation probability became destructive.

This package should be used as the main hyperparameter screening evidence instead of the older coarse selection grid. The older ratio-based selection analysis was moved to soft-delete because the refined `parent_count x elite_count` grid is more directly interpretable.

## Package Contents

- `runs/`: raw run directories for the selection and mutation refinement grids.
- `analysis/ga_hyperparam_refinement_20260504/`: generated summary tables, heatmaps, and detailed reports.
- `scripts/`: launch scripts and analyzer used to reproduce the package.

The redundant `combined_individual_metrics.csv` was not kept in the package because it was very large and all per-run individual metrics remain available inside `runs/`.

## Recommended Next Step

Use the configuration below as the next baseline:

`population=48`, `parent_count=14`, `elite_count=1`, `mutation_prob=0.10`, `mutation_sigma=0.25`

For a confirmation run with another seed, compare it against:

- `parent_count=10`, `elite_count=1`, `mutation_prob=0.10`, `mutation_sigma=0.25`
- `parent_count=14`, `elite_count=1`, `mutation_prob=0.05`, `mutation_sigma=0.325`
- optionally `parent_count=14`, `elite_count=1`, `mutation_prob=0.05`, `mutation_sigma=0.35`
