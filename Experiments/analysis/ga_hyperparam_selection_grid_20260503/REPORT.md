# GA Hyperparameter Sweep Analysis

## Input Roots

- `Experiments\runs_ga_hyperparam\pc2_selection_grid_seed_2026050311`

## Summary

This analysis compares GA hyperparameters for the fixed reward tuple `(finished, progress, -time, -crashes)`.
The environment baseline is fixed FPS 100, AABB-clearance lidar, binary gas/brake, max time 30, dense progress, and disabled elite cache.

Loaded runs: `25`.
Incomplete runs: `0`.
Runs with cached elite evaluations: `0`.

This is a screening experiment. The tables below identify promising regions; they are not a thesis-final proof until the best candidates are repeated with another seed.

## Best Overall Candidates

| grid | grid_x | grid_y | compromise_score | mutation_prob | mutation_sigma | parent_count | elite_count | first_finish_generation | best_finish_time | last50_finish_rate | last50_mean_dense_progress | last50_crash_rate | last50_timeout_rate | last50_penalized_mean_time | max_cached_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selection | 0.300 | 0.100 | 0.368 | 0.200 | 0.200 | 14 | 1 | 113.000 | 18.140 | 0.384 | 58.992 | 0.603 | 0.013 | 27.199 | 0 |
| selection | 0.200 | 0.100 | 0.180 | 0.200 | 0.200 | 10 | 1 | 118.000 | 18.020 | 0.317 | 53.241 | 0.664 | 0.019 | 28.120 | 0 |
| selection | 0.300 | 0.400 | -0.211 | 0.200 | 0.200 | 14 | 6 | 152.000 | 22.160 | 0.163 | 41.288 | 0.771 | 0.066 | 29.172 | 0 |
| selection | 0.100 | 0.200 | -0.354 | 0.200 | 0.200 | 4 | 1 | 124.000 | 18.030 | 0.141 | 35.124 | 0.842 | 0.017 | 28.961 | 0 |
| selection | 0.100 | 0.500 | -0.388 | 0.200 | 0.200 | 4 | 2 | 120.000 | 17.840 | 0.135 | 33.056 | 0.850 | 0.015 | 28.846 | 0 |
| selection | 0.100 | 0.300 | -0.414 | 0.200 | 0.200 | 4 | 1 | 78.000 | 17.040 | 0.132 | 32.213 | 0.868 | 0.001 | 28.627 | 0 |
| selection | 0.200 | 0.400 | -0.381 | 0.200 | 0.200 | 10 | 4 | 146.000 | 23.620 | 0.116 | 37.257 | 0.865 | 0.019 | 29.489 | 0 |
| selection | 0.100 | 0.100 | -0.445 | 0.200 | 0.200 | 4 | 1 | 76.000 | 19.020 | 0.113 | 32.134 | 0.877 | 0.009 | 29.188 | 0 |
| selection | 0.100 | 0.400 | -0.468 | 0.200 | 0.200 | 4 | 2 | 70.000 | 16.780 | 0.113 | 30.521 | 0.885 | 0.002 | 28.775 | 0 |
| selection | 0.200 | 0.200 | -0.412 | 0.200 | 0.200 | 10 | 2 | 168.000 | 23.220 | 0.085 | 37.500 | 0.858 | 0.058 | 29.769 | 0 |

## Best Compromise Candidates

Compromise score is `last50_finish_rate + last50_mean_dense_progress / 100 - last50_crash_rate - 0.25 * last50_timeout_rate`.
It is a screening helper, not a final fitness value.

| grid | grid_x | grid_y | compromise_score | mutation_prob | mutation_sigma | parent_count | elite_count | first_finish_generation | best_finish_time | last50_finish_rate | last50_mean_dense_progress | last50_crash_rate | last50_timeout_rate | last50_penalized_mean_time | max_cached_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selection | 0.300 | 0.100 | 0.368 | 0.200 | 0.200 | 14 | 1 | 113.000 | 18.140 | 0.384 | 58.992 | 0.603 | 0.013 | 27.199 | 0 |
| selection | 0.200 | 0.100 | 0.180 | 0.200 | 0.200 | 10 | 1 | 118.000 | 18.020 | 0.317 | 53.241 | 0.664 | 0.019 | 28.120 | 0 |
| selection | 0.300 | 0.400 | -0.211 | 0.200 | 0.200 | 14 | 6 | 152.000 | 22.160 | 0.163 | 41.288 | 0.771 | 0.066 | 29.172 | 0 |
| selection | 0.100 | 0.200 | -0.354 | 0.200 | 0.200 | 4 | 1 | 124.000 | 18.030 | 0.141 | 35.124 | 0.842 | 0.017 | 28.961 | 0 |
| selection | 0.200 | 0.400 | -0.381 | 0.200 | 0.200 | 10 | 4 | 146.000 | 23.620 | 0.116 | 37.257 | 0.865 | 0.019 | 29.489 | 0 |
| selection | 0.100 | 0.500 | -0.388 | 0.200 | 0.200 | 4 | 2 | 120.000 | 17.840 | 0.135 | 33.056 | 0.850 | 0.015 | 28.846 | 0 |
| selection | 0.200 | 0.200 | -0.412 | 0.200 | 0.200 | 10 | 2 | 168.000 | 23.220 | 0.085 | 37.500 | 0.858 | 0.058 | 29.769 | 0 |
| selection | 0.100 | 0.300 | -0.414 | 0.200 | 0.200 | 4 | 1 | 78.000 | 17.040 | 0.132 | 32.213 | 0.868 | 0.001 | 28.627 | 0 |
| selection | 0.100 | 0.100 | -0.445 | 0.200 | 0.200 | 4 | 1 | 76.000 | 19.020 | 0.113 | 32.134 | 0.877 | 0.009 | 29.188 | 0 |
| selection | 0.100 | 0.400 | -0.468 | 0.200 | 0.200 | 4 | 2 | 70.000 | 16.780 | 0.113 | 30.521 | 0.885 | 0.002 | 28.775 | 0 |

## Mutation Grid Candidates

_No data._

## Mutation Grid Compromise Candidates

_No data._

## Selection Pressure Grid Candidates

| grid | grid_x | grid_y | compromise_score | mutation_prob | mutation_sigma | parent_count | elite_count | first_finish_generation | best_finish_time | last50_finish_rate | last50_mean_dense_progress | last50_crash_rate | last50_timeout_rate | last50_penalized_mean_time | max_cached_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selection | 0.300 | 0.100 | 0.368 | 0.200 | 0.200 | 14 | 1 | 113.000 | 18.140 | 0.384 | 58.992 | 0.603 | 0.013 | 27.199 | 0 |
| selection | 0.200 | 0.100 | 0.180 | 0.200 | 0.200 | 10 | 1 | 118.000 | 18.020 | 0.317 | 53.241 | 0.664 | 0.019 | 28.120 | 0 |
| selection | 0.300 | 0.400 | -0.211 | 0.200 | 0.200 | 14 | 6 | 152.000 | 22.160 | 0.163 | 41.288 | 0.771 | 0.066 | 29.172 | 0 |
| selection | 0.100 | 0.200 | -0.354 | 0.200 | 0.200 | 4 | 1 | 124.000 | 18.030 | 0.141 | 35.124 | 0.842 | 0.017 | 28.961 | 0 |
| selection | 0.100 | 0.500 | -0.388 | 0.200 | 0.200 | 4 | 2 | 120.000 | 17.840 | 0.135 | 33.056 | 0.850 | 0.015 | 28.846 | 0 |
| selection | 0.100 | 0.300 | -0.414 | 0.200 | 0.200 | 4 | 1 | 78.000 | 17.040 | 0.132 | 32.213 | 0.868 | 0.001 | 28.627 | 0 |
| selection | 0.200 | 0.400 | -0.381 | 0.200 | 0.200 | 10 | 4 | 146.000 | 23.620 | 0.116 | 37.257 | 0.865 | 0.019 | 29.489 | 0 |
| selection | 0.100 | 0.100 | -0.445 | 0.200 | 0.200 | 4 | 1 | 76.000 | 19.020 | 0.113 | 32.134 | 0.877 | 0.009 | 29.188 | 0 |

## Selection Pressure Compromise Candidates

| grid | grid_x | grid_y | compromise_score | mutation_prob | mutation_sigma | parent_count | elite_count | first_finish_generation | best_finish_time | last50_finish_rate | last50_mean_dense_progress | last50_crash_rate | last50_timeout_rate | last50_penalized_mean_time | max_cached_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selection | 0.300 | 0.100 | 0.368 | 0.200 | 0.200 | 14 | 1 | 113.000 | 18.140 | 0.384 | 58.992 | 0.603 | 0.013 | 27.199 | 0 |
| selection | 0.200 | 0.100 | 0.180 | 0.200 | 0.200 | 10 | 1 | 118.000 | 18.020 | 0.317 | 53.241 | 0.664 | 0.019 | 28.120 | 0 |
| selection | 0.300 | 0.400 | -0.211 | 0.200 | 0.200 | 14 | 6 | 152.000 | 22.160 | 0.163 | 41.288 | 0.771 | 0.066 | 29.172 | 0 |
| selection | 0.100 | 0.200 | -0.354 | 0.200 | 0.200 | 4 | 1 | 124.000 | 18.030 | 0.141 | 35.124 | 0.842 | 0.017 | 28.961 | 0 |
| selection | 0.200 | 0.400 | -0.381 | 0.200 | 0.200 | 10 | 4 | 146.000 | 23.620 | 0.116 | 37.257 | 0.865 | 0.019 | 29.489 | 0 |
| selection | 0.100 | 0.500 | -0.388 | 0.200 | 0.200 | 4 | 2 | 120.000 | 17.840 | 0.135 | 33.056 | 0.850 | 0.015 | 28.846 | 0 |
| selection | 0.200 | 0.200 | -0.412 | 0.200 | 0.200 | 10 | 2 | 168.000 | 23.220 | 0.085 | 37.500 | 0.858 | 0.058 | 29.769 | 0 |
| selection | 0.100 | 0.300 | -0.414 | 0.200 | 0.200 | 4 | 1 | 78.000 | 17.040 | 0.132 | 32.213 | 0.868 | 0.001 | 28.627 | 0 |

## Edge Check

- `selection` best by stability ranking is on grid edge (minimum y): `parents_ratio_030_elites_ratio_010_p14_e1`.
- `selection` best by compromise score is on grid edge (minimum y): `parents_ratio_030_elites_ratio_010_p14_e1`.

## Generated Plots

- `candidate_stability_vs_speed.png`
- `heatmap_selection_best_finish_time.png`
- `heatmap_selection_first_finish_generation.png`
- `heatmap_selection_last50_crash_rate.png`
- `heatmap_selection_last50_finish_rate.png`
- `heatmap_selection_last50_mean_dense_progress.png`
- `heatmap_selection_last50_penalized_mean_time.png`
- `heatmap_selection_last50_timeout_rate.png`

## Reading Guide

- `first_finish_generation` measures how quickly a configuration discovers a complete lap.
- `last50_finish_rate` measures late training stability, not only one lucky finisher.
- `best_finish_time` measures speed, but should not override stability by itself.
- `last50_penalized_mean_time` treats unfinished individuals as `max_time`, so it combines finish quality and failure rate.
- If the best value lies on a grid edge, the next experiment should be a smaller refinement grid around that edge.
