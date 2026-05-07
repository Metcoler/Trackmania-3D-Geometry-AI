# GA Hyperparameter Sweep Analysis

## Input Roots

- `Experiments\runs_ga_hyperparam\pc1_selection_refinement_seed_2026050401`
- `Experiments\runs_ga_hyperparam\pc2_mutation_refinement_seed_2026050401`

## Summary

This analysis compares GA hyperparameters for the fixed reward tuple `(finished, progress, -time, -crashes)`.
The environment baseline is fixed FPS 100, AABB-clearance lidar, binary gas/brake, max time 30, dense progress, and disabled elite cache.

Loaded runs: `54`.
Incomplete runs: `0`.
Runs with cached elite evaluations: `0`.

This is a screening experiment. The tables below identify promising regions; they are not a thesis-final proof until the best candidates are repeated with another seed.

## Best Overall Candidates

| grid | grid_x | grid_y | compromise_score | mutation_prob | mutation_sigma | parent_count | elite_count | first_finish_generation | best_finish_time | last50_finish_rate | last50_mean_dense_progress | last50_crash_rate | last50_timeout_rate | last50_penalized_mean_time | max_cached_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selection | 14.000 | 1.000 | 0.378 | 0.200 | 0.200 | 14 | 1 | 105.000 | 17.700 | 0.408 | 55.896 | 0.588 | 0.004 | 26.229 | 0 |
| selection | 10.000 | 1.000 | 0.207 | 0.200 | 0.200 | 10 | 1 | 148.000 | 18.170 | 0.318 | 56.283 | 0.670 | 0.013 | 27.500 | 0 |
| selection | 12.000 | 2.000 | 0.099 | 0.200 | 0.200 | 12 | 2 | 86.000 | 18.590 | 0.302 | 47.622 | 0.672 | 0.025 | 27.788 | 0 |
| selection | 14.000 | 3.000 | 0.080 | 0.200 | 0.200 | 14 | 3 | 145.000 | 18.210 | 0.288 | 49.569 | 0.702 | 0.010 | 27.855 | 0 |
| mutation | 0.100 | 0.250 | 0.106 | 0.100 | 0.250 | 16 | 4 | 132.000 | 18.800 | 0.276 | 53.183 | 0.695 | 0.029 | 28.133 | 0 |
| mutation | 0.050 | 0.325 | 0.124 | 0.050 | 0.325 | 16 | 4 | 154.000 | 18.700 | 0.274 | 57.438 | 0.724 | 0.003 | 28.051 | 0 |
| mutation | 0.075 | 0.300 | 0.041 | 0.075 | 0.300 | 16 | 4 | 148.000 | 19.020 | 0.260 | 48.939 | 0.697 | 0.043 | 28.381 | 0 |
| selection | 8.000 | 2.000 | -0.036 | 0.200 | 0.200 | 8 | 2 | 104.000 | 18.690 | 0.250 | 45.110 | 0.733 | 0.017 | 28.329 | 0 |
| selection | 10.000 | 2.000 | -0.057 | 0.200 | 0.200 | 10 | 2 | 150.000 | 18.930 | 0.230 | 47.118 | 0.754 | 0.017 | 28.590 | 0 |
| selection | 10.000 | 4.000 | -0.088 | 0.200 | 0.200 | 10 | 4 | 137.000 | 18.530 | 0.220 | 44.422 | 0.743 | 0.036 | 28.643 | 0 |

## Best Compromise Candidates

Compromise score is `last50_finish_rate + last50_mean_dense_progress / 100 - last50_crash_rate - 0.25 * last50_timeout_rate`.
It is a screening helper, not a final fitness value.

| grid | grid_x | grid_y | compromise_score | mutation_prob | mutation_sigma | parent_count | elite_count | first_finish_generation | best_finish_time | last50_finish_rate | last50_mean_dense_progress | last50_crash_rate | last50_timeout_rate | last50_penalized_mean_time | max_cached_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selection | 14.000 | 1.000 | 0.378 | 0.200 | 0.200 | 14 | 1 | 105.000 | 17.700 | 0.408 | 55.896 | 0.588 | 0.004 | 26.229 | 0 |
| selection | 10.000 | 1.000 | 0.207 | 0.200 | 0.200 | 10 | 1 | 148.000 | 18.170 | 0.318 | 56.283 | 0.670 | 0.013 | 27.500 | 0 |
| mutation | 0.050 | 0.325 | 0.124 | 0.050 | 0.325 | 16 | 4 | 154.000 | 18.700 | 0.274 | 57.438 | 0.724 | 0.003 | 28.051 | 0 |
| mutation | 0.100 | 0.250 | 0.106 | 0.100 | 0.250 | 16 | 4 | 132.000 | 18.800 | 0.276 | 53.183 | 0.695 | 0.029 | 28.133 | 0 |
| selection | 12.000 | 2.000 | 0.099 | 0.200 | 0.200 | 12 | 2 | 86.000 | 18.590 | 0.302 | 47.622 | 0.672 | 0.025 | 27.788 | 0 |
| selection | 14.000 | 3.000 | 0.080 | 0.200 | 0.200 | 14 | 3 | 145.000 | 18.210 | 0.288 | 49.569 | 0.702 | 0.010 | 27.855 | 0 |
| mutation | 0.075 | 0.300 | 0.041 | 0.075 | 0.300 | 16 | 4 | 148.000 | 19.020 | 0.260 | 48.939 | 0.697 | 0.043 | 28.381 | 0 |
| mutation | 0.075 | 0.275 | -0.021 | 0.075 | 0.275 | 16 | 4 | 161.000 | 21.890 | 0.216 | 52.481 | 0.754 | 0.030 | 28.970 | 0 |
| selection | 8.000 | 2.000 | -0.036 | 0.200 | 0.200 | 8 | 2 | 104.000 | 18.690 | 0.250 | 45.110 | 0.733 | 0.017 | 28.329 | 0 |
| selection | 10.000 | 2.000 | -0.057 | 0.200 | 0.200 | 10 | 2 | 150.000 | 18.930 | 0.230 | 47.118 | 0.754 | 0.017 | 28.590 | 0 |

## Mutation Grid Candidates

| grid | grid_x | grid_y | compromise_score | mutation_prob | mutation_sigma | parent_count | elite_count | first_finish_generation | best_finish_time | last50_finish_rate | last50_mean_dense_progress | last50_crash_rate | last50_timeout_rate | last50_penalized_mean_time | max_cached_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mutation | 0.100 | 0.250 | 0.106 | 0.100 | 0.250 | 16 | 4 | 132.000 | 18.800 | 0.276 | 53.183 | 0.695 | 0.029 | 28.133 | 0 |
| mutation | 0.050 | 0.325 | 0.124 | 0.050 | 0.325 | 16 | 4 | 154.000 | 18.700 | 0.274 | 57.438 | 0.724 | 0.003 | 28.051 | 0 |
| mutation | 0.075 | 0.300 | 0.041 | 0.075 | 0.300 | 16 | 4 | 148.000 | 19.020 | 0.260 | 48.939 | 0.697 | 0.043 | 28.381 | 0 |
| mutation | 0.075 | 0.275 | -0.021 | 0.075 | 0.275 | 16 | 4 | 161.000 | 21.890 | 0.216 | 52.481 | 0.754 | 0.030 | 28.970 | 0 |
| mutation | 0.100 | 0.275 | -0.096 | 0.100 | 0.275 | 16 | 4 | 132.000 | 21.610 | 0.200 | 46.273 | 0.745 | 0.055 | 29.023 | 0 |
| mutation | 0.125 | 0.225 | -0.141 | 0.125 | 0.225 | 16 | 4 | 155.000 | 21.520 | 0.177 | 48.111 | 0.790 | 0.033 | 29.101 | 0 |
| mutation | 0.100 | 0.300 | -0.236 | 0.100 | 0.300 | 16 | 4 | 138.000 | 19.970 | 0.159 | 42.617 | 0.815 | 0.026 | 28.982 | 0 |
| mutation | 0.125 | 0.300 | -0.355 | 0.125 | 0.300 | 16 | 4 | 146.000 | 20.180 | 0.121 | 39.352 | 0.866 | 0.013 | 29.391 | 0 |

## Mutation Grid Compromise Candidates

| grid | grid_x | grid_y | compromise_score | mutation_prob | mutation_sigma | parent_count | elite_count | first_finish_generation | best_finish_time | last50_finish_rate | last50_mean_dense_progress | last50_crash_rate | last50_timeout_rate | last50_penalized_mean_time | max_cached_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mutation | 0.050 | 0.325 | 0.124 | 0.050 | 0.325 | 16 | 4 | 154.000 | 18.700 | 0.274 | 57.438 | 0.724 | 0.003 | 28.051 | 0 |
| mutation | 0.100 | 0.250 | 0.106 | 0.100 | 0.250 | 16 | 4 | 132.000 | 18.800 | 0.276 | 53.183 | 0.695 | 0.029 | 28.133 | 0 |
| mutation | 0.075 | 0.300 | 0.041 | 0.075 | 0.300 | 16 | 4 | 148.000 | 19.020 | 0.260 | 48.939 | 0.697 | 0.043 | 28.381 | 0 |
| mutation | 0.075 | 0.275 | -0.021 | 0.075 | 0.275 | 16 | 4 | 161.000 | 21.890 | 0.216 | 52.481 | 0.754 | 0.030 | 28.970 | 0 |
| mutation | 0.100 | 0.275 | -0.096 | 0.100 | 0.275 | 16 | 4 | 132.000 | 21.610 | 0.200 | 46.273 | 0.745 | 0.055 | 29.023 | 0 |
| mutation | 0.125 | 0.225 | -0.141 | 0.125 | 0.225 | 16 | 4 | 155.000 | 21.520 | 0.177 | 48.111 | 0.790 | 0.033 | 29.101 | 0 |
| mutation | 0.100 | 0.300 | -0.236 | 0.100 | 0.300 | 16 | 4 | 138.000 | 19.970 | 0.159 | 42.617 | 0.815 | 0.026 | 28.982 | 0 |
| mutation | 0.125 | 0.300 | -0.355 | 0.125 | 0.300 | 16 | 4 | 146.000 | 20.180 | 0.121 | 39.352 | 0.866 | 0.013 | 29.391 | 0 |

## Selection Pressure Grid Candidates

| grid | grid_x | grid_y | compromise_score | mutation_prob | mutation_sigma | parent_count | elite_count | first_finish_generation | best_finish_time | last50_finish_rate | last50_mean_dense_progress | last50_crash_rate | last50_timeout_rate | last50_penalized_mean_time | max_cached_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selection | 14.000 | 1.000 | 0.378 | 0.200 | 0.200 | 14 | 1 | 105.000 | 17.700 | 0.408 | 55.896 | 0.588 | 0.004 | 26.229 | 0 |
| selection | 10.000 | 1.000 | 0.207 | 0.200 | 0.200 | 10 | 1 | 148.000 | 18.170 | 0.318 | 56.283 | 0.670 | 0.013 | 27.500 | 0 |
| selection | 12.000 | 2.000 | 0.099 | 0.200 | 0.200 | 12 | 2 | 86.000 | 18.590 | 0.302 | 47.622 | 0.672 | 0.025 | 27.788 | 0 |
| selection | 14.000 | 3.000 | 0.080 | 0.200 | 0.200 | 14 | 3 | 145.000 | 18.210 | 0.288 | 49.569 | 0.702 | 0.010 | 27.855 | 0 |
| selection | 8.000 | 2.000 | -0.036 | 0.200 | 0.200 | 8 | 2 | 104.000 | 18.690 | 0.250 | 45.110 | 0.733 | 0.017 | 28.329 | 0 |
| selection | 10.000 | 2.000 | -0.057 | 0.200 | 0.200 | 10 | 2 | 150.000 | 18.930 | 0.230 | 47.118 | 0.754 | 0.017 | 28.590 | 0 |
| selection | 10.000 | 4.000 | -0.088 | 0.200 | 0.200 | 10 | 4 | 137.000 | 18.530 | 0.220 | 44.422 | 0.743 | 0.036 | 28.643 | 0 |
| selection | 8.000 | 3.000 | -0.163 | 0.200 | 0.200 | 8 | 3 | 98.000 | 17.590 | 0.219 | 38.857 | 0.768 | 0.013 | 28.042 | 0 |

## Selection Pressure Compromise Candidates

| grid | grid_x | grid_y | compromise_score | mutation_prob | mutation_sigma | parent_count | elite_count | first_finish_generation | best_finish_time | last50_finish_rate | last50_mean_dense_progress | last50_crash_rate | last50_timeout_rate | last50_penalized_mean_time | max_cached_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selection | 14.000 | 1.000 | 0.378 | 0.200 | 0.200 | 14 | 1 | 105.000 | 17.700 | 0.408 | 55.896 | 0.588 | 0.004 | 26.229 | 0 |
| selection | 10.000 | 1.000 | 0.207 | 0.200 | 0.200 | 10 | 1 | 148.000 | 18.170 | 0.318 | 56.283 | 0.670 | 0.013 | 27.500 | 0 |
| selection | 12.000 | 2.000 | 0.099 | 0.200 | 0.200 | 12 | 2 | 86.000 | 18.590 | 0.302 | 47.622 | 0.672 | 0.025 | 27.788 | 0 |
| selection | 14.000 | 3.000 | 0.080 | 0.200 | 0.200 | 14 | 3 | 145.000 | 18.210 | 0.288 | 49.569 | 0.702 | 0.010 | 27.855 | 0 |
| selection | 8.000 | 2.000 | -0.036 | 0.200 | 0.200 | 8 | 2 | 104.000 | 18.690 | 0.250 | 45.110 | 0.733 | 0.017 | 28.329 | 0 |
| selection | 10.000 | 2.000 | -0.057 | 0.200 | 0.200 | 10 | 2 | 150.000 | 18.930 | 0.230 | 47.118 | 0.754 | 0.017 | 28.590 | 0 |
| selection | 10.000 | 4.000 | -0.088 | 0.200 | 0.200 | 10 | 4 | 137.000 | 18.530 | 0.220 | 44.422 | 0.743 | 0.036 | 28.643 | 0 |
| selection | 16.000 | 4.000 | -0.140 | 0.200 | 0.200 | 16 | 4 | 158.000 | 17.970 | 0.207 | 40.135 | 0.735 | 0.058 | 28.618 | 0 |

## Edge Check

- `mutation` best by stability ranking is inside the tested grid: `prob_0100_sigma_0250`.
- `mutation` best by compromise score is on grid edge (minimum x, maximum y): `prob_0050_sigma_0325`.
- `selection` best by stability ranking is on grid edge (minimum y): `parent_count_014_elite_count_001`.
- `selection` best by compromise score is on grid edge (minimum y): `parent_count_014_elite_count_001`.

## Generated Plots

- `candidate_stability_vs_speed.png`
- `heatmap_mutation_best_finish_time.png`
- `heatmap_mutation_first_finish_generation.png`
- `heatmap_mutation_last50_crash_rate.png`
- `heatmap_mutation_last50_finish_rate.png`
- `heatmap_mutation_last50_mean_dense_progress.png`
- `heatmap_mutation_last50_penalized_mean_time.png`
- `heatmap_mutation_last50_timeout_rate.png`
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
