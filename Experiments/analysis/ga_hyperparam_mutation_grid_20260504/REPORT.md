# GA Hyperparameter Sweep Analysis

## Input Roots

- `Experiments\runs_ga_hyperparam\pc1_mutation_grid_seed_2026050311`

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
| mutation | 0.100 | 0.300 | 0.208 | 0.100 | 0.300 | 16 | 4 | 114.000 | 19.120 | 0.328 | 53.259 | 0.645 | 0.027 | 27.861 | 0 |
| mutation | 0.100 | 0.250 | 0.243 | 0.100 | 0.250 | 16 | 4 | 124.000 | 18.900 | 0.328 | 58.163 | 0.664 | 0.009 | 27.800 | 0 |
| mutation | 0.150 | 0.200 | 0.200 | 0.150 | 0.200 | 16 | 4 | 137.000 | 21.270 | 0.302 | 56.835 | 0.660 | 0.038 | 28.393 | 0 |
| mutation | 0.150 | 0.250 | 0.118 | 0.150 | 0.250 | 16 | 4 | 105.000 | 17.000 | 0.300 | 50.571 | 0.684 | 0.016 | 27.309 | 0 |
| mutation | 0.300 | 0.150 | 0.034 | 0.300 | 0.150 | 16 | 4 | 104.000 | 20.640 | 0.267 | 48.808 | 0.718 | 0.015 | 28.469 | 0 |
| mutation | 0.250 | 0.300 | -0.258 | 0.250 | 0.300 | 16 | 4 | 102.000 | 18.960 | 0.167 | 37.736 | 0.792 | 0.041 | 28.747 | 0 |
| mutation | 0.150 | 0.300 | -0.225 | 0.150 | 0.300 | 16 | 4 | 125.000 | 19.710 | 0.167 | 41.812 | 0.803 | 0.030 | 28.885 | 0 |
| mutation | 0.200 | 0.300 | -0.212 | 0.200 | 0.300 | 16 | 4 | 103.000 | 20.010 | 0.161 | 41.061 | 0.766 | 0.073 | 28.940 | 0 |
| mutation | 0.150 | 0.150 | -0.218 | 0.150 | 0.150 | 16 | 4 | 167.000 | 22.780 | 0.125 | 49.854 | 0.830 | 0.045 | 29.565 | 0 |
| mutation | 0.250 | 0.250 | -0.437 | 0.250 | 0.250 | 16 | 4 | 138.000 | 19.790 | 0.120 | 31.227 | 0.866 | 0.013 | 29.130 | 0 |

## Best Compromise Candidates

Compromise score is `last50_finish_rate + last50_mean_dense_progress / 100 - last50_crash_rate - 0.25 * last50_timeout_rate`.
It is a screening helper, not a final fitness value.

| grid | grid_x | grid_y | compromise_score | mutation_prob | mutation_sigma | parent_count | elite_count | first_finish_generation | best_finish_time | last50_finish_rate | last50_mean_dense_progress | last50_crash_rate | last50_timeout_rate | last50_penalized_mean_time | max_cached_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mutation | 0.100 | 0.250 | 0.243 | 0.100 | 0.250 | 16 | 4 | 124.000 | 18.900 | 0.328 | 58.163 | 0.664 | 0.009 | 27.800 | 0 |
| mutation | 0.100 | 0.300 | 0.208 | 0.100 | 0.300 | 16 | 4 | 114.000 | 19.120 | 0.328 | 53.259 | 0.645 | 0.027 | 27.861 | 0 |
| mutation | 0.150 | 0.200 | 0.200 | 0.150 | 0.200 | 16 | 4 | 137.000 | 21.270 | 0.302 | 56.835 | 0.660 | 0.038 | 28.393 | 0 |
| mutation | 0.150 | 0.250 | 0.118 | 0.150 | 0.250 | 16 | 4 | 105.000 | 17.000 | 0.300 | 50.571 | 0.684 | 0.016 | 27.309 | 0 |
| mutation | 0.300 | 0.150 | 0.034 | 0.300 | 0.150 | 16 | 4 | 104.000 | 20.640 | 0.267 | 48.808 | 0.718 | 0.015 | 28.469 | 0 |
| mutation | 0.200 | 0.300 | -0.212 | 0.200 | 0.300 | 16 | 4 | 103.000 | 20.010 | 0.161 | 41.061 | 0.766 | 0.073 | 28.940 | 0 |
| mutation | 0.150 | 0.150 | -0.218 | 0.150 | 0.150 | 16 | 4 | 167.000 | 22.780 | 0.125 | 49.854 | 0.830 | 0.045 | 29.565 | 0 |
| mutation | 0.150 | 0.300 | -0.225 | 0.150 | 0.300 | 16 | 4 | 125.000 | 19.710 | 0.167 | 41.812 | 0.803 | 0.030 | 28.885 | 0 |
| mutation | 0.250 | 0.300 | -0.258 | 0.250 | 0.300 | 16 | 4 | 102.000 | 18.960 | 0.167 | 37.736 | 0.792 | 0.041 | 28.747 | 0 |
| mutation | 0.300 | 0.200 | -0.409 | 0.300 | 0.200 | 16 | 4 | 142.000 | 23.510 | 0.108 | 36.331 | 0.876 | 0.016 | 29.491 | 0 |

## Mutation Grid Candidates

| grid | grid_x | grid_y | compromise_score | mutation_prob | mutation_sigma | parent_count | elite_count | first_finish_generation | best_finish_time | last50_finish_rate | last50_mean_dense_progress | last50_crash_rate | last50_timeout_rate | last50_penalized_mean_time | max_cached_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mutation | 0.100 | 0.300 | 0.208 | 0.100 | 0.300 | 16 | 4 | 114.000 | 19.120 | 0.328 | 53.259 | 0.645 | 0.027 | 27.861 | 0 |
| mutation | 0.100 | 0.250 | 0.243 | 0.100 | 0.250 | 16 | 4 | 124.000 | 18.900 | 0.328 | 58.163 | 0.664 | 0.009 | 27.800 | 0 |
| mutation | 0.150 | 0.200 | 0.200 | 0.150 | 0.200 | 16 | 4 | 137.000 | 21.270 | 0.302 | 56.835 | 0.660 | 0.038 | 28.393 | 0 |
| mutation | 0.150 | 0.250 | 0.118 | 0.150 | 0.250 | 16 | 4 | 105.000 | 17.000 | 0.300 | 50.571 | 0.684 | 0.016 | 27.309 | 0 |
| mutation | 0.300 | 0.150 | 0.034 | 0.300 | 0.150 | 16 | 4 | 104.000 | 20.640 | 0.267 | 48.808 | 0.718 | 0.015 | 28.469 | 0 |
| mutation | 0.250 | 0.300 | -0.258 | 0.250 | 0.300 | 16 | 4 | 102.000 | 18.960 | 0.167 | 37.736 | 0.792 | 0.041 | 28.747 | 0 |
| mutation | 0.150 | 0.300 | -0.225 | 0.150 | 0.300 | 16 | 4 | 125.000 | 19.710 | 0.167 | 41.812 | 0.803 | 0.030 | 28.885 | 0 |
| mutation | 0.200 | 0.300 | -0.212 | 0.200 | 0.300 | 16 | 4 | 103.000 | 20.010 | 0.161 | 41.061 | 0.766 | 0.073 | 28.940 | 0 |

## Mutation Grid Compromise Candidates

| grid | grid_x | grid_y | compromise_score | mutation_prob | mutation_sigma | parent_count | elite_count | first_finish_generation | best_finish_time | last50_finish_rate | last50_mean_dense_progress | last50_crash_rate | last50_timeout_rate | last50_penalized_mean_time | max_cached_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mutation | 0.100 | 0.250 | 0.243 | 0.100 | 0.250 | 16 | 4 | 124.000 | 18.900 | 0.328 | 58.163 | 0.664 | 0.009 | 27.800 | 0 |
| mutation | 0.100 | 0.300 | 0.208 | 0.100 | 0.300 | 16 | 4 | 114.000 | 19.120 | 0.328 | 53.259 | 0.645 | 0.027 | 27.861 | 0 |
| mutation | 0.150 | 0.200 | 0.200 | 0.150 | 0.200 | 16 | 4 | 137.000 | 21.270 | 0.302 | 56.835 | 0.660 | 0.038 | 28.393 | 0 |
| mutation | 0.150 | 0.250 | 0.118 | 0.150 | 0.250 | 16 | 4 | 105.000 | 17.000 | 0.300 | 50.571 | 0.684 | 0.016 | 27.309 | 0 |
| mutation | 0.300 | 0.150 | 0.034 | 0.300 | 0.150 | 16 | 4 | 104.000 | 20.640 | 0.267 | 48.808 | 0.718 | 0.015 | 28.469 | 0 |
| mutation | 0.200 | 0.300 | -0.212 | 0.200 | 0.300 | 16 | 4 | 103.000 | 20.010 | 0.161 | 41.061 | 0.766 | 0.073 | 28.940 | 0 |
| mutation | 0.150 | 0.150 | -0.218 | 0.150 | 0.150 | 16 | 4 | 167.000 | 22.780 | 0.125 | 49.854 | 0.830 | 0.045 | 29.565 | 0 |
| mutation | 0.150 | 0.300 | -0.225 | 0.150 | 0.300 | 16 | 4 | 125.000 | 19.710 | 0.167 | 41.812 | 0.803 | 0.030 | 28.885 | 0 |

## Selection Pressure Grid Candidates

_No data._

## Selection Pressure Compromise Candidates

_No data._

## Edge Check

- `mutation` best by stability ranking is on grid edge (minimum x, maximum y): `prob_010_sigma_030`.
- `mutation` best by compromise score is on grid edge (minimum x): `prob_010_sigma_025`.

## Generated Plots

- `candidate_stability_vs_speed.png`
- `heatmap_mutation_best_finish_time.png`
- `heatmap_mutation_first_finish_generation.png`
- `heatmap_mutation_last50_crash_rate.png`
- `heatmap_mutation_last50_finish_rate.png`
- `heatmap_mutation_last50_mean_dense_progress.png`
- `heatmap_mutation_last50_penalized_mean_time.png`
- `heatmap_mutation_last50_timeout_rate.png`

## Reading Guide

- `first_finish_generation` measures how quickly a configuration discovers a complete lap.
- `last50_finish_rate` measures late training stability, not only one lucky finisher.
- `best_finish_time` measures speed, but should not override stability by itself.
- `last50_penalized_mean_time` treats unfinished individuals as `max_time`, so it combines finish quality and failure rate.
- If the best value lies on a grid edge, the next experiment should be a smaller refinement grid around that edge.
