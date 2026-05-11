# Final GA Training Runs 20260507

## Summary

This curated package summarizes the three final GA training runs used at the end of thesis chapter 7.
The runs demonstrate the same lexicographic neuroevolution pipeline on a flat track, a height track, and a multi-surface track.

## Source Runs

### Flat track s jedným povrchom

- Source: `logs\tm_finetune_runs\20260510_092555_tm_finetune_map_single_surface_flat_v2d_asphalt_h48x24_p48_src_resume_population_gen_0170`
- Map: `single_surface_flat`
- Observation dimension: `34`
- Network: `48x24 relu,tanh`
- Ranking: `(finished, progress, -time, -crashes)`
- Population/parents/elites: `48/14/2`
- Mutation: `p=0.15`, `sigma=0.3`
- Logged generations: `202`
- First finish generation: `49`
- Total finishing individuals: `654`
- Best finish time: `25,92 s`
- Best progress: `100,00 %`
- Best trajectory: `trajectories/gen_0202_rank_001_idx_028_finish_1.npz`

### Track with height changes

- Source: `logs\tm_finetune_runs\20260506_160030_tm_seed_map_single_surface_height_v3d_asphalt_h48x24_p48_src_best_model`
- Map: `single_surface_height`
- Observation dimension: `48`
- Network: `48x24 relu,tanh`
- Ranking: `(finished, progress, -time, -crashes)`
- Population/parents/elites: `48/14/2`
- Mutation: `p=0.15`, `sigma=0.3`
- Logged generations: `110`
- First finish generation: `48`
- Total finishing individuals: `351`
- Best finish time: `32,55 s`
- Best progress: `100,00 %`
- Best trajectory: `trajectories/gen_0081_rank_001_idx_038_finish_1.npz`

### Flat track s rôznymi povrchmi

- Source: `logs\tm_finetune_runs\20260507_090226_tm_seed_map_multi_surface_flat_v2d_surface_h48x24_p48_src_best_model`
- Map: `multi_surface_flat`
- Observation dimension: `39`
- Network: `48x24 relu,tanh`
- Ranking: `(finished, progress, -time, -crashes)`
- Population/parents/elites: `48/14/2`
- Mutation: `p=0.15`, `sigma=0.3`
- Logged generations: `158`
- First finish generation: `128`
- Total finishing individuals: `125`
- Best finish time: `35,10 s`
- Best progress: `100,00 %`
- Best trajectory: `trajectories/gen_0151_rank_001_idx_042_finish_1.npz`

## Generated Files

- `summary.csv`
- `metadata.json`
- `Masters thesis\Latex\images\training_policy\final_ga_training_progress.pdf`
- `Masters thesis\Latex\images\training_policy\final_ga_training_trajectories.pdf`

## Interpretation Notes

- The x-axis in the progress figure is normalized because the logged runs have different lengths.
- These runs are final training evidence, not the final evaluation chapter.
- Trajectory colors use speed in km/h: colder colors indicate faster sections and warmer colors slower sections.
