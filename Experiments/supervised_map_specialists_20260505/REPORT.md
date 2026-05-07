# Supervised Map Specialists 20260505

## Summary
This package contains the supervised map-specialist result used to visualize how a small map-specific imitation model behaves compared with human teacher trajectories.

Three supervised specialists were trained from newly collected v3d/surface datasets:

- `single_surface_flat`
- `multi_surface_flat`
- `single_surface_height`

Each model uses observation dimension `53`, hidden architecture `48,24`, hidden activations `relu,tanh`, target action mode, height features, surface features and mirror augmentation.

## Main Thesis Figure
The currently validated thesis-ready figure is:

`analysis/single_surface_flat/single_surface_flat_teacher_agent_paths.png`

It overlays:

- Track map background with start, finish, road and edge legend.
- Ten teacher trajectories from supervised data.
- The trained supervised agent rollout with the same global speed color scale.
- Crash/touch markers as black dots.

The shared speed gradient makes it possible to discuss where the agent differs from teacher driving, for example where it slows down, turns differently, or touches the wall.

## Result Status
For `single_surface_flat`, the deterministic TM2D replay finished the track:

- finished: `1`
- progress: `100.0`
- time: `43.39`
- crashes/touches: `32`
- left road surface: `0`

This is useful as a qualitative supervised-learning diagnostic figure, not as proof of a final autonomous racing policy. The graph is mainly evidence for the limitation of pure supervised imitation: the model can follow the track, but small trajectory deviations can accumulate into wall contacts.

## Included Files
- `analysis/`: generated figures, rollout metrics, trajectory file and training summaries.
- `runs/`: small supervised training outputs and best models for the three map specialists.
- `scripts/`: reproduction scripts used to train and analyze this result.

Raw supervised datasets are not duplicated here to avoid unnecessary package size. Their source paths are listed in `metadata.json`.
