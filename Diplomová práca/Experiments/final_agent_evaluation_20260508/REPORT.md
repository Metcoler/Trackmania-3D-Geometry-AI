# Final Agent Evaluation 20260508

This package prepares thesis figures for chapter 8.

## small_map transfer comparison

- Human reference: 17,89 s
- Diploma agent: 19,68 s
- Bachelor agent: 23,06 s
- Improvement over bachelor agent: 3,38 s
- Gap to human reference: 1,79 s

## Ranked final training finishers

- `single_surface_flat`: 82 finishers, best 27,96 s
- `single_surface_height`: 351 finishers, best 32,55 s
- `multi_surface_flat`: 125 finishers, best 35,10 s

## Human context on `single_surface_flat`

- Player time sample: 100 times from `Maps/GameFiles/casy_single_surface_flat.csv`
- Best player time: 23,05 s
- Median player time: 25,12 s
- Slowest player time: 29,68 s
- Diploma agent: 27,96 s
- Player times faster than agent: 96
- Player times slower than or equal to agent: 4
