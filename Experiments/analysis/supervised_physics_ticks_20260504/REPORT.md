# Supervised Physics Tick Analysis

- Attempt files: 12
- Valid attempt files: 12
- Valid frame deltas: 28486
- Recommended `--physics-tick-probs`: `1:0.938285,2:0.060381,3:0.000562,4:0.000772`
- Thesis plot: `physics_tick_distribution_thesis.png`

## Distribution

| ticks | physics Hz | probability | count |
| ---: | ---: | ---: | ---: |
| 1 | 100.00 | 0.9383 | 26728 |
| 2 | 50.00 | 0.0604 | 1720 |
| 3 | 33.33 | 0.0006 | 16 |
| 4 | 25.00 | 0.0008 | 22 |

Interpretation: the observation should encode physics-tick delay, not render FPS. `physics_delay_norm = 1 - 1 / tick_count` is zero at 100 Hz and grows when game physics updates are skipped.

Elite-cache motivation: fixed-100 Hz TM2D experiments are deterministic, but live Trackmania can miss physics updates. Re-evaluating an elite in a different tick sequence can therefore change the outcome even when the genotype did not change. This does not prove that elite cache is always better, but it explains why it is a meaningful training variant to test under variable physics timing.
