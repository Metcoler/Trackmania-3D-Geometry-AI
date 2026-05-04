# Supervised Physics Tick Analysis

- Attempt files: 12
- Valid attempt files: 12
- Valid frame deltas: 28486
- Recommended `--physics-tick-probs`: `1:0.938285,2:0.060381,3:0.000562,4:0.000772`

## Distribution

| ticks | physics Hz | probability | count |
| ---: | ---: | ---: | ---: |
| 1 | 100.00 | 0.9383 | 26728 |
| 2 | 50.00 | 0.0604 | 1720 |
| 3 | 33.33 | 0.0006 | 16 |
| 4 | 25.00 | 0.0008 | 22 |

Interpretation: the observation should encode physics-tick delay, not render FPS. `physics_delay_norm = 1 - 1 / tick_count` is zero at 100 Hz and grows when game physics updates are skipped.
