# Supervised Architecture Sweep 20260503

This curated package summarizes the supervised architecture-capacity sweep used in the thesis chapter on policy training.

## Source Run

- Raw run: `logs\supervised_architecture_sweep\20260503_214049_v2d_asphalt_capacity_sweep`
- Dataset: v2d/asphalt human-driving observations on `AI Training #5`
- Train samples: 43898
- Validation samples: 10858
- Observation dimension: 34
- Action dimension: 3
- Epochs: 120
- Optimizer: Adam
- Loss: SmoothL1Loss with action weights `[1, 1, 3]`

## Selected Candidates

| architecture | activation | params | val loss | steer MAE | gas acc. | brake acc. | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32x16 | relu,relu | 1699 | 0.065913 | 0.2218 | 0.925 | 0.961 | 119 |
| 32x16 | relu,tanh | 1699 | 0.066909 | 0.2247 | 0.922 | 0.961 | 119 |
| 48x24 | relu,relu | 2931 | 0.060599 | 0.2161 | 0.940 | 0.967 | 119 |
| 48x24 | relu,tanh | 2931 | 0.061774 | 0.2163 | 0.938 | 0.966 | 120 |
| 128x64 | relu,relu | 12931 | 0.057776 | 0.2012 | 0.946 | 0.967 | 114 |
| 128x64 | relu,tanh | 12931 | 0.056275 | 0.1993 | 0.949 | 0.973 | 118 |

## Interpretation

This sweep is a representation-capacity test. It shows how well a fixed MLP architecture can approximate human actions on recorded data. It does not prove closed-loop driving quality. Larger networks reach lower validation loss, but they also create a larger parameter vector for later evolutionary search. The thesis therefore keeps `32x16 relu,tanh` as a cheap experimental baseline and `48x24 relu,tanh` as the stronger practical candidate.

## Generated Figures

- `supervised_architecture_training_curves_all.pdf`
- `supervised_architecture_training_curves_practical.pdf`
- `supervised_architecture_val_loss_vs_params.pdf`
- `supervised_architecture_focus_32x16_48x24.pdf`
