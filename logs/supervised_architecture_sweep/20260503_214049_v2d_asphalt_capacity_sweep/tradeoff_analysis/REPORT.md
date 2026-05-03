# Supervised Architecture Tradeoff Analysis

Source run: `logs\supervised_architecture_sweep\20260503_214049_v2d_asphalt_capacity_sweep`

This analysis treats the supervised sweep as a representation-capacity test, not as closed-loop driving validation. The goal is to find the smallest MLP that is close enough to the best validation loss, because in GA every parameter becomes part of the evolved genome.

## Dataset And Sweep

- Train attempts: 8
- Validation attempts: 4
- Train frames after preprocessing: 43898
- Validation frames after preprocessing: 10858
- Observation/action dimensions: 34 -> 3
- Train mirroring: enabled
- Validation mirroring: disabled
- Epochs: 120
- Candidates: 40

## Absolute Best

| candidate | arch | activation | params | val loss | loss inc. | steer MAE | gas acc | brake acc | epoch |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `c038_arch_128x64_act_relu_tanh` | `128x64` | `relu,tanh` | 12931 | 0.05627 | +0.0% | 0.19928 | 0.949 | 0.973 | 118 |

The absolute best model is the overkill reference `128x64 relu,tanh`. It is useful as an upper-capacity reference, but it has 12931 parameters, so it is a large GA genome.

## Smallest Models Within Loss Tolerances

| tolerance from best | candidate | arch | activation | params | val loss | loss inc. | steer MAE | gas acc | brake acc | epoch |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5% | `c029_arch_64x32_act_relu_relu` | `64x32` | `relu,relu` | 4419 | 0.05838 | +3.7% | 0.20574 | 0.945 | 0.967 | 120 |
| 10% | `c025_arch_48x24_act_relu_relu` | `48x24` | `relu,relu` | 2931 | 0.06060 | +7.7% | 0.21606 | 0.940 | 0.967 | 119 |
| 15% | `c035_arch_32x16x8_act_relu_relu_tanh` | `32x16x8` | `relu,relu,tanh` | 1811 | 0.06444 | +14.5% | 0.22229 | 0.924 | 0.963 | 119 |
| 20% | `c021_arch_32x16_act_relu_relu` | `32x16` | `relu,relu` | 1699 | 0.06591 | +17.1% | 0.22184 | 0.925 | 0.961 | 119 |
| 25% | `c018_arch_24x12_act_relu_tanh` | `24x12` | `relu,tanh` | 1179 | 0.06895 | +22.5% | 0.23375 | 0.922 | 0.963 | 120 |
| 30% | `c013_arch_16x8_act_relu_relu` | `16x8` | `relu,relu` | 723 | 0.07186 | +27.7% | 0.23507 | 0.915 | 0.949 | 119 |

## Practical Recommendation

- Best quality/reference: `128x64` with `relu,tanh`. Use it only as an upper bound, not as GA default.
- Strong GA candidate: `48x24` with `relu,relu`. It stays within 7.7% of the best loss while using only 2931 parameters.
- Smaller robust candidate: `32x16x8` with `relu,relu,tanh`. It stays within 14.5% and uses 1811 parameters.
- Cheapest defensible baseline: `32x16` with `relu,relu`. It stays within 17.1% and uses 1699 parameters.

For the thesis, the clean argument is that `48x24 relu,relu` is the best tradeoff if we want quality, while `32x16 relu,relu` or `32x16x8 relu,relu,tanh` are defensible compact GA baselines. The old `32x16 relu,tanh` is usable but not the best activation choice in this sweep; `relu,relu` was better for `32x16`.

## Pareto Frontier

| candidate | arch | activation | params | val loss | loss inc. | steer MAE | gas acc | brake acc | epoch |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `c002_arch_8_act_tanh` | `8` | `tanh` | 307 | 0.08132 | +44.5% | 0.26506 | 0.915 | 0.946 | 120 |
| `c005_arch_16_act_tanh` | `16` | `tanh` | 611 | 0.07752 | +37.8% | 0.26034 | 0.922 | 0.953 | 120 |
| `c013_arch_16x8_act_relu_relu` | `16x8` | `relu,relu` | 723 | 0.07186 | +27.7% | 0.23507 | 0.915 | 0.949 | 119 |
| `c018_arch_24x12_act_relu_tanh` | `24x12` | `relu,tanh` | 1179 | 0.06895 | +22.5% | 0.23375 | 0.922 | 0.963 | 120 |
| `c021_arch_32x16_act_relu_relu` | `32x16` | `relu,relu` | 1699 | 0.06591 | +17.1% | 0.22184 | 0.925 | 0.961 | 119 |
| `c035_arch_32x16x8_act_relu_relu_tanh` | `32x16x8` | `relu,relu,tanh` | 1811 | 0.06444 | +14.5% | 0.22229 | 0.924 | 0.963 | 119 |
| `c025_arch_48x24_act_relu_relu` | `48x24` | `relu,relu` | 2931 | 0.06060 | +7.7% | 0.21606 | 0.940 | 0.967 | 119 |
| `c029_arch_64x32_act_relu_relu` | `64x32` | `relu,relu` | 4419 | 0.05838 | +3.7% | 0.20574 | 0.945 | 0.967 | 120 |
| `c038_arch_128x64_act_relu_tanh` | `128x64` | `relu,tanh` | 12931 | 0.05627 | +0.0% | 0.19928 | 0.949 | 0.973 | 118 |

## Activation Notes

| activation | count | best arch | best params | best val loss | median val loss |
| --- | ---: | --- | ---: | ---: | ---: |
| `relu` | 4 | `32` | 1219 | 0.07111 | 0.07817 |
| `relu,relu` | 6 | `128x64` | 12931 | 0.05778 | 0.06591 |
| `relu,relu,relu` | 1 | `32x16x8` | 1811 | 0.06679 | 0.06679 |
| `relu,relu,tanh` | 1 | `32x16x8` | 1811 | 0.06444 | 0.06444 |
| `relu,tanh` | 6 | `128x64` | 12931 | 0.05627 | 0.06691 |
| `sigmoid` | 4 | `32` | 1219 | 0.09045 | 0.09998 |
| `sigmoid,sigmoid` | 6 | `128x64` | 12931 | 0.07861 | 0.09759 |
| `sigmoid,sigmoid,sigmoid` | 1 | `32x16x8` | 1811 | 0.09842 | 0.09842 |
| `tanh` | 4 | `32` | 1219 | 0.07519 | 0.07752 |
| `tanh,tanh` | 6 | `128x64` | 12931 | 0.06240 | 0.07051 |
| `tanh,tanh,tanh` | 1 | `32x16x8` | 1811 | 0.06908 | 0.06908 |

Hidden sigmoid remains a useful negative/control activation: it is usually worse than ReLU/Tanh combinations at comparable sizes. The strongest serious activations are ReLU/ReLU and ReLU/Tanh, with ReLU/ReLU giving the best compact models in this run.

## Generated Files

- `tradeoff_candidates.csv`
- `pareto_candidates.csv`
- `activation_summary.csv`
- `tradeoff_val_loss_vs_params.png`
- `tradeoff_steer_mae_vs_params.png`
- `tradeoff_gas_brake_accuracy_vs_params.png`
