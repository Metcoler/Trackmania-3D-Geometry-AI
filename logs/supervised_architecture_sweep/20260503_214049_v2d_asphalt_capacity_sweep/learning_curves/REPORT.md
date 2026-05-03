# Supervised Architecture Learning Curves

Source run: `logs\supervised_architecture_sweep\20260503_214049_v2d_asphalt_capacity_sweep`

## Generated Plots

- `key_validation_loss_learning_curves.png`
- `key_validation_loss_learning_curves_zoom_epoch20.png`
- `train_vs_validation_key_candidates.png`
- `convergence_summary.csv`

## Key Candidates

| candidate | arch | activation | params | best val | best epoch | val@50 | val@80 | val@100 | val@120 | improve 100->120 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `c038_arch_128x64_act_relu_tanh` | `128x64` | `relu,tanh` | 12931 | 0.05627 | 118 | 0.06093 | 0.05780 | 0.05755 | 0.05677 | 1.37% |
| `c029_arch_64x32_act_relu_relu` | `64x32` | `relu,relu` | 4419 | 0.05838 | 120 | 0.06697 | 0.06111 | 0.05938 | 0.05838 | 1.68% |
| `c025_arch_48x24_act_relu_relu` | `48x24` | `relu,relu` | 2931 | 0.06060 | 119 | 0.07012 | 0.06550 | 0.06334 | 0.06081 | 4.00% |
| `c035_arch_32x16x8_act_relu_relu_tanh` | `32x16x8` | `relu,relu,tanh` | 1811 | 0.06444 | 119 | 0.07535 | 0.06851 | 0.06637 | 0.06478 | 2.38% |
| `c021_arch_32x16_act_relu_relu` | `32x16` | `relu,relu` | 1699 | 0.06591 | 119 | 0.07375 | 0.06889 | 0.06731 | 0.06602 | 1.92% |
| `c022_arch_32x16_act_relu_tanh` | `32x16` | `relu,tanh` | 1699 | 0.06691 | 119 | 0.07568 | 0.07115 | 0.06890 | 0.06698 | 2.80% |
| `c018_arch_24x12_act_relu_tanh` | `24x12` | `relu,tanh` | 1179 | 0.06895 | 120 | 0.07805 | 0.07304 | 0.07062 | 0.06895 | 2.36% |
| `c013_arch_16x8_act_relu_relu` | `16x8` | `relu,relu` | 723 | 0.07186 | 119 | 0.08993 | 0.07923 | 0.07452 | 0.07187 | 3.55% |
| `c002_arch_8_act_tanh` | `8` | `tanh` | 307 | 0.08132 | 120 | 0.09425 | 0.08540 | 0.08294 | 0.08132 | 1.96% |

## Reading

Most strong candidates were still improving near the end of the 120-epoch run. That means final validation loss is a capacity-and-training-budget measurement together, not a pure asymptotic capacity measurement. For the thesis this is acceptable if we define a fixed training budget, because GA is also constrained by time and parameter count.
