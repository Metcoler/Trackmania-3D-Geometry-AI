# Supervised Architecture Capacity Sweep

## Purpose

This experiment estimates whether a small MLP can approximate the mapping `observation -> human action` on a fixed supervised dataset. It is not a closed-loop driving validation.

## Dataset

- Data roots: `['logs/supervised_data/20260502_153421_map_AI Training #5_v2d_asphalt_target_dataset', 'logs/supervised_data/20260502_154227_map_AI Training #5_v2d_asphalt_target_dataset']`
- Train attempts: `8`
- Validation attempts: `4`
- Train frames after preprocessing: `43898`
- Validation frames after preprocessing: `10858`
- Observation dim: `34`
- Action dim: `3`
- Train mirroring: `True`
- Validation mirroring: `False`

## Best Candidates By Validation Loss

| candidate_id | architecture | activation | parameter_count | best_val_loss | best_val_steer_mae | best_val_gas_accuracy | best_val_brake_accuracy | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c038_arch_128x64_act_relu_tanh | 128x64 | relu,tanh | 12931 | 0.05627 | 0.19928 | 0.94907 | 0.97311 | 118 |
| c037_arch_128x64_act_relu_relu | 128x64 | relu,relu | 12931 | 0.05778 | 0.20122 | 0.94557 | 0.96666 | 114 |
| c029_arch_64x32_act_relu_relu | 64x32 | relu,relu | 4419 | 0.05838 | 0.20574 | 0.94456 | 0.96675 | 120 |
| c030_arch_64x32_act_relu_tanh | 64x32 | relu,tanh | 4419 | 0.05881 | 0.20891 | 0.94778 | 0.97062 | 120 |
| c025_arch_48x24_act_relu_relu | 48x24 | relu,relu | 2931 | 0.06060 | 0.21606 | 0.93977 | 0.96657 | 119 |
| c026_arch_48x24_act_relu_tanh | 48x24 | relu,tanh | 2931 | 0.06177 | 0.21627 | 0.93783 | 0.96648 | 120 |
| c039_arch_128x64_act_tanh_tanh | 128x64 | tanh,tanh | 12931 | 0.06240 | 0.21860 | 0.94235 | 0.96509 | 117 |
| c035_arch_32x16x8_act_relu_relu_tanh | 32x16x8 | relu,relu,tanh | 1811 | 0.06444 | 0.22229 | 0.92356 | 0.96334 | 119 |
| c031_arch_64x32_act_tanh_tanh | 64x32 | tanh,tanh | 4419 | 0.06531 | 0.22570 | 0.93995 | 0.96298 | 117 |
| c027_arch_48x24_act_tanh_tanh | 48x24 | tanh,tanh | 2931 | 0.06559 | 0.22541 | 0.93765 | 0.96160 | 120 |
| c021_arch_32x16_act_relu_relu | 32x16 | relu,relu | 1699 | 0.06591 | 0.22184 | 0.92476 | 0.96095 | 119 |
| c033_arch_32x16x8_act_relu_relu_relu | 32x16x8 | relu,relu,relu | 1811 | 0.06679 | 0.21779 | 0.91923 | 0.95460 | 119 |

## Smallest Candidates Within Loss Tolerances

| tolerance | candidate_id | architecture | activation | parameter_count | best_val_loss | loss_ratio | best_val_steer_mae | best_val_gas_accuracy | best_val_brake_accuracy | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5% | c029_arch_64x32_act_relu_relu | 64x32 | relu,relu | 4419 | 0.05838 | 1.03748 | 0.20574 | 0.94456 | 0.96675 | 120 |
| 10% | c025_arch_48x24_act_relu_relu | 48x24 | relu,relu | 2931 | 0.06060 | 1.07684 | 0.21606 | 0.93977 | 0.96657 | 119 |
| 15% | c035_arch_32x16x8_act_relu_relu_tanh | 32x16x8 | relu,relu,tanh | 1811 | 0.06444 | 1.14505 | 0.22229 | 0.92356 | 0.96334 | 119 |
| 20% | c021_arch_32x16_act_relu_relu | 32x16 | relu,relu | 1699 | 0.06591 | 1.17127 | 0.22184 | 0.92476 | 0.96095 | 119 |

## Skipped Or Duplicate Configurations

Skipped incompatible or duplicate architecture/activation combinations: `0`.

## Generated Plots

- `heatmap_architecture_activation_val_loss.png`
- `validation_loss_vs_parameter_count.png`
- `steer_mae_vs_parameter_count.png`
- `gas_accuracy_vs_parameter_count.png`
- `brake_accuracy_vs_parameter_count.png`

## Interpretation Guide

Prefer the smallest architecture close to the best validation loss, not the largest network with the absolute best score. For GA, parameter count is genome size, so bigger networks directly increase the evolutionary search space.
