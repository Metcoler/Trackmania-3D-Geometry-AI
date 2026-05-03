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
| arch_128x64_act_relu_relu | 128x64 | relu,relu | 12931 | 0.05796 | 0.20530 | 0.94539 | 0.96979 | 80 |
| arch_128x64_act_relu_tanh | 128x64 | relu,tanh | 12931 | 0.05861 | 0.20783 | 0.94520 | 0.96970 | 80 |
| arch_128x64_act_relu_relu | 128x64 | relu,relu | 12931 | 0.05895 | 0.20634 | 0.94649 | 0.96694 | 77 |
| arch_64x32_act_relu_relu | 64x32 | relu,relu | 4419 | 0.06245 | 0.21841 | 0.93719 | 0.96261 | 80 |
| arch_128x64_act_tanh_tanh | 128x64 | tanh,tanh | 12931 | 0.06312 | 0.21865 | 0.94207 | 0.96371 | 80 |
| arch_64x32_act_relu_tanh | 64x32 | relu,tanh | 4419 | 0.06371 | 0.22298 | 0.93415 | 0.96620 | 80 |
| arch_64x32_act_relu_relu | 64x32 | relu,relu | 4419 | 0.06381 | 0.22330 | 0.93793 | 0.96574 | 79 |
| arch_128x64_act_tanh_tanh | 128x64 | tanh,tanh | 12931 | 0.06400 | 0.22215 | 0.93885 | 0.96381 | 79 |
| arch_48x24_act_relu_relu | 48x24 | relu,relu | 2931 | 0.06499 | 0.22129 | 0.92503 | 0.96288 | 80 |
| arch_48x24_act_relu_tanh | 48x24 | relu,tanh | 2931 | 0.06642 | 0.22629 | 0.92430 | 0.96353 | 80 |
| arch_48x24_act_relu_relu | 48x24 | relu,relu | 2931 | 0.06680 | 0.22256 | 0.92116 | 0.95763 | 80 |
| arch_64x32_act_tanh_tanh | 64x32 | tanh,tanh | 4419 | 0.06894 | 0.23804 | 0.93194 | 0.96371 | 79 |

## Smallest Candidate Within 5% Of Best Loss

| candidate_id | architecture | activation | parameter_count | best_val_loss | best_val_steer_mae | best_val_gas_accuracy | best_val_brake_accuracy | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arch_128x64_act_relu_relu | 128x64 | relu,relu | 12931 | 0.05796 | 0.20530 | 0.94539 | 0.96979 | 80 |

## Skipped Configurations

Skipped incompatible architecture/activation combinations: `15`.

## Generated Plots

- `heatmap_architecture_activation_val_loss.png`
- `validation_loss_vs_parameter_count.png`
- `steer_mae_vs_parameter_count.png`
- `gas_accuracy_vs_parameter_count.png`
- `brake_accuracy_vs_parameter_count.png`

## Interpretation Guide

Prefer the smallest architecture close to the best validation loss, not the largest network with the absolute best score. For GA, parameter count is genome size, so bigger networks directly increase the evolutionary search space.
