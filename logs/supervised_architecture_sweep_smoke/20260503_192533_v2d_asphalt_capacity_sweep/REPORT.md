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
| arch_16x8_act_relu_tanh | 16x8 | relu,tanh | 723 | 0.19994 | 0.46229 | 0.86397 | 0.87419 | 2 |
| arch_16x8_act_tanh_tanh | 16x8 | tanh,tanh | 723 | 0.20418 | 0.45902 | 0.66218 | 0.69672 | 2 |
| arch_8_act_tanh | 8 | tanh | 307 | 0.30917 | 0.62749 | 0.77381 | 0.87198 | 2 |

## Smallest Candidate Within 5% Of Best Loss

| candidate_id | architecture | activation | parameter_count | best_val_loss | best_val_steer_mae | best_val_gas_accuracy | best_val_brake_accuracy | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arch_16x8_act_tanh_tanh | 16x8 | tanh,tanh | 723 | 0.20418 | 0.45902 | 0.66218 | 0.69672 | 2 |

## Skipped Configurations

Skipped incompatible architecture/activation combinations: `1`.

## Generated Plots

- `heatmap_architecture_activation_val_loss.png`
- `validation_loss_vs_parameter_count.png`
- `steer_mae_vs_parameter_count.png`
- `gas_accuracy_vs_parameter_count.png`
- `brake_accuracy_vs_parameter_count.png`

## Interpretation Guide

Prefer the smallest architecture close to the best validation loss, not the largest network with the absolute best score. For GA, parameter count is genome size, so bigger networks directly increase the evolutionary search space.
