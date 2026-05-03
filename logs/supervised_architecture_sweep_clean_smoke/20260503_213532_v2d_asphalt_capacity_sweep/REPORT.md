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
| c009_arch_32x16x8_act_tanh_tanh_tanh | 32x16x8 | tanh,tanh,tanh | 1811 | 0.33133 | 0.65013 | 0.86213 | 0.86821 | 1 |
| c002_arch_8_act_tanh | 8 | tanh | 307 | 0.33493 | 0.65849 | 0.86010 | 0.87392 | 1 |
| c008_arch_32x16x8_act_relu_relu_relu | 32x16x8 | relu,relu,relu | 1811 | 0.35921 | 0.68257 | 0.86213 | 0.87419 | 1 |
| c006_arch_16x8_act_tanh_tanh | 16x8 | tanh,tanh | 723 | 0.36083 | 0.67881 | 0.86195 | 0.12581 | 1 |
| c010_arch_32x16x8_act_relu_relu_tanh | 32x16x8 | relu,relu,tanh | 1811 | 0.36114 | 0.68413 | 0.86213 | 0.56594 | 1 |
| c005_arch_16x8_act_relu_tanh | 16x8 | relu,tanh | 723 | 0.38781 | 0.71671 | 0.84030 | 0.87419 | 1 |
| c001_arch_8_act_relu | 8 | relu | 307 | 0.39625 | 0.71456 | 0.13234 | 0.50810 | 1 |
| c003_arch_8_act_sigmoid | 8 | sigmoid | 307 | 0.40020 | 0.73521 | 0.86213 | 0.87419 | 1 |
| c007_arch_16x8_act_sigmoid_sigmoid | 16x8 | sigmoid,sigmoid | 723 | 0.40522 | 0.75140 | 0.86213 | 0.87419 | 1 |
| c004_arch_16x8_act_relu_relu | 16x8 | relu,relu | 723 | 0.40525 | 0.73713 | 0.86213 | 0.87419 | 1 |
| c011_arch_32x16x8_act_sigmoid_sigmoid_sigmoid | 32x16x8 | sigmoid,sigmoid,sigmoid | 1811 | 0.41654 | 0.75286 | 0.86213 | 0.87419 | 1 |

## Smallest Candidates Within Loss Tolerances

| tolerance | candidate_id | architecture | activation | parameter_count | best_val_loss | loss_ratio | best_val_steer_mae | best_val_gas_accuracy | best_val_brake_accuracy | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5% | c002_arch_8_act_tanh | 8 | tanh | 307 | 0.33493 | 1.01089 | 0.65849 | 0.86010 | 0.87392 | 1 |
| 10% | c002_arch_8_act_tanh | 8 | tanh | 307 | 0.33493 | 1.01089 | 0.65849 | 0.86010 | 0.87392 | 1 |
| 15% | c002_arch_8_act_tanh | 8 | tanh | 307 | 0.33493 | 1.01089 | 0.65849 | 0.86010 | 0.87392 | 1 |
| 20% | c001_arch_8_act_relu | 8 | relu | 307 | 0.39625 | 1.19595 | 0.71456 | 0.13234 | 0.50810 | 1 |

## Skipped Configurations

Skipped incompatible architecture/activation combinations: `0`.

## Generated Plots

- `heatmap_architecture_activation_val_loss.png`
- `validation_loss_vs_parameter_count.png`
- `steer_mae_vs_parameter_count.png`
- `gas_accuracy_vs_parameter_count.png`
- `brake_accuracy_vs_parameter_count.png`

## Interpretation Guide

Prefer the smallest architecture close to the best validation loss, not the largest network with the absolute best score. For GA, parameter count is genome size, so bigger networks directly increase the evolutionary search space.
