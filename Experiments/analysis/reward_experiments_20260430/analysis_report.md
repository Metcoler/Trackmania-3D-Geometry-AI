# Reward Experiment Analysis

## Completion
- ga_terminal_reward_sweep: err_bytes=0, out_tail=Saved metrics to Experiments\runs\ga_terminal_reward_sweep_20260430_114834\terminal_fitness\20260430_160914_tm2d_ga_AI_Training_5\generation_metrics.csv
- ga_moo_trackmania_racing: err_bytes=0, out_tail=Saved metrics to Experiments\runs\ga_moo_trackmania_racing_20260430_120449\20260430_120457_tm2d_ga_moo_AI_Training_5_trackmania_racing\generation_metrics.csv
- sac_reward_sweep: err_bytes=0, out_tail=[TM2D SAC] Finished. Artifacts saved in: Experiments\runs\sac_reward_sweep_20260430_132154\terminal_progress_time_efficiency\20260430_161608_tm2d_sac_AI_Training__5_terminal_progress_time_efficiency_gas_brake_steer

## GA / MOO Ranked Summary
- GA `terminal_lexicographic_no_distance`: best_finish_time=15.985s, first_finish_gen=95, last50_mean_dense=50.13%, auc_mean_dense=37.14%
- GA `terminal_lexicographic`: best_finish_time=15.991s, first_finish_gen=105, last50_mean_dense=49.75%, auc_mean_dense=33.32%
- MOO_GA `moo_trackmania_racing`: best_finish_time=15.998s, first_finish_gen=81, last50_mean_dense=50.15%, auc_mean_dense=36.47%
- GA `terminal_lexicographic_progress20`: best_finish_time=16.001s, first_finish_gen=109, last50_mean_dense=52.33%, auc_mean_dense=32.48%
- GA `terminal_progress_time_efficiency`: best_finish_time=16.051s, first_finish_gen=129, last50_mean_dense=45.53%, auc_mean_dense=28.06%
- GA `terminal_fitness`: best_finish_time=16.406s, first_finish_gen=198, last50_mean_dense=51.38%, auc_mean_dense=20.48%

## SAC Ranked Summary
- SAC `individual_dense`: max_dense=2.857%, last100_mean_dense=1.979%, timeouts=74, crashes=1841, rows=1915
- SAC `progress_primary_delta`: max_dense=0.895%, last100_mean_dense=0.574%, timeouts=0, crashes=2000, rows=2000
- SAC `terminal_progress_time_efficiency`: max_dense=0.562%, last100_mean_dense=0.335%, timeouts=0, crashes=2000, rows=2000
- SAC `delta_lexicographic`: max_dense=0.540%, last100_mean_dense=0.343%, timeouts=0, crashes=2000, rows=2000
- SAC `delta_progress_time_efficiency`: max_dense=0.512%, last100_mean_dense=0.356%, timeouts=0, crashes=2000, rows=2000

## Generated Graphs
- ga_best_dense_progress.png
- ga_finish_time.png
- ga_mean_dense_progress.png
- ga_summary_bars.png
- moo_front0_size.png
- moo_objectives.png
- reward_experiment_dashboard.png
- sac_cumulative_best_dense_progress.png
- sac_dense_progress_rolling50.png
- sac_episode_reward_rolling50.png
- sac_summary_bars.png