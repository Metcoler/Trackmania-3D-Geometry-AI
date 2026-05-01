# Fast Trackmania 2D Experiments

This folder is a local sandbox for testing reward functions, observations and GA/RL ideas without waiting for one realtime Trackmania instance.

The simulator intentionally reuses the main project pieces where it matters:

- `Map.py` parses map block layouts from `Maps/BlockLayouts/*.txt` (legacy fallback: `Maps/ExportedBlocks/*.txt`) and resolves Trackmania block meshes from `Assets/BlockMeshes/` (legacy fallback: `Meshes/`).
- `.obj` road/wall meshes are projected into 2D XZ geometry.
- `ObservationEncoder.py` builds the same flat observation shape used by live training.
- `NeuralPolicy.py` and `Individual.py` are reused for networks, GA individuals and scalar fitness.

The simulator is flat asphalt by default. Surface instructions are still emitted as map traction values and height instructions as zero/flat map values, so the observation remains compatible with the current 2D Trackmania setup.

## Quick Commands

Run a fast GA experiment:

```powershell
python Experiments/train_ga.py --map-name "AI Training #5" --generations 100 --population-size 48
```

Run the current preferred lexicographic GA setup. Dense progress is the default progress source in the 2D simulator:

```powershell
python Experiments/train_ga.py --map-name "AI Training #5" --generations 300 --population-size 48 --elite-count 4 --parent-count 16 --max-time 45 --hidden-dim "32,16" --hidden-activation "relu,tanh" --mutation-prob 0.18 --mutation-sigma 0.22 --fitness-mode ranking --ranking-key "(finished, progress, -time)" --collision-mode corners --num-workers 4 --log-dir Experiments/runs/lex_finished_progress_time_g300_p48_corners_variable_fps
```

Run a Pareto/NSGA-II style multi-objective GA experiment:

```powershell
python Experiments/train_ga_moo.py --map-name "AI Training #5" --generations 100 --population-size 48 --num-workers 4
```

Run a local SAC experiment over the same 2D simulator:

```powershell
python Experiments/train_sac.py --map-name "AI Training #5" --reward-mode progress_delta --episodes 200
```

Try the conservative dense `Individual`-style reward with SAC:

```powershell
python Experiments/train_sac.py --map-name "AI Training #5" --reward-mode individual_dense --episodes 200
```

Evaluate individuals in parallel when the population is large enough to amortize worker startup:

```powershell
python Experiments/train_ga.py --map-name "AI Training #5" --generations 100 --population-size 48 --num-workers 4
```

For stricter collision checks, add `--collision-mode corners`. The default `center` mode is intentionally faster for reward experiments.

Use `--fixed-fps 60` for deterministic stepping. Leave `--fixed-fps` unset to sample frame times from the configured 100-30 FPS range, which is closer to realtime Trackmania jitter.

Compare reward score tables:

```powershell
python Experiments/reward_lab.py --map-name "AI Training #5"
```

Recompute physics suggestions from supervised attempts:

```powershell
python Experiments/calibrate_tm2d_physics.py --data-root logs/supervised_data
```

Visualize the 2D simulator with a simple heuristic driver:

```powershell
python Experiments/visualize_tm2d.py --map-name "AI Training #5"
```

Visualize a saved policy:

```powershell
python Experiments/visualize_tm2d.py --map-name "AI Training #5" --model-path Experiments/runs/<run>/best_policy.pt
```

## GA Ranking

The most important GA path is now `--fitness-mode ranking`. In this mode the scalar `fitness` column is kept mostly for plots/logging, while selection uses `Individual.__lt__` and the explicit lexicographic tuple from `--ranking-key`.

Useful tuple symbols:

- `finished`: `1` when the policy reaches finish, otherwise `0`.
- `progress`: progress value selected by `--ranking-progress-source`.
- `dense_progress`: continuous geometric progress along the path.
- `discrete_progress`: checkpoint/block-style progress.
- `time`: episode time in seconds.
- `crashes`: number of crashes/touches.
- `distance`: driven ground distance, useful as a late tie-breaker against S-turns.

The `progress` symbol is deliberately configurable. In the 2D simulator the default is `dense_progress`, because discrete checkpoint/block progress leaves too many early policies tied.

- `--ranking-progress-source discrete_progress` makes `progress` use discrete checkpoint/block progress.
- `--ranking-progress-source dense_progress` makes `progress` use continuous path progress.
- If the tuple directly contains `dense_progress`, the trainer automatically uses dense progress even when the source flag was not set.

Current working candidate after the May 2026 local comparison:

```text
(finished, progress, -time)
```

Interpretation: first prefer policies that finish, then those that get farther, and among comparable progress levels prefer lower time. In the latest four-way comparison on `AI Training #5`, this tuple found the first finish earliest and produced the fastest best finish.

Other useful comparison tuples:

```text
(finished, progress)
(finished, progress, -crashes, -time)
(finished, progress, -time, -crashes)
```

These are useful thesis baselines because they expose the tradeoff between progress-only behavior, safer driving, and more aggressive time pressure.

## Reward Modes

Reward modes are still used by `TM2DSimEnv` and SAC. For GA ranking experiments they are logged and can still shape per-step environment reward, but GA selection is controlled by `--ranking-key` when `--fitness-mode ranking` is used.

- `progress_delta`: score is only current progress.
- `progress_primary_delta`: score is progress minus at most one map progress bucket over the full timeout. This keeps progress dominant and uses time only as a tie-breaker.
- `pace_delta`: reproduces the stricter pace reward idea that can make fast early failure attractive.
- `terminal_fitness`: uses `Individual.compute_scalar_fitness_for` only at terminal states.
- `individual_dense`: uses `Individual.compute_scalar_fitness_for`, but feeds it continuous geometric progress along the current path segment. This is the conservative mode to try when discrete progress buckets leave most random agents tied at `0%`.
- `terminal_lexicographic`: terminal-only bounded score: `finished`, progress, time tie-break, crash count, and distance tie-break.
- `terminal_lexicographic_no_distance`: same terminal score without the distance tie-break.
- `terminal_lexicographic_progress20`: terminal score that ignores time/distance tie-breaks below `20%` progress.
- `delta_lexicographic`: dense bounded score with progress primary, time secondary, distance tertiary, and finish bonus; crash does not retroactively erase early progress.
- `delta_lexicographic_terminal`: same dense score, but collision/timeout also apply the terminal failure component at episode end.
- `terminal_progress_time_efficiency`: terminal-only score where normalized progress is primary, time is only a progress-scaled tie-breaker, finish adds one full normalized completion unit, and distance only penalizes excess path length.
- `delta_progress_time_efficiency`: potential-difference version of `terminal_progress_time_efficiency`, intended as the current clean SAC candidate because it keeps the same final ordering while giving incremental signal.
- `progress_rate`: average progress rate style score. This is intentionally kept as a warning/negative-control mode because it can prefer fast early crashes.

Use `--fitness-mode reward` in `train_ga.py` when you want GA selection to optimize the selected step reward directly. The default `--fitness-mode scalar` keeps the live-project lexicographic scalar fitness.

`train_sac.py` uses the same reward modes directly as Gym step rewards. This makes it useful for quick reward-shaping experiments before spending realtime Trackmania hours in `RL_test/train_sac_trackmania.py`.

## Logging And Analysis

GA runs write:

- `config.json`: exact experiment configuration.
- `generation_metrics.csv`: per-generation best values, percentiles, outcome counts, cache/evaluation counts, wall time and virtual driving time.
- `individual_metrics.csv`: one row per individual per generation with ranking key, progress, dense progress, time, finished/crashes/timeout, distance, reward and step count.
- `best_policy.pt`: best policy according to the configured selection mode.

The detailed individual log is intentionally verbose. It lets us build thesis-grade plots such as progress distributions, finish-count curves, safety/time scatter plots, crash rates, and virtual-time learning efficiency.

Latest generated comparison:

```text
Experiments/analysis/ga_lexicographic_reward_comparison_20260501
```

This folder contains:

- `REPORT.md`: compact interpretation of the four reward/ranking candidates.
- `summary.csv`: headline metrics for each run.
- `combined_generation_metrics.csv`: merged generation metrics.
- `combined_individual_metrics.csv`: merged individual metrics.
- `01_best_dense_progress.png`
- `02_population_progress_distribution.png`
- `03_finish_count_and_best_finish_time.png`
- `04_outcome_rates.png`
- `05_final_population_scatter.png`
- `06_summary_bars.png`
- `07_virtual_time_efficiency.png`

Latest local conclusion:

- `(finished, progress, -time)` was the strongest tuple in the four-way dense-progress comparison.
- `(finished, progress, -crashes, -time)` is a useful safer alternative.
- `(finished, progress)` is a good baseline, but tends to learn slower driving because it ignores time.
- `(finished, progress, -time, -crashes)` can be more aggressive because crash safety is considered only after time.

## Multi-Objective GA

`train_ga_moo.py` is an optional add-on inspired by Pareto/NSGA-II neuroevolution. It does not replace `train_ga.py`; it only changes selection. Each individual is still evaluated in the same TM2D simulator, but instead of collapsing everything into one scalar fitness, selection sorts the population into Pareto fronts.

Current `trackmania_racing` objectives are all maximized:

- `progress`: continuous progress along the path, normalized to `[0, 1]`.
- `finish`: `1` only when the agent reaches finish.
- `speed_for_progress`: `progress * (1 - time / max_time)`, so time pressure only matters after real progress exists.
- `safe_progress`: progress scaled by remaining crash budget, so one touch is tolerated when the experiment allows multiple touches but cleaner runs stay preferred.
- `path_efficiency`: progress penalized by excess distance beyond the estimated path distance, so S-turns are discouraged without rewarding raw distance.

The default within-front priority is `finish,progress,speed_for_progress,safe_progress,path_efficiency`. This mirrors the racing goal: prefer a finish when a comparable solution exists, otherwise get farther, then get there sooner, then prefer non-crashing and cleaner trajectories. Use `--pareto-tiebreak crowding` if you want a more diversity-focused NSGA-II ordering.

Local SAC defaults:

- continuous action layout is `gas_brake_steer`
- default policy is SB3 `MlpPolicy` with `[128, 128]` hidden layers and ReLU
- default training frequency is one update batch per episode, matching the live SAC experiment style
- default discount is `gamma=0.9995`, because realtime-style racing needs a long horizon and `0.995` can discount 20-40 second outcomes almost away
- default entropy coefficient is fixed at `ent_coef=0.01`, because `ent_coef="auto"` collapsed exploration in live SAC experiments
- artifacts are saved under `Experiments/runs/<timestamp>_tm2d_sac_...`

Current reward-sweep conclusion:

- `progress_rate` is unsafe: it can rank a faster crash ahead of a slower rollout that reaches farther.
- `terminal_lexicographic` is sane for GA-style selection but too sparse for short SAC runs.
- `delta_lexicographic_terminal` looked strongest in a short GA sweep, but the terminal failure drop is too harsh for short SAC experiments.
- `delta_progress_time_efficiency` is the best current clean SAC candidate: dense progress signal, progress-first ordering, bounded time pressure, and no incentive to idle before the first progress bucket.
- For SAC reward experiments, `--action-layout gas_steer` is often a better diagnostic than `gas_brake_steer`, because independent random gas and brake exploration can cancel out near the start.

## Physics Calibration

`TM2DPhysicsConfig` is calibrated from the supervised `attempt_*.npz` files where possible:

- speed comes from the historical observation prefix at index `25`
- actions come from recorded `gas`, `brake`, `steer`
- frame timing comes from `game_times`
- local simulation samples `dt` uniformly from `1/100s` to `1/30s`, matching the intended 100-30 FPS realtime jitter range
- longitudinal acceleration is fitted as `accel ~= intercept + gas*G + brake*B + speed*D`

Older supervised attempts do not contain raw heading/yaw directly, so steering remains a curve-feasibility approximation rather than a fitted Trackmania model.

## Performance Notes

Runtime lidar does not raycast against `.obj` meshes directly. Maps are loaded through `Map.py`, wall meshes are projected once into 2D XZ line segments, and each lidar frame uses analytic ray/segment intersections against those cached segments.

Current profiling shows the main cost is per-frame environment stepping, especially observation construction, clipping/normalization, lidar raycasts, and road-containment checks. Torch policy inference is not the bottleneck in this sandbox because `train_ga.py` uses a NumPy policy view during rollouts.

`--num-workers` uses process-based parallelism across independent individuals. This is useful for longer generations and larger populations, but small smoke tests can be slower because Windows has to start worker processes and each worker loads the map once.
