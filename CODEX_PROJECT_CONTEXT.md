## Purpose

This file is a handoff/context document for another Codex/GPT-5.4 instance working on this repository.
It summarizes:

- what the application does
- how data flows through the system
- what each important file is responsible for
- which files should be indexed first
- how the project evolved since the first 2024 prototype
- which experiments were tried and what conclusions were reached
- what the current baseline state is


## Project Summary

This repository implements an autonomous driving agent for Trackmania.

There are currently three main learning paths in the codebase:

- a Genetic Algorithm / neuroevolution path for live training in Trackmania
- a supervised learning path that records player driving data and trains a torch policy offline
- a Stable-Baselines3 SAC reinforcement-learning experiment path under `RL_test/`
- a fast local 2D experiment sandbox under `Experiments/`

The project also contains Trackmania map extraction assets and an OpenPlanet plugin that streams game state over TCP to Python.

### Fast 2D experiment sandbox

`Experiments/` was added because live Trackmania is a poor place to iterate on reward design:
there is only one realtime game instance, no headless mode, no parallel agents, and every GA/RL
attempt costs real wall-clock time. The goal of this folder is to test reward functions,
observation changes, simple physics assumptions, and GA behavior in minutes instead of hours.

The sandbox intentionally reuses the main project pieces where possible:

- `Experiments/tm2d_geometry.py` loads the same exported TM maps through `Map.py`, projects road/wall `.obj` meshes into 2D XZ geometry, keeps the 32-unit TM grid, path tiles, path instructions, surface instructions, and height instructions.
- `Experiments/tm2d_env.py` implements a synchronous but variable-dt 2D car environment. The random dt simulates Trackmania running at different frame rates, while still being fully local and fast.
- The flat observation is compatible with `ObservationEncoder.total_obs_dim(vertical_mode=False)`: lasers, path instructions, speed/side speed, signed segment heading errors, dt ratio, slip mean, surface instructions, height instructions, and temporal derivative features. Surface defaults to asphalt-like values and height defaults to flat values on flat maps.
- `Experiments/train_ga.py` uses `Individual`/`NeuralPolicy` for the same MLP architecture and genome semantics, but evaluates policies through a fast NumPy forward pass instead of calling torch every simulated frame.
- `Experiments/train_ga.py` also supports `--num-workers N` for process-based parallel evaluation of independent individuals. Each worker creates its own read-only map/environment once and receives flat genomes to evaluate.
- `Experiments/train_ga.py`, `Experiments/train_sac.py`, and `Experiments/visualize_tm2d.py` support `--fixed-fps FPS`. When set, the local simulator uses deterministic `min_dt = max_dt = 1 / FPS`; when omitted, it keeps the default variable-dt range to mimic variable Trackmania FPS.
- `Experiments/train_sac.py` wraps the same `TM2DSimEnv` as a Gymnasium environment and runs Stable-Baselines3 SAC locally, so reward modes can be tested quickly before using the realtime Trackmania SAC script.
- `Experiments/reward_lab.py` prints reward score tables for one map, including `progress_delta`, `progress_primary_delta`, `pace_delta`, and `terminal_fitness`.
- `Experiments/visualize_tm2d.py` opens a small tkinter visualizer for the projected map, car, and lidar rays.
- `Experiments/calibrate_tm2d_physics.py` reads supervised `attempt_*.npz` files and estimates simple 2D physics parameters from recorded `speed`, `game_times`, `distances`, and `gas/brake/steer` actions.

Metric-parity note:

- `TM2DSimEnv` mirrors the live `Car.get_data()` metric names for reward work:
  `discrete_progress`, `dense_progress`, `finished`, `crashes`, `done`,
  `time`, and `distance`.
- `done` in the 2D env follows the OpenPlanet meaning used by live TM: it is
  `1.0` only for finish, not for every terminated rollout.
- The old three-valued `term` metric was removed from the current runtime data
  model. Current code represents outcome as `finished` (`0/1`) and `crashes`
  (`0..max_touches`). Timeout is derived for logging as
  `finished == 0 and crashes == 0`, not optimized as its own objective.
- Historical aliases such as `total_progress`, `tile_progress`,
  `block_progress`, and `map_progress` were removed from the current runtime
  data flow. Old logs/checkpoints may still contain those names, but new code
  should use only `discrete_progress` for confirmed path-tile progress and
  `dense_progress` for projected between-tile progress.
- Runtime/log progress values stay in percent (`0..100`) because that is much
  easier to read in graphs and console output. Ranking/objective code also
  exposes normalized aliases (`progress_norm`, `discrete_progress_norm`,
  `dense_progress_norm`, `ranking_progress_norm`) for formulas that should work
  in clean `[0, 1]` units.
- 2D rollout diagnostic `fitness` now uses `dense_progress` when
  `Individual.RANKING_PROGRESS_SOURCE == "dense_progress"`, matching
  `GeneticTrainer._evaluate_single_rollout()`.
- The remaining intentional mismatch is termination physics: local 2D uses
  road-containment collision (`center` or `corners`), while live TM uses
  lidar contact, stall, wall-ride, heading-error, timeout, and OpenPlanet
  finish signals. Reward functions that work locally should therefore be
  re-tested with the strictest local settings before being trusted in live TM.

Default experiment outputs go to ignored `Experiments/runs/`.

Local 2D GA logging is intentionally research-grade now. `train_ga.py` and
`train_ga_moo.py` write:

- `generation_metrics.csv`: one row per generation with best/mean metrics plus
  distribution statistics (`min`, `p10`, `p25`, `median`, `mean`, `std`, `p75`,
  `p90`, `max`) for progress, dense progress, time, distance, reward, fitness,
  and rollout steps.
- `individual_metrics.csv`: one row per individual per generation with rank,
  elite/parent flags, cached-evaluation flag where applicable, progress,
  dense progress, time, finished/crashes/timeout fields, distance,
  reward, fitness, steps, and selection-specific values such as ranking keys
  or Pareto objectives.
- Virtual experience counters: generation and cumulative virtual driving time
  plus rollout step counts, so thesis plots can show both wall-clock training
  cost and how many simulated driving hours the algorithm consumed.

Performance notes for the sandbox:

- runtime lidar no longer raycasts against `.obj` meshes directly
- `tm2d_geometry.py` loads map meshes once, projects walls into cached 2D XZ line segments, and uses analytic ray/segment intersections for lidar
- profiling shows the main costs are environment stepping, observation construction/clipping, lidar raycasts, and road-containment checks
- policy inference is not the bottleneck during GA experiments because rollouts use the NumPy policy view
- multiprocessing helps most on larger populations/generations; tiny smoke tests can be slower due to Windows process startup and per-worker map loading

The current 2D physics defaults are calibrated from the available supervised data where possible:

- observed speed median is about `66` TM units/s
- observed speed p95 is about `91` from the packet speed feature and about `108` from distance derivative
- observed speed p99 is about `104` from packet speed and about `137` from distance derivative
- median dt is about `0.010s`, while p95/p99 are about `0.030s`/`0.040s`
- fitted longitudinal model is approximately `accel = -15.985 + 28.187*gas - 16.648*brake - 0.0966*speed`

This calibrates acceleration/drag/braking better than the original toy physics.
Older supervised attempts do not contain enough direct heading/yaw data to fit steering exactly,
so `max_yaw_rate` and lateral grip are still pragmatic curve-feasibility parameters.

The sandbox reward modes currently include:

- `progress_delta`: score is current progress only
- `progress_primary_delta`: score is progress minus at most one progress bucket over the full timeout
- `pace_delta`: old aggressive pace reward, useful mainly as a negative control because it can prefer fast early failure
- `terminal_fitness`: terminal-only `Individual.compute_scalar_fitness_for`
- `individual_dense`: conservative `Individual.compute_scalar_fitness_for` style score using continuous geometric progress between path tiles; this was added because pure discrete progress left random GA populations tied at `0%`
- `terminal_lexicographic`: bounded terminal score with `finished`, progress, time tie-break, crash count, and distance tie-break
- `terminal_lexicographic_no_distance`: same terminal score without the distance tie-break
- `terminal_lexicographic_progress20`: terminal score that ignores time/distance tie-breaks below `20%` progress
- `delta_lexicographic`: dense bounded score with progress primary, time secondary, distance tertiary, and finish bonus; failure does not retroactively erase earlier progress
- `delta_lexicographic_terminal`: same dense score, but collision/timeout also apply the terminal failure component at episode end
- `terminal_progress_time_safety`: terminal-only SAC/debug score in progress-percent units; finish adds one full track (`+100`) and crash/timeout subtract one full track (`-100`)
- `delta_progress_time_safety`: potential-difference version of `terminal_progress_time_safety`; useful to test whether SAC can learn from dense progress when failed episodes are still globally bad
- `terminal_progress_time_block_penalty`: terminal-only SAC curriculum score; failed episodes subtract one map-derived progress bucket instead of one full track
- `delta_progress_time_block_penalty`: potential-difference version of `terminal_progress_time_block_penalty`; current best SAC debug candidate because it keeps progress dense while making timeout/crash non-zero failures
- `progress_rate`: average progress-rate score, retained as a negative-control mode because it can prefer fast early crashes

The same reward modes are usable from both `Experiments/train_ga.py` and `Experiments/train_sac.py`.
For SAC, the local wrapper exposes a continuous `gas_brake_steer` action space by default and logs
`episode_metrics.csv`, `monitor.csv`, `latest_model.zip`, `best_model.zip`, and periodic checkpoints
under `Experiments/runs/`.

Reward-sweep notes from the local sandbox:

- static scenario ranking confirmed that `progress_rate` is unsafe, because a fast early crash can outrank a slower run that reaches farther
- `terminal_lexicographic` and its variants preserve the desired ordering for terminal GA-style selection, but are sparse for SAC
- a short GA sweep on `AI Training #5` found the best early progress with `delta_lexicographic_terminal` (`14.29%` best progress in the tested 10-generation/24-population run)
- a short SAC sweep with `gas_brake_steer` mostly stayed near the start because random gas and brake cancel each other too often
- a short SAC sweep with diagnostic `gas_steer` made reward comparison clearer: `delta_lexicographic` reached `6.80%` dense progress in 30 episodes, while `progress_delta` reached only `2.86%`
- current recommendation:
  - GA reward experiments: start with `delta_lexicographic_terminal`, compare against `terminal_lexicographic`
  - SAC reward experiments: start with `delta_lexicographic`; avoid `progress_rate`
  - use `gas_brake_steer` for live SAC so the trained policy learns the real `gas`, `brake`, `steer` layout; keep `gas_steer` only as a diagnostic switch when debugging early-start exploration

### 2026-04-30 SAC debug follow-up

After the large reward experiment showed that SAC from scratch failed on `AI Training #5`, a smaller SAC debug sweep was run on `small_map_test_2` with a `32x32` ReLU policy, `gamma=0.9995`, fixed entropy, and short 3-minute runs. The goal was to isolate whether the issue was reward sign, timeout handling, action layout, or SAC credit assignment.

Key checks:

- Stable-Baselines3 SAC maximizes expected return; internally it minimizes actor/critic losses, but higher environment reward is still preferred.
- Timeout must not be treated as neutral `0` in SAC rewards. Modes based directly on old `terminal_fitness`/`Individual.compute_scalar_fitness_for` can make timeout at low progress much better than crash because `term=0` outranks `term=-1`.
- Pure terminal rewards such as `terminal_lexicographic` are safe but too sparse for SAC from scratch: they reached only about the first progress bucket in the small-map smoke tests.
- `progress_primary_delta` gives SAC a strong dense signal and quickly reached about `50%` on `small_map_test_2`, but it rewards kamikaze checkpoint rushing because a crash after progress can still be a large positive return.
- `delta_progress_time_safety` fixes the kamikaze sign problem by subtracting one full track on crash/timeout, but that made most failed trajectories strongly negative and the policy collapsed back toward early failures.
- `delta_progress_time_block_penalty` was the best diagnostic compromise: crash/timeout are not neutral, but the failure penalty is exactly one map progress bucket. With `gas_steer`, it reached a deterministic `50%` on `small_map_test_2`; with `throttle_steer` or `gas_brake_steer`, learning was worse.

Best debug SAC configuration so far:

```text
reward_mode = delta_progress_time_block_penalty
action_layout = gas_steer
net_arch = [32, 32]
activation = relu
gamma = 0.9995
ent_coef = 0.01
train_freq = (1, "episode")
gradient_steps = 8
```

Current interpretation:

- SAC is not failing because rewards are minimized; the sign convention is correct.
- The first major SAC failure mode was reward design: timeout/crash and progress had inconsistent incentives.
- The second major failure mode is action-space complexity: `gas_brake_steer` makes random exploration weak because gas and brake can cancel, while `throttle_steer` did not explore well enough in the tested setup.
- The third likely failure mode is sparse/long-horizon credit assignment. Even with a better dense reward, SAC still gets stuck around early checkpoints from scratch.
- Practical next step is to apply this first in the local `Experiments/train_sac.py` simulator on `AI Training #5`, then only move back to live Trackmania after it shows meaningful progress locally.
- This setup has now been applied to `Experiments/train_sac.py` for `AI Training #5`:
  `DEFAULT_MAP_NAME="AI Training #5"`, `DEFAULT_REWARD_MODE="delta_progress_time_block_penalty"`, `DEFAULT_ACTION_LAYOUT="gas_steer"`, `DEFAULT_NET_ARCH=[32, 32]`, default `gamma=0.9995`, default `ent_coef="0.01"`, default `gradient_steps=8`.

### 2026-04-30 reproducible reward-function experiment report

This experiment was created after several live Trackmania SAC/GA runs got stuck in local minima. The question was:

> How should the human goal "get as far as possible, and for equal progress get there as fast as possible" be translated into a fitness/reward function without arbitrary magic constants?

The experiment was successful in the local simulator:

- scalar GA reward variants produced working finishers
- a stable/safe baseline exists through the old `terminal_fitness` style
- multi-objective GA, abbreviated as MOO GA in the code and sometimes discussed as MOD GA, reached complete-map solutions fastest
- SAC did not solve the map from scratch under the tested settings, which is itself useful evidence that SAC likely needs pretraining/curriculum rather than more reward guessing in live Trackmania

Experiment environment:

- simulator: `Experiments/tm2d_env.py`
- map: `AI Training #5`
- path tile count: `36`
- discrete progress bucket: `100 / (36 - 1) = 2.857142857%`
- collision mode: `center`
- action layout: `gas`, `brake`, `steer`
- canonical local observation dimension for this experiment: `44`
- surface values were all asphalt-like on this map
- height values were all flat on this map
- generated analysis folder:
  - `Experiments/analysis/reward_experiments_20260430/`
- generated graph files:
  - `reward_experiment_dashboard.png`
  - `ga_best_dense_progress.png`
  - `ga_mean_dense_progress.png`
  - `ga_finish_time.png`
  - `ga_summary_bars.png`
  - `moo_objectives.png`
  - `moo_front0_size.png`
  - `sac_dense_progress_rolling50.png`
  - `sac_episode_reward_rolling50.png`
  - `sac_cumulative_best_dense_progress.png`
  - `sac_summary_bars.png`
  - `best_policy_multiseed_robustness.png`
  - `ga_moo_best_policy_trajectories.png`

Important reproducibility warning:

- the exact physics/dt settings are stored in each run's `config.json`
- the scalar GA and SAC sweeps used `min_dt=0.01`, `max_dt=0.0333333333`
- the MOO GA run was launched before the later `30..100 FPS` default change and its saved config records `min_dt=0.0083333333`, `max_dt=0.04`
- future reproductions should report the saved `config.json` values, not only current defaults in the source file

Scalar GA sweep configuration:

- root: `Experiments/runs/ga_terminal_reward_sweep_20260430_114834`
- generations: `300`
- population size: `64`
- elite count: `4`
- parent count: `16`
- hidden layers: `[32, 16]`
- hidden activations: `["relu", "tanh"]`
- action mode/layout: target-style `gas`, `brake`, `steer`
- mutation probability: `0.18`
- mutation sigma: `0.22`
- fitness mode: `reward`
- workers: `4`
- max episode time: `45.0s`
- tested reward modes:
  - `terminal_progress_time_efficiency`
  - `terminal_lexicographic_progress20`
  - `terminal_lexicographic_no_distance`
  - `terminal_lexicographic`
  - `terminal_fitness`

The scalar GA runs can be reproduced with commands of this form:

```powershell
python Experiments\train_ga.py --map-name "AI Training #5" --generations 300 --population-size 64 --elite-count 4 --parent-count 16 --hidden-dim "32,16" --hidden-activation "relu,tanh" --mutation-prob 0.18 --mutation-sigma 0.22 --fitness-mode reward --reward-mode terminal_lexicographic --collision-mode center --num-workers 4 --max-time 45 --log-dir Experiments\runs\<run_root>\<reward_mode>
```

Multi-objective GA configuration:

- root: `Experiments/runs/ga_moo_trackmania_racing_20260430_120449`
- run folder: `20260430_120457_tm2d_ga_moo_AI_Training_5_trackmania_racing`
- generations: `300`
- population size: `64`
- elite count: `4`
- parent count: `16`
- hidden layers: `[32, 16]`
- hidden activations: `["relu", "tanh"]`
- mutation probability: `0.18`
- mutation sigma: `0.22`
- selection mode: `pareto`
- objective mode: `trackmania_racing`
- Pareto tiebreak: `priority`
- max episode time: `45.0s`
- workers: `4`

The MOO GA run can be reproduced with a command of this form:

```powershell
python Experiments\train_ga_moo.py --map-name "AI Training #5" --generations 300 --population-size 64 --elite-count 4 --parent-count 16 --hidden-dim "32,16" --hidden-activation "relu,tanh" --mutation-prob 0.18 --mutation-sigma 0.22 --selection-mode pareto --objective-mode trackmania_racing --objective-priority "progress,finish,speed_for_progress,safe_progress,path_efficiency" --pareto-tiebreak priority --collision-mode center --num-workers 4 --max-time 45 --log-dir Experiments\runs\<run_root>
```

SAC sweep configuration:

- root: `Experiments/runs/sac_reward_sweep_20260430_132154`
- episodes per mode: `2000`
- max runtime per mode: `45min`
- max episode time: `45.0s`
- action layout: `gas_brake_steer`
- net architecture: `[128, 128]`
- activation: `relu`
- learning rate: `0.0003`
- gamma: `0.9995`
- entropy coefficient: fixed `0.01`
- train frequency: one update batch per episode
- gradient steps: `16`
- collision mode: `center`
- seed: `123`
- tested reward modes:
  - `delta_progress_time_efficiency`
  - `delta_lexicographic`
  - `progress_primary_delta`
  - `individual_dense`
  - `terminal_progress_time_efficiency`

The SAC runs can be reproduced with commands of this form:

```powershell
python Experiments\train_sac.py --map-name "AI Training #5" --reward-mode delta_progress_time_efficiency --episodes 2000 --max-runtime-minutes 45 --env-max-time 45 --action-layout gas_brake_steer --net-arch "128,128" --activation-fn relu --learning-rate 0.0003 --gamma 0.9995 --ent-coef 0.01 --train-freq-episode --gradient-steps 16 --collision-mode center --seed 123 --log-dir Experiments\runs\<run_root>\<reward_mode>
```

Reward and fitness functions tested:

The goal of this section is to make the reward functions readable even for
someone seeing this project for the first time. In all GA scalar modes,
higher score is better. In tuple-style pseudocode, Python compares the tuple
from left to right, so the first element is the most important criterion.

Important implementation note:
`fitness` was originally useful mainly as a scalar value for logging, plotting,
and quick summaries. The more faithful GA comparison can be the tuple returned
by `Individual.ranking_key()`, because `Individual.__lt__` and `Individual.__eq__`
fall back to tuple comparison whenever `fitness` is `None`. The local
`Experiments/train_ga.py` script now has `--fitness-mode ranking` for this:
it stores rollout metrics on each `Individual`, leaves `fitness=None`, and lets
`population.sort(reverse=True)` use the tuple ordering directly. The scalar
`best_fitness` column in CSV should then be read only as a diagnostic plot
value, not as the actual selection mechanism.

Tuple ranking candidates are intentionally configured separately from
`--reward-mode`. The current syntax is an explicit tuple expression via
`--ranking-key`; old `--ranking-key-mode` names were removed from the active
workflow because they hid the real lexicographic comparison behind strings like
`progress_term_time_distance`. Historical aliases were:

```text
term_progress_time_distance              = (term, progress, -time, -distance)
progress                                 = (progress,)
progress_time                            = (progress, -time)
progress_time_distance                   = (progress, -time, -distance)
finished_progress_time                   = (finished, progress, -time)
progress_finished_time                   = (progress, finished, -time)
progress_finished_time_distance          = (progress, finished, -time, -distance)
progress_finished_crashed_time           = (progress, finished, -crashed, -time)
progress_finished_crashed_time_distance  = (progress, finished, -crashed, -time, -distance)
finished_crashed_progress_time           = (finished, -crashed, progress, -time)
crashed_finished_progress_time           = (-crashed, finished, progress, -time)
finished_progress_crashed_time           = (finished, progress, -crashed, -time)
finished_progress_crashed_time_distance  = (finished, progress, -crashed, -time, -distance)
finished_progress_time_crashed           = (finished, progress, -time, -crashed)
finished_progress_time_distance          = (finished, progress, -time, -distance)
progress_term_time                       = (progress, term, -time)
progress_term_time_distance              = (progress, term, -time, -distance)  # legacy comparison
```

Current explicit syntax:

```python
ranking_key = "(dense_progress, term, -time, -distance)"
```

`RankingKey.py` parses this without using `eval`: it accepts a comma-separated
tuple of known metric names and an optional unary `-`. Legacy underscore names
now raise an error with an example tuple, so new experiments should be
self-documenting in their command line and `config.json`.

Allowed metric names:

```text
term, finished, crashed, progress, ranking_progress,
discrete_progress, dense_progress, time, distance
```

The `progress` slot in these tuple modes can now be selected with
`--ranking-progress-source`:

```text
discrete_progress = confirmed checkpoint/path-tile progress in percent
dense_progress    = geometric progress projected onto the current path segment, also in percent
```

Naming note:
The runtime metric cleanup intentionally removed the old aliases
`total_progress`, `tile_progress`, `block_progress`, and `map_progress`. They
were useful during the transition, but they made reward discussions ambiguous.
Current code should read `discrete_progress` or `dense_progress` explicitly.
Some persisted artifacts still keep older field names such as checkpoint
`progresses`/`best_progress` for compatibility with existing files; those are
storage compatibility names, not live observation keys.

This lets us test a very simple dense-progress ranking without creating a new
reward function. For example:

```powershell
python Experiments\train_ga.py --map-name "AI Training #5" --fitness-mode ranking --ranking-key "(dense_progress, -time)"
```

This ranks agents as:

```python
return (dense_progress, -time)
```

Interpretation:
The agent is first rewarded for getting as far as possible along the route,
including smooth in-between progress inside a block. If two agents reach the
same dense progress, the faster one wins. This is intentionally minimal and is
a useful baseline for checking whether the extra `term`, crash, and distance
criteria are actually helping or only adding noise.

Current real Trackmania GA overnight test:

`GeneticTrainer.py` is prepared for a real-game overnight GA run using tuple
selection rather than scalar fitness:

```python
Individual.COMPARE_BY_RANKING_KEY = True
Individual.RANKING_KEY = "(dense_progress, term, -time, -distance)"
Individual.RANKING_PROGRESS_SOURCE = "dense_progress"
```

The effective selection tuple is:

```python
return (dense_progress, term, -time, -distance)
```

Interpretation:
Dense progress is the primary objective. `term` is only a tie-break when two
agents reach effectively the same route position, so a timeout can no longer
beat a much farther crash just because `term=0 > -1`. Time and distance remain
later tie-breakers. This is the real Trackmania version of the local simulator
result where dense progress was much more robust than discrete tile progress.
`discrete_progress` is logged as the confirmed path-tile progress and
`dense_progress` as the smoother projected progress. `generation_summary.csv`
logs both `best_dense_progress` and `mean_dense_progress`. New real GA run
directory names include the progress source, for example
`_dense_progress_term_neg_time_neg_distance`, to avoid mixing these results
with older discrete-progress runs.

Elite evaluation cache:
Live Trackmania evaluation is expensive and non-headless, so unchanged elite
copies are no longer re-driven every generation. `Individual.copy()` preserves
the previous rollout metrics and `evaluation_valid=True`; genome changes
through mutation reset that flag. `GeneticTrainer.evaluate_population()` skips
valid non-mirrored individuals and logs the count as `cached_evaluations` in
`generation_summary.csv`. This should save roughly the elite fraction of live
evaluation time when mirror evaluation is disabled. It intentionally does not
reuse metrics for mirrored/evaluate-both-mirrors rollouts.

This cache is a time optimization, not a robustness guarantee. Because live
Trackmania is noisy (variable FPS, reset timing, controller timing, small
position differences), a risky individual can occasionally get one lucky strong
run and then stay protected by cached elite metrics. That can help speed, but it
can also propagate a policy that does not reliably reproduce its best result.
For fast exploratory training this is acceptable; for scientific comparison or
final model selection we should explicitly validate without cache.

Recommended cache protocol:

- Fast training mode: keep elite cache enabled to save real wall-clock time.
- Robust evaluation mode: re-drive elite individuals every generation or every
  `N` generations and compare against the cached variant.
- Final validation: take top `K` policies, run each policy multiple times
  without cache, and report mean, median, best, worst, finish rate, crash rate,
  and timeout rate.
- Thesis experiment: compare cache ON/OFF/hybrid as an explicit research
  variable. Cache ON should train faster; cache OFF or periodic re-evaluation
  should better reject lucky/risk-seeking policies.

The same cache is also implemented in the local 2D GA sandbox. In
`Experiments/train_ga.py`, unchanged valid elites are converted back to cached
metric dictionaries before worker payloads are built, so process-based
parallelism remains simple: workers only receive genomes that really need a new
rollout. `generation_metrics.csv` logs `cached_evaluations`. A smoke run with
`--num-workers 2`, `--population-size 6`, and `--elite-count 2` confirmed the
expected behavior: generation 1 cached `0` evaluations and generation 2 cached
`2` evaluations.

Current outcome metrics are `finished` and `crashes`. `finished = 1` only when
the agent reaches finish. `crashes` is the number of counted touches/crashes in
the rollout and can be greater than `1` when `max_touches > 1`. Timeout is
derived as `finished == 0 and crashes == 0`.

Common metrics used by the reward functions:

`term`:
This is the terminal result of a rollout. `term = 1` means the agent reached
the finish. `term = 0` means the episode ended by timeout or is still running
inside a non-terminal scoring call. `term = -1` means crash/failure. In the
old `Individual` ranking, `term` is the strongest criterion, so finish beats
timeout and timeout beats crash.

`terminal`:
This is true when the episode is over. In the local simulator this means
`term != 0` or `time >= max_time`. Terminal-only reward modes return `0`
while the episode is still running and only produce a meaningful score at
crash, timeout, or finish.

`progress`:
This is track progress in percent, usually `0..100`. On exported Trackmania
maps it is mostly discrete at path-tile boundaries. For `AI Training #5`, the
path has `36` path tiles, so one discrete bucket is
`100 / (36 - 1) = 2.857142857%`.

`dense_progress`:
This is a smoother progress estimate between path tiles. In the local
simulator it comes from the projected route position; in real Trackmania it is
computed in `Car.py` by projecting the car position onto the centerline segment
from the current path tile to the next path tile. It is still expressed in
percent, but it can produce values between the discrete progress buckets. It is
clamped so it never goes below confirmed `discrete_progress`.

`progress_norm`:
This is normalized progress:

```text
progress_norm = clamp(progress / 100, 0, 1)
```

It converts `0..100%` progress to `0..1`, which makes it easier to mix with
other normalized terms.

`path_tile_count`:
The number of path tiles in the map path. It is used to derive the smallest
meaningful progress unit from the map itself, instead of inventing a magic
constant.

`tile_unit`:
One map progress bucket in normalized units:

```text
tile_unit = 1 / (path_tile_count - 1)
```

For `AI Training #5`, `tile_unit = 1 / 35 = 0.02857142857`. A key design idea
of the bounded lexicographic rewards is that one `tile_unit` of progress
should dominate the entire time tie-break.

`tile_percent`:
The same one-tile bucket in percent units:

```text
tile_percent = 100 * tile_unit
```

For `AI Training #5`, this is `2.857142857%`.

`time` / `time_value`:
The time in seconds from the start of the episode. Lower is better only after
the agent has made comparable progress. We try to avoid rewarding a policy
only because it crashes quickly.

`max_time`:
The episode timeout in seconds. In the main reward experiment it was `45s`.
It is used to normalize time:

```text
time_norm = clamp(time / max_time, 0, 1)
```

`distance`:
The physical distance driven by the car. This is not the same as progress.
A car can drive a long zig-zag while making little progress. Some reward
functions use distance as a final tie-break or as an excess-distance penalty.

`distance_limit` / `max_episode_distance`:
A map-derived or physics-derived upper bound used to normalize distance. This
keeps distance terms small and prevents them from dominating progress.

`estimated_path_length`:
The estimated ideal path length of the map. It is used to estimate how much
distance would be reasonable for a given `progress_norm`.

`excess_distance`:
Only the distance driven beyond the progress-scaled ideal distance:

```text
ideal_distance = progress_norm * estimated_path_length
excess_distance = max(0, distance - ideal_distance)
```

This discourages zig-zagging without forcing the agent to take the
geometrically shortest line in every situation.

1. `terminal_fitness`

This is the old `Individual.compute_scalar_fitness_for` style. It first builds
a lexicographic tuple with `Individual.ranking_key_for`, then converts that
tuple into one large scalar number.

Pseudocode:

```python
term = int(term)
progress = float(progress)
dist = float(distance)
t = float(time_value)

if is_finite(t):
    time_bucket = floor(t)
else:
    time_bucket = INF

if progress < 20:
    ranking_key = (term, progress, 0, 0)
elif term <= 0:
    ranking_key = (term, progress, -t, 0)
else:
    ranking_key = (term, progress, -time_bucket, -dist)

term_key, progress_key, neg_time_key, neg_dist_key = ranking_key
time_key = -neg_time_key
dist_key = -neg_dist_key

fitness = (
    term_key * 1_000_000_000
    + progress_key * 1_000_000
    - time_key * 10_000
    - dist_key
)
```

`term`:
Because `term` is multiplied by `1_000_000_000`, terminal status dominates
everything else. A finished run is much better than a timeout/crash. A timeout
also outranks a crash because `0 > -1`.

`progress`:
Progress is the second strongest component. More progress is better, but only
after `term` has already been compared.

`time_bucket`:
For finished runs, time is bucketed to integer seconds using `floor(time)`.
This makes tiny sub-frame timing noise less important. For unfinished runs
above `20%`, exact `time` is used.

`distance`:
Distance is only used for finished runs as the last tie-break. Shorter
finished runs get slightly better score.

Interpretation:
This reward is conservative and strongly lexicographic. It solved the map in
the GA experiment, but it was the slowest scalar function to discover a
finisher: first finish around generation `198`, best time `16.406s`. Its
strength is safety/robustness; its weakness is a coarse and very discontinuous
learning signal.

2. `terminal_lexicographic`

This is a terminal-only bounded scalar. It encodes the same human preference
as a lexicographic comparison, but with map-derived scales instead of huge
weights.

Pseudocode:

```python
if not terminal:
    return 0

progress_norm = clamp(progress / 100, 0, 1)
tile_unit = 1 / (path_tile_count - 1)
time_norm = clamp(time / max_time, 0, 1)
distance_norm = clamp(distance / distance_limit, 0, 1)

score = progress_norm

if term == 1:
    score += 1
elif include_failure_term:
    score -= 1

score += tile_unit * (1 - time_norm)
score += tile_unit**2 * (1 - distance_norm)

return score
```

`progress_norm`:
This is the primary non-terminal criterion. More progress gives a higher
score.

`term` / finish bonus:
If `term == 1`, the score gets `+1`, which is equivalent to one full track.
This makes a finish clearly better than any partial run.

`include_failure_term`:
For terminal lexicographic modes used in the GA sweep, this is true, so crash
or timeout gets `-1`. This makes a failed terminal state globally worse than a
successful finish, while progress still differentiates failed attempts.

`tile_unit * (1 - time_norm)`:
This is the time tie-break. It is intentionally smaller than one progress
bucket, so a faster but shorter run should not beat a slower but farther run.

`tile_unit**2 * (1 - distance_norm)`:
This is the distance tie-break. It is even smaller than the time tie-break, so
distance cannot dominate time or progress.

Interpretation:
This function directly encodes "go farther first, then be faster, then be
more efficient". It worked very well in GA: first finish around generation
`105`, best time `15.991s`. It is a strong scalar baseline.

3. `terminal_lexicographic_no_distance`

This is the same as `terminal_lexicographic`, but removes the distance
tie-break entirely.

Pseudocode:

```python
if not terminal:
    return 0

progress_norm = clamp(progress / 100, 0, 1)
tile_unit = 1 / (path_tile_count - 1)
time_norm = clamp(time / max_time, 0, 1)

score = progress_norm

if term == 1:
    score += 1
elif include_failure_term:
    score -= 1

score += tile_unit * (1 - time_norm)

return score
```

`progress_norm`:
Primary objective. More progress is better.

`term`:
Finish gets the full-track bonus. Failed terminal states get the failure
penalty.

`time_norm`:
Only used as a tie-break after progress. Faster is better when progress is
effectively equal.

`distance`:
Distance is intentionally ignored. This avoids penalizing racing lines that
are longer geometrically but faster dynamically.

Interpretation:
This was the fastest scalar function by best time in the GA sweep. It found
the first finish around generation `95` and reached best time `15.985s`.
However, multi-seed re-evaluation showed it was less robust than
`terminal_lexicographic` and `terminal_fitness`, probably because removing the
distance tie-break gives less pressure toward stable efficient paths.

4. `terminal_lexicographic_progress20`

This is a terminal-only lexicographic score that delays time/distance pressure
until the agent reaches at least `20%` progress.

Pseudocode:

```python
if not terminal:
    return 0

progress_norm = clamp(progress / 100, 0, 1)
tile_unit = 1 / (path_tile_count - 1)

score = progress_norm

if term == 1:
    score += 1
elif include_failure_term:
    score -= 1

if progress >= 20:
    time_norm = clamp(time / max_time, 0, 1)
    score += tile_unit * (1 - time_norm)

if progress >= 20 and term == 1:
    distance_norm = clamp(distance / distance_limit, 0, 1)
    score += tile_unit**2 * (1 - distance_norm)

return score
```

`progress < 20`:
Below `20%`, the function mostly asks only "did the agent get farther?".
This avoids punishing slow early exploration before the agent has learned
basic road following.

`progress >= 20`:
After the agent can drive a meaningful part of the map, time becomes a
tie-break and faster runs start getting better score.

`distance`:
Distance is only used very conservatively, mainly for finished/meaningful
runs.

Interpretation:
This is a curriculum-like scalar reward. It worked well in training: first
finish around generation `109`, best time `16.001s`. However, saved-policy
multi-seed re-evaluation was weaker, so it may discover finishers but produce
less robust policies.

5. `terminal_progress_time_efficiency`

This is a terminal-only version of a progress-primary score. It is designed
to avoid arbitrary huge weights while still saying: progress first, then time,
then avoid unnecessary extra distance.

Pseudocode:

```python
if not terminal:
    return 0

progress_norm = clamp(progress / 100, 0, 1)
tile_unit = 1 / (path_tile_count - 1)
time_norm = clamp(time / max_time, 0, 1)

score = progress_norm

if term == 1:
    score += 1

if progress_norm > 0:
    score += tile_unit * progress_norm * (1 - time_norm)

    ideal_distance = progress_norm * estimated_path_length
    excess_distance = max(0, distance - ideal_distance)
    excess_norm = clamp(excess_distance / max_episode_distance, 0, 1)

    score -= tile_unit**2 * progress_norm * excess_norm

return score
```

`progress_norm`:
Primary objective. A run that gets farther should generally score higher.

`term`:
Finish gets `+1`, so completing the map is strongly preferred.

`tile_unit * progress_norm * (1 - time_norm)`:
This is the time bonus, but it is gated by progress. If the agent makes no
progress, it cannot win by crashing quickly. If it makes more progress, speed
matters more.

`ideal_distance`:
The expected distance for the achieved progress, based on the map length.

`excess_distance`:
Only distance beyond the expected amount is penalized. This avoids rewarding
zig-zagging, but it does not blindly force the shortest geometric line.

Interpretation:
This function is conceptually clean and avoids most magic constants. It
solved the GA map, but slower than lexicographic variants: first finish around
generation `129`, best time `16.051s`. It is useful, but not the strongest
scalar GA reward from the experiment.

6. `delta_progress_time_efficiency`

This is the potential-difference version of
`terminal_progress_time_efficiency`. The score formula is the same, but the
environment reward at each step is the change in score.

Pseudocode:

```python
current_score = progress_time_efficiency_score(
    term=term,
    progress=dense_progress,
    time=time,
    distance=distance,
    terminal_only=False,
)

reward = current_score - previous_score
previous_score = current_score

return reward
```

`current_score`:
The current potential value of the state according to the progress/time/
efficiency formula.

`previous_score`:
The score from the previous simulator step.

`reward`:
Positive when the state improved, negative when the score dropped. This is
mainly useful for RL/SAC because it provides dense feedback during the run.

Interpretation:
This was included mainly for SAC-style learning. For GA, terminal whole-run
selection worked better. The delta formulation is not automatically bad, but
it can make local step incentives more important than whole-run success.

7. `delta_lexicographic`

This is the dense potential-difference version of the bounded lexicographic
score. It gives reward continuously as the potential score changes.

Pseudocode:

```python
current_score = lexicographic_score(
    term=term,
    progress=dense_progress,
    time=time,
    distance=distance,
    include_failure_term=False,
    distance_mode="all",
)

reward = current_score - previous_score
previous_score = current_score

return reward
```

`dense_progress`:
Used instead of purely discrete progress so the agent can receive feedback
between path tiles.

`include_failure_term=False`:
A crash does not retroactively erase all earlier progress. The episode ends,
so the agent loses future reward opportunities, but it does not receive a huge
terminal drop.

`time` and `distance`:
They remain small tie-breaks inside the potential score.

Interpretation:
This can be useful for RL because it provides dense reward. In the GA
experiments, however, delta modes did not outperform terminal selection. GA
does not need per-frame credit assignment as much as SAC does; it can rank a
whole rollout directly.

8. `delta_lexicographic_terminal`

This is the same as `delta_lexicographic`, but failed terminal states also
apply the failure term.

Pseudocode:

```python
current_score = lexicographic_score(
    term=term,
    progress=dense_progress,
    time=time,
    distance=distance,
    include_failure_term=True,
    distance_mode="all",
)

reward = current_score - previous_score
previous_score = current_score

return reward
```

`include_failure_term=True`:
Crash or timeout creates a terminal score drop. This makes failure explicitly
bad instead of merely ending future reward.

`reward`:
The final step can become strongly negative if the agent crashes or times out.

Interpretation:
This looked promising in short early sweeps because it rewards progress but
still marks failure. It did not become the best GA direction in the larger
experiment. For SAC it can be too harsh early in training because most random
episodes fail.

9. `individual_dense`

This uses the old `Individual.compute_scalar_fitness_for` idea, but replaces
discrete progress with `dense_progress`.

Pseudocode:

```python
dense_progress = geometric_progress_between_path_tiles()

score = Individual.compute_scalar_fitness_for(
    term=term,
    progress=dense_progress,
    time_value=time,
    distance=distance,
)

return score / terminal_fitness_scale
```

`dense_progress`:
Prevents many random policies from tying at exactly `0%`.

`terminal_fitness_scale`:
Only rescales the large old scalar to a smaller numeric range. It should not
change ordering.

`term`, `time`, and `distance`:
They behave like the old `terminal_fitness` logic.

Interpretation:
This was added as a conservative dense variant of the old fitness. It is
useful diagnostically, but it did not become the strongest GA reward.

10. `terminal_progress_time_safety`

This score is in progress-percent units rather than normalized `0..1` units.
It was created mainly for SAC/debugging to make progress visually obvious in
logs while making crash/timeout globally bad.

Pseudocode:

```python
if terminal_only and not terminal:
    return 0

progress = clamp(progress, 0, 100)
progress_norm = progress / 100
tile_percent = 100 / (path_tile_count - 1)
time_norm = clamp(time / max_time, 0, 1)

score = progress

if progress_norm > 0:
    score += tile_percent * progress_norm * (1 - time_norm)

if terminal:
    if term == 1:
        score += 100
    else:
        score -= 100

return score
```

`progress`:
The base score is direct progress percent. `50%` progress starts from score
`50`.

`time bonus`:
The time bonus is at most one path-tile bucket and is scaled by progress.

`finish bonus`:
Finish adds `+100`, equivalent to another full track.

`failure penalty`:
Crash or timeout subtracts `100`, equivalent to one full track. This makes all
failures strongly bad.

Interpretation:
This fixed the sign problem where timeout/crash could be too attractive, but
it was too harsh for early SAC from scratch. Most random policies fail, so the
learning signal can collapse toward avoiding motion instead of exploring.

11. `terminal_progress_time_block_penalty`

This is a softer curriculum variant of `terminal_progress_time_safety`.
Instead of subtracting one full track on failure, it subtracts one map-derived
progress bucket.

Pseudocode:

```python
if terminal_only and not terminal:
    return 0

progress = clamp(progress, 0, 100)
progress_norm = progress / 100
tile_percent = 100 / (path_tile_count - 1)
time_norm = clamp(time / max_time, 0, 1)

score = progress

if progress_norm > 0:
    score += tile_percent * progress_norm * (1 - time_norm)

if terminal:
    if term == 1:
        score += 100
    else:
        score -= tile_percent

return score
```

`tile_percent`:
This is the key difference from the safety version. Failure costs one block of
progress, not the whole map.

`progress`:
Progress remains dense and easy for SAC to learn from.

`time bonus`:
Faster progress still gets a small reward, but it cannot dominate a full
progress bucket.

Interpretation:
This became the best SAC debug candidate. It keeps timeout/crash non-neutral,
but it does not destroy the reward of every failed exploratory run. On
`small_map_test_2`, the diagnostic SAC policy reached about `50%`
deterministically. On `AI Training #5`, it improved over the start but still
got stuck around the early buckets from scratch.

12. `delta_progress_time_block_penalty`

This is the potential-difference version of
`terminal_progress_time_block_penalty`.

Pseudocode:

```python
current_score = progress_time_block_penalty_score(
    term=term,
    progress=dense_progress,
    time=time,
    distance=distance,
    terminal_only=False,
)

reward = current_score - previous_score
previous_score = current_score

return reward
```

`dense_progress`:
Allows partial progress feedback inside a path tile.

`terminal failure`:
At crash or timeout, the score loses one `tile_percent`. This is enough to
make failure bad, but not so large that all failed exploration becomes
worthless.

Interpretation:
This is the current best local SAC debug mode, not the best GA mode. It is a
compromise between dense learning and avoiding the old local minimum where
timeout/crash handling made standing still attractive.

13. `progress_rate`

This is an average progress-rate score. It is kept mostly as a negative
control.

Pseudocode:

```python
if time <= 0:
    return 0

progress_norm = clamp(progress / 100, 0, 1)
time_norm = max(time / max_time, tile_unit)

score = progress_norm / time_norm

return score
```

`progress_norm / time_norm`:
This rewards progress per normalized time, so a fast early progress jump can
look very good.

`tile_unit` minimum:
This prevents division by near-zero time, but it does not solve the deeper
incentive problem.

Interpretation:
This is unsafe for racing as a main reward. A fast kamikaze crash after a tiny
amount of progress can outrank a slower but much more useful trajectory. It
should be treated as a negative-control mode, not a recommended objective.

14. MOO GA `trackmania_racing`

This is not one scalar reward. It is a multi-objective GA selection scheme
based on Pareto dominance. A policy is better if it is at least as good in all
objectives and better in at least one objective.

Pseudocode:

```python
progress_obj = dense_progress / 100

finish_obj = finished

speed_for_progress_obj = progress_obj * (1 - time / max_time)

safe_progress_obj = progress_obj * (1 - min(crashes / max_crashes, 1))

ideal_distance = progress_obj * estimated_path_length
excess_distance = max(0, distance - ideal_distance)
path_efficiency_obj = progress_obj * (1 - excess_distance / max_episode_distance)

objectives = (
    finish_obj,
    progress_obj,
    speed_for_progress_obj,
    safe_progress_obj,
    path_efficiency_obj,
)
```

`finish_obj`:
Whether the agent reached the finish. This separates complete-map policies
from partial policies.

`progress_obj`:
How far the agent got. This acts like finish split into smaller measurable
steps while the policy cannot finish yet.

`speed_for_progress_obj`:
How quickly the agent achieved its progress. It is multiplied by progress, so
a parked car cannot score well just because time handling is favorable.

`safe_progress_obj`:
Progress scaled by remaining crash budget. This discourages pure kamikaze
progress while still supporting real Trackmania runs where `max_touches > 1`.

`path_efficiency_obj`:
Rewards progress with limited excess distance. It is also gated by progress,
so a parked car is not an "efficient" solution.

Default within-front priority is:

```text
finish -> progress -> speed_for_progress -> safe_progress -> path_efficiency
```

Pareto fronts preserve multi-objective diversity first; this priority is only
used inside the same non-dominated front.

Pareto comparison:
A dominates B if A is at least as good in every objective and strictly better
in at least one objective. When policies are on the same Pareto front, the
priority tiebreak used in the experiment was:

```text
progress > finish > speed_for_progress > safe_progress > path_efficiency
```

Interpretation:
This was the most promising GA direction. It found a finisher earliest,
around generation `81`, with best time `15.998s`. The main advantage is that
we do not need to compress progress, finish, speed, safety, and path
efficiency into one fragile hand-weighted scalar.

Experiment results:

| Method | First finish | Best finish time | Last-50 mean dense progress | Notes |
| --- | ---: | ---: | ---: | --- |
| MOO GA `trackmania_racing` | generation `81` | `15.998s` | `50.15%` | fastest to discover a finisher |
| GA `terminal_lexicographic_no_distance` | generation `95` | `15.985s` | `50.13%` | fastest scalar best-time, but less robust on re-eval |
| GA `terminal_lexicographic` | generation `105` | `15.991s` | `49.75%` | strong scalar baseline |
| GA `terminal_lexicographic_progress20` | generation `109` | `16.001s` | `52.33%` | good population mean, weak re-eval robustness |
| GA `terminal_progress_time_efficiency` | generation `129` | `16.051s` | `45.53%` | solved but less robust |
| GA `terminal_fitness` | generation `198` | `16.406s` | `51.38%` | slowest to solve, but safest re-eval |

SAC results:

| SAC reward mode | Max dense progress | Last-100 mean dense progress | Terminal behavior |
| --- | ---: | ---: | --- |
| `individual_dense` | `2.857%` | `1.979%` | 74 timeouts, 1841 crashes, stopped at runtime limit |
| `progress_primary_delta` | `0.895%` | `0.574%` | 2000 crashes |
| `terminal_progress_time_efficiency` | `0.562%` | `0.335%` | 2000 crashes |
| `delta_lexicographic` | `0.540%` | `0.343%` | 2000 crashes |
| `delta_progress_time_efficiency` | `0.512%` | `0.356%` | 2000 crashes |

Interpretation:

- GA clearly benefited from terminal selection because a whole rollout can be evaluated and selected without credit assignment through every individual frame.
- MOO GA was the most interesting result because it found a full-map solution earliest while keeping the objective definitions interpretable.
- Scalar `terminal_lexicographic` is still valuable as a simple baseline because it also solved the map and encodes the intended ranking cleanly.
- `terminal_fitness` is conservative: it solved late but was the most robust when the saved best policies were re-evaluated under multiple simulator seeds.
- SAC from scratch did not learn meaningful road-following in this experiment. The best SAC mode only reached the first progress bucket. This suggests the bottleneck is not just the exact reward formula, but SAC exploration/credit assignment from an untrained continuous policy.
- For SAC, the next sensible path is not another live Trackmania reward guess. It should be either:
  - initialize SAC from a GA/supervised policy
  - use curriculum maps that are much shorter than `AI Training #5`
  - pretrain basic road-following in the local simulator first

Multi-seed re-evaluation of saved best GA/MOO policies:

| Policy source | Finish rate over 12 seeds | Mean dense progress | Best finish time | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `terminal_fitness` | `100.0%` | `100.00%` | `16.410s` | safest/stablest but slower |
| MOO GA `trackmania_racing` | `91.7%` | `98.39%` | `15.999s` | best speed/discovery trade-off |
| `terminal_lexicographic` | `91.7%` | `95.09%` | `16.000s` | strong scalar alternative |
| `terminal_lexicographic_no_distance` | `66.7%` | `89.34%` | `15.983s` | can produce very fast runs, less stable |
| `terminal_lexicographic_progress20` | `16.7%` | `66.29%` | `16.038s` | weaker robustness despite good training CSV |
| `terminal_progress_time_efficiency` | `0.0%` | `35.42%` | n/a | not robust enough as best-policy choice |

Practical conclusion:

- for local GA reward research, use MOO GA `trackmania_racing` as the main experimental direction
- keep `terminal_lexicographic` as the scalar control/baseline
- keep `terminal_fitness` as the safety baseline
- do not judge a run only by its best generation row; always re-evaluate the saved `best_policy.pt` over multiple simulator seeds
- do not use SAC-from-scratch results in the local simulator as evidence that a reward is bad for all algorithms; in this experiment SAC failed much earlier than reward ranking could become meaningful

Recommended extra logging for future reproductions:

- log wall-clock training duration
- log total simulated game time per run:
  - for GA: sum of all individual rollout times across all generations
  - for SAC: sum of all episode times
- log per-generation cumulative simulated game time
- include simulated game time on plots, e.g. `generation 300 = 30h virtual driving time`
- this matters because a local GA might run for `45min` of wall-clock time while evaluating tens of hours of virtual driving; thesis/report graphs should show both wall-clock efficiency and virtual experience consumed
- recommended CSV fields:
  - `wall_time_elapsed_seconds`
  - `generation_sim_time_seconds`
  - `cumulative_sim_time_seconds`
  - `generation_rollouts`
  - `cumulative_rollouts`

Recommended evaluation plots for lexicographic GA behavior:

The four lexicographic candidates
`(finished, progress)`, `(finished, progress, -time)`,
`(finished, progress, -time, -crashes)`, and
`(finished, progress, -crashes, -time)` should not be judged by final time
alone. They intentionally trade off speed, progress, and safety differently.

Suggested safety metrics:

- `finish_rate`: fraction of validation runs that reach the finish.
- `crash_rate`: fraction of validation runs with `crashes > 0`.
- `mean_crashes`: average crash/touch count per validation run.
- `timeout_rate`: derived as `finished == 0 and crashes == 0`.
- `safe_finish_rate`: fraction of validation runs with `finished == 1` and
  `crashes == 0`.
- `crash_free_progress`: mean progress over runs with `crashes == 0`.
- `robust_progress`: median progress and 10th percentile progress over repeated
  validation seeds; this shows whether a policy is reproducible or only has a
  lucky best run.

Good single-figure options:

- Bubble scatter:
  - x-axis: mean/median final time of successful runs
  - y-axis: finish rate
  - bubble size: mean crashes or crash rate
  - color: safe finish rate or timeout rate
  This shows fast-but-risky policies as large/dark bubbles and safe reliable
  policies as small/light bubbles.
- Progress-time-safety scatter:
  - x-axis: mean time
  - y-axis: median progress
  - color gradient: crash rate
  - marker shape: reward/ranking key
  This works even when some methods do not finish often enough for time-only
  comparison.
- Pareto-style plot:
  - x-axis: median time among finishes
  - y-axis: safe finish rate
  - annotate each lexicographic key
  This visually motivates why NSGA-II/Pareto selection is useful: there is no
  universally best scalar ordering when speed and safety conflict.

Avoid compressing safety into one arbitrary weighted score for the main thesis
plot. It is better to show time, progress, finish rate, and crash behavior as
separate dimensions. If a single summary number is needed, use it only as a
secondary helper and define it from observable quantities, for example
`safe_finish_rate` rather than a hand-weighted sum.


## Current High-Level Architecture

### Runtime dataflow

1. Trackmania runs with the OpenPlanet plugin in `Plugins/get_data_driver/main.as`.
2. The plugin streams one fixed-size packet per game frame over TCP on `127.0.0.1:9002`.
3. `Car.py` connects to the socket, reads packets in a background thread, keeps only the latest decoded packet, and exposes it to the rest of the Python app.
4. `Map.py` loads track geometry and logical path data from map block layouts and block meshes.
   Preferred cleaned-up names are `Maps/BlockLayouts/*.txt` and
   `Assets/BlockMeshes/*.obj`; the current code still falls back to the legacy
   `Maps/ExportedBlocks/*.txt` and `Meshes/*.obj` folders until the physical
   directory move is done.
5. `Car.py` combines the live packet with map/path state and lidar-style raycasts to produce:
   - laser distances
   - upcoming path instructions
   - progress info
   - signed near/far heading alignment against upcoming path segments
6. `ObservationEncoder.py` standardizes those values into the neural-network observation vector.
7. A policy from `NeuralPolicy.py` maps observation -> action.
8. `Enviroment.py` applies that action through `vgamepad` to Trackmania and enforces training guards such as timeout, touches, idle detection, and wall-ride detection.
9. `Enviroment.reset()` now performs a confirmed track restart handshake:
   - press `B` on the virtual gamepad
   - poll live telemetry until a negative `time` is observed
   - retry the `B` press several times if negative time is not seen
   This prevents the next individual from inheriting stale state from the previous attempt when Trackmania does not restart on the first button press.

### GA training dataflow

1. `GeneticTrainer.py` initializes a population of `Individual` objects.
2. Each `Individual` contains a `NeuralPolicy` and a flattened genome view over model parameters.
3. `GeneticTrainer.py` evaluates each individual sequentially in Trackmania through `RacingGameEnviroment`.
4. The environment returns terminal status and telemetry.
5. The individual is ranked by:
   - averaged rollout fitness across normal and mirrored evaluation
   - telemetry summaries still keep representative aggregate metrics for logging
6. The GA applies elitism, selection from the top half, arithmetic-mean crossover, mutation, and annealed mutation schedules.
   Parent pairing now happens without repetition inside each pairing round; once the round is exhausted,
   the top-half parent pool is reshuffled and paired again if more children are still needed.
7. Training logs and checkpoints are stored under `logs/ga_runs/...`.
8. Population checkpoints now also store `current_mutation_prob` and `current_mutation_sigma`
   so resume can continue the annealing schedule from the actual checkpoint state instead of
   reusing only the values from the trainer script.

### GA trainer features worth documenting

The real Trackmania GA trainer accumulated several practical improvements that
are useful for the thesis because they show the engineering problems caused by
slow, non-headless, real-time evaluation:

- Lexicographic ranking keys:
  `Individual.ranking_key()` can compare policies by explicit tuples such as
  `(dense_progress, finished, -time, -crashes, -distance)`. This avoided
  arbitrary weighted sums, but also exposed a real limitation: changing the
  metric order changes the learned behavior. This later motivated the
  multi-objective/Pareto GA experiments.
- Dense progress:
  beside discrete path-tile progress, the trainer can rank by projected
  between-tile progress. This reduces the sparse-reward problem where many
  individuals are tied at the same checkpoint percentage.
- Mirrored evaluation:
  `mirror_episode_prob` can randomly evaluate some individuals on mirrored
  observations/actions, while `evaluate_both_mirrors=True` evaluates every
  individual in both normal and mirrored mode and averages the result. This was
  introduced to reduce left/right track bias.
- Multiple touches:
  `max_touches` allows experiments where a policy may survive a small number
  of wall contacts. The outcome metric is now `crashes` rather than a single
  binary crash flag, so safety can be optimized as a count.
- Mutation annealing:
  `mutation_prob` and `mutation_sigma` can decay toward configured minimums.
  Checkpoints store the current mutation state, so resumed runs continue from
  the real exploration/fine-tuning point instead of restarting the schedule.
- Arithmetic-mean crossover:
  children are produced by averaging parent genomes before mutation. Parent
  pairing is done without repetition inside each pairing round, then reshuffled
  for the next round. This gives the top parent pool more even genetic usage.
- Elite cache:
  unchanged elite copies can keep their previous rollout metrics, saving live
  Trackmania evaluation time. This is a speed optimization, not a robustness
  guarantee, so final validation should re-drive top policies without cache.
- Supervised/model seeding:
  the population can start from a supervised `.pt` policy or a population
  checkpoint instead of pure random genomes. Seeded copies can be mixed with
  increasingly mutated variants.
- Confirmed reset handshake:
  `Enviroment.reset()` repeatedly presses `B` and waits for telemetry evidence
  of a real restart. This avoids leaking stale Trackmania state from one
  individual into the next.
- Detailed experiment logs:
  generation summaries, individual metrics, checkpoints, best-policy payloads,
  current mutation state, cached-evaluation counts, and training config are
  stored so that runs are reproducible and analyzable after long overnight
  experiments.

### Supervised dataflow

1. `Actor.py` reads both:
   - Trackmania state through `Car.py`
   - real Xbox controller state through `XboxController.py`
2. While the human player is driving, it records attempts into `logs/supervised_data/...`.
3. `SupervisedTraining.py` loads all saved attempts, preprocesses them, applies mirror augmentation, and trains a torch MLP policy in target-action mode.
4. Trained models are stored under `logs/supervised_runs/.../best_model.pt`.
5. `Driver.py` can load the latest supervised model and replay it in Trackmania.
6. `GeneticTrainer.py` can also seed a GA population from a `.pt` supervised model.

Naming convention:

- In the thesis text, use "genetic algorithm (GA)" for the concrete algorithm.
- Explain once that GA is a subtype of evolutionary algorithms and that, because
  the genome encodes neural-network weights, this is also neuroevolution.
- In code, `NeuralPolicy.py` is the shared model implementation, `Individual.py`
  is the GA genome/evaluation wrapper, and `GeneticTrainer.py` is the live
  Trackmania GA trainer.
- Historical names `EvolutionPolicy.py` and `EvolutionTrainer.py` were removed
  from the active code after the rename to avoid mixing "evolutionary" and
  "genetic algorithm" terminology. Old commits still document that transition.

Directory naming convention:

- Preferred block mesh library path: `Assets/BlockMeshes/`
  - legacy fallback still used on disk for now: `Meshes/`
- Preferred exported block-layout path: `Maps/BlockLayouts/`
  - legacy fallback still used on disk for now: `Maps/ExportedBlocks/`
- Preferred original Trackmania map path: `Maps/Gbx/`
  - legacy fallback still used on disk for now: `Maps/GameFiles/`
- Preferred full exported track-mesh path: `Maps/TrackMeshes/`
  - legacy fallback still used on disk for now: `Maps/Meshes/`

`ProjectPaths.py` centralizes these locations. The code can already read from
the cleaned-up names if the directories exist, while still supporting the
legacy layout. The physical folder move should be done only when no long-running
experiments are active, because old processes may still read the legacy paths.


## Important Current Semantics

### Observation

The current observation is built in `ObservationEncoder.py`.

Current observation layout:

- `15` laser distances
- `5` path instructions
  - same feature name as before, but current semantics are signed curvature:
    `straight = 0`, `Curve1 = +/-1.0`, `Curve2 = +/-0.5`,
    `Curve3 = +/-0.333`, `Curve4 = +/-0.25`
  - sign encodes left/right turn direction; absolute value encodes turn
    sharpness, so tighter turns produce stronger steering cues
- `speed`
- `side_speed`
- `segment_heading_error`
- `next_segment_heading_error`
- `dt_ratio`
- `slip_mean`, computed as the clipped average of FL/FR/RL/RR slip coefficients
- `5` `surface_instruction_*` traction estimates aligned with the path lookahead
- `5` `height_instruction_*` values aligned with the path lookahead
  - `0.0` means same height
  - `+0.5` / `-0.5` means one logical height step up/down
  - `+1.0` / `-1.0` means two or more logical height steps up/down
- `longitudinal_accel`
- `lateral_accel`
- `yaw_rate`
- `5` overlapping laser clearance-rate sector averages

Current observation dimension:

- canonical/default training observation: `53`
- legacy/debug flat observation: `44`

Historical observation layout milestones:

- earlier 2D supervised/GA layouts used `10` path instructions and no surface/height instructions
- after adding per-wheel slip and temporal motion features, the observation reached:
  - `42` dims in the 10-lookahead version
  - then `41` dims during the transition before path lookahead was shortened
- after shortening lookahead to `5` and adding the first vertical-mode experiment:
  - flat observation was `37`
  - vertical observation was `46`
  - vertical lidar tried exact triangle-by-triangle road traversal
- after adding surface and height instructions while still keeping all four wheel slips:
  - flat observation was `47`
  - vertical/canonical observation was `56`
- the current `44` / `53` layout is intentionally not checkpoint-compatible with those older SAC/GA/supervised models
- the latest reduction from four slip inputs to one `slip_mean` was made because all four wheel slip values were usually highly correlated in practice, and the smaller input should reduce noise and model size
- old SAC `.zip` checkpoints from the 56-dim observation were removed because they cannot be safely loaded into the new 53-dim policy

Current canonical training standard:

- new supervised data collection uses `vertical_mode=True`
- new supervised models are trained against the 53-dim 3D-compatible observation
- new GA runs/checkpoints use `vertical_mode=True`
- new SAC/RL runs use `vertical_mode=True`
- surface grip instructions are always part of the canonical observation
- on all-asphalt maps the surface instructions naturally stay at `1.0`, which keeps the input compatible without needing a separate surface toggle

Current short-horizon settings:

- path lookahead horizon is `5` tiles
- lidar max range is `160` world units

Optional vertical-mode extension:

- `vertical_mode=False`
  - debug/performance fallback only
  - uses the legacy flat observation at `44`
  - uses the legacy flat wall-only lidar
- `vertical_mode=True`
  - default/canonical mode for training and data collection
  - keeps the same leading `44` features
  - appends:
    - `vertical_speed`
    - `forward_y`
    - `support_normal_y`
    - `cross_slope`
    - `5` overlapping `surface_elevation_sector_*` features
  - current vertical-mode observation dimension is `53`

Current vertical-mode sensor semantics:

- `Car.py` can run in a simplified block-grid surface-following lidar mode
- the sensor first picks the current `MapBlock` from the car's X/Z grid cell and nearest fitted road plane height
- if the upcoming `SIGHT_TILES + 1` path tiles have no height change and the current support plane is flat,
  `vertical_mode=True` automatically uses the fast legacy flat wall raycast for that frame
- each `MapBlock` fits its road surface as a simple plane `y = ax + bz + c`
  - flat blocks become horizontal planes
  - slope blocks become one sloped plate
- on sloped blocks, laser directions are generated around the fitted road-plane normal and projected through the block-grid traversal
- each laser walks from grid cell to grid cell instead of triangle to triangle
  - within a cell, the ray follows the current block's fitted road plane at `surface_ray_lift`
  - wall checks are performed only against that block's `sensor_walls_mesh`
  - `sensor_walls_mesh` keeps the original wall mesh intact
  - slope blocks add simple vertical side-curtain polygons along road boundary edges
  - the curtains leave slope entry/exit open and only make the side catch surface taller
  - transitions to another block are allowed only when those blocks are connected in the logical path
  - if the ray exits the known block grid without a wall hit, the result is treated as `grid_gap`
  - if the ray reaches `LASER_MAX_DISTANCE`, the result is `grid_open`
  - if it hits padded block walls, the result is `grid_wall`
  - if it tries to enter a non-connected neighboring block, the result is `grid_blocked_transition`
- when `vertical_mode=False`, the old flat `walls_mesh`-only raycast remains active
- historical note: the first vertical lidar prototype used fixed-step marching
  over the road surface (`surface_step_size`). It was slow and unreliable
  because correctness depended on the chosen step size. The active code no
  longer exposes or logs `surface_step_size`; the replacement is block-grid
  surface-following traversal with analytic wall checks per grid cell.
- `surface_probe_height` and `surface_ray_lift` are still active parameters:
  the former controls how high support probes start above the fitted road
  surface and the latter keeps surface-following laser paths slightly above the
  fitted road plane.

Supported height-changing road blocks:

- `RoadTechSlopeBase`
  - 1x1 straight slope, height delta `+1` logical level when driven low-to-high
- `RoadTechSlopeBase2`
  - 1x1 steeper straight slope, height delta `+2`
- `RoadTechSlopeBase2x1`
  - 1x2 gentler straight slope, height delta `+1`
  - this uses a non-square footprint and has custom orientation handling in `MapBlock`
- `RoadTechSlopeBaseCurve2Left` / `RoadTechSlopeBaseCurve2Right`
  - 2x2 curve2 slope, height delta `+1`
  - left/right variants preserve the usual curve instruction sign while also adding height instruction `+0.5`
- `RoadTechSlope2BaseCurve2Left` / `RoadTechSlope2BaseCurve2Right`
  - 2x2 curve2 slope, height delta `+2`
  - height instruction reaches `+1.0`
- entering any of these blocks from the opposite side uses `MapBlock.swap_in_out()`,
  so the same metadata also supports downhill `-0.5` / `-1.0` height instructions
- vertical-mode lidar adds side-curtain helper polygons for all slope-like blocks,
  not only straight slopes; entry and exit edges remain open for connected block traversal

Current phased observation roadmap:

- current 2D upgrade
  - add one compact `slip_mean` feature derived from all four wheel slip coefficients
  - add `5` compact future surface traction instructions:
    - `RoadTech` / asphalt: `1.00`
    - `PlatformGrass`: `0.70`
    - `RoadDirt` / `PlatformDirt`: `0.50`
    - `PlatformPlastic`: `0.75`
    - `PlatformIce`: `0.05`
    - `PlatformSnow`: `0.15` initial estimate
  - add compact temporal summary:
    - `longitudinal_accel`
    - `lateral_accel`
    - `yaw_rate`
    - `5` overlapping clearance-rate sectors derived from the lidar fan
  - add compact future height instructions:
    - `height_instruction_0..4`
    - `+0.5/-0.5` for one logical level step
    - `+1.0/-1.0` for two or more logical level steps
- next 2D/surface-aware upgrade
  - add gear
  - add rpm
- later 3D upgrade
  - current first 3D step
    - add toggleable `vertical_mode`
    - add block-grid surface-following lidar distances
    - add compact vertical block:
      - `vertical_speed`
      - `forward_y`
      - `support_normal_y`
      - `cross_slope`
      - `surface_elevation_sector_0..4`
  - later 3D expansion
    - add airborne/contact metrics
    - add richer orientation metrics
    - add more advanced surface/material state when needed

Important history:

- `previous_action` used to be part of the observation.
- It was removed from the supervised-target pipeline because it created strong label leakage:
  the network learned to repeat the previous action instead of reacting to state, especially failing at the very first frame after race start.
- `next_point_direction` used to be a mostly unsigned/alignment-like cue.
- It was replaced by signed current and next segment heading errors because the agent needed explicit left/right steering information.
- `path_instructions` used to be stored at block granularity while the car advanced by tile index.
- This created offsets on multi-tile curves; `Map.py` now expands block instructions into tile-aligned path entries.
- `path_instructions` also used to encode signed curve size:
  `Curve1 = +/-0.25`, `Curve2 = +/-0.5`, `Curve3 = +/-0.75`,
  `Curve4 = +/-1.0` after observation normalization.
- That meant larger absolute values actually described wider/easier turns, not
  sharper turns. The current version keeps the same observation slot name but
  changes the value to signed curvature, approximately `sign / curve_size`.
  This is more physically meaningful for steering because `Curve1` is the
  sharpest turn and therefore now has the largest absolute instruction.
- Per-wheel slip was originally added as four separate observation values: `slip_fl`, `slip_fr`, `slip_rl`, `slip_rr`.
- It was later compressed into `slip_mean` to reduce observation dimension after live inspection showed the four values are usually redundant for the current agent.

### Action modes

Two action semantics exist in the project:

- `delta`
  - policy outputs a delta action
  - environment integrates it into the previous applied action
  - `dt_ratio` is used to scale the delta
- `target`
  - policy outputs the target action directly
  - this is the active direction for supervised learning

### Stable-Baselines3 SAC RL path

The active SAC experiment lives in `RL_test/train_sac_trackmania.py`.

Current prepared default:

- map: `AI Training #5`
- max wall-clock runtime: `8` hours
- episode cap: `10000`
- timestep cap: `50_000_000`
- `vertical_mode=True`
- `action_layout="gas_brake_steer"`
- reward mode: `delta_lexicographic`
- initial model: `None`, because this run intentionally starts from scratch with the latest observation/reward setup
- policy architecture: SB3 `MlpPolicy` with hidden layers `[128, 128]`
- hidden activation: `relu`
- learning rate: `3e-4` by default, exposed as `--learning-rate` for quick sweeps without editing the script
- train frequency: one update batch after each episode via `SAC_TRAIN_FREQ = (1, "episode")`
- gradient steps after each episode: `64`

Current `delta_lexicographic` reward:

- shared implementation lives in `Individual.compute_delta_lexicographic_score_for`
- emitted SAC reward is `score_now - score_previous`
- progress is normalized to `[0, 1]` and is the primary term
- one path-tile progress unit is always larger than the whole time tie-break range
- the time tie-break is always larger than the whole distance tie-break range
- finish adds `+1` through the terminal term
- crash/timeout do not receive a large terminal penalty in the SAC delta mode; they end the episode and remove future reward opportunities
- this was chosen because local SAC tests found terminal failure drops too harsh early in training

This intentionally avoids raw `delta_progress / delta_time` because that ratio can over-reward a very fast first block followed by an immediate crash.

SAC/RL history notes:

- an older hand-written run-based RL trainer existed separately from GA, but the first long run learned very slowly and stalled around early-map progress
- Stable-Baselines3 SAC was introduced as a testbed under `RL_test/` to check whether the task can be learned with a stronger off-policy algorithm
- early SAC reward modes included:
  - pure terminal fitness
  - progress-delta shaping
  - hybrid progress plus terminal fitness
- pure terminal fitness was too sparse for efficient learning
- progress-delta learned early movement faster but did not pressure enough for shortest-time driving
- a raw `delta_progress / delta_time` idea was considered and rejected because it can reward sprinting into the first wall
- the old pace-normalized `fitness_delta` was replaced by bounded `delta_lexicographic` after local sandbox tests showed it is less prone to rewarding fast early failure
- a temporary `gas_steer` default was used to diagnose early movement because local SAC tests showed independent random gas/brake exploration can cancel movement near the start
- the live default is now back to `gas_brake_steer`, because brake is still useful and we want the policy to learn the real control layout from scratch
- a short local learning-rate sweep with `gas_brake_steer` compared `1e-4`, `3e-4`, and `1e-3`; it was too short to prove an optimum, but `1e-3` reduced entropy much faster, so `3e-4` remains the safer overnight default
- older SAC models/checkpoints are expected to become incompatible whenever the observation dimension changes
- after switching four wheel-slip inputs to one `slip_mean`, SAC training is intentionally restarted from scratch with `INITIAL_MODEL_PATH = None`

### Current target-action semantics

In target mode:

- `gas` is a sigmoid output in `[0, 1]`
- `brake` is a sigmoid output in `[0, 1]`
- `steer` is a tanh output in `[-1, 1]`

At environment/controller application time:

- `gas` is thresholded at `0.5`
- `brake` is thresholded at `0.5`
- both may be active simultaneously
- `steer` remains analog in `[-1, 1]`

The same binary pedal semantics are used when collecting supervised data in `Actor.py`.


## Core Files To Index First

If another Codex instance needs to understand the project efficiently, index files in this order:

1. `CODEX_PROJECT_CONTEXT.md`
2. `ObservationEncoder.py`
3. `Car.py`
4. `Map.py`
5. `Enviroment.py`
6. `NeuralPolicy.py`
7. `Individual.py`
8. `GeneticTrainer.py`
9. `Driver.py`
10. `Actor.py`
11. `SupervisedTraining.py`
12. `XboxController.py`
13. `Plugins/get_data_driver/main.as`
14. `README.md`

Secondary files:

- `GraphView.py`
- `Vizualizer.py`
- `installation.txt`


## File Responsibilities

### `Plugins/get_data_driver/main.as`

OpenPlanet plugin.

Responsibilities:

- opens TCP server on `127.0.0.1:9002`
- streams 37 floats every Trackmania frame
- includes:
  - speed
  - side speed
  - distance
  - position
  - steer/gas/brake inputs
  - finish flag
  - gear / rpm
  - direction vector
  - game time
  - FL/FR/RL/RR slip coefficients
  - FL/FR/RL/RR ground material diagnostics
  - FL/FR/RL/RR icing and dirt diagnostics
  - wetness

Historical note:

- older versions streamed about 20 floats and did not include the later material/icing/dirt/wetness diagnostics
- the extra diagnostics were added while investigating surface-aware driving and whether Trackmania exposes traction-like information directly

This is the root of the live runtime data stream.

### `Car.py`

Bridge between Trackmania packets and Python-side state.

Responsibilities:

- connect to the OpenPlanet TCP stream
- keep latest packet only
- derive map/path progress
- derive future path instructions
- derive future surface traction instructions
- derive future height-change instructions
- compute signed heading errors for the current and next future path segment
- compute lidar-style laser distances against map walls
- in `vertical_mode`, compute block-grid surface-following laser distances over fitted block road planes
- expose support-normal / slope debug data for the observation encoder and vizualizer

Important implementation detail:

- the reader thread stores only the latest decoded packet, so the system does not intentionally process an old packet backlog frame by frame

### `Map.py`

Map geometry and logical path representation.

Responsibilities:

- parse exported block layout files from `Maps/BlockLayouts/*.txt`
  with fallback to legacy `Maps/ExportedBlocks/*.txt`
- instantiate mesh blocks from `Assets/BlockMeshes/*.obj`
  with fallback to legacy `Meshes/*.obj`
- construct the logical path from start to finish
- expand block-level turn semantics into tile-aligned `path_instructions`
- expand block-level surface semantics into tile-aligned `path_surface_instructions`
- expand tile-level height deltas into tile-aligned `path_height_instructions`
- provide road mesh and wall mesh for geometry queries
- provide fitted road planes, X/Z block-grid lookup, logical path block transitions, and per-block sensor walls/side curtains for 3D laser walking

### `ObservationEncoder.py`

Canonical observation builder.

Responsibilities:

- standardize distances and motion values
- standardize wheel slip into one compact `slip_mean` feature
- standardize future surface traction instructions
- standardize future height-change instructions
- derive compact temporal motion features from previous vs current frame
- optionally append compact vertical terrain features in `vertical_mode`
- compute `dt_ratio = dt / dt_ref`
- expose observation bounds
- provide mirror helpers for observations and actions

This file should be treated as the single source of truth for observation format.

### `Enviroment.py`

Trackmania environment wrapper.

Responsibilities:

- hold the `Map`, `Car`, and `vgamepad` controller
- reset the game state
- build observations through `ObservationEncoder`
- apply actions in delta or target mode
- enforce termination/truncation conditions

Important guards currently implemented:

- `max_time`
- `wrong-way`
- `start_idle`
- `stuck_after_progress`
- `max_touches`
- `wall_ride`

Important note:

- old RL reward logic was removed
- current reward returned by `step()` is neutral (`0.0`)
- GA optimization does not use per-step reward

### `NeuralPolicy.py`

Torch policy network.

Responsibilities:

- define the MLP policy
- support one or more hidden layers with per-layer activations
- support `delta` and `target` action modes
- expose flattened genome view for GA
- save/load `.pt` policy files

Important:

- this is now the canonical model implementation
- older numpy policy versions are preserved in `Backup/numpy_logic_20260317_133951`

### `Individual.py`

GA individual wrapper around the policy.

Responsibilities:

- hold evaluation metrics
- expose ranking key
- provide mutation and crossover
- provide scalar fitness only as a log-friendly numeric proxy

Current ranking logic:

- `finished`
  - `1` only when the agent reaches finish
  - `0` otherwise
- `crashes`
  - counted wall touches/crashes during the rollout
  - can be greater than `1` when `max_touches > 1`
- timeout is derived as `finished == 0 and crashes == 0`

Current ranking policy in `Individual.ranking_key()`:

- ranking is configured by an explicit tuple expression such as
  `(dense_progress, finished, -time, -crashes, -distance)` or
  `(finished, progress, -time, -crashes)`
- `progress` inside the ranking key can refer to either discrete or dense
  progress according to `Individual.RANKING_PROGRESS_SOURCE`
- scalar `fitness` remains a log-friendly numeric proxy; GA selection can
  compare the tuple directly through `Individual.__lt__`

Reason for this design:

- explicit tuples avoid arbitrary weighted sums and make the priority order
  visible in experiments
- the trade-off is that tuple order matters: putting `-crashes` before `-time`
  can favor safe but slow timeout policies, while putting `-time` first can
  favor faster but riskier policies

### `GeneticTrainer.py`

Main GA trainer.

Responsibilities:

- population initialization
- optional seeding from supervised `.pt` model
- optional resume from population `.npz`
- individual evaluation in Trackmania
- logging/checkpointing
- mutation annealing

Current baseline default in `__main__`:

- map: `AI Training #3`
- hidden dim: `32`
- population: `64`
- generations: `100`
- action mode: `target`
- no supervised pretraining
- no mirroring
- `max_touches = 1`
- `env_max_time = 60`
- mutation starts exploratory and anneals down

### `Driver.py`

Evaluation and replay tool.

Responsibilities:

- drive a single `.pt` supervised model
- or replay individuals from a population `.npz`
- auto-pick latest supervised model if configured

Useful for sanity checks before starting long GA runs.

### `Actor.py`

Supervised data collection tool.

Responsibilities:

- read Trackmania state
- read physical Xbox controller state
- record attempts into `.npz`

Attempt workflow:

- recording starts when game time becomes `> 0`
- pressing `B` during a run discards the attempt
- after finish:
  - `A` saves the attempt
  - `B` discards it

### `SupervisedTraining.py`

Offline torch training script for imitation learning.

Responsibilities:

- load all attempts from `logs/supervised_data`
- preprocess frames
- optionally filter boring frames
- mirror-augment the dataset
- train a target-action MLP
- save `best_model.pt`

Current simplification trend:

- one hidden layer with `16` neurons
- no validation split
- all frames pooled together and shuffled

### `XboxController.py`

Dedicated Xbox controller reader using `inputs`.

Responsibilities:

- read gas / brake / steer
- read `A` and `B`
- apply steer deadzone

### `Vizualizer.py`

Legacy/auxiliary visualization script for scene inspection and debugging.

### `GraphView.py`

Plotting and post-run analysis separated from the GA trainer.


## Historical Evolution Since Project Start

The git history currently spans 71 commits from `2024-03-08` to `2026-04-29`.
The sections below intentionally include old and superseded designs, because they explain why the current code looks the way it does.

### March 2024: first prototype, map extraction, and mesh visualization

Relevant commits:

- `1e6e233` - initial README commit
- `a584a14` - first full project import
- `0bc9630` - map visualization

Main additions:

- the repository started around a C# map extractor using GBX/TmEssentials-related tooling
- early `Map.py` parsed exported Trackmania block files and loaded `.obj` meshes
- early meshes and map files were committed directly, including RoadTech blocks and small test maps
- `Player.py` / early visualizer-style code was used to inspect the generated map scene
- this phase established the core idea that Python would reason over exported Trackmania geometry rather than only live game pixels

Lessons:

- map extraction and coordinate conventions became a long-running source of complexity
- several generated binaries and extracted assets were committed early, then later cleaned or ignored
- this is why the repo still contains both Python runtime code and C#/mesh extraction tooling

### March 2024: first lidar/raycast and live telemetry loop

Relevant commits:

- `5279490` - raycast from car added
- `23d972a` - raycasting from car point to all directions

Main additions:

- `Car.py` was introduced as the bridge between live Trackmania telemetry and Python logic
- OpenPlanet plugin data started feeding Python with live car state
- lidar-style raycasts were added from the car into the map walls
- raycasts evolved from one forward ray toward a fan of rays around the car

Lessons:

- the agent's perception was intentionally built as geometric lidar over exported meshes, not image-based vision
- this early choice still shapes the entire project: most later work is about making the geometric observation more truthful and compact

### March 2024: first environment wrapper and training attempts

Relevant commits:

- `155b616` - added environment and FPS-limited rendering
- `044ec3d` - car responsibility fix, supervised data gathering, map path
- `8970a9d` - global path points, speed optimization, real-time delay fix
- `989ecdd` - first training attempts

Main additions:

- `Enviroment.py` appeared as a Gym-like wrapper around Trackmania, map, car, and controller actions
- `Training.py` represented the early RL/training path before the later GA/supervised/SAC split
- `Actor.py` appeared for supervised data gathering from human/controller driving
- `Map.py` started building global path points and progress-related information from map geometry
- real-time data delay was noticed early and optimized by keeping only fresh telemetry rather than processing an old backlog
- old `logs/model_*_steps.zip` artifacts show that Stable-Baselines-style `.zip` models existed in the first era too

Lessons:

- asynchronous real-time Trackmania behavior was a problem from the beginning
- stale telemetry and reset timing can invalidate training, so later reset handshakes and latest-packet-only design are not incidental details
- the current `RL_test/` SAC path is a new RL testbed, but it is not the first time this project tried RL `.zip` models

### June 2024: early elevation support

Relevant commit:

- `7d9835c` - map elevation level added

Main additions:

- map blocks started carrying an elevation/height level
- `RoadTechSlopeBase.obj` and a `loop_test` map were added
- this was the first clear signal that flat 2D assumptions would eventually be insufficient

Lessons:

- height-aware driving was deferred for a long time because flat maps were easier to debug
- the later vertical lidar work in 2026 builds on this older realization that maps are grid/block/elevation structured

### May 2025: thesis/documentation phase

Relevant commits:

- `627c97c` - thesis skeleton
- `fde409a` through `ddb5d0c` - README updates

Main additions:

- diploma/thesis-related documentation entered the repository
- README and project framing were updated several times

Lessons:

- this period was more about project explanation and thesis structure than runtime architecture
- it helps explain why the repo sometimes mixes implementation, experiments, generated artifacts, and documentation

### November-December 2025: neuroevolution revival

Relevant commits:

- `adabe65` - evolution algorithm added
- `d282cc2` - training history and graph plot
- `4e6976f` - training results and README analysis

Main additions:

- `EvolutionPolicy.py`, `EvolutionTrainer.py`, `Individual.py`, and `GADriver.py` introduced the first major GA/neuroevolution workflow
- old `ppo_racing_game.zip` was removed when the GA direction became more important
- GA history CSVs, population checkpoints, and graph images were added
- `Individual` introduced lexicographic-style fitness concepts around terminal status, progress, time, and distance

Lessons:

- GA became the main direction because it fit the real-time/asynchronous environment better than classic step-heavy RL at that time
- old distance minimization could produce a bad local optimum where an unfinished agent preferred crashing close to the start
- later ranking changes keep distance meaningful mainly for finished runs

### December 2025 baseline

Relevant commit:

- `d282cc2` - training history and graph plot

Project state around this period:

- simpler GA pipeline
- no torch policy
- no supervised learning pipeline
- no advanced guard logic
- no resume/checkpoint system comparable to current state

This period is important because the user reported that a simpler earlier trainer could sometimes train a finisher more reliably than the later experimental versions.

### February 2026: GA infrastructure expansion

Relevant commits:

- `606e559`
- `c459f3a`

Main additions:

- separated graphing from trainer via `GraphView`
- added per-run logging
- added resumable population checkpoints
- added persistent `global_best`
- added more robust experiment management

### February 2026: runtime/control experiments

Relevant commit:

- `b3102f8`

Main additions:

- normalized observation
- added `dt_ratio`
- introduced dt-aware control semantics
- added Xbox controller debug reader

### March 2026: torch migration and supervised pipeline

Relevant commit:

- `9ffd709`

Main additions:

- replaced numpy policy with torch-based policy
- introduced shared policy representation for supervised, GA, and driver
- added supervised attempt collection
- added supervised training script
- added seeding GA population from a `.pt` supervised model
- backed up the old numpy logic under `Backup/numpy_logic_20260317_133951`

### March 2026: supervised target pipeline refinement

Relevant commit:

- `f062277`

Main additions:

- refined supervised target-action workflow
- aligned Actor, Environment, Driver, and GA around target semantics
- several iterations on pedal thresholding and model structure

### March 2026: current baseline cleanup

Relevant commits:

- `61e28f2`
- `5742638`
- `04bf526`

Main changes:

- baseline training defaults for cleaner GA runs
- old reward function removed from environment
- focus shifted from feature accumulation to establishing a stable baseline again

### March-April 2026: observation expansion and path-index fixes

Relevant commits:

- `0f29e88` - expanded observations with per-wheel slip and temporal motion features
- `900c8f9` - fixed tile-aligned path instructions and mirrored/normal GA evaluation
- `111507f` - added surface-aware observations, reset handling, and broader dataflow changes

Main additions and lessons:

- per-wheel slip was added first as four separate NN inputs
- `longitudinal_accel`, `lateral_accel`, `yaw_rate`, and clearance-rate sectors were added to give the agent motion derivatives
- path lookahead was reduced from `10` tiles to `5` tiles to keep the input compact and focus on near-horizon decisions
- lidar range was reduced from `320` to `160` world units to match the shorter lookahead horizon
- signed segment heading errors replaced the older direction cue so the agent can distinguish left vs right
- block-level path instructions caused offsets on multi-tile curves; tile-aligned expansion fixed this
- surface instructions were added as compact traction coefficients instead of one-hot surface categories

### April 2026: vertical lidar experiments

Relevant commits:

- `7484f4a` - added surface-aware observations and optimized vertical lidar
- `681ccea` - improved height-aware SAC training pipeline

Main additions and lessons:

- the first vertical-mode lidar used exact triangle traversal over the road mesh
- that approach was theoretically elegant but too slow and unreliable in live testing
- it was replaced by a simpler block-grid surface-following lidar:
  - fit each road block as a plane
  - walk lasers across logical grid blocks
  - check walls only inside the current logical block
  - allow transitions only across connected path blocks
- flat sections automatically fall back to fast 2D wall raycasting when no upcoming height change is relevant
- slope blocks use helper side-curtain polygons to make walls catch surface-following rays without closing valid road entry/exit edges
- supported slope metadata now includes straight slopes, steeper slopes, 2x1 slopes, and curve2 slope variants

### April 2026: SAC RL reactivation

Main additions and lessons:

- the project reintroduced RL through Stable-Baselines3 SAC under `RL_test/`
- SAC uses continuous `gas`, `brake`, and `steer` actions mapped through `ContinuousTargetRacingEnv`
- training uses the same `Enviroment.py`, `Car.py`, and `ObservationEncoder.py` gateway as GA/supervised code
- early SAC runs showed progress-delta feedback learns faster than terminal-only reward
- reward shaping is still experimental and should be judged by actual live training curves, not only offline intuition
- after reducing slip inputs from four values to `slip_mean`, all old SAC checkpoints were deleted and current SAC defaults restart from scratch

### Training-map and artifact-management notes from the full history

Relevant commits:

- `8ea1214`, `6e7cb29`, `1135e2a` - training-map additions around late March 2026
- `39d5a88` - SAC training artifacts committed on 2026-04-29

Main lessons:

- the repo history contains several committed model artifacts, videos, thesis PDFs, graph images, and generated logs
- some old model artifacts were useful historically but are not safe as current defaults when observation dimensions or reward semantics change
- generated training outputs should usually stay local unless they are deliberately preserved as evidence for an experiment
- `.gitignore` has gradually become more important as the project moved from prototype to repeatable experiments
- when reviewing history, distinguish code architecture commits from experiment-output commits; both are informative, but only code/config commits should usually drive current behavior


## Important Experiments Already Tried

This section is critical. Another Codex instance should not rediscover these from scratch.

### 1. Mirror augmentation

Tried in both the mini project and Trackmania GA.

Goal:

- reduce one-sided overfitting
- teach left/right symmetry

Status:

- mechanism exists
- currently disabled in the baseline trainer because the user wants a simpler baseline first

### 2. Multi-touch instead of instant crash

Goal:

- allow a few small contacts before terminating

Implementation:

- `max_touches`
- touch debounce
- wall-ride guard

Status:

- still implemented
- baseline currently uses `max_touches = 1`

### 3. Target vs delta action

This has been one of the biggest experimental branches.

Observations:

- delta mode historically felt more stable in some GA runs
- target mode is more natural for supervised imitation learning
- target mode initially failed badly due to action semantics inconsistencies and overly strong shortcuts

Current status:

- supervised path is target-mode oriented
- baseline GA currently also defaults to target mode
- this is still an area of uncertainty and comparison

### 4. Previous action in the observation

Originally added to provide temporal context.

Result:

- in supervised target training it became a harmful shortcut
- the network learned to copy previous action instead of initiating correct start behavior
- especially bad at race start: no gas/brake on first frame

Decision:

- removed from the observation
- observation dimension reduced to `29`

### 5. dt_ratio input

Added because Trackmania/OpenPlanet produces variable frame timing.

Current belief:

- keeping `dt_ratio` in the observation is reasonable
- in delta mode it should scale the delta action
- in target mode it is still useful context but not used to scale outputs

### 6. Supervised validation split

A validation split existed previously.

Issues encountered:

- validation could become misleading
- real usefulness is determined by Driver replay in Trackmania, not by abstract validation loss
- map/run-based splitting created confusion in interpreting generalization

Current state:

- validation split was removed
- training uses all pooled frames

### 7. Large supervised model vs small supervised model

A larger model was tried first.

Current simplification:

- reduced to a much smaller MLP: one hidden layer, `16` neurons

Reason:

- simplify the hypothesis space
- test whether the pipeline works before increasing model capacity

### 8. Mini 2D pretraining project

Historically, a separate lightweight mini project existed and was used heavily for:

- cheap pretraining
- mirror experiments
- exporting TM-compatible checkpoints

Current repo state:

- the mini-project source is not present in the current top-level tracked files
- references to mini-project population checkpoints still exist in loaders and historical workflow discussions

Important:

- treat mini-project pretraining as a historical branch of experimentation, not as the current core runtime in this checkout


## Current Known Problems / Open Questions

These are active research/debug topics, not solved truths.

- The user reports difficulty training a reliable finisher despite adding many improvements.
- The simpler historical trainer sometimes seemed to work better.
- It is unclear whether target mode is truly better for GA than delta mode in Trackmania runtime.
- Supervised policies have sometimes:
  - failed to start properly
  - turned too weakly
  - behaved conservatively
- The exact amount of steer needed in Trackmania relative to the learned policy remains a practical issue.
- The influence of observation design vs policy architecture is still unresolved.
- The project has accumulated many safety/guard mechanisms; some may help, some may distort selection pressure.
- The steering cue in `Car.py` now uses signed current/next segment heading errors instead of a pure dot-product alignment scalar.
  This should preserve left/right information, but it is still worth verifying on live maps that the sign convention matches intuitive steering direction.
- A previous bug came from `path_instructions` being stored at block granularity while `path_tile_index` advanced at tile granularity.
  `Map.py` now expands instructions to tile-aligned entries so the lookahead slice in `Car.py` stays synchronized through multi-tile corners.


## Current Recommended Debugging Order

If continuing experimentation, do not start by adding more complexity.

Recommended order:

1. Verify raw runtime data is sane:
   - plugin packets
   - `Car.py` derived values
   - observation ranges
2. Verify `Driver.py` behavior of the latest supervised model.
3. Verify action semantics end to end:
   - policy output
   - environment thresholding/clipping
   - vgamepad behavior in Trackmania
4. Only after runtime sanity is confirmed, launch GA runs.
5. Compare baseline `target` vs `delta` mode cleanly rather than mixing many new features at once.


## Current Practical Entry Points

### GA baseline

Run:

```powershell
python GeneticTrainer.py
```

This currently uses the baseline config from `GeneticTrainer.py`.

### Driver replay

Run:

```powershell
python Driver.py
```

By default, this auto-loads the latest supervised model.

### Supervised data collection

Run:

```powershell
python Actor.py
```

### Supervised training

Run:

```powershell
python SupervisedTraining.py
```


## Current Logs/Artifacts Layout

- `logs/ga_runs/...`
  - GA runs
  - summaries
  - population checkpoints
- `logs/supervised_data/...`
  - recorded human driving attempts
- `logs/supervised_runs/...`
  - trained supervised torch models
- `Backup/numpy_logic_20260317_133951/...`
  - backup of pre-torch numpy logic


## Environment / Dependencies

See `installation.txt`.

Important runtime dependencies:

- `torch`
- `numpy`
- `gymnasium`
- `trimesh`
- `vgamepad`
- `inputs`
- OpenPlanet plugin in Trackmania
- ViGEmBus driver for virtual gamepad


## Guidance For Another Codex Instance

When opening this project on another machine:

1. Read this file first.
2. Index the files listed in the "Core Files To Index First" section.
3. Assume the current priority is baseline reliability, not novelty.
4. Do not remove existing experimental features unless explicitly asked.
5. Treat supervised and GA as two connected but not yet fully stabilized pipelines.
6. Prefer simple A/B experiments over stacking multiple new ideas at once.


## Suggested Handoff Prompt

If another Codex/GPT-5.4 instance needs a starting prompt, use something like:

> Read `CODEX_PROJECT_CONTEXT.md` first, then index `ObservationEncoder.py`, `Car.py`, `Map.py`, `Enviroment.py`, `NeuralPolicy.py`, `Individual.py`, `GeneticTrainer.py`, `Driver.py`, `Actor.py`, and `SupervisedTraining.py`. This repository is a Trackmania autonomous driving project with a live GA/neuroevolution pipeline and a newer supervised-learning pipeline. The current priority is to restore a reliable baseline training workflow, not to add new complex features. Preserve existing experimental mechanisms, but reason from the current baseline defaults and from the historical experiments summarized in `CODEX_PROJECT_CONTEXT.md`.
