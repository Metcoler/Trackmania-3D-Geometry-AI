# Trackmania Autonomous Driving Agent

This repository contains the code, experiments, figures, and thesis materials for a diploma project about an autonomous driving agent for **Trackmania**.  
Instead of treating the game only as a stream of pixels, the project reconstructs a geometric copy of the track and gives the agent compact observations derived from that geometry.  
The final policy is a small neural network trained mainly through lexicographic neuroevolution, with supervised learning, imitation learning, reinforcement learning, and Pareto experiments used as supporting comparisons.

The full thesis methodology, citations, and detailed discussion are in `Diplomová práca/Latex/main.pdf`.

## Demo Videos

Videos are intended to be attached as GitHub release assets. If a link opens the release page, choose the matching video file from the release assets.

| Demo | Link | What it shows |
|---|---|---|
| `single_surface_flat` | [release video](https://github.com/Metcoler/Trackmania-BC/releases) | Final agent on a flat single-surface map. |
| `single_surface_height` | [release video](https://github.com/Metcoler/Trackmania-BC/releases) | Agent using height-aware observations. |
| `multi_surface_flat` | [release video](https://github.com/Metcoler/Trackmania-BC/releases) | Agent driving on a map with multiple surface types. |
| `small_map` transfer | [trajectory figure](Diplomov%C3%A1%20pr%C3%A1ca/Latex/images/evaluation/evaluation_small_map_trajectory.png) | Transfer evaluation on a map not used for training. |

## How The System Works

The agent runs in a closed loop. Trackmania provides the real-time environment, a game-side script streams car state, Python projects the state into a reconstructed virtual map, the observation encoder builds a compact input vector, a neural policy chooses an action, and the action is applied through a virtual gamepad.

![Game-agent loop](Diplomov%C3%A1%20pr%C3%A1ca/Latex/images/solution_proposal/solution_game_agent_loop.png)

In short:

1. Trackmania is the real environment.
2. Game telemetry gives car position, orientation, velocity, time, and state flags.
3. Python keeps a geometric copy of the map.
4. Raycasting and track-progress logic produce the observation.
5. `NeuralPolicy.py` maps observation to gas, brake, and steering.
6. `XboxController.py` applies the action as a virtual controller input.

## Environment And Geometry

Trackmania maps are built from blocks. The project exports block layouts and mesh data, then rebuilds the map as a geometric world in Python. This lets the agent reason about track shape, walls, surfaces, height changes, and progress without processing the full screen image.

<p align="center">
  <img src="Diplomov%C3%A1%20pr%C3%A1ca/Latex/images/solution_proposal/solution_blocks_editor_view.png" width="49%" alt="Trackmania blocks">
  <img src="Diplomov%C3%A1%20pr%C3%A1ca/Latex/images/solution_proposal/solution_game_map_overview.png" width="49%" alt="Trackmania map">
</p>

Individual blocks are represented as triangle meshes. By placing the exported meshes according to the block layout, the runtime builds a virtual copy of the track.

<p align="center">
  <img src="Diplomov%C3%A1%20pr%C3%A1ca/Latex/images/solution_proposal/mesh_straight.png" width="32%" alt="Straight mesh">
  <img src="Diplomov%C3%A1%20pr%C3%A1ca/Latex/images/solution_proposal/mesh_curve.png" width="32%" alt="Curve mesh">
  <img src="Diplomov%C3%A1%20pr%C3%A1ca/Latex/images/solution_proposal/map_mesh.png" width="32%" alt="Map mesh">
</p>

## What The Agent Observes

The agent does not receive the whole map. It receives a compact vector of local and state-based measurements:

- progress along an approximated centerline,
- distances to track edges from ray-like sensors,
- forward and lateral velocity components,
- heading relative to the track,
- upcoming curve instructions,
- optional surface and traction information,
- optional height-profile signals.

![Observation from geometry](Diplomov%C3%A1%20pr%C3%A1ca/Latex/images/solution_proposal/solution_observation_from_geometry.png)

This representation keeps the input small and interpretable. It also separates the perception problem from the policy-learning problem: the neural network learns how to drive from geometric measurements, not from raw pixels.

## Training Story

The thesis follows a step-by-step experimental path:

- **Supervised learning:** human driving data was used to check whether a small neural network can approximate the mapping from observation to action.
- **Imitation learning:** mixed human-agent control was tested to collect recovery states that pure supervised learning misses.
- **Lexicographic genetic algorithm:** the main training method evolved neural-network weights directly from driving outcomes using the tuple `(finished, progress, -time, -crashes)`.
- **Activation and hyperparameter studies:** closed-loop experiments supported `ReLU,tanh` networks and a practical GA baseline.
- **RL and Pareto branches:** PPO/SAC/TD3 and NSGA-II style experiments were tested as comparison paths, but the lexicographic GA remained the most practical direction in these experiments.
- **Final training:** the selected setup was tested on flat, height-aware, and multi-surface maps.

![Final GA progress](Diplomov%C3%A1%20pr%C3%A1ca/Latex/images/training_policy/final_ga_training_progress.png)

![Final GA trajectories](Diplomov%C3%A1%20pr%C3%A1ca/Latex/images/training_policy/final_ga_training_trajectories.png)

## Results Snapshot

These numbers summarize selected thesis experiments. They are practical experimental results from the tested runs, not a broad statistical benchmark across many independent seeds.

| Scenario | Best time | Finishers | Note |
|---|---:|---:|---|
| `single_surface_flat` | `25.92 s` | `654` | Flat map with one surface type. |
| `single_surface_height` | `32.55 s` | `351` | Map with height changes and height-aware observation. |
| `multi_surface_flat` | `35.10 s` | `125` | Flat map with multiple surface types. |
| `small_map` transfer | `19.68 s` | - | Agent transferred from the flat-map generation. |

On `small_map`, the transferred diploma agent improved over the original bachelor implementation:

| Driver | Time |
|---|---:|
| Human reference | `17.89 s` |
| Diploma agent | `19.68 s` |
| Bachelor agent | `23.064 s` |

<p align="center">
  <img src="Diplomov%C3%A1%20pr%C3%A1ca/Latex/images/evaluation/evaluation_small_map_comparison.png" width="45%" alt="Small map comparison">
  <img src="Diplomov%C3%A1%20pr%C3%A1ca/Latex/images/evaluation/evaluation_small_map_trajectory.png" width="45%" alt="Small map trajectory">
</p>

The final evaluation also compares the trained agent against a wider set of player times on the `single_surface_flat` map. In the tested run, the agent reached `25.92 s`, which is close to the average stored player time for that map.

![Human vs agent times](Diplomov%C3%A1%20pr%C3%A1ca/Latex/images/evaluation/evaluation_human_vs_agent_single_surface_flat.png)

## Repository Guide

### Runtime and agent

- `Driver.py` - replay and live driving entry point for a trained policy.
- `Enviroment.py` - Trackmania environment wrapper used by the agent.
- `Car.py` - car state, movement, and runtime measurements.
- `ObservationEncoder.py` - converts car state and geometry into the neural-network input vector.
- `NeuralPolicy.py` - neural network policy used by supervised, imitation, and GA paths.
- `XboxController.py` - virtual controller output used to apply actions in the game.

### Training

- `GeneticTrainer.py` - main neuroevolution trainer.
- `Individual.py` - one population member, including policy weights and metrics.
- `RankingKey.py` - lexicographic ranking logic.
- `SupervisedTraining.py` - supervised training from recorded human data.
- `ImitationTrainer.py` - mixed human-agent imitation learning.

### Maps and geometry

- `Maps/` - game maps, exported block layouts, player times, and map meshes.
- `Meshes/` - block-level mesh assets used for reconstruction.
- `Map.py` - map loading, block graph logic, geometry, and progress handling.
- `Map Extractor C#/` - extractor tooling used to obtain map/block data.

### Game integration

- `Plugins/` - Trackmania/OpenPlanet-side scripts for game-state communication.
- `ProjectPaths.py` - central path helpers used by the Python runtime.

### Experiments and thesis assets

- `Experiments/` - raw and working experiment tools.
- `Diplomová práca/Experiments/` - curated thesis experiment packages with reports and summaries.
- `Diplomová práca/Latex/images/` - thesis-ready figures used in the PDF and this README.
- `Diplomová práca/Latex/` - thesis source and final PDF build.

## Notes

- This is research code developed for a diploma thesis, not a polished general-purpose Trackmania AI framework.
- Some large logs, trained checkpoints, and videos may be distributed through GitHub releases or electronic appendix assets instead of being committed directly to the repository.
- The full academic argument, citations, limitations, and detailed experiment interpretation are in the thesis PDF.
