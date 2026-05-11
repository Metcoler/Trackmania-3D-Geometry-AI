# Trackmania 3D Geometry AI

Autonomous Trackmania driving agent built around a **geometry-first** idea: instead of learning directly from screen pixels, the project reconstructs the track as a 3D geometric world, projects the live car into that world, and trains a neural policy from compact observations.

The repository contains the Trackmania runtime, map/mesh tooling, supervised and imitation-learning experiments, neuroevolution training, evaluation scripts, videos, and the English thesis PDF:

[**Trackmania-3D-Geometry-AI-Masters-Thesis.pdf**](Trackmania-3D-Geometry-AI-Masters-Thesis.pdf)

## Demo Videos

| Demo | Video | What it shows |
|---|---|---|
| Flat track | [single_surface_flat.mp4](Videos/single_surface_flat.mp4) | Final agent on a flat map with one road surface. |
| Height-aware track | [single_surface_height.mp4](Videos/single_surface_height.mp4) | Agent using height-aware observations. |
| Multi-surface track | [multi_surface_flat.mp4](Videos/multi_surface_flat.mp4) | Agent driving with different surface types. |
| Bachelor baseline | [bachelor_thesis_version.mp4](Videos/bachelor_thesis_version.mp4) | Original bachelor-thesis version. |
| Transfer map | [diploma_agent_small_map.mp4](Videos/diploma_agent_small_map.mp4) | Diploma agent transferred to a different map. |

## Project In One Picture

The agent is wrapped into the classic environment-agent loop, but the observation path is custom: game telemetry and exported geometry are combined inside Python before the neural policy chooses the next gamepad action.

<p align="center">
  <img src="docs/images/system_loop.png" width="88%" alt="Trackmania agent-environment loop">
</p>

**Runtime loop:** Trackmania environment -> game script / telemetry -> Python runtime -> virtual geometry -> observation -> neural policy -> virtual gamepad action.

## The Core Approach: Blocks -> Meshes -> Geometry -> Observation

Trackmania maps are assembled from reusable blocks. The project uses that structure directly: if we can understand the placed blocks, we can rebuild the track outside the game.

<p align="center">
  <img src="docs/images/trackmania_blocks.png" width="48%" alt="Separate Trackmania blocks">
  <img src="docs/images/trackmania_map_overview.png" width="48%" alt="Composed Trackmania map">
</p>

The same blocks are exported as 3D meshes. A map is then reconstructed by placing those meshes into a common coordinate system.

<p align="center">
  <img src="docs/images/mesh_straight.png" width="30%" alt="Straight block mesh">
  <img src="docs/images/mesh_curve.png" width="30%" alt="Curve block mesh">
  <img src="docs/images/map_mesh.png" width="30%" alt="Full reconstructed map mesh">
</p>

This gives us a geometric copy of the Trackmania world that can be queried by the agent.

<p align="center">
  <img src="docs/images/virtual_map_render.png" width="42%" alt="Virtual reconstructed map">
  <img src="docs/images/game_vs_agent_view.png" width="52%" alt="Game view versus agent view">
</p>

## Driving Task And Map Variability

The goal is still simple to state: drive from start to finish as fast as possible. The important part is that the track can change shape, surface, and height.

<p align="center">
  <img src="docs/images/track_drive_preview.png" width="72%" alt="Track traversal preview">
</p>

Different surfaces matter because they change traction. Height changes matter because a flat 2D ray is not enough when the road climbs or breaks over a crest.

<p align="center">
  <img src="docs/images/surface_blocks.png" width="45%" alt="Different Trackmania surfaces">
  <img src="docs/images/height_blocks.png" width="45%" alt="Trackmania height blocks">
</p>

<p align="center">
  <img src="docs/images/height_raycast_profile.png" width="70%" alt="Height-aware raycast profile">
</p>

## What The Agent Sees

The policy does not receive the full map. It receives a compact observation vector derived from local geometry and car state:

- progress along the track,
- distance rays to the road boundaries,
- forward and lateral velocity components,
- heading relative to the track,
- upcoming curve instructions,
- optional surface/traction features,
- optional height-profile features,
- timing and previous-action features when useful.

<p align="center">
  <img src="docs/images/observation_from_geometry.png" width="90%" alt="Observation from virtual geometry">
</p>

Actions are applied through a virtual gamepad, so steering can be continuous rather than only keyboard-like left/right taps.

<p align="center">
  <img src="docs/images/keyboard_vs_controller.png" width="74%" alt="Keyboard versus controller action space">
</p>

## Training Story

The thesis does not jump directly to one final algorithm. It builds the agent step by step.

### 1. Supervised learning: can a small network imitate driving?

Human driving data is first used to test whether a small neural network can approximate:

```text
observation -> action
```

<p align="center">
  <img src="docs/images/supervised_architecture_training.png" width="84%" alt="Supervised architecture training curves">
</p>

This proves that the observation is usable, but not that the agent is robust. A small early deviation can put the policy into a state not covered by the human data.

<p align="center">
  <img src="docs/images/supervised_teacher_agent_paths.png" width="48%" alt="Teacher and agent paths">
  <img src="docs/images/butterfly_effect.png" width="48%" alt="Small action change causing a different trajectory">
</p>

### 2. Lexicographic neuroevolution: optimize driving directly

The main method evolves the neural-network weights. Instead of hiding the objective inside one weighted reward sum, individuals are ranked by interpretable priorities:

```text
(finished, progress, -time, -crashes)
```

Meaning: finish first, then get farther, then be faster, then avoid crashes.

<p align="center">
  <img src="docs/images/lexicographic_reward_progress.png" width="84%" alt="Lexicographic reward progress">
</p>

Closed-loop tests also supported the final activation choice: ReLU followed by tanh behaved better in the tested driving setting than ReLU/ReLU variants.

<p align="center">
  <img src="docs/images/activation_ablation.png" width="84%" alt="Closed-loop activation comparison">
</p>

### 3. Comparison branches: RL and Pareto search

Reinforcement learning and Pareto/NSGA-II style multi-objective search were tested as comparison branches. They helped diagnose the problem, but the lexicographic genetic algorithm remained the clearest practical path in the thesis experiments.

<p align="center">
  <img src="docs/images/rl_comparison.png" width="48%" alt="Reinforcement learning comparison">
  <img src="docs/images/pareto_comparison.png" width="48%" alt="Pareto comparison">
</p>

### 4. GA settings and training improvements

Mutation strength, parent count, elite count, and training improvements were selected through practical sweeps. The goal was not to claim universal optimality, but to find a robust working configuration for this project.

<p align="center">
  <img src="docs/images/ga_mutation_grid.png" width="31%" alt="GA mutation grid">
  <img src="docs/images/ga_selection_pressure.png" width="31%" alt="GA selection pressure">
  <img src="docs/images/ga_training_improvements.png" width="31%" alt="GA training improvements">
</p>

## Final Training Runs

The final setup was tested on three track variants.

| Scenario | Best time in thesis run | What changes |
|---|---:|---|
| `single_surface_flat` | `25.92 s` | Flat map, one surface type. |
| `single_surface_height` | `32.55 s` | Height-aware observation. |
| `multi_surface_flat` | `35.10 s` | Different surface types and traction cues. |

<p align="center">
  <img src="docs/images/final_ga_training_progress.png" width="88%" alt="Final GA training progress">
</p>

<p align="center">
  <img src="docs/images/trajectory_single_surface_flat.png" width="31%" alt="Final flat-track trajectory">
  <img src="docs/images/trajectory_single_surface_height.png" width="31%" alt="Final height-track trajectory">
  <img src="docs/images/trajectory_multi_surface_flat.png" width="31%" alt="Final multi-surface trajectory">
</p>

The final trained agents are also compared by ranked finishing times.

<p align="center">
  <img src="docs/images/final_training_ranked_times.png" width="78%" alt="Ranked final training times">
</p>

## Evaluation Snapshot

The diploma agent was tested on `small_map`, a different map from the main training track. It improved over the bachelor-thesis version and moved closer to the human reference.

| Driver | Time on `small_map` |
|---|---:|
| Human reference | `17.89 s` |
| Diploma agent | `19.68 s` |
| Bachelor agent | `23.064 s` |

<p align="center">
  <img src="docs/images/small_map_comparison.png" width="42%" alt="Small map comparison">
  <img src="docs/images/small_map_trajectory.png" width="50%" alt="Small map trajectory">
</p>

The thesis also places the flat-track agent into a wider distribution of stored player times.

<p align="center">
  <img src="docs/images/human_vs_agent_times.png" width="74%" alt="Human versus agent finishing times">
</p>

## Repository Guide

### Runtime and agent

- `Driver.py` - replay and live driving entry point for a trained policy.
- `Enviroment.py` - Trackmania environment wrapper.
- `Car.py` - car state, movement, and runtime measurements.
- `ObservationEncoder.py` - converts car state and geometry into neural-network inputs.
- `NeuralPolicy.py` - neural policy used by supervised, imitation, and GA training.
- `XboxController.py` - virtual controller output.

### Training

- `GeneticTrainer.py` - main neuroevolution trainer.
- `Individual.py` - one population member, including policy weights and metrics.
- `RankingKey.py` - lexicographic ranking logic.
- `SupervisedTraining.py` - supervised training from recorded human data.
- `ImitationTrainer.py` - mixed human-agent imitation learning.

### Maps and geometry

- `Maps/` - maps, exported layouts, player times, and map meshes.
- `Meshes/` - block-level mesh assets.
- `Map.py` - map loading, block graph logic, geometry, and progress handling.
- `Map Extractor C#/` - helper tooling used to obtain map/block data.

### Game integration

- `Plugins/` - Trackmania/OpenPlanet-side scripts for game-state communication.
- `ProjectPaths.py` - path helpers used by the Python runtime.

### Thesis and documentation

- `Trackmania-3D-Geometry-AI-Masters-Thesis.pdf` - English thesis PDF in the repository root.
- `Masters thesis/Latex/` - English LaTeX project.
- `Diplomová práca/Latex/` - original Slovak LaTeX project.
- `docs/images/` - README-friendly visual assets.
- `Diplomová práca/Experiments/` - curated thesis experiment packages.

## Notes

- This is research code developed for a diploma thesis, not a polished general-purpose Trackmania AI framework.
- Some raw logs and checkpoints are large and may be omitted or distributed separately.
- The thesis PDF contains the full methodology, citations, limitations, and detailed experiment interpretation.
