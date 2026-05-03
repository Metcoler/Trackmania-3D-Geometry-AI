from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NeuralPolicy import normalize_hidden_activations, normalize_hidden_dims
from Individual import Individual

from Experiments.multiobjective import (
    objective_names_for_mode,
    objectives_from_metrics,
    pareto_order,
)
from Experiments.tm2d_env import TM2DRewardConfig, TM2DSimEnv
from Experiments.train_ga import (
    NumpyPolicyView,
    _evaluate_genome_worker,
    _init_worker,
    generation_metric_fieldnames,
    individual_metric_fieldnames,
    make_child_pool,
    metric_stats,
    parse_list,
)


def evaluate_individual(env: TM2DSimEnv, individual: Individual) -> dict:
    return env.rollout_policy(NumpyPolicyView(individual.policy))


def apply_metrics_to_individual(individual: Individual, metrics: dict, selection_score: float) -> None:
    individual.fitness = float(selection_score)
    individual.discrete_progress = float(metrics["progress"])
    individual.dense_progress = float(metrics.get("dense_progress", metrics["progress"]))
    individual.time = float(metrics["time"])
    individual.finished = int(metrics.get("finished", 0))
    individual.crashes = int(metrics.get("crashes", 0))
    individual.distance = float(metrics["distance"])
    individual.reward = float(metrics.get("reward", metrics.get("fitness", 0.0)))
    individual.evaluation_steps = int(metrics.get("steps", 0))
    individual.evaluation_terminated = bool(metrics.get("terminated", False))
    individual.evaluation_truncated = bool(metrics.get("truncated", False))
    individual.evaluation_valid = True


def objective_priority_indices(names: list[str], priority: str) -> list[int]:
    priority_names = [part.strip() for part in str(priority).split(",") if part.strip()]
    if not priority_names or [name.lower() for name in priority_names] == ["auto"]:
        return list(range(len(names)))
    missing = [name for name in priority_names if name not in names]
    if missing:
        raise ValueError(f"Unknown objective priority names: {missing}. Available: {names}")
    return [names.index(name) for name in priority_names]


def select_objective_subset(names: list[str], subset: str) -> list[int]:
    subset_names = [part.strip() for part in str(subset).split(",") if part.strip()]
    if not subset_names or [name.lower() for name in subset_names] == ["auto"]:
        return list(range(len(names)))
    missing = [name for name in subset_names if name not in names]
    if missing:
        raise ValueError(f"Unknown objective subset names: {missing}. Available: {names}")
    return [names.index(name) for name in subset_names]


def build_run_dir(log_dir: str, map_name: str, objective_mode: str) -> Path:
    run_name = (
        time.strftime("%Y%m%d_%H%M%S")
        + f"_tm2d_ga_moo_{map_name.replace(' ', '_').replace('#', '')}_{objective_mode}"
    )
    run_dir = Path(log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast 2D Trackmania-like multi-objective GA experiments.")
    parser.add_argument("--map-name", default="AI Training #5")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=48)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--parent-count", type=int, default=16)
    parser.add_argument("--max-time", type=float, default=45.0)
    parser.add_argument("--hidden-dim", default="32,16")
    parser.add_argument("--hidden-activation", default="relu,tanh")
    parser.add_argument("--mutation-prob", type=float, default=0.18)
    parser.add_argument("--mutation-sigma", type=float, default=0.22)
    parser.add_argument("--collision-mode", choices=["center", "corners", "laser", "lidar"], default="laser")
    parser.add_argument(
        "--collision-distance-threshold",
        type=float,
        default=2.0,
        help="Laser/lidar collision threshold used when --collision-mode laser is selected.",
    )
    parser.add_argument(
        "--vertical-mode",
        action="store_true",
        help=(
            "Use the 3D-compatible observation layout with neutral "
            "vertical features while keeping TM2D physics flat."
        ),
    )
    parser.add_argument(
        "--multi-surface-mode",
        action="store_true",
        help="Append surface traction instructions to the observation.",
    )
    parser.add_argument(
        "--continuous-gas-brake",
        action="store_true",
        help="Disable live-TM-style gas/brake binarization in TM2D diagnostics.",
    )
    parser.add_argument(
        "--objective-mode",
        choices=["trackmania_racing", "lexicographic_primitives"],
        default="trackmania_racing",
        help="Objective vector used for Pareto selection.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=[
            "terminal_progress_time_efficiency",
            "delta_progress_time_efficiency",
            "terminal_lexicographic",
            "terminal_lexicographic_no_distance",
            "terminal_lexicographic_progress20",
            "terminal_fitness",
        ],
        default="terminal_progress_time_efficiency",
        help="Only used for reward logging; Pareto selection uses --objective-mode.",
    )
    parser.add_argument(
        "--objective-priority",
        default="auto",
        help="Comma-separated objective names for within-front ordering.",
    )
    parser.add_argument(
        "--objective-subset",
        default="auto",
        help=(
            "Optional comma-separated objective names to keep before Pareto sorting. "
            "Use this to compare simpler objective sets such as "
            "finished,progress,neg_time."
        ),
    )
    parser.add_argument(
        "--pareto-tiebreak",
        choices=["priority", "crowding"],
        default="priority",
        help="How to order individuals inside the same Pareto front.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel individual evaluation. Use 1 to disable.",
    )
    parser.add_argument("--log-dir", default="Experiments/runs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--render-best", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    hidden_dim = tuple(parse_list(args.hidden_dim, int))
    hidden_activation = tuple(parse_list(args.hidden_activation, str))
    hidden_dim = normalize_hidden_dims(hidden_dim)
    hidden_activation = normalize_hidden_activations(hidden_activation, len(hidden_dim))

    reward_config = TM2DRewardConfig(mode=args.reward_mode)
    env = TM2DSimEnv(
        map_name=args.map_name,
        max_time=args.max_time,
        reward_config=reward_config,
        seed=args.seed,
        collision_mode=args.collision_mode,
        collision_distance_threshold=args.collision_distance_threshold,
        vertical_mode=args.vertical_mode,
        multi_surface_mode=args.multi_surface_mode,
        binary_gas_brake=not args.continuous_gas_brake,
    )
    action_scale = np.array([0.2, 0.2, 0.2], dtype=np.float32)
    population = [
        Individual(
            obs_dim=env.obs_dim,
            hidden_dim=hidden_dim,
            act_dim=env.act_dim,
            action_scale=action_scale,
            action_mode="target",
            hidden_activation=hidden_activation,
        )
        for _ in range(args.population_size)
    ]

    available_objective_names = objective_names_for_mode(args.objective_mode)
    objective_subset_indices = select_objective_subset(available_objective_names, args.objective_subset)
    objective_names = [available_objective_names[index] for index in objective_subset_indices]
    priority_indices = objective_priority_indices(objective_names, args.objective_priority)
    max_episode_distance = float(env.physics.max_speed) * float(args.max_time)
    run_dir = Path(args.run_dir) if args.run_dir else build_run_dir(args.log_dir, args.map_name, args.objective_mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "map_name": args.map_name,
        "generations": args.generations,
        "population_size": args.population_size,
        "elite_count": args.elite_count,
        "parent_count": args.parent_count,
        "max_time": args.max_time,
        "hidden_dim": list(hidden_dim),
        "hidden_activation": list(hidden_activation),
        "mutation_prob": args.mutation_prob,
        "mutation_sigma": args.mutation_sigma,
        "selection_mode": "pareto",
        "objective_mode": args.objective_mode,
        "available_objective_names": available_objective_names,
        "objective_subset": objective_names,
        "objective_names": objective_names,
        "objective_priority": [objective_names[index] for index in priority_indices],
        "pareto_tiebreak": args.pareto_tiebreak,
        "reward_mode": args.reward_mode,
        "collision_mode": args.collision_mode,
        "collision_distance_threshold": float(args.collision_distance_threshold),
        "lidar_mode": "aabb_clearance",
        "vehicle_hitbox": env.vehicle_hitbox.as_dict(),
        "vertical_mode": bool(args.vertical_mode),
        "multi_surface_mode": bool(args.multi_surface_mode),
        "binary_gas_brake": bool(not args.continuous_gas_brake),
        "obs_dim": env.obs_dim,
        "act_dim": env.act_dim,
        "progress_bucket": env.progress_bucket,
        "estimated_path_length": env.geometry.estimated_path_length,
        "max_episode_distance": max_episode_distance,
        "physics": asdict(env.physics),
        "num_workers": int(args.num_workers),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n[TM2D MOO GA] Prepared multi-objective GA experiment")
    print(f"[TM2D MOO GA] run_dir={run_dir}")
    print(f"[TM2D MOO GA] objectives={objective_names}")
    print(f"[TM2D MOO GA] priority={[objective_names[index] for index in priority_indices]}")
    print(f"[TM2D MOO GA] pareto_tiebreak={args.pareto_tiebreak}")

    csv_path = run_dir / "generation_metrics.csv"
    individual_csv_path = run_dir / "individual_metrics.csv"
    common_generation_fields = generation_metric_fieldnames()
    moo_generation_fields = [
        "front0_size",
        "best_rank",
        "best_crowding",
        "best_priority_score",
        "best_scalar_fitness",
    ] + [f"best_obj_{name}" for name in objective_names] + [f"mean_obj_{name}" for name in objective_names]
    fieldnames = common_generation_fields + [
        field for field in moo_generation_fields if field not in common_generation_fields
    ]
    for prefix in (
        "progress",
        "dense_progress",
        "ranking_progress",
        "time",
        "distance",
        "reward",
        "fitness",
        "steps",
    ):
        for field in metric_stats(prefix, [0.0]).keys():
            if field not in fieldnames:
                fieldnames.append(field)

    common_individual_fields = individual_metric_fieldnames()
    moo_individual_fields = [
        "front_rank",
        "crowding",
        "front0",
        "priority_score",
    ] + [f"obj_{name}" for name in objective_names]
    individual_fieldnames = common_individual_fields + [
        field for field in moo_individual_fields if field not in common_individual_fields
    ]

    with (
        csv_path.open("w", newline="", encoding="utf-8") as handle,
        individual_csv_path.open("w", newline="", encoding="utf-8") as individual_handle,
    ):
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        individual_writer = csv.DictWriter(individual_handle, fieldnames=individual_fieldnames)
        writer.writeheader()
        individual_writer.writeheader()

        worker_pool = None
        best_overall: Individual | None = None
        best_overall_key: tuple[float, ...] | None = None
        training_wall_start = time.perf_counter()
        cumulative_virtual_time = 0.0
        cumulative_virtual_steps = 0
        if int(args.num_workers) > 1:
            worker_config = {
                "map_name": args.map_name,
                "max_time": args.max_time,
                "reward_mode": args.reward_mode,
                "collision_mode": args.collision_mode,
                "collision_distance_threshold": float(args.collision_distance_threshold),
                "vertical_mode": bool(args.vertical_mode),
                "multi_surface_mode": bool(args.multi_surface_mode),
                "binary_gas_brake": bool(not args.continuous_gas_brake),
                "seed": args.seed,
                "obs_dim": env.obs_dim,
                "hidden_dim": list(hidden_dim),
                "hidden_activation": list(hidden_activation),
                "act_dim": env.act_dim,
                "action_mode": "target",
                "action_scale": action_scale.tolist(),
            }
            worker_pool = mp.get_context("spawn").Pool(
                processes=int(args.num_workers),
                initializer=_init_worker,
                initargs=(worker_config,),
            )

        try:
            for generation in range(1, args.generations + 1):
                generation_wall_start = time.perf_counter()
                original_index_by_id: dict[int, int] = {
                    id(individual): idx for idx, individual in enumerate(population)
                }
                if worker_pool is None:
                    metrics = [evaluate_individual(env, individual) for individual in population]
                else:
                    payloads = [
                        (idx, individual.genome.copy())
                        for idx, individual in enumerate(population)
                    ]
                    metrics = [None for _ in population]
                    for idx, metric in worker_pool.map(_evaluate_genome_worker, payloads):
                        metrics[idx] = metric
                    metrics = [metric for metric in metrics if metric is not None]

                objective_matrix = np.vstack(
                    [
                        objectives_from_metrics(
                            metric,
                            mode=args.objective_mode,
                            max_time=args.max_time,
                            estimated_path_length=env.geometry.estimated_path_length,
                            max_episode_distance=max_episode_distance,
                        )[objective_subset_indices]
                        for metric in metrics
                    ]
                )
                ordering = pareto_order(
                    objective_matrix,
                    priority_indices=priority_indices,
                    tiebreak=args.pareto_tiebreak,
                    objective_names=objective_names,
                )
                population = [population[index] for index in ordering.order]
                metrics = [metrics[index] for index in ordering.order]
                ordered_objectives = ordering.objectives[ordering.order]
                ordered_crowding = ordering.crowding[ordering.order]
                ordered_ranks = ordering.ranks[ordering.order]
                ordered_original_indices = list(ordering.order)

                for local_idx, individual in enumerate(population):
                    priority_score = tuple(float(ordered_objectives[local_idx, idx]) for idx in priority_indices)
                    scalar_priority = float(sum((10.0 ** -i) * value for i, value in enumerate(priority_score)))
                    apply_metrics_to_individual(individual, metrics[local_idx], selection_score=scalar_priority)

                best = population[0]
                best_metric = metrics[0]
                best_objectives = ordered_objectives[0]
                best_key = tuple(float(best_objectives[index]) for index in priority_indices)
                if best_overall_key is None or best_key > best_overall_key:
                    best_overall_key = best_key
                    best_overall = best.copy()

                progress_values = np.asarray([float(metric["progress"]) for metric in metrics], dtype=np.float64)
                dense_progress_values = np.asarray(
                    [float(metric["dense_progress"]) for metric in metrics],
                    dtype=np.float64,
                )
                # MOO uses dense progress as the common analysis progress source.
                ranking_progress_values = dense_progress_values.copy()
                reward_values = np.asarray([float(metric["reward"]) for metric in metrics], dtype=np.float64)
                fitness_values = np.asarray([float(metric["fitness"]) for metric in metrics], dtype=np.float64)
                time_values = np.asarray([float(metric["time"]) for metric in metrics], dtype=np.float64)
                distance_values = np.asarray([float(metric["distance"]) for metric in metrics], dtype=np.float64)
                step_values = np.asarray([int(metric.get("steps", 0)) for metric in metrics], dtype=np.float64)
                finished_values = np.asarray([int(metric.get("finished", 0)) for metric in metrics], dtype=np.int32)
                crash_values = np.asarray([int(metric.get("crashes", 0)) for metric in metrics], dtype=np.int32)
                timeout_values = np.asarray(
                    [
                        int(int(metric.get("finished", 0)) <= 0 and int(metric.get("crashes", 0)) <= 0)
                        for metric in metrics
                    ],
                    dtype=np.int32,
                )
                virtual_time_sum = float(np.sum(time_values))
                virtual_steps_sum = int(np.sum(step_values))
                cumulative_virtual_time += virtual_time_sum
                cumulative_virtual_steps += virtual_steps_sum
                generation_wall_seconds = float(time.perf_counter() - generation_wall_start)
                cumulative_wall_seconds = float(time.perf_counter() - training_wall_start)
                row = {
                    "generation": generation,
                    "best_fitness": float(best.fitness),
                    "best_progress": float(best_metric["progress"]),
                    "best_dense_progress": float(best_metric["dense_progress"]),
                    "best_ranking_progress": float(best_metric["dense_progress"]),
                    "best_time": float(best_metric["time"]),
                    "best_finished": int(best_metric.get("finished", 0)),
                    "best_crashes": int(best_metric.get("crashes", 0)),
                    "best_distance": float(best_metric["distance"]),
                    "best_steps": int(best_metric.get("steps", 0)),
                    "best_reward": float(best_metric["reward"]),
                    "best_ranking_key": json.dumps([float(value) for value in best_objectives]),
                    "cached_evaluations": 0,
                    "evaluated_count": int(len(population)),
                    "population_size": int(len(population)),
                    "finish_count": int(np.sum(finished_values > 0)),
                    "crash_count": int(np.sum(crash_values > 0)),
                    "timeout_count": int(np.sum(timeout_values > 0)),
                    "terminated_count": int(sum(bool(metric.get("terminated", False)) for metric in metrics)),
                    "truncated_count": int(sum(bool(metric.get("truncated", False)) for metric in metrics)),
                    "virtual_time_sum": virtual_time_sum,
                    "cumulative_virtual_time": cumulative_virtual_time,
                    "virtual_steps_sum": virtual_steps_sum,
                    "cumulative_virtual_steps": int(cumulative_virtual_steps),
                    "generation_wall_seconds": generation_wall_seconds,
                    "cumulative_wall_seconds": cumulative_wall_seconds,
                    "mean_progress": float(np.mean(progress_values)),
                    "mean_dense_progress": float(np.mean(dense_progress_values)),
                    "mean_ranking_progress": float(np.mean(ranking_progress_values)),
                    "mean_reward": float(np.mean(reward_values)),
                    "mean_time": float(np.mean(time_values)),
                    "front0_size": len(ordering.fronts[0]) if ordering.fronts else 0,
                    "best_rank": int(ordered_ranks[0]),
                    "best_crowding": float(ordered_crowding[0]),
                    "best_priority_score": float(best.fitness),
                    "best_scalar_fitness": float(best_metric["fitness"]),
                }
                for objective_idx, name in enumerate(objective_names):
                    row[f"best_obj_{name}"] = float(best_objectives[objective_idx])
                    row[f"mean_obj_{name}"] = float(np.mean(ordered_objectives[:, objective_idx]))
                row.update(metric_stats("progress", progress_values))
                row.update(metric_stats("dense_progress", dense_progress_values))
                row.update(metric_stats("ranking_progress", ranking_progress_values))
                row.update(metric_stats("time", time_values))
                row.update(metric_stats("distance", distance_values))
                row.update(metric_stats("reward", reward_values))
                row.update(metric_stats("fitness", fitness_values))
                row.update(metric_stats("steps", step_values))
                writer.writerow(row)
                handle.flush()

                for rank, (individual, metric, objective_row) in enumerate(
                    zip(population, metrics, ordered_objectives),
                    start=1,
                ):
                    individual_row = {
                        "generation": generation,
                        "rank": rank,
                        "original_index": int(
                            original_index_by_id.get(id(individual), ordered_original_indices[rank - 1])
                        ),
                        "cached": 0,
                        "is_elite": int(rank <= int(args.elite_count)),
                        "is_parent": int(rank <= int(args.parent_count)),
                        "fitness": float(metric["fitness"]),
                        "ranking_key": json.dumps([float(value) for value in objective_row]),
                        "ranking_progress": float(metric["dense_progress"]),
                        "progress": float(metric["progress"]),
                        "dense_progress": float(metric["dense_progress"]),
                        "time": float(metric["time"]),
                        "finished": int(metric.get("finished", 0)),
                        "crashes": int(metric.get("crashes", 0)),
                        "timeout": int(
                            int(metric.get("finished", 0)) <= 0
                            and int(metric.get("crashes", 0)) <= 0
                        ),
                        "distance": float(metric["distance"]),
                        "reward": float(metric["reward"]),
                        "steps": int(metric.get("steps", 0)),
                        "terminated": int(bool(metric.get("terminated", False))),
                        "truncated": int(bool(metric.get("truncated", False))),
                        "front_rank": int(ordered_ranks[rank - 1]),
                        "crowding": float(ordered_crowding[rank - 1]),
                        "front0": int(int(ordered_ranks[rank - 1]) == 0),
                        "priority_score": float(individual.fitness),
                    }
                    for objective_idx, name in enumerate(objective_names):
                        individual_row[f"obj_{name}"] = float(objective_row[objective_idx])
                    individual_writer.writerow(individual_row)
                individual_handle.flush()

                objective_summary = " ".join(
                    f"{name}={row[f'best_obj_{name}']:.3f}"
                    for name in objective_names[:3]
                )
                print(
                    f"gen={generation:04d} front0={row['front0_size']:02d} "
                    f"best_dense={row['best_dense_progress']:.2f}% "
                    f"best_time={row['best_time']:.2f}s "
                    f"fin={row['best_finished']} crashes={row['best_crashes']} "
                    f"{objective_summary}"
                )

                if generation < args.generations:
                    elites = [individual.copy() for individual in population[: args.elite_count]]
                    parents = [individual.copy() for individual in population[: args.parent_count]]
                    children = make_child_pool(parents, args.population_size - len(elites))
                    for child in children:
                        child.mutate(args.mutation_prob, args.mutation_sigma)
                    population = elites + children
        finally:
            if worker_pool is not None:
                worker_pool.close()
                worker_pool.join()

    best_path = run_dir / "best_policy.pt"
    if best_overall is None:
        best_overall = population[0]
    best_overall.policy.save(str(best_path), extra={"config": config})
    print(f"Saved best policy to {best_path}")
    print(f"Saved metrics to {csv_path}")
    print(f"Saved individual metrics to {individual_csv_path}")

    if args.render_best:
        from Experiments.tm2d_viewer import TM2DViewer

        viewer_env = TM2DSimEnv(
            map_name=args.map_name,
            max_time=args.max_time,
            reward_config=reward_config,
            seed=args.seed + 1,
            collision_mode=args.collision_mode,
            collision_distance_threshold=args.collision_distance_threshold,
            vertical_mode=args.vertical_mode,
            multi_surface_mode=args.multi_surface_mode,
            binary_gas_brake=not args.continuous_gas_brake,
        )
        viewer = TM2DViewer(viewer_env)
        obs, info = viewer_env.reset()
        while True:
            action = best_overall.act(obs)
            obs, _, terminated, truncated, info = viewer_env.step(action)
            viewer.update(info)
            time.sleep(0.01)
            if terminated or truncated:
                time.sleep(1.0)
                obs, info = viewer_env.reset()


if __name__ == "__main__":
    main()
