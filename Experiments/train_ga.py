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

from EvolutionPolicy import normalize_hidden_activations, normalize_hidden_dims
from Individual import Individual

from Experiments.tm2d_env import TM2DPhysicsConfig, TM2DRewardConfig, TM2DSimEnv


_WORKER_ENV: TM2DSimEnv | None = None
_WORKER_CONFIG: dict | None = None


class NumpyPolicyView:
    """Fast inference view over EvolutionPolicy weights for local simulations."""

    def __init__(self, policy) -> None:
        self.action_mode = str(policy.action_mode)
        self.hidden_activations = tuple(policy.hidden_activations)
        self.action_scale = policy.action_scale.detach().cpu().numpy().astype(np.float32)
        linear_layers = [module for module in policy.model if module.__class__.__name__ == "Linear"]
        self.weights = [
            module.weight.detach().cpu().numpy().astype(np.float32, copy=True)
            for module in linear_layers
        ]
        self.biases = [
            module.bias.detach().cpu().numpy().astype(np.float32, copy=True)
            for module in linear_layers
        ]

    @staticmethod
    def _activate(values: np.ndarray, activation: str) -> np.ndarray:
        if activation == "relu":
            return np.maximum(values, 0.0)
        if activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-values))
        if activation == "tanh":
            return np.tanh(values)
        raise ValueError(f"Unsupported activation: {activation}")

    def act(self, obs) -> np.ndarray:
        state = np.asarray(obs, dtype=np.float32)
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            state = weight @ state + bias
            if idx < len(self.weights) - 1:
                state = self._activate(state, self.hidden_activations[idx])
        if self.action_mode == "delta":
            return (np.tanh(state) * self.action_scale).astype(np.float32, copy=False)
        gas = 1.0 / (1.0 + np.exp(-state[0]))
        brake = 1.0 / (1.0 + np.exp(-state[1]))
        steer = np.tanh(state[2])
        return np.asarray([gas, brake, steer], dtype=np.float32)


def parse_list(value: str, cast):
    return [cast(part.strip()) for part in str(value).split(",") if part.strip()]


def evaluate_individual(env: TM2DSimEnv, individual: Individual, fitness_mode: str) -> dict:
    metrics = env.rollout_policy(NumpyPolicyView(individual.policy))
    apply_metrics_to_individual(individual, metrics, fitness_mode)
    return metrics


def apply_metrics_to_individual(individual: Individual, metrics: dict, fitness_mode: str) -> None:
    individual.total_progress = float(metrics["progress"])
    individual.dense_progress = float(metrics.get("dense_progress", metrics["progress"]))
    individual.time = float(metrics["time"])
    individual.term = int(metrics["term"])
    individual.distance = float(metrics["distance"])
    if fitness_mode == "reward":
        individual.fitness = float(metrics["reward"])
    elif fitness_mode == "scalar":
        individual.fitness = float(metrics["fitness"])
    elif fitness_mode == "ranking":
        # Keep fitness only as a plot/log concept. Selection falls back to
        # Individual.__lt__/ranking_key(), which compares the tuple directly.
        individual.fitness = None
    else:
        raise ValueError("fitness_mode must be scalar, reward, or ranking.")


def _init_worker(config: dict) -> None:
    global _WORKER_ENV, _WORKER_CONFIG
    _WORKER_CONFIG = dict(config)
    Individual.RANKING_KEY_MODE = str(config.get("ranking_key_mode", Individual.RANKING_KEY_MODE))
    Individual.RANKING_PROGRESS_SOURCE = str(
        config.get("ranking_progress_source", Individual.RANKING_PROGRESS_SOURCE)
    )
    reward_config = TM2DRewardConfig(mode=str(config["reward_mode"]))
    _WORKER_ENV = TM2DSimEnv(
        map_name=str(config["map_name"]),
        max_time=float(config["max_time"]),
        reward_config=reward_config,
        physics_config=TM2DPhysicsConfig().with_fixed_fps(config.get("fixed_fps")),
        seed=int(config["seed"]),
        collision_mode=str(config["collision_mode"]),
    )


def _evaluate_genome_worker(payload: tuple[int, np.ndarray]) -> tuple[int, dict]:
    if _WORKER_ENV is None or _WORKER_CONFIG is None:
        raise RuntimeError("Worker environment was not initialized.")
    index, genome = payload
    policy = NumpyGenomePolicyView(
        genome=np.asarray(genome, dtype=np.float32),
        obs_dim=int(_WORKER_CONFIG["obs_dim"]),
        hidden_dim=tuple(int(v) for v in _WORKER_CONFIG["hidden_dim"]),
        hidden_activation=tuple(str(v) for v in _WORKER_CONFIG["hidden_activation"]),
        act_dim=int(_WORKER_CONFIG["act_dim"]),
        action_mode=str(_WORKER_CONFIG["action_mode"]),
        action_scale=np.asarray(_WORKER_CONFIG["action_scale"], dtype=np.float32),
    )
    metrics = _WORKER_ENV.rollout_policy(policy)
    return int(index), metrics


class NumpyGenomePolicyView:
    """Policy view built directly from a flat genome; used in worker processes."""

    def __init__(
        self,
        genome: np.ndarray,
        obs_dim: int,
        hidden_dim: tuple[int, ...],
        hidden_activation: tuple[str, ...],
        act_dim: int,
        action_mode: str,
        action_scale: np.ndarray,
    ) -> None:
        self.hidden_activations = tuple(hidden_activation)
        self.action_mode = str(action_mode)
        self.action_scale = np.asarray(action_scale, dtype=np.float32)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        dims = (int(obs_dim), *[int(v) for v in hidden_dim], int(act_dim))
        flat = np.asarray(genome, dtype=np.float32).reshape(-1)
        offset = 0
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            weight_size = int(in_dim) * int(out_dim)
            bias_size = int(out_dim)
            weight = flat[offset:offset + weight_size].reshape(int(out_dim), int(in_dim))
            offset += weight_size
            bias = flat[offset:offset + bias_size]
            offset += bias_size
            self.weights.append(weight)
            self.biases.append(bias)
        if offset != flat.size:
            raise ValueError(f"Genome size mismatch: consumed {offset}, got {flat.size}.")

    @staticmethod
    def _activate(values: np.ndarray, activation: str) -> np.ndarray:
        if activation == "relu":
            return np.maximum(values, 0.0)
        if activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-values))
        if activation == "tanh":
            return np.tanh(values)
        raise ValueError(f"Unsupported activation: {activation}")

    def act(self, obs) -> np.ndarray:
        state = np.asarray(obs, dtype=np.float32)
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            state = weight @ state + bias
            if idx < len(self.weights) - 1:
                state = self._activate(state, self.hidden_activations[idx])
        if self.action_mode == "delta":
            return (np.tanh(state) * self.action_scale).astype(np.float32, copy=False)
        gas = 1.0 / (1.0 + np.exp(-state[0]))
        brake = 1.0 / (1.0 + np.exp(-state[1]))
        steer = np.tanh(state[2])
        return np.asarray([gas, brake, steer], dtype=np.float32)


def make_child_pool(parents: list[Individual], child_count: int) -> list[Individual]:
    children: list[Individual] = []
    if len(parents) == 1:
        while len(children) < child_count:
            children.append(parents[0].copy())
        return children
    while len(children) < child_count:
        order = list(range(len(parents)))
        random.shuffle(order)
        for i in range(0, len(order) - 1, 2):
            if len(children) >= child_count:
                break
            children.append(parents[order[i]].crossover(parents[order[i + 1]]))
    return children


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast 2D Trackmania-like GA experiments.")
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
    parser.add_argument(
        "--fixed-fps",
        type=float,
        default=None,
        help="Use deterministic fixed simulation FPS by setting min_dt=max_dt=1/fps.",
    )
    parser.add_argument("--fitness-mode", choices=["scalar", "reward", "ranking"], default="scalar")
    parser.add_argument(
        "--ranking-key-mode",
        choices=list(Individual.RANKING_KEY_MODES),
        default="term_progress_time_distance",
        help="Tuple ordering used by Individual.__lt__ when --fitness-mode ranking.",
    )
    parser.add_argument(
        "--ranking-progress-source",
        choices=list(Individual.RANKING_PROGRESS_SOURCES),
        default="progress",
        help="Progress value used inside ranking tuples: discrete checkpoints or dense geometry.",
    )
    parser.add_argument("--collision-mode", choices=["center", "corners"], default="center")
    parser.add_argument(
        "--reward-mode",
        choices=[
            "progress_delta",
            "progress_primary_delta",
            "pace_delta",
            "terminal_fitness",
            "individual_dense",
            "terminal_lexicographic",
            "terminal_lexicographic_no_distance",
            "terminal_lexicographic_progress20",
            "delta_lexicographic",
            "delta_lexicographic_terminal",
            "terminal_progress_time_efficiency",
            "delta_progress_time_efficiency",
            "progress_rate",
        ],
        default="progress_primary_delta",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel individual evaluation. Use 1 to disable.",
    )
    parser.add_argument("--log-dir", default="Experiments/runs")
    parser.add_argument("--render-best", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    Individual.RANKING_KEY_MODE = str(args.ranking_key_mode)
    Individual.RANKING_PROGRESS_SOURCE = str(args.ranking_progress_source)

    hidden_dim = tuple(parse_list(args.hidden_dim, int))
    hidden_activation = tuple(parse_list(args.hidden_activation, str))
    hidden_dim = normalize_hidden_dims(hidden_dim)
    hidden_activation = normalize_hidden_activations(hidden_activation, len(hidden_dim))

    reward_config = TM2DRewardConfig(mode=args.reward_mode)
    physics_config = TM2DPhysicsConfig().with_fixed_fps(args.fixed_fps)
    env = TM2DSimEnv(
        map_name=args.map_name,
        max_time=args.max_time,
        reward_config=reward_config,
        physics_config=physics_config,
        seed=args.seed,
        collision_mode=args.collision_mode,
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

    run_name = time.strftime("%Y%m%d_%H%M%S") + f"_tm2d_ga_{args.map_name.replace(' ', '_').replace('#', '')}"
    run_dir = Path(args.log_dir) / run_name
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
        "fixed_fps": args.fixed_fps,
        "fitness_mode": args.fitness_mode,
        "ranking_key_mode": args.ranking_key_mode,
        "ranking_progress_source": args.ranking_progress_source,
        "reward_mode": args.reward_mode,
        "collision_mode": args.collision_mode,
        "obs_dim": env.obs_dim,
        "act_dim": env.act_dim,
        "progress_bucket": env.progress_bucket,
        "physics": asdict(env.physics),
        "num_workers": int(args.num_workers),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    csv_path = run_dir / "generation_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "generation",
                "best_fitness",
                "best_progress",
                "best_dense_progress",
                "best_ranking_progress",
                "best_time",
                "best_term",
                "best_reward",
                "mean_progress",
                "mean_dense_progress",
                "mean_ranking_progress",
                "mean_reward",
                "mean_time",
            ],
        )
        writer.writeheader()

        worker_pool = None
        best_overall: Individual | None = None
        if int(args.num_workers) > 1:
            worker_config = {
                "map_name": args.map_name,
                "max_time": args.max_time,
                "reward_mode": args.reward_mode,
                "ranking_key_mode": args.ranking_key_mode,
                "ranking_progress_source": args.ranking_progress_source,
                "collision_mode": args.collision_mode,
                "seed": args.seed,
                "fixed_fps": args.fixed_fps,
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
                if worker_pool is None:
                    metrics = [
                        evaluate_individual(env, individual, args.fitness_mode)
                        for individual in population
                    ]
                else:
                    payloads = [
                        (idx, individual.genome.copy())
                        for idx, individual in enumerate(population)
                    ]
                    metrics = [None for _ in population]
                    for idx, metric in worker_pool.map(_evaluate_genome_worker, payloads):
                        metrics[idx] = metric
                        apply_metrics_to_individual(population[idx], metric, args.fitness_mode)
                    metrics = [metric for metric in metrics if metric is not None]

                population.sort(reverse=True)
                best = population[0]
                if best_overall is None or best_overall < best:
                    best_overall = best.copy()
                best_fitness_for_plot = (
                    float(best.fitness)
                    if best.fitness is not None and np.isfinite(best.fitness)
                    else float(best.compute_scalar_fitness())
                )
                row = {
                    "generation": generation,
                    "best_fitness": best_fitness_for_plot,
                    "best_progress": float(best.total_progress),
                    "best_dense_progress": float(max(metric["dense_progress"] for metric in metrics)),
                    "best_ranking_progress": float(best.ranking_progress()),
                    "best_time": float(best.time),
                    "best_term": int(best.term),
                    "best_reward": float(max(metric["reward"] for metric in metrics)),
                    "mean_progress": float(np.mean([metric["progress"] for metric in metrics])),
                    "mean_dense_progress": float(np.mean([metric["dense_progress"] for metric in metrics])),
                    "mean_ranking_progress": float(
                        np.mean(
                            [
                                metric["dense_progress"]
                                if args.ranking_progress_source == "dense_progress"
                                else metric["progress"]
                                for metric in metrics
                            ]
                        )
                    ),
                    "mean_reward": float(np.mean([metric["reward"] for metric in metrics])),
                    "mean_time": float(np.mean([metric["time"] for metric in metrics])),
                }
                writer.writerow(row)
                handle.flush()
                print(
                    f"gen={generation:04d} best_fit={row['best_fitness']:.2f} "
                    f"best_prog={row['best_progress']:.2f}% "
                    f"best_dense={row['best_dense_progress']:.2f}% "
                    f"best_rank={row['best_ranking_progress']:.2f}% "
                    f"best_time={row['best_time']:.2f}s mean_prog={row['mean_progress']:.2f}%"
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
        population.sort(reverse=True)
        best_overall = population[0]
    best_overall.policy.save(str(best_path), extra={"config": config})
    print(f"Saved best policy to {best_path}")
    print(f"Saved metrics to {csv_path}")

    if args.render_best:
        from Experiments.tm2d_viewer import TM2DViewer

        viewer_env = TM2DSimEnv(
            map_name=args.map_name,
            max_time=args.max_time,
            reward_config=reward_config,
            physics_config=physics_config,
            seed=args.seed + 1,
            collision_mode=args.collision_mode,
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
