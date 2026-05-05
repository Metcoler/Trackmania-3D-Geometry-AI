from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NeuralPolicy import normalize_hidden_activations, normalize_hidden_dims
from Individual import Individual
from RankingKey import canonical_ranking_key_expression
from TrajectoryLogger import TrajectoryLogger, should_log_trajectory

from Experiments.tm2d_env import TM2DPhysicsConfig, TM2DRewardConfig, TM2DSimEnv


_WORKER_ENV: TM2DSimEnv | None = None
_WORKER_CONFIG: dict | None = None


class NumpyPolicyView:
    """Fast inference view over NeuralPolicy weights for local simulations."""

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
        if activation == "leaky_relu":
            return np.where(values >= 0.0, values, 0.01 * values)
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


def parse_physics_tick_probs(value: str | None) -> tuple[tuple[int, ...], tuple[float, ...]] | tuple[None, None]:
    if value is None or str(value).strip() == "":
        return None, None
    tick_values: list[int] = []
    tick_probs: list[float] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("Physics tick probabilities must use 'ticks:prob' items.")
        tick_text, prob_text = item.split(":", 1)
        tick_values.append(max(1, int(tick_text.strip())))
        tick_probs.append(float(prob_text.strip()))
    return tuple(tick_values), tuple(tick_probs)


def make_physics_config(
    physics_tick_profile: str,
    physics_tick_probs: str | None,
    fixed_fps: float | None = None,
) -> TM2DPhysicsConfig:
    base = TM2DPhysicsConfig()
    if fixed_fps is not None and float(fixed_fps) > 0.0:
        if abs(float(fixed_fps) - 100.0) > 1e-6:
            raise ValueError("Legacy --fixed-fps is only supported for 100 Hz after the tick-profile migration.")
        return base.with_tick_profile("fixed100")
    tick_values, tick_probabilities = parse_physics_tick_probs(physics_tick_probs)
    return base.with_tick_profile(
        physics_tick_profile,
        tick_values=tick_values,
        tick_probabilities=tick_probabilities,
    )


def evaluation_context_key(*, mirrored: bool, evaluate_both_mirrors: bool) -> str:
    return f"both={int(bool(evaluate_both_mirrors))};mirror={int(bool(mirrored))}"


def aggregate_mirror_metrics(normal: dict, mirrored: dict) -> dict:
    rollout_metrics = [normal, mirrored]
    finished = int(min(int(metric.get("finished", 0)) for metric in rollout_metrics))
    crashes = int(max(int(metric.get("crashes", 0)) for metric in rollout_metrics))
    progress_values = [
        float(metric.get("progress", metric.get("dense_progress", metric.get("block_progress", 0.0))))
        for metric in rollout_metrics
    ]
    block_progress_values = [
        float(metric.get("block_progress", metric.get("discrete_progress", metric.get("progress", 0.0))))
        for metric in rollout_metrics
    ]
    metrics = {
        "progress": float(np.mean(progress_values)),
        "block_progress": float(np.mean(block_progress_values)),
        "dense_progress": float(np.mean(progress_values)),
        "discrete_progress": float(np.mean(block_progress_values)),
        "time": float(np.mean([float(metric["time"]) for metric in rollout_metrics])),
        "distance": float(np.mean([float(metric["distance"]) for metric in rollout_metrics])),
        "reward": float(np.mean([float(metric["reward"]) for metric in rollout_metrics])),
        "fitness": float(np.mean([float(metric["fitness"]) for metric in rollout_metrics])),
        "steps": int(round(float(np.mean([int(metric.get("steps", 0)) for metric in rollout_metrics])))),
        "finished": finished,
        "crashes": crashes,
        "timeout": int(finished <= 0 and crashes <= 0),
        "terminated": bool(any(bool(metric.get("terminated", False)) for metric in rollout_metrics)),
        "truncated": bool(any(bool(metric.get("truncated", False)) for metric in rollout_metrics)),
        "mirrored": 0,
        "evaluate_both_mirrors": 1,
        "rollout_count": 2,
        "mirror_rollout_count": 1,
        "normal_progress": float(normal.get("progress", normal.get("dense_progress", 0.0))),
        "mirrored_progress": float(mirrored.get("progress", mirrored.get("dense_progress", 0.0))),
    }
    for key in ("physics_tick_mean", "physics_hz_mean", "physics_delay_norm_mean"):
        metrics[key] = float(np.mean([float(metric.get(key, 0.0)) for metric in rollout_metrics]))
    return metrics


def evaluate_policy_with_context(
    env: TM2DSimEnv,
    policy,
    *,
    mirrored: bool = False,
    evaluate_both_mirrors: bool = False,
) -> dict:
    if evaluate_both_mirrors:
        normal_metrics = env.rollout_policy(policy, mirrored=False)
        mirrored_metrics = env.rollout_policy(policy, mirrored=True)
        return aggregate_mirror_metrics(normal_metrics, mirrored_metrics)
    metrics = env.rollout_policy(policy, mirrored=bool(mirrored))
    metrics["mirrored"] = int(bool(mirrored))
    metrics["evaluate_both_mirrors"] = 0
    metrics["rollout_count"] = 1
    metrics["mirror_rollout_count"] = int(bool(mirrored))
    return metrics


def evaluate_individual(
    env: TM2DSimEnv,
    individual: Individual,
    fitness_mode: str,
    *,
    mirrored: bool = False,
    evaluate_both_mirrors: bool = False,
) -> dict:
    metrics = evaluate_policy_with_context(
        env,
        NumpyPolicyView(individual.policy),
        mirrored=mirrored,
        evaluate_both_mirrors=evaluate_both_mirrors,
    )
    apply_metrics_to_individual(
        individual,
        metrics,
        fitness_mode,
        evaluation_context=evaluation_context_key(
            mirrored=mirrored,
            evaluate_both_mirrors=evaluate_both_mirrors,
        ),
    )
    return metrics


def apply_metrics_to_individual(
    individual: Individual,
    metrics: dict,
    fitness_mode: str,
    evaluation_context: str = "",
) -> None:
    block_progress = float(metrics.get("block_progress", metrics.get("discrete_progress", metrics["progress"])))
    progress = float(metrics.get("progress", metrics.get("dense_progress", block_progress)))
    individual.discrete_progress = block_progress
    individual.dense_progress = progress
    individual.time = float(metrics["time"])
    individual.finished = int(metrics.get("finished", 0))
    individual.crashes = int(metrics.get("crashes", 0))
    individual.distance = float(metrics["distance"])
    individual.reward = float(metrics.get("reward", metrics.get("fitness", 0.0)))
    individual.evaluation_steps = int(metrics.get("steps", 0))
    individual.evaluation_terminated = bool(metrics.get("terminated", False))
    individual.evaluation_truncated = bool(metrics.get("truncated", False))
    individual.evaluation_context = str(evaluation_context)
    individual.physics_tick_mean = float(metrics.get("physics_tick_mean", 0.0))
    individual.physics_hz_mean = float(metrics.get("physics_hz_mean", 0.0))
    individual.physics_delay_norm_mean = float(metrics.get("physics_delay_norm_mean", 0.0))
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
    individual.evaluation_valid = True


def cached_metrics_from_individual(individual: Individual) -> dict:
    context = str(getattr(individual, "evaluation_context", ""))
    evaluate_both_mirrors = "both=1" in context
    mirrored = "mirror=1" in context
    fitness = (
        float(individual.fitness)
        if individual.fitness is not None and np.isfinite(float(individual.fitness))
        else float(individual.compute_scalar_fitness())
    )
    return {
        "progress": float(individual.dense_progress),
        "block_progress": float(individual.discrete_progress),
        "dense_progress": float(individual.dense_progress),
        "discrete_progress": float(individual.discrete_progress),
        "time": float(individual.time),
        "finished": int(individual.finished),
        "crashes": int(individual.crashes),
        "timeout": int(int(individual.finished) <= 0 and int(individual.crashes) <= 0),
        "distance": float(individual.distance),
        "reward": float(individual.reward),
        "fitness": fitness,
        "steps": int(getattr(individual, "evaluation_steps", 0)),
        "terminated": bool(getattr(individual, "evaluation_terminated", False)),
        "truncated": bool(getattr(individual, "evaluation_truncated", False)),
        "evaluation_context": context,
        "mirrored": int(mirrored),
        "evaluate_both_mirrors": int(evaluate_both_mirrors),
        "rollout_count": 2 if evaluate_both_mirrors else 1,
        "mirror_rollout_count": 1 if (evaluate_both_mirrors or mirrored) else 0,
        "physics_tick_mean": float(getattr(individual, "physics_tick_mean", 0.0)),
        "physics_hz_mean": float(getattr(individual, "physics_hz_mean", 0.0)),
        "physics_delay_norm_mean": float(getattr(individual, "physics_delay_norm_mean", 0.0)),
    }


def make_tm2d_env_from_args(args, reward_config, physics_config, seed: int) -> TM2DSimEnv:
    return TM2DSimEnv(
        map_name=args.map_name,
        max_time=args.max_time,
        reward_config=reward_config,
        physics_config=physics_config,
        seed=int(seed),
        collision_mode=args.collision_mode,
        collision_distance_threshold=args.collision_distance_threshold,
        vertical_mode=args.vertical_mode,
        multi_surface_mode=args.multi_surface_mode,
        binary_gas_brake=not args.continuous_gas_brake,
        max_touches=args.max_touches,
        collision_bounce_speed_retention=args.collision_bounce_speed_retention,
        collision_bounce_backoff=args.collision_bounce_backoff,
        touch_release_clearance_threshold=args.touch_release_clearance_threshold,
        mask_physics_delay_observation=args.mask_physics_delay_observation,
    )


def metric_stats(prefix: str, values) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        array = np.asarray([0.0], dtype=np.float64)
    return {
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_p10": float(np.percentile(array, 10)),
        f"{prefix}_p25": float(np.percentile(array, 25)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_std": float(np.std(array)),
        f"{prefix}_p75": float(np.percentile(array, 75)),
        f"{prefix}_p90": float(np.percentile(array, 90)),
        f"{prefix}_max": float(np.max(array)),
    }


def finish_triggered_decay_factor(current_value: float, target_value: float, remaining_generations: int) -> float:
    if remaining_generations <= 0:
        return 1.0
    current = float(current_value)
    target = float(target_value)
    if current <= target or current <= 0.0:
        return 1.0
    if target <= 0.0:
        return 0.0
    return float((target / current) ** (1.0 / float(remaining_generations)))


def generation_metric_fieldnames() -> list[str]:
    base_fields = [
        "generation",
        "best_fitness",
        "best_progress",
        "best_block_progress",
        "best_ranking_progress",
        "best_time",
        "best_finished",
        "best_crashes",
        "best_distance",
        "best_steps",
        "best_reward",
        "best_ranking_key",
        "current_mutation_prob",
        "current_mutation_sigma",
        "mutation_decay_trigger",
        "mutation_decay_active",
        "mutation_decay_trigger_generation",
        "effective_mutation_prob_decay",
        "effective_mutation_sigma_decay",
        "cached_evaluations",
        "evaluated_count",
        "rollout_count",
        "mirror_rollout_count",
        "mirror_individual_count",
        "population_size",
        "finish_count",
        "crash_count",
        "timeout_count",
        "terminated_count",
        "truncated_count",
        "virtual_time_sum",
        "cumulative_virtual_time",
        "virtual_steps_sum",
        "cumulative_virtual_steps",
        "generation_wall_seconds",
        "cumulative_wall_seconds",
        "mean_progress",
        "mean_block_progress",
        "mean_ranking_progress",
        "mean_reward",
        "mean_time",
        "mean_physics_tick_count",
        "mean_physics_hz",
        "mean_physics_delay_norm",
        "generalization_test_map",
        "generalization_best_finished",
        "generalization_best_crashes",
        "generalization_best_progress",
        "generalization_best_block_progress",
        "generalization_best_time",
        "generalization_best_distance",
    ]
    stat_fields: list[str] = []
    for prefix in (
        "progress",
        "block_progress",
        "ranking_progress",
        "time",
        "distance",
        "reward",
        "fitness",
        "steps",
        "physics_tick",
        "physics_hz",
        "physics_delay_norm",
    ):
        stat_fields.extend(metric_stats(prefix, [0.0]).keys())
    return base_fields + [field for field in stat_fields if field not in base_fields]


def individual_metric_fieldnames() -> list[str]:
    return [
        "generation",
        "rank",
        "original_index",
        "cached",
        "evaluation_context",
        "mirrored",
        "evaluate_both_mirrors",
        "is_elite",
        "is_parent",
        "fitness",
        "ranking_key",
        "ranking_progress",
        "progress",
        "block_progress",
        "time",
        "finished",
        "crashes",
        "timeout",
        "distance",
        "reward",
        "steps",
        "physics_tick_mean",
        "physics_hz_mean",
        "physics_delay_norm_mean",
        "terminated",
        "truncated",
    ]


def generalization_metric_fieldnames() -> list[str]:
    return [
        "generation",
        "train_rank",
        "original_index",
        "test_map",
        "train_finished",
        "train_crashes",
        "train_time",
        "train_progress",
        "train_block_progress",
        "train_distance",
        "test_finished",
        "test_crashes",
        "test_timeout",
        "test_time",
        "test_progress",
        "test_block_progress",
        "test_distance",
        "test_steps",
        "test_reward",
        "test_fitness",
        "test_physics_tick_mean",
        "test_physics_hz_mean",
        "test_physics_delay_norm_mean",
        "test_terminated",
        "test_truncated",
    ]


def _init_worker(config: dict) -> None:
    global _WORKER_ENV, _WORKER_CONFIG
    _WORKER_CONFIG = dict(config)
    Individual.RANKING_KEY = str(config.get("ranking_key", Individual.RANKING_KEY))
    Individual.RANKING_PROGRESS_SOURCE = str(
        config.get("ranking_progress_source", Individual.RANKING_PROGRESS_SOURCE)
    )
    reward_config = TM2DRewardConfig(mode=str(config["reward_mode"]))
    _WORKER_ENV = TM2DSimEnv(
        map_name=str(config["map_name"]),
        max_time=float(config["max_time"]),
        reward_config=reward_config,
        physics_config=make_physics_config(
            physics_tick_profile=str(config.get("physics_tick_profile", "supervised_v2d")),
            physics_tick_probs=config.get("physics_tick_probs"),
            fixed_fps=config.get("fixed_fps"),
        ),
        seed=int(config["seed"]),
        collision_mode=str(config["collision_mode"]),
        collision_distance_threshold=float(config.get("collision_distance_threshold", 2.0)),
        vertical_mode=bool(config.get("vertical_mode", False)),
        multi_surface_mode=bool(config.get("multi_surface_mode", False)),
        binary_gas_brake=bool(config.get("binary_gas_brake", True)),
        max_touches=int(config.get("max_touches", 1)),
        collision_bounce_speed_retention=float(config.get("collision_bounce_speed_retention", 0.40)),
        collision_bounce_backoff=float(config.get("collision_bounce_backoff", 0.05)),
        touch_release_clearance_threshold=float(config.get("touch_release_clearance_threshold", 0.50)),
        mask_physics_delay_observation=bool(config.get("mask_physics_delay_observation", False)),
    )


def _evaluate_genome_worker(payload: tuple[int, np.ndarray, bool, bool]) -> tuple[int, dict]:
    if _WORKER_ENV is None or _WORKER_CONFIG is None:
        raise RuntimeError("Worker environment was not initialized.")
    index, genome, mirrored, evaluate_both_mirrors = payload
    policy = NumpyGenomePolicyView(
        genome=np.asarray(genome, dtype=np.float32),
        obs_dim=int(_WORKER_CONFIG["obs_dim"]),
        hidden_dim=tuple(int(v) for v in _WORKER_CONFIG["hidden_dim"]),
        hidden_activation=tuple(str(v) for v in _WORKER_CONFIG["hidden_activation"]),
        act_dim=int(_WORKER_CONFIG["act_dim"]),
        action_mode=str(_WORKER_CONFIG["action_mode"]),
        action_scale=np.asarray(_WORKER_CONFIG["action_scale"], dtype=np.float32),
    )
    metrics = evaluate_policy_with_context(
        _WORKER_ENV,
        policy,
        mirrored=bool(mirrored),
        evaluate_both_mirrors=bool(evaluate_both_mirrors),
    )
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
        if activation == "leaky_relu":
            return np.where(values >= 0.0, values, 0.01 * values)
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
        "--mutation-prob-decay",
        type=float,
        default=1.0,
        help="Multiplicative mutation probability decay applied after each generation.",
    )
    parser.add_argument(
        "--mutation-prob-min",
        type=float,
        default=None,
        help="Lower bound for mutation probability decay. Defaults to --mutation-prob.",
    )
    parser.add_argument(
        "--mutation-sigma-decay",
        type=float,
        default=1.0,
        help="Multiplicative mutation sigma decay applied after each generation.",
    )
    parser.add_argument(
        "--mutation-sigma-min",
        type=float,
        default=None,
        help="Lower bound for mutation sigma decay. Defaults to --mutation-sigma.",
    )
    parser.add_argument(
        "--mutation-decay-trigger",
        choices=["immediate", "first_finish"],
        default="immediate",
        help=(
            "When to start applying mutation decay. immediate preserves the legacy "
            "behavior; first_finish keeps exploration values until the first finisher "
            "and then computes a decay that reaches the min values by the final generation."
        ),
    )
    parser.add_argument(
        "--physics-tick-profile",
        choices=["fixed100", "supervised_v2d", "custom"],
        default="supervised_v2d",
        help="Discrete physics tick profile used by TM2D. fixed100 keeps 100 Hz; supervised_v2d mimics measured live tick skips.",
    )
    parser.add_argument(
        "--physics-tick-probs",
        default=None,
        help="Custom tick distribution as '1:0.938,2:0.060,3:0.001,4:0.001'. Requires --physics-tick-profile custom.",
    )
    parser.add_argument(
        "--mask-physics-delay-observation",
        action="store_true",
        help="Keep variable physics ticks but force the observation physics_delay_norm feature to zero.",
    )
    parser.add_argument("--fixed-fps", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fps-min", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fps-max", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fitness-mode", choices=["scalar", "reward", "ranking"], default="scalar")
    parser.add_argument(
        "--ranking-mode",
        choices=["lexicographic"],
        default="lexicographic",
        help="Ranking strategy. Currently only lexicographic tuple comparison is supported.",
    )
    parser.add_argument(
        "--ranking-key",
        default="(finished, progress, -time, -crashes, -distance)",
        help=(
            "Lexicographic tuple expression, e.g. "
            "'(progress, finished, -time, -crashes, -distance)'. "
        ),
    )
    parser.add_argument(
        "--ranking-progress-source",
        choices=list(Individual.RANKING_PROGRESS_SOURCES),
        default="progress",
        help="Progress value used inside ranking tuples: progress=dense geometry, block_progress=discrete checkpoints.",
    )
    parser.add_argument("--collision-mode", choices=["center", "corners", "laser", "lidar"], default="laser")
    parser.add_argument(
        "--collision-distance-threshold",
        type=float,
        default=2.0,
        help=(
            "Legacy diagnostic value kept for config visibility. Lidar collision now "
            "uses AABB-relative clearance instead of a global raw-distance threshold."
        ),
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
        "--max-touches",
        type=int,
        default=1,
        help="Number of AABB-lidar touches allowed before terminating the TM2D episode.",
    )
    parser.add_argument(
        "--collision-bounce-speed-retention",
        type=float,
        default=0.40,
        help="Velocity multiplier after a non-terminal TM2D lidar touch when --max-touches > 1.",
    )
    parser.add_argument(
        "--collision-bounce-backoff",
        type=float,
        default=0.05,
        help="Small push away from the wall after a non-terminal TM2D lidar touch.",
    )
    parser.add_argument(
        "--touch-release-clearance-threshold",
        type=float,
        default=0.50,
        help="Required lidar clearance before another TM2D touch can be counted.",
    )
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
    parser.add_argument(
        "--disable-elite-cache",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Elite cache is disabled by default; "
            "use --enable-elite-cache to opt in."
        ),
    )
    parser.add_argument(
        "--enable-elite-cache",
        action="store_true",
        help=(
            "Reuse copied elite evaluations in the next generation. "
            "Disabled by default because it saves little time and can amplify lucky runs."
        ),
    )
    parser.add_argument(
        "--evaluate-both-mirrors",
        action="store_true",
        help="Evaluate every individual on normal and mirrored observations/actions, then aggregate metrics.",
    )
    parser.add_argument(
        "--mirror-episode-prob",
        type=float,
        default=0.0,
        help="Probability that an individual is evaluated in mirrored mode instead of normal mode.",
    )
    parser.add_argument(
        "--trajectory-log-mode",
        choices=["off", "top", "top-finishers-final", "all"],
        default="off",
        help="Optional compact NPZ trajectory logging. Default off to keep sweeps cheap.",
    )
    parser.add_argument(
        "--trajectory-top-k",
        type=int,
        default=1,
        help="Number of top-ranked individuals to log per generation when trajectory logging is enabled.",
    )
    parser.add_argument(
        "--trajectory-log-actions",
        action="store_true",
        help="Also store gas/brake/steer arrays in trajectory NPZ files.",
    )
    parser.add_argument(
        "--generalization-test-map",
        default="",
        help="Optional holdout map for post-generation top-K evaluation. Does not affect training fitness.",
    )
    parser.add_argument(
        "--generalization-test-top-k",
        type=int,
        default=1,
        help="Number of top-ranked individuals to evaluate on the holdout map.",
    )
    parser.add_argument(
        "--generalization-test-every",
        type=int,
        default=1,
        help="Run holdout-map evaluation every N generations.",
    )
    parser.add_argument(
        "--generalization-test-max-time",
        type=float,
        default=None,
        help="Max episode time for holdout-map evaluation. Defaults to --max-time.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    ranking_key = str(args.ranking_key)
    ranking_key_expression = canonical_ranking_key_expression(ranking_key)
    ranking_progress_source = str(args.ranking_progress_source)
    if ranking_progress_source == "dense_progress":
        ranking_progress_source = "progress"
    elif ranking_progress_source == "discrete_progress":
        ranking_progress_source = "block_progress"
    if (
        args.ranking_key is not None
        and "dense_progress" in ranking_key_expression
        and ranking_progress_source != "progress"
    ):
        ranking_progress_source = "progress"
    if (
        args.ranking_key is not None
        and ("block_progress" in ranking_key_expression or "discrete_progress" in ranking_key_expression)
    ):
        ranking_progress_source = "block_progress"
    Individual.RANKING_KEY = ranking_key
    Individual.RANKING_PROGRESS_SOURCE = ranking_progress_source
    elite_cache_enabled = bool(args.enable_elite_cache and not args.disable_elite_cache)
    mutation_prob_min = float(args.mutation_prob if args.mutation_prob_min is None else args.mutation_prob_min)
    mutation_sigma_min = float(args.mutation_sigma if args.mutation_sigma_min is None else args.mutation_sigma_min)
    if float(args.mutation_prob_decay) <= 0.0 or float(args.mutation_sigma_decay) <= 0.0:
        raise ValueError("Mutation decay factors must be positive.")
    if mutation_prob_min < 0.0 or mutation_sigma_min < 0.0:
        raise ValueError("Mutation minimum values must be non-negative.")
    mirror_episode_prob = float(np.clip(float(args.mirror_episode_prob), 0.0, 1.0))
    evaluate_both_mirrors = bool(args.evaluate_both_mirrors)
    if evaluate_both_mirrors:
        mirror_episode_prob = 0.0
    generalization_test_map = str(args.generalization_test_map).strip()
    generalization_test_enabled = bool(generalization_test_map)
    generalization_test_top_k = max(1, int(args.generalization_test_top_k))
    generalization_test_every = max(1, int(args.generalization_test_every))
    generalization_test_max_time = float(
        args.max_time if args.generalization_test_max_time is None else args.generalization_test_max_time
    )

    hidden_dim = tuple(parse_list(args.hidden_dim, int))
    hidden_activation = tuple(parse_list(args.hidden_activation, str))
    hidden_dim = normalize_hidden_dims(hidden_dim)
    hidden_activation = normalize_hidden_activations(hidden_activation, len(hidden_dim))

    reward_config = TM2DRewardConfig(mode=args.reward_mode)
    if args.fps_min is not None or args.fps_max is not None:
        raise ValueError(
            "--fps-min/--fps-max were removed. Use --physics-tick-profile supervised_v2d "
            "or --physics-tick-profile custom --physics-tick-probs instead."
        )
    physics_config = make_physics_config(
        physics_tick_profile=args.physics_tick_profile,
        physics_tick_probs=args.physics_tick_probs,
        fixed_fps=args.fixed_fps,
    )
    effective_physics_tick_profile = (
        "fixed100"
        if args.fixed_fps is not None and float(args.fixed_fps) > 0.0
        else str(args.physics_tick_profile)
    )
    env = make_tm2d_env_from_args(args, reward_config, physics_config, seed=args.seed)
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
        "mutation_prob_decay": float(args.mutation_prob_decay),
        "mutation_prob_min": float(mutation_prob_min),
        "mutation_sigma_decay": float(args.mutation_sigma_decay),
        "mutation_sigma_min": float(mutation_sigma_min),
        "mutation_decay_trigger": str(args.mutation_decay_trigger),
        "physics_tick_profile": effective_physics_tick_profile,
        "physics_tick_probs": args.physics_tick_probs,
        "mask_physics_delay_observation": bool(args.mask_physics_delay_observation),
        "legacy_fixed_fps": args.fixed_fps,
        "fitness_mode": args.fitness_mode,
        "ranking_mode": args.ranking_mode,
        "ranking_key": ranking_key,
        "ranking_key_expression": ranking_key_expression,
        "ranking_progress_source": ranking_progress_source,
        "reward_mode": args.reward_mode,
        "seed": int(args.seed),
        "collision_mode": args.collision_mode,
        "collision_distance_threshold": float(args.collision_distance_threshold),
        "lidar_mode": "aabb_clearance",
        "vehicle_hitbox": env.vehicle_hitbox.as_dict(),
        "vertical_mode": bool(args.vertical_mode),
        "multi_surface_mode": bool(args.multi_surface_mode),
        "binary_gas_brake": bool(not args.continuous_gas_brake),
        "max_touches": int(args.max_touches),
        "collision_bounce_speed_retention": float(args.collision_bounce_speed_retention),
        "collision_bounce_backoff": float(args.collision_bounce_backoff),
        "touch_release_clearance_threshold": float(args.touch_release_clearance_threshold),
        "elite_cache_enabled": bool(elite_cache_enabled),
        "evaluate_both_mirrors": bool(evaluate_both_mirrors),
        "mirror_episode_prob": float(mirror_episode_prob),
        "obs_dim": env.obs_dim,
        "act_dim": env.act_dim,
        "progress_bucket": env.progress_bucket,
        "physics": asdict(env.physics),
        "num_workers": int(args.num_workers),
        "trajectory_log_mode": str(args.trajectory_log_mode),
        "trajectory_top_k": int(args.trajectory_top_k),
        "trajectory_log_actions": bool(args.trajectory_log_actions),
        "generalization_test_map": generalization_test_map,
        "generalization_test_top_k": int(generalization_test_top_k),
        "generalization_test_every": int(generalization_test_every),
        "generalization_test_max_time": float(generalization_test_max_time),
        "generalization_test_note": (
            "Holdout-map rollouts are diagnostic only and never affect ranking, "
            "selection, elite cache, mutation, or best policy selection."
        ),
        "trajectory_logging_note": (
            "TM2D trajectories are replayed after selection to avoid worker IPC. "
            "With stochastic FPS they are diagnostic replays of the same genome, "
            "not necessarily the exact original evaluation."
        ),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    trajectory_logger = (
        TrajectoryLogger(run_dir)
        if str(args.trajectory_log_mode).strip().lower() != "off"
        else None
    )
    trajectory_env = (
        make_tm2d_env_from_args(args, reward_config, physics_config, seed=args.seed + 999_000)
        if trajectory_logger is not None
        else None
    )
    generalization_reward_config = TM2DRewardConfig(mode=args.reward_mode)
    generalization_env = (
        TM2DSimEnv(
            map_name=generalization_test_map,
            max_time=generalization_test_max_time,
            reward_config=generalization_reward_config,
            physics_config=physics_config,
            seed=args.seed + 2_000_000,
            collision_mode=args.collision_mode,
            collision_distance_threshold=args.collision_distance_threshold,
            vertical_mode=args.vertical_mode,
            multi_surface_mode=args.multi_surface_mode,
            binary_gas_brake=bool(not args.continuous_gas_brake),
            max_touches=int(args.max_touches),
            collision_bounce_speed_retention=float(args.collision_bounce_speed_retention),
            collision_bounce_backoff=float(args.collision_bounce_backoff),
            touch_release_clearance_threshold=float(args.touch_release_clearance_threshold),
            mask_physics_delay_observation=bool(args.mask_physics_delay_observation),
        )
        if generalization_test_enabled
        else None
    )

    csv_path = run_dir / "generation_metrics.csv"
    individual_csv_path = run_dir / "individual_metrics.csv"
    generalization_csv_path = run_dir / "generalization_metrics.csv"
    with (
        csv_path.open("w", newline="", encoding="utf-8") as handle,
        individual_csv_path.open("w", newline="", encoding="utf-8") as individual_handle,
        (
            generalization_csv_path.open("w", newline="", encoding="utf-8")
            if generalization_test_enabled
            else nullcontext()
        ) as generalization_handle,
    ):
        writer = csv.DictWriter(handle, fieldnames=generation_metric_fieldnames())
        individual_writer = csv.DictWriter(
            individual_handle,
            fieldnames=individual_metric_fieldnames(),
        )
        generalization_writer = (
            csv.DictWriter(generalization_handle, fieldnames=generalization_metric_fieldnames())
            if generalization_test_enabled
            else None
        )
        writer.writeheader()
        individual_writer.writeheader()
        if generalization_writer is not None:
            generalization_writer.writeheader()

        worker_pool = None
        best_overall: Individual | None = None
        training_wall_start = time.perf_counter()
        cumulative_virtual_time = 0.0
        cumulative_virtual_steps = 0
        current_mutation_prob = float(args.mutation_prob)
        current_mutation_sigma = float(args.mutation_sigma)
        mutation_decay_trigger = str(args.mutation_decay_trigger)
        mutation_decay_active = mutation_decay_trigger == "immediate"
        mutation_decay_trigger_generation = 0 if mutation_decay_active else None
        effective_mutation_prob_decay = float(args.mutation_prob_decay)
        effective_mutation_sigma_decay = float(args.mutation_sigma_decay)
        if int(args.num_workers) > 1:
            worker_config = {
                "map_name": args.map_name,
                "max_time": args.max_time,
                "reward_mode": args.reward_mode,
                "ranking_key": ranking_key,
                "ranking_progress_source": ranking_progress_source,
                "collision_mode": args.collision_mode,
                "collision_distance_threshold": float(args.collision_distance_threshold),
                "vertical_mode": bool(args.vertical_mode),
                "multi_surface_mode": bool(args.multi_surface_mode),
                "binary_gas_brake": bool(not args.continuous_gas_brake),
                "max_touches": int(args.max_touches),
                "collision_bounce_speed_retention": float(args.collision_bounce_speed_retention),
                "collision_bounce_backoff": float(args.collision_bounce_backoff),
                "touch_release_clearance_threshold": float(args.touch_release_clearance_threshold),
                "seed": args.seed,
                "physics_tick_profile": effective_physics_tick_profile,
                "physics_tick_probs": args.physics_tick_probs,
                "mask_physics_delay_observation": bool(args.mask_physics_delay_observation),
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
                generation_wall_start = time.perf_counter()
                cached_evaluations = 0
                cached_by_id: dict[int, bool] = {}
                original_index_by_id: dict[int, int] = {
                    id(individual): idx for idx, individual in enumerate(population)
                }
                if evaluate_both_mirrors:
                    mirror_flags = np.zeros(len(population), dtype=bool)
                elif mirror_episode_prob > 0.0:
                    mirror_flags = np.random.rand(len(population)) < mirror_episode_prob
                else:
                    mirror_flags = np.zeros(len(population), dtype=bool)
                requested_context_by_id: dict[int, str] = {
                    id(individual): evaluation_context_key(
                        mirrored=bool(mirror_flags[idx]),
                        evaluate_both_mirrors=evaluate_both_mirrors,
                    )
                    for idx, individual in enumerate(population)
                }
                if worker_pool is None:
                    metrics = []
                    for idx, individual in enumerate(population):
                        # Elite cache is intentionally elite-safe: a valid cached elite
                        # is not re-evaluated just because mirror_probability sampled
                        # the opposite context in this generation.
                        can_use_cache = (
                            elite_cache_enabled
                            and bool(getattr(individual, "evaluation_valid", False))
                        )
                        if can_use_cache:
                            cached_evaluations += 1
                            cached_by_id[id(individual)] = True
                            metrics.append(cached_metrics_from_individual(individual))
                        else:
                            cached_by_id[id(individual)] = False
                            metrics.append(
                                evaluate_individual(
                                    env,
                                    individual,
                                    args.fitness_mode,
                                    mirrored=bool(mirror_flags[idx]),
                                    evaluate_both_mirrors=evaluate_both_mirrors,
                                )
                            )
                else:
                    metrics = [None for _ in population]
                    for idx, individual in enumerate(population):
                        # Elite cache is intentionally elite-safe; see sequential branch.
                        can_use_cache = (
                            elite_cache_enabled
                            and bool(getattr(individual, "evaluation_valid", False))
                        )
                        if can_use_cache:
                            cached_evaluations += 1
                            cached_by_id[id(individual)] = True
                            metrics[idx] = cached_metrics_from_individual(individual)
                        else:
                            cached_by_id[id(individual)] = False

                    payloads = [
                        (
                            idx,
                            individual.genome.copy(),
                            bool(mirror_flags[idx]),
                            bool(evaluate_both_mirrors),
                        )
                        for idx, individual in enumerate(population)
                        if metrics[idx] is None
                    ]
                    if payloads:
                        for idx, metric in worker_pool.map(_evaluate_genome_worker, payloads):
                            metrics[idx] = metric
                            apply_metrics_to_individual(
                                population[idx],
                                metric,
                                args.fitness_mode,
                                evaluation_context=requested_context_by_id[id(population[idx])],
                            )
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
                progress_values = np.asarray(
                    [
                        float(metric.get("progress", metric.get("dense_progress", metric.get("block_progress", 0.0))))
                        for metric in metrics
                    ],
                    dtype=np.float64,
                )
                block_progress_values = np.asarray(
                    [
                        float(metric.get("block_progress", metric.get("discrete_progress", metric.get("progress", 0.0))))
                        for metric in metrics
                    ],
                    dtype=np.float64,
                )
                ranking_progress_values = np.asarray(
                    [
                        float(metric.get("block_progress", metric.get("discrete_progress", metric.get("progress", 0.0))))
                        if ranking_progress_source in {"block_progress", "discrete_progress"}
                        else float(metric.get("progress", metric.get("dense_progress", 0.0)))
                        for metric in metrics
                    ],
                    dtype=np.float64,
                )
                reward_values = np.asarray([float(metric["reward"]) for metric in metrics], dtype=np.float64)
                fitness_values = np.asarray([float(metric["fitness"]) for metric in metrics], dtype=np.float64)
                time_values = np.asarray([float(metric["time"]) for metric in metrics], dtype=np.float64)
                distance_values = np.asarray([float(metric["distance"]) for metric in metrics], dtype=np.float64)
                step_values = np.asarray([int(metric.get("steps", 0)) for metric in metrics], dtype=np.float64)
                physics_tick_values = np.asarray(
                    [float(metric.get("physics_tick_mean", 1.0)) for metric in metrics],
                    dtype=np.float64,
                )
                physics_hz_values = np.asarray(
                    [float(metric.get("physics_hz_mean", 100.0)) for metric in metrics],
                    dtype=np.float64,
                )
                physics_delay_values = np.asarray(
                    [float(metric.get("physics_delay_norm_mean", 0.0)) for metric in metrics],
                    dtype=np.float64,
                )
                finished_values = np.asarray([int(metric.get("finished", 0)) for metric in metrics], dtype=np.int32)
                crash_values = np.asarray([int(metric.get("crashes", 0)) for metric in metrics], dtype=np.int32)
                timeout_values = np.asarray(
                    [
                        int(int(metric.get("finished", 0)) <= 0 and int(metric.get("crashes", 0)) <= 0)
                        for metric in metrics
                    ],
                    dtype=np.int32,
                )
                rollout_count_sum = int(sum(int(metric.get("rollout_count", 1)) for metric in metrics))
                mirror_rollout_count = int(sum(int(metric.get("mirror_rollout_count", 0)) for metric in metrics))
                mirror_individual_count = int(
                    sum(
                        int(bool(metric.get("mirrored", 0)) or bool(metric.get("evaluate_both_mirrors", 0)))
                        for metric in metrics
                    )
                )
                finish_count = int(np.sum(finished_values > 0))
                crash_count = int(np.sum(crash_values > 0))
                timeout_count = int(np.sum(timeout_values > 0))
                if (
                    mutation_decay_trigger == "first_finish"
                    and not mutation_decay_active
                    and finish_count > 0
                ):
                    mutation_decay_active = True
                    mutation_decay_trigger_generation = int(generation)
                    remaining_decay_generations = max(0, int(args.generations) - int(generation))
                    effective_mutation_prob_decay = finish_triggered_decay_factor(
                        current_mutation_prob,
                        mutation_prob_min,
                        remaining_decay_generations,
                    )
                    effective_mutation_sigma_decay = finish_triggered_decay_factor(
                        current_mutation_sigma,
                        mutation_sigma_min,
                        remaining_decay_generations,
                    )
                generalization_summary = {
                    "generalization_test_map": generalization_test_map if generalization_test_enabled else "",
                    "generalization_best_finished": "",
                    "generalization_best_crashes": "",
                    "generalization_best_progress": "",
                    "generalization_best_block_progress": "",
                    "generalization_best_time": "",
                    "generalization_best_distance": "",
                }
                if (
                    generalization_env is not None
                    and generalization_writer is not None
                    and generation % generalization_test_every == 0
                ):
                    top_k = min(generalization_test_top_k, len(population))
                    for train_rank, individual in enumerate(population[:top_k], start=1):
                        original_index = int(original_index_by_id.get(id(individual), -1))
                        test_seed = int(args.seed) + 2_000_000 + generation * 10_000 + train_rank
                        test_metrics = generalization_env.rollout_policy(
                            NumpyPolicyView(individual.policy),
                            reset_seed=test_seed,
                            mirrored=False,
                        )
                        generalization_writer.writerow(
                            {
                                "generation": int(generation),
                                "train_rank": int(train_rank),
                                "original_index": original_index,
                                "test_map": generalization_test_map,
                                "train_finished": int(individual.finished),
                                "train_crashes": int(individual.crashes),
                                "train_time": float(individual.time),
                                "train_progress": float(individual.dense_progress),
                                "train_block_progress": float(individual.discrete_progress),
                                "train_distance": float(individual.distance),
                                "test_finished": int(test_metrics.get("finished", 0)),
                                "test_crashes": int(test_metrics.get("crashes", 0)),
                                "test_timeout": int(test_metrics.get("timeout", 0)),
                                "test_time": float(test_metrics.get("time", 0.0)),
                                "test_progress": float(test_metrics.get("progress", 0.0)),
                                "test_block_progress": float(
                                    test_metrics.get("block_progress", test_metrics.get("discrete_progress", 0.0))
                                ),
                                "test_distance": float(test_metrics.get("distance", 0.0)),
                                "test_steps": int(test_metrics.get("steps", 0)),
                                "test_reward": float(test_metrics.get("reward", 0.0)),
                                "test_fitness": float(test_metrics.get("fitness", 0.0)),
                                "test_physics_tick_mean": float(test_metrics.get("physics_tick_mean", 1.0)),
                                "test_physics_hz_mean": float(test_metrics.get("physics_hz_mean", 100.0)),
                                "test_physics_delay_norm_mean": float(
                                    test_metrics.get("physics_delay_norm_mean", 0.0)
                                ),
                                "test_terminated": int(bool(test_metrics.get("terminated", False))),
                                "test_truncated": int(bool(test_metrics.get("truncated", False))),
                            }
                        )
                        if train_rank == 1:
                            generalization_summary.update(
                                {
                                    "generalization_best_finished": int(test_metrics.get("finished", 0)),
                                    "generalization_best_crashes": int(test_metrics.get("crashes", 0)),
                                    "generalization_best_progress": float(test_metrics.get("progress", 0.0)),
                                    "generalization_best_block_progress": float(
                                        test_metrics.get("block_progress", test_metrics.get("discrete_progress", 0.0))
                                    ),
                                    "generalization_best_time": float(test_metrics.get("time", 0.0)),
                                    "generalization_best_distance": float(test_metrics.get("distance", 0.0)),
                                }
                            )
                    generalization_handle.flush()
                virtual_time_sum = float(np.sum(time_values))
                virtual_steps_sum = int(np.sum(step_values))
                cumulative_virtual_time += virtual_time_sum
                cumulative_virtual_steps += virtual_steps_sum
                generation_wall_seconds = float(time.perf_counter() - generation_wall_start)
                cumulative_wall_seconds = float(time.perf_counter() - training_wall_start)
                row = {
                    "generation": generation,
                    "best_fitness": best_fitness_for_plot,
                    "best_progress": float(np.max(progress_values)),
                    "best_block_progress": float(best.discrete_progress),
                    "best_ranking_progress": float(best.ranking_progress()),
                    "best_time": float(best.time),
                    "best_finished": int(best.finished),
                    "best_crashes": int(best.crashes),
                    "best_distance": float(best.distance),
                    "best_steps": int(getattr(best, "evaluation_steps", 0)),
                    "best_reward": float(np.max(reward_values)),
                    "best_ranking_key": json.dumps([float(value) for value in best.ranking_key()]),
                    "current_mutation_prob": float(current_mutation_prob),
                    "current_mutation_sigma": float(current_mutation_sigma),
                    "mutation_decay_trigger": mutation_decay_trigger,
                    "mutation_decay_active": int(bool(mutation_decay_active)),
                    "mutation_decay_trigger_generation": (
                        "" if mutation_decay_trigger_generation is None else int(mutation_decay_trigger_generation)
                    ),
                    "effective_mutation_prob_decay": float(effective_mutation_prob_decay),
                    "effective_mutation_sigma_decay": float(effective_mutation_sigma_decay),
                    "cached_evaluations": int(cached_evaluations),
                    "evaluated_count": int(len(population) - cached_evaluations),
                    "rollout_count": int(rollout_count_sum),
                    "mirror_rollout_count": int(mirror_rollout_count),
                    "mirror_individual_count": int(mirror_individual_count),
                    "population_size": int(len(population)),
                    "finish_count": finish_count,
                    "crash_count": crash_count,
                    "timeout_count": timeout_count,
                    "terminated_count": int(sum(bool(metric.get("terminated", False)) for metric in metrics)),
                    "truncated_count": int(sum(bool(metric.get("truncated", False)) for metric in metrics)),
                    "virtual_time_sum": virtual_time_sum,
                    "cumulative_virtual_time": cumulative_virtual_time,
                    "virtual_steps_sum": virtual_steps_sum,
                    "cumulative_virtual_steps": int(cumulative_virtual_steps),
                    "generation_wall_seconds": generation_wall_seconds,
                    "cumulative_wall_seconds": cumulative_wall_seconds,
                    "mean_progress": float(np.mean(progress_values)),
                    "mean_block_progress": float(np.mean(block_progress_values)),
                    "mean_ranking_progress": float(np.mean(ranking_progress_values)),
                    "mean_reward": float(np.mean(reward_values)),
                    "mean_time": float(np.mean(time_values)),
                    "mean_physics_tick_count": float(np.mean(physics_tick_values)),
                    "mean_physics_hz": float(np.mean(physics_hz_values)),
                    "mean_physics_delay_norm": float(np.mean(physics_delay_values)),
                }
                row.update(generalization_summary)
                row.update(metric_stats("progress", progress_values))
                row.update(metric_stats("block_progress", block_progress_values))
                row.update(metric_stats("ranking_progress", ranking_progress_values))
                row.update(metric_stats("time", time_values))
                row.update(metric_stats("distance", distance_values))
                row.update(metric_stats("reward", reward_values))
                row.update(metric_stats("fitness", fitness_values))
                row.update(metric_stats("steps", step_values))
                row.update(metric_stats("physics_tick", physics_tick_values))
                row.update(metric_stats("physics_hz", physics_hz_values))
                row.update(metric_stats("physics_delay_norm", physics_delay_values))
                writer.writerow(row)
                handle.flush()

                for rank, individual in enumerate(population, start=1):
                    fitness_for_plot = (
                        float(individual.fitness)
                        if individual.fitness is not None and np.isfinite(individual.fitness)
                        else float(individual.compute_scalar_fitness())
                    )
                    timeout = int(int(individual.finished) <= 0 and int(individual.crashes) <= 0)
                    individual_writer.writerow(
                        {
                            "generation": generation,
                            "rank": rank,
                            "original_index": int(original_index_by_id.get(id(individual), -1)),
                            "cached": int(bool(cached_by_id.get(id(individual), False))),
                            "evaluation_context": str(getattr(individual, "evaluation_context", "")),
                            "mirrored": int("mirror=1" in str(getattr(individual, "evaluation_context", ""))),
                            "evaluate_both_mirrors": int("both=1" in str(getattr(individual, "evaluation_context", ""))),
                            "is_elite": int(rank <= int(args.elite_count)),
                            "is_parent": int(rank <= int(args.parent_count)),
                            "fitness": fitness_for_plot,
                            "ranking_key": json.dumps([float(value) for value in individual.ranking_key()]),
                            "ranking_progress": float(individual.ranking_progress()),
                            "progress": float(individual.dense_progress),
                            "block_progress": float(individual.discrete_progress),
                            "time": float(individual.time),
                            "finished": int(individual.finished),
                            "crashes": int(individual.crashes),
                            "timeout": timeout,
                            "distance": float(individual.distance),
                            "reward": float(individual.reward),
                            "steps": int(getattr(individual, "evaluation_steps", 0)),
                            "physics_tick_mean": float(getattr(individual, "physics_tick_mean", 0.0)),
                            "physics_hz_mean": float(getattr(individual, "physics_hz_mean", 0.0)),
                            "physics_delay_norm_mean": float(getattr(individual, "physics_delay_norm_mean", 0.0)),
                            "terminated": int(bool(getattr(individual, "evaluation_terminated", False))),
                            "truncated": int(bool(getattr(individual, "evaluation_truncated", False))),
                        }
                    )
                individual_handle.flush()

                if trajectory_logger is not None and trajectory_env is not None:
                    for rank, individual in enumerate(population, start=1):
                        original_index = int(original_index_by_id.get(id(individual), -1))
                        if not should_log_trajectory(
                            mode=args.trajectory_log_mode,
                            generation=generation,
                            rank=rank,
                            finished=int(individual.finished),
                            final_generation=int(args.generations),
                            top_k=int(args.trajectory_top_k),
                        ):
                            continue
                        replay_seed = int(args.seed) + 1_000_000 + generation * 10_000 + max(0, original_index)
                        replay_metrics = trajectory_env.rollout_policy(
                            NumpyPolicyView(individual.policy),
                            collect_trajectory=True,
                            trajectory_log_actions=bool(args.trajectory_log_actions),
                            reset_seed=replay_seed,
                            mirrored=(
                                "mirror=1" in str(getattr(individual, "evaluation_context", ""))
                                and "both=1" not in str(getattr(individual, "evaluation_context", ""))
                            ),
                        )
                        trajectory_logger.save(
                            generation=generation,
                            rank=rank,
                            original_index=original_index,
                            finished=int(individual.finished),
                            crashes=int(individual.crashes),
                            time_value=float(individual.time),
                            dense_progress=float(individual.dense_progress),
                            trajectory=replay_metrics.get("trajectory", []),
                            log_actions=bool(args.trajectory_log_actions),
                        )
                print(
                    f"gen={generation:04d} best_fit={row['best_fitness']:.2f} "
                    f"best_prog={row['best_progress']:.2f}% "
                    f"block={row['best_block_progress']:.2f}% "
                    f"best_rank={row['best_ranking_progress']:.2f}% "
                    f"best_time={row['best_time']:.2f}s mean_prog={row['mean_progress']:.2f}% "
                    f"mut_p={current_mutation_prob:.4f} sigma={current_mutation_sigma:.4f} "
                    f"decay={'active' if mutation_decay_active else 'waiting'} "
                    f"mirror_rollouts={mirror_rollout_count} cached={cached_evaluations}"
                )

                if generation < args.generations:
                    elites = [individual.copy() for individual in population[: args.elite_count]]
                    if not elite_cache_enabled:
                        for elite in elites:
                            elite.invalidate_evaluation()
                    parents = [individual.copy() for individual in population[: args.parent_count]]
                    children = make_child_pool(parents, args.population_size - len(elites))
                    for child in children:
                        child.mutate(current_mutation_prob, current_mutation_sigma)
                    population = elites + children
                    if mutation_decay_active:
                        current_mutation_prob = max(
                            float(mutation_prob_min),
                            float(current_mutation_prob) * float(effective_mutation_prob_decay),
                        )
                        current_mutation_sigma = max(
                            float(mutation_sigma_min),
                            float(current_mutation_sigma) * float(effective_mutation_sigma_decay),
                        )
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
    print(f"Saved individual metrics to {individual_csv_path}")

    if args.render_best:
        from Experiments.tm2d_viewer import TM2DViewer

        viewer_env = TM2DSimEnv(
            map_name=args.map_name,
            max_time=args.max_time,
            reward_config=reward_config,
            physics_config=physics_config,
            seed=args.seed + 1,
            collision_mode=args.collision_mode,
            collision_distance_threshold=args.collision_distance_threshold,
            vertical_mode=args.vertical_mode,
            multi_surface_mode=args.multi_surface_mode,
            binary_gas_brake=not args.continuous_gas_brake,
            max_touches=args.max_touches,
            collision_bounce_speed_retention=args.collision_bounce_speed_retention,
            collision_bounce_backoff=args.collision_bounce_backoff,
            touch_release_clearance_threshold=args.touch_release_clearance_threshold,
            mask_physics_delay_observation=args.mask_physics_delay_observation,
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
