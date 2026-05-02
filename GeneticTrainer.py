import csv
import glob
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from Car import Car
from NeuralPolicy import NeuralPolicy
from Individual import Individual
from ObservationEncoder import ObservationEncoder
from Experiments.multiobjective import (
    objective_names_for_mode,
    objectives_from_metrics,
    pareto_order,
)
from RankingKey import canonical_ranking_key_expression


HiddenDims = Union[int, Sequence[int]]
HiddenActivations = Union[str, Sequence[str]]


def normalize_hidden_dims(hidden_dim: HiddenDims) -> Tuple[int, ...]:
    if isinstance(hidden_dim, (tuple, list)):
        dims = tuple(int(dim) for dim in hidden_dim)
    else:
        dims = (int(hidden_dim),)
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("hidden_dim must contain positive integers.")
    return dims


def hidden_dims_tag(hidden_dims: HiddenDims) -> str:
    dims = normalize_hidden_dims(hidden_dims)
    return "x".join(str(dim) for dim in dims)


def safe_ranking_key_tag(ranking_key: str) -> str:
    tag = canonical_ranking_key_expression(ranking_key)
    replacements = {
        "(": "",
        ")": "",
        ",": "",
        "+": "",
        "-": "neg_",
        " ": "_",
    }
    for old, new in replacements.items():
        tag = tag.replace(old, new)
    while "__" in tag:
        tag = tag.replace("__", "_")
    return tag.strip("_")


def _parse_name_list(value: str) -> List[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _resolve_objective_indices(available_names: Sequence[str], value: str) -> List[int]:
    names = list(available_names)
    requested = _parse_name_list(value)
    if not requested or [name.lower() for name in requested] == ["auto"]:
        return list(range(len(names)))
    missing = [name for name in requested if name not in names]
    if missing:
        raise ValueError(f"Unknown objective names: {missing}. Available: {names}")
    return [names.index(name) for name in requested]


def safe_objective_tag(value: str) -> str:
    names = _parse_name_list(value)
    if not names or [name.lower() for name in names] == ["auto"]:
        return "auto"
    tag = "_".join(names)
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return "".join(ch if ch in allowed else "_" for ch in tag).strip("_")


def normalize_hidden_activations(
    hidden_activation: HiddenActivations,
    num_hidden_layers: int,
) -> Tuple[str, ...]:
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive.")

    if isinstance(hidden_activation, str):
        activations = [hidden_activation]
    else:
        activations = list(hidden_activation)

    if not activations:
        raise ValueError("hidden_activation must contain at least one activation name.")

    normalized: List[str] = []
    for activation in activations:
        value = str(activation).strip().lower()
        if value == "tann":
            value = "tanh"
        if value not in {"tanh", "relu", "sigmoid"}:
            raise ValueError(
                f"Unsupported hidden activation '{activation}'. "
                "Use 'tanh', 'relu', or 'sigmoid'."
            )
        normalized.append(value)

    if len(normalized) == 1 and num_hidden_layers > 1:
        normalized = normalized * num_hidden_layers
    if len(normalized) != num_hidden_layers:
        raise ValueError(
            "hidden_activation must provide either one activation or exactly "
            f"{num_hidden_layers} activations, got {len(normalized)}."
        )
    return tuple(normalized)


class TrainingLogger:
    INDIVIDUAL_HEADERS = [
        "timestamp_utc",
        "generation",
        "individual_index",
        "finished",
        "crashes",
        "timeout",
        "progress",
        "discrete_progress",
        "dense_progress",
        "ranking_progress",
        "distance",
        "time",
        "reward",
        "fitness",
        "ranking_key",
        "selection_mode",
        "selection_rank",
        "selection_crowding",
        "selection_objectives",
        "selection_objective_names",
        "evaluation_steps",
        "evaluation_terminated",
        "evaluation_truncated",
        "evaluation_valid",
    ]

    SUMMARY_HEADERS = [
        "timestamp_utc",
        "generation",
        "cached_evaluations",
        "evaluated_count",
        "population_size",
        "dist_avg",
        "dist_best_gen",
        "dist_best_global",
        "time_avg",
        "time_best_gen",
        "time_best_global",
        "finish_count",
        "crash_count",
        "timeout_count",
        "finish_rate",
        "crash_rate",
        "timeout_rate",
        "best_finished",
        "best_crashes",
        "best_progress",
        "best_dense_progress",
        "best_ranking_progress",
        "best_ranking_key",
        "mean_discrete_progress",
        "mean_dense_progress",
        "mean_ranking_progress",
        "std_dense_progress",
        "std_ranking_progress",
        "best_distance",
        "best_time",
        "best_reward",
        "best_fitness",
        "selection_mode",
        "front0_size",
        "best_selection_rank",
        "best_selection_crowding",
        "best_selection_objectives",
        "selection_objective_names",
        "virtual_time_sum",
        "virtual_distance_sum",
        "evaluation_steps_sum",
    ]

    def __init__(
        self,
        base_dir: str = "logs/ga_runs",
        run_name: Optional[str] = None,
        run_dir: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        if run_dir is None:
            os.makedirs(base_dir, exist_ok=True)
            if run_name is None:
                run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(base_dir, self._sanitize_name(run_name))
            os.makedirs(run_dir, exist_ok=False)
        else:
            os.makedirs(run_dir, exist_ok=True)

        self.run_dir = run_dir
        self.config_path = os.path.join(run_dir, "config.json")
        self.individual_metrics_path = os.path.join(run_dir, "individual_metrics.csv")
        self.generation_summary_path = os.path.join(run_dir, "generation_summary.csv")
        self.checkpoints_dir = os.path.join(run_dir, "checkpoints")
        self.best_individual_path = os.path.join(run_dir, "best_individual.npz")
        self.best_individual_model_path = os.path.join(run_dir, "best_individual.pt")
        self.global_best_path = os.path.join(run_dir, "global_best.npz")
        self.global_best_model_path = os.path.join(run_dir, "global_best.pt")
        self.final_population_path = os.path.join(run_dir, "final_population.npz")
        self.final_population_model_path = os.path.join(run_dir, "final_population_best.pt")

        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self._init_csv_if_missing(self.individual_metrics_path, self.INDIVIDUAL_HEADERS)
        self._init_csv_if_missing(self.generation_summary_path, self.SUMMARY_HEADERS)

        if config is not None:
            self.write_config(config, merge=True)

    @staticmethod
    def _sanitize_name(value: str) -> str:
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        return "".join(ch if ch in allowed else "_" for ch in value)

    @staticmethod
    def _timestamp_utc() -> str:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _init_csv_if_missing(path: str, headers: List[str]) -> None:
        if os.path.exists(path):
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    def write_config(self, updates: Dict, merge: bool = True) -> None:
        data: Dict = {}
        if merge and os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        data.update(updates)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=True)

    def log_individual_batch(self, rows: List[Dict]) -> None:
        if not rows:
            return
        with open(self.individual_metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.INDIVIDUAL_HEADERS)
            writer.writerows(
                {key: row.get(key, "") for key in self.INDIVIDUAL_HEADERS}
                for row in rows
            )

    def log_generation_summary(self, row: Dict) -> None:
        with open(self.generation_summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.SUMMARY_HEADERS)
            writer.writerow({key: row.get(key, "") for key in self.SUMMARY_HEADERS})

    def save_population_checkpoint(
        self,
        population: List[Individual],
        generation: int,
        obs_dim: int,
        hidden_dim: HiddenDims,
        act_dim: int,
        best_individual: Optional[Individual] = None,
        current_mutation_prob: Optional[float] = None,
        current_mutation_sigma: Optional[float] = None,
        observation_layout: Optional[Sequence[str]] = None,
        vertical_mode: bool = False,
    ) -> str:
        checkpoint_path = os.path.join(
            self.checkpoints_dir, f"population_gen_{generation:04d}.npz"
        )

        genomes = np.stack([ind.genome for ind in population]).astype(np.float32)
        progresses = np.array(
            [float(ind.discrete_progress) for ind in population], dtype=np.float32
        )
        dense_progresses = np.array(
            [float(ind.dense_progress) for ind in population], dtype=np.float32
        )
        times = np.array([float(ind.time) for ind in population], dtype=np.float32)
        finisheds = np.array([int(ind.finished) for ind in population], dtype=np.int32)
        crashes = np.array([int(ind.crashes) for ind in population], dtype=np.int32)
        distances = np.array([float(ind.distance) for ind in population], dtype=np.float32)
        rewards = np.array([float(ind.reward) for ind in population], dtype=np.float32)
        evaluation_steps = np.array(
            [int(ind.evaluation_steps) for ind in population], dtype=np.int32
        )
        evaluation_terminated = np.array(
            [int(bool(ind.evaluation_terminated)) for ind in population], dtype=np.int32
        )
        evaluation_truncated = np.array(
            [int(bool(ind.evaluation_truncated)) for ind in population], dtype=np.int32
        )
        evaluation_valid = np.array(
            [int(bool(ind.evaluation_valid)) for ind in population], dtype=np.int32
        )
        fitnesses = np.array(
            [np.nan if ind.fitness is None else float(ind.fitness) for ind in population],
            dtype=np.float32,
        )

        payload = dict(
            generation=np.array([generation], dtype=np.int32),
            genomes=genomes,
            progresses=progresses,
            dense_progresses=dense_progresses,
            times=times,
            finisheds=finisheds,
            crashes=crashes,
            distances=distances,
            rewards=rewards,
            evaluation_steps=evaluation_steps,
            evaluation_terminated=evaluation_terminated,
            evaluation_truncated=evaluation_truncated,
            evaluation_valid=evaluation_valid,
            fitnesses=fitnesses,
            obs_dim=np.array([obs_dim], dtype=np.int32),
            act_dim=np.array([act_dim], dtype=np.int32),
            hidden_dims=np.array(normalize_hidden_dims(hidden_dim), dtype=np.int32),
            vertical_mode=np.array([int(bool(vertical_mode))], dtype=np.int32),
        )
        if observation_layout is not None:
            payload["observation_layout"] = np.array(list(observation_layout), dtype=str)
        if current_mutation_prob is not None:
            payload["current_mutation_prob"] = np.array(
                [float(current_mutation_prob)],
                dtype=np.float32,
            )
        if current_mutation_sigma is not None:
            payload["current_mutation_sigma"] = np.array(
                [float(current_mutation_sigma)],
                dtype=np.float32,
            )
        hidden_dims = normalize_hidden_dims(hidden_dim)
        if len(hidden_dims) == 1:
            payload["hidden_dim"] = np.array([hidden_dims[0]], dtype=np.int32)
        if population:
            hidden_activations = tuple(population[0].policy.hidden_activations)
            payload.update(
                action_scale=population[0].policy.action_scale.detach().cpu().numpy().astype(np.float32),
                action_mode=np.array([population[0].policy.action_mode]),
                hidden_activations=np.array(hidden_activations, dtype=str),
            )
            if len(hidden_activations) == 1:
                payload["hidden_activation"] = np.array([hidden_activations[0]], dtype=str)

        if best_individual is not None:
            payload.update(
                best_genome=best_individual.genome.astype(np.float32),
                best_progress=np.array([float(best_individual.discrete_progress)], dtype=np.float32),
                best_dense_progress=np.array([float(best_individual.dense_progress)], dtype=np.float32),
                best_time=np.array([float(best_individual.time)], dtype=np.float32),
                best_finished=np.array([int(best_individual.finished)], dtype=np.int32),
                best_crashes=np.array([int(best_individual.crashes)], dtype=np.int32),
                best_distance=np.array([float(best_individual.distance)], dtype=np.float32),
                best_reward=np.array([float(best_individual.reward)], dtype=np.float32),
                best_evaluation_steps=np.array([int(best_individual.evaluation_steps)], dtype=np.int32),
                best_evaluation_terminated=np.array(
                    [int(bool(best_individual.evaluation_terminated))],
                    dtype=np.int32,
                ),
                best_evaluation_truncated=np.array(
                    [int(bool(best_individual.evaluation_truncated))],
                    dtype=np.int32,
                ),
                best_fitness=np.array(
                    [np.nan if best_individual.fitness is None else float(best_individual.fitness)],
                    dtype=np.float32,
                ),
            )

        np.savez(checkpoint_path, **payload)
        return checkpoint_path

    def save_final_population(
        self,
        population: List[Individual],
        generation: int,
        obs_dim: int,
        hidden_dim: HiddenDims,
        act_dim: int,
        best_individual: Optional[Individual] = None,
        observation_layout: Optional[Sequence[str]] = None,
        vertical_mode: bool = False,
    ) -> str:
        final_path = self.final_population_path
        genomes = np.stack([ind.genome for ind in population]).astype(np.float32)
        progresses = np.array(
            [float(ind.discrete_progress) for ind in population], dtype=np.float32
        )
        dense_progresses = np.array(
            [float(ind.dense_progress) for ind in population], dtype=np.float32
        )
        times = np.array([float(ind.time) for ind in population], dtype=np.float32)
        finisheds = np.array([int(ind.finished) for ind in population], dtype=np.int32)
        crashes = np.array([int(ind.crashes) for ind in population], dtype=np.int32)
        distances = np.array([float(ind.distance) for ind in population], dtype=np.float32)
        rewards = np.array([float(ind.reward) for ind in population], dtype=np.float32)
        evaluation_steps = np.array(
            [int(ind.evaluation_steps) for ind in population], dtype=np.int32
        )
        evaluation_terminated = np.array(
            [int(bool(ind.evaluation_terminated)) for ind in population], dtype=np.int32
        )
        evaluation_truncated = np.array(
            [int(bool(ind.evaluation_truncated)) for ind in population], dtype=np.int32
        )
        evaluation_valid = np.array(
            [int(bool(ind.evaluation_valid)) for ind in population], dtype=np.int32
        )
        fitnesses = np.array(
            [np.nan if ind.fitness is None else float(ind.fitness) for ind in population],
            dtype=np.float32,
        )

        payload = dict(
            generation=np.array([generation], dtype=np.int32),
            genomes=genomes,
            progresses=progresses,
            dense_progresses=dense_progresses,
            times=times,
            finisheds=finisheds,
            crashes=crashes,
            distances=distances,
            rewards=rewards,
            evaluation_steps=evaluation_steps,
            evaluation_terminated=evaluation_terminated,
            evaluation_truncated=evaluation_truncated,
            evaluation_valid=evaluation_valid,
            fitnesses=fitnesses,
            obs_dim=np.array([obs_dim], dtype=np.int32),
            act_dim=np.array([act_dim], dtype=np.int32),
            hidden_dims=np.array(normalize_hidden_dims(hidden_dim), dtype=np.int32),
            vertical_mode=np.array([int(bool(vertical_mode))], dtype=np.int32),
        )
        if observation_layout is not None:
            payload["observation_layout"] = np.array(list(observation_layout), dtype=str)
        hidden_dims = normalize_hidden_dims(hidden_dim)
        if len(hidden_dims) == 1:
            payload["hidden_dim"] = np.array([hidden_dims[0]], dtype=np.int32)
        if population:
            hidden_activations = tuple(population[0].policy.hidden_activations)
            payload.update(
                action_scale=population[0].policy.action_scale.detach().cpu().numpy().astype(np.float32),
                action_mode=np.array([population[0].policy.action_mode]),
                hidden_activations=np.array(hidden_activations, dtype=str),
            )
            if len(hidden_activations) == 1:
                payload["hidden_activation"] = np.array([hidden_activations[0]], dtype=str)
        if best_individual is not None:
            payload.update(best_genome=best_individual.genome.astype(np.float32))
        np.savez(final_path, **payload)
        if best_individual is not None:
            best_individual.policy.save(
                self.final_population_model_path,
                extra=self._policy_extra(
                    best_individual,
                    generation,
                    observation_layout=observation_layout,
                    vertical_mode=vertical_mode,
                ),
            )
        return final_path

    def save_best_individual(
        self,
        best: Individual,
        generation: Optional[int] = None,
        observation_layout: Optional[Sequence[str]] = None,
        vertical_mode: bool = False,
    ) -> str:
        payload = dict(
            genome=best.genome.astype(np.float32),
            discrete_progress=float(best.discrete_progress),
            dense_progress=float(best.dense_progress),
            time=float(best.time),
            finished=int(best.finished),
            crashes=int(best.crashes),
            distance=float(best.distance),
            reward=float(best.reward),
            ranking_progress=float(best.ranking_progress()),
            ranking_key=np.array([json.dumps([float(value) for value in best.ranking_key()])]),
            selection_mode=np.array([getattr(best, "selection_mode", "")]),
            selection_rank=np.array(
                [-1 if best.selection_rank is None else int(best.selection_rank)],
                dtype=np.int32,
            ),
            selection_crowding=np.array([float(best.selection_crowding)], dtype=np.float32),
            selection_objectives=np.array(
                [
                    json.dumps(
                        [] if best.selection_objectives is None else list(best.selection_objectives)
                    )
                ]
            ),
            selection_objective_names=np.array(
                [json.dumps(list(best.selection_objective_names))]
            ),
            evaluation_steps=int(best.evaluation_steps),
            evaluation_terminated=int(bool(best.evaluation_terminated)),
            evaluation_truncated=int(bool(best.evaluation_truncated)),
            evaluation_valid=int(bool(best.evaluation_valid)),
            fitness=np.nan if best.fitness is None else float(best.fitness),
        )
        if generation is not None:
            payload["generation"] = np.array([int(generation)], dtype=np.int32)
        payload["saved_utc"] = np.array([self._timestamp_utc()])

        # Primary continuously-updated artifact requested for long-running training.
        np.savez(self.global_best_path, **payload)
        # Backward compatibility for older tooling.
        np.savez(self.best_individual_path, **payload)
        extra = self._policy_extra(
            best,
            generation,
            observation_layout=observation_layout,
            vertical_mode=vertical_mode,
        )
        best.policy.save(self.global_best_model_path, extra=extra)
        best.policy.save(self.best_individual_model_path, extra=extra)
        return self.global_best_path

    @staticmethod
    def _policy_extra(
        best: Individual,
        generation: Optional[int],
        observation_layout: Optional[Sequence[str]] = None,
        vertical_mode: bool = False,
    ) -> Dict:
        extra: Dict = dict(
            discrete_progress=float(best.discrete_progress),
            dense_progress=float(best.dense_progress),
            time=float(best.time),
            finished=int(best.finished),
            crashes=int(best.crashes),
            distance=float(best.distance),
            reward=float(best.reward),
            ranking_progress=float(best.ranking_progress()),
            ranking_key=json.dumps([float(value) for value in best.ranking_key()]),
            selection_rank=best.selection_rank,
            selection_crowding=float(best.selection_crowding),
            selection_objectives=(
                []
                if best.selection_objectives is None
                else list(best.selection_objectives)
            ),
            selection_objective_names=list(best.selection_objective_names),
            evaluation_steps=int(best.evaluation_steps),
            evaluation_terminated=bool(best.evaluation_terminated),
            evaluation_truncated=bool(best.evaluation_truncated),
            evaluation_valid=bool(best.evaluation_valid),
            fitness=np.nan if best.fitness is None else float(best.fitness),
            observation_layout=list(
                observation_layout
                if observation_layout is not None
                else ObservationEncoder.feature_names(vertical_mode=vertical_mode)
            ),
            vertical_mode=bool(vertical_mode),
        )
        if generation is not None:
            extra["generation"] = int(generation)
        return extra


class GeneticTrainer:
    def __init__(
        self,
        env,
        obs_dim: int,
        hidden_dim: HiddenDims = 16,
        act_dim: int = 3,
        pop_size: int = 16,
        max_steps: Optional[int] = 2000,
        policy_action_scale: Optional[np.ndarray] = None,
        policy_action_mode: str = "delta",
        hidden_activation: HiddenActivations = "tanh",
        target_steer_deadzone: float = 0.0,
        logger: Optional[TrainingLogger] = None,
        observation_layout: Optional[Sequence[str]] = None,
        selection_mode: str = "lexicographic",
        moo_objective_mode: str = "lexicographic_primitives",
        moo_objective_subset: str = "auto",
        moo_objective_priority: str = "auto",
        pareto_tiebreak: str = "priority",
    ) -> None:
        self.env = env
        self.obs_dim = obs_dim
        self.hidden_dims = normalize_hidden_dims(hidden_dim)
        self.hidden_dim = self.hidden_dims[0] if len(self.hidden_dims) == 1 else self.hidden_dims
        self.act_dim = act_dim
        self.pop_size = pop_size
        self.max_steps = None if max_steps is None else int(max_steps)
        self.policy_action_scale = None if policy_action_scale is None else np.asarray(policy_action_scale, dtype=np.float32)
        self.policy_action_mode = str(policy_action_mode).strip().lower()
        self.hidden_activations = normalize_hidden_activations(
            hidden_activation=hidden_activation,
            num_hidden_layers=len(self.hidden_dims),
        )
        self.hidden_activation = (
            self.hidden_activations[0]
            if len(self.hidden_activations) == 1
            else list(self.hidden_activations)
        )
        self.vertical_mode = bool(
            getattr(getattr(env, "obs_encoder", None), "vertical_mode", False)
        )
        self.observation_layout = list(
            observation_layout
            if observation_layout is not None
            else ObservationEncoder.feature_names(vertical_mode=self.vertical_mode)
        )
        self.target_steer_deadzone = float(target_steer_deadzone)
        self.logger = logger
        self.selection_mode = str(selection_mode).strip().lower()
        if self.selection_mode in {"ranking", "lexicographic"}:
            self.selection_mode = "lexicographic"
        if self.selection_mode not in {"lexicographic", "pareto"}:
            raise ValueError("selection_mode must be 'lexicographic' or 'pareto'.")
        self.moo_objective_mode = str(moo_objective_mode).strip().lower()
        self.moo_available_objective_names = objective_names_for_mode(self.moo_objective_mode)
        self.moo_objective_subset_indices = _resolve_objective_indices(
            self.moo_available_objective_names,
            moo_objective_subset,
        )
        self.moo_objective_names = [
            self.moo_available_objective_names[index]
            for index in self.moo_objective_subset_indices
        ]
        self.moo_priority_indices = _resolve_objective_indices(
            self.moo_objective_names,
            moo_objective_priority,
        )
        self.pareto_tiebreak = str(pareto_tiebreak).strip().lower()
        if self.pareto_tiebreak not in {"priority", "crowding"}:
            raise ValueError("pareto_tiebreak must be 'priority' or 'crowding'.")
        self._last_front0_size: Optional[int] = None

        self.population: List[Individual] = [
            Individual(
                obs_dim,
                self.hidden_dims,
                act_dim,
                action_scale=self.policy_action_scale,
                action_mode=self.policy_action_mode,
                hidden_activation=self.hidden_activations,
            )
            for _ in range(pop_size)
        ]
        self.best_individual: Optional[Individual] = None
        self.last_cached_evaluations: int = 0

        # Počet už vyhodnotených generácií.
        self.generation: int = 0

        # Ak načítame checkpoint vyhodnotenej generácie, prvý krok run() má vytvoriť ďalšiu.
        self._loaded_checkpoint_evaluated: bool = False
        self._pending_initial_downselect: bool = False
        self._checkpoint_current_mutation_prob: Optional[float] = None
        self._checkpoint_current_mutation_sigma: Optional[float] = None

    @staticmethod
    def _outcome_status_text(finished: int, crashes: int) -> str:
        if int(finished) > 0:
            return "FINISH"
        if int(crashes) > 0:
            return f"CRASHx{int(crashes)}"
        return "TIMEOUT"

    def _wait_for_positive_game_time(
        self,
        observation: np.ndarray,
        info: Dict,
        timeout_seconds: float = 5.0,
    ) -> Tuple[np.ndarray, Dict]:
        if float(info.get("time", 0.0)) > 0.0:
            return observation, info

        deadline = datetime.now().timestamp() + float(timeout_seconds)
        last_info = dict(info)
        while datetime.now().timestamp() < deadline:
            distances, instructions, info = self.env.observation_info()
            last_info = info
            if float(info.get("time", 0.0)) > 0.0:
                observation = self.env.build_observation(
                    distances=distances,
                    instructions=instructions,
                    info=info,
                )
                self.env.previous_observation_info = (distances, instructions, info)
                self.env.previous_observation = observation
                return observation, info
        return observation, last_info

    def evaluate_individual(
        self,
        individual: Individual,
        index: Optional[int] = None,
        total: Optional[int] = None,
        verbose: bool = False,
        mirrored: bool = False,
        evaluate_both_mirrors: bool = False,
    ) -> float:
        if evaluate_both_mirrors:
            normal_metrics = self._evaluate_single_rollout(
                individual=individual,
                mirrored=False,
            )
            mirrored_metrics = self._evaluate_single_rollout(
                individual=individual,
                mirrored=True,
            )
            rollout_metrics = [normal_metrics, mirrored_metrics]
            mean_discrete_progress = float(
                np.mean([metrics["discrete_progress"] for metrics in rollout_metrics])
            )
            mean_dense_progress = float(
                np.mean([metrics["dense_progress"] for metrics in rollout_metrics])
            )
            mean_time = float(np.mean([metrics["time"] for metrics in rollout_metrics]))
            mean_distance = float(
                np.mean([metrics["distance"] for metrics in rollout_metrics])
            )
            mean_reward = float(np.mean([metrics["reward"] for metrics in rollout_metrics]))
            mean_fitness = float(np.mean([metrics["fitness"] for metrics in rollout_metrics]))
            mean_steps = int(round(float(np.mean([metrics["steps"] for metrics in rollout_metrics]))))
            representative_finished = int(
                min(int(metrics["finished"]) for metrics in rollout_metrics)
            )
            representative_crashes = int(
                round(float(np.mean([metrics["crashes"] for metrics in rollout_metrics])))
            )
            representative_terminated = bool(
                any(bool(metrics["terminated"]) for metrics in rollout_metrics)
            )
            representative_truncated = bool(
                any(bool(metrics["truncated"]) for metrics in rollout_metrics)
            )

            individual.discrete_progress = mean_discrete_progress
            individual.dense_progress = mean_dense_progress
            individual.time = mean_time
            individual.finished = representative_finished
            individual.crashes = representative_crashes
            individual.distance = mean_distance
            individual.reward = mean_reward
            individual.fitness = mean_fitness
            individual.evaluation_valid = True
            individual.evaluation_steps = mean_steps
            individual.evaluation_terminated = representative_terminated
            individual.evaluation_truncated = representative_truncated

            if verbose and index is not None and total is not None:
                normal_status = self._outcome_status_text(
                    int(normal_metrics["finished"]),
                    int(normal_metrics["crashes"]),
                )
                mirrored_status = self._outcome_status_text(
                    int(mirrored_metrics["finished"]),
                    int(mirrored_metrics["crashes"]),
                )
                print(
                    f"{index + 1}/{total} "
                    f"N:{normal_status} {normal_metrics['discrete_progress']:.1f}%/{normal_metrics['time']:.2f}s | "
                    f"M:{mirrored_status} {mirrored_metrics['discrete_progress']:.1f}%/{mirrored_metrics['time']:.2f}s | "
                    f"AVG progress={mean_discrete_progress:.1f}% | time={mean_time:.2f}s | score={mean_fitness:.2f}"
                )
            return mean_fitness

        rollout_metrics = self._evaluate_single_rollout(
            individual=individual,
            mirrored=mirrored,
        )
        individual.discrete_progress = float(rollout_metrics["discrete_progress"])
        individual.dense_progress = float(rollout_metrics["dense_progress"])
        individual.time = float(rollout_metrics["time"])
        individual.finished = int(rollout_metrics["finished"])
        individual.crashes = int(rollout_metrics["crashes"])
        individual.distance = float(rollout_metrics["distance"])
        individual.reward = float(rollout_metrics["reward"])
        individual.fitness = float(rollout_metrics["fitness"])
        individual.evaluation_valid = True
        individual.evaluation_steps = int(rollout_metrics["steps"])
        individual.evaluation_terminated = bool(rollout_metrics["terminated"])
        individual.evaluation_truncated = bool(rollout_metrics["truncated"])

        if verbose and index is not None and total is not None:
            status = self._outcome_status_text(individual.finished, individual.crashes)
            mirror_tag = " [MIRROR]" if mirrored else ""
            print(
                f"{index + 1}/{total} "
                f"{status} | progress={individual.discrete_progress:.1f}% | "
                f"dense={individual.dense_progress:.1f}% | "
                f"time={individual.time:.2f}s | score={individual.fitness:.2f}{mirror_tag}"
            )

        return float(individual.fitness)

    def _evaluate_single_rollout(
        self,
        individual: Individual,
        mirrored: bool = False,
    ) -> Dict[str, float]:
        obs, info = self.env.reset()
        while info["done"] != 0:
            obs, info = self.env.reset()
        if self.policy_action_mode == "target":
            obs, info = self._wait_for_positive_game_time(obs, info)

        last_info = info

        step_count = 0
        terminated = False
        truncated = False
        while True:
            if self.max_steps is not None and step_count >= self.max_steps:
                truncated = True
                break
            policy_obs = self._mirror_observation(obs) if mirrored else obs
            action = individual.act(policy_obs)
            if mirrored:
                action = self._mirror_action_delta(action)
            action = self._apply_target_steer_deadzone(action)
            obs, reward, done, truncated, info = self.env.step(action)
            last_info = info
            step_count += 1

            race_finished = int(getattr(self.env, "finished", 0))
            race_term = int(getattr(self.env, "race_terminated", 0))
            info_done = info.get("done", 0.0) == 1.0

            terminated = done or truncated or info_done or race_finished > 0 or race_term != 0
            if terminated:
                break

        if hasattr(self.env, "raw_metrics_from_info"):
            raw_metrics = self.env.raw_metrics_from_info(
                last_info,
                timed_out=bool(truncated),
                step_count=step_count,
                terminated=bool(terminated and not truncated),
                truncated=bool(truncated),
            )
        else:
            discrete_progress = float(last_info.get("discrete_progress", 0.0))
            dense_progress = float(last_info.get("dense_progress", discrete_progress))
            raw_metrics = dict(
                progress=discrete_progress,
                discrete_progress=discrete_progress,
                dense_progress=dense_progress,
                time=float(last_info.get("time", 0.0)),
                distance=float(last_info.get("distance", 0.0)),
                finished=int(last_info.get("finished", int(getattr(self.env, "finished", 0)))),
                crashes=int(last_info.get("crashes", int(getattr(self.env, "crashes", 0)))),
                timeout=int(bool(truncated)),
                steps=step_count,
                terminated=bool(terminated and not truncated),
                truncated=bool(truncated),
            )

        discrete_progress = float(raw_metrics["discrete_progress"])
        dense_progress = float(raw_metrics["dense_progress"])
        t = float(raw_metrics["time"])
        if t <= 0:
            t = 1e-3

        distance = float(raw_metrics["distance"])
        finished = int(raw_metrics["finished"])
        crashes = int(raw_metrics["crashes"])

        scalar = Individual.compute_scalar_fitness_for(
            finished=finished,
            crashes=crashes,
            progress=dense_progress
            if Individual.RANKING_PROGRESS_SOURCE == "dense_progress"
            else discrete_progress,
            time_value=t,
            distance=distance,
        )
        return dict(
            discrete_progress=discrete_progress,
            dense_progress=dense_progress,
            time=t,
            finished=finished,
            crashes=crashes,
            distance=distance,
            reward=float(scalar),
            fitness=scalar,
            timeout=int(raw_metrics.get("timeout", 0)),
            steps=int(raw_metrics.get("steps", step_count)),
            terminated=bool(raw_metrics.get("terminated", terminated)),
            truncated=bool(raw_metrics.get("truncated", truncated)),
        )

    def _mirror_observation(self, obs: np.ndarray) -> np.ndarray:
        return ObservationEncoder.mirror_observation(
            obs,
            vertical_mode=self.vertical_mode,
        )

    @staticmethod
    def _mirror_action_delta(action: np.ndarray) -> np.ndarray:
        return ObservationEncoder.mirror_action(action)

    def _apply_target_steer_deadzone(self, action: np.ndarray) -> np.ndarray:
        if self.policy_action_mode != "target" or self.target_steer_deadzone <= 0.0:
            return action
        adjusted = np.asarray(action, dtype=np.float32).copy()
        if adjusted.shape == (3,) and abs(float(adjusted[2])) < self.target_steer_deadzone:
            adjusted[2] = 0.0
        return adjusted

    @staticmethod
    def _sample_mirror_flags(count: int, mirror_episode_prob: float) -> np.ndarray:
        if count <= 0 or mirror_episode_prob <= 0.0:
            return np.zeros(max(count, 0), dtype=bool)
        if mirror_episode_prob >= 1.0:
            return np.ones(count, dtype=bool)
        return (np.random.rand(count) < mirror_episode_prob)

    def evaluate_population(
        self,
        verbose: bool = False,
        mirror_flags: Optional[np.ndarray] = None,
        evaluate_both_mirrors: bool = False,
        reuse_valid_evaluations: bool = True,
    ) -> np.ndarray:
        n = len(self.population)
        fitnesses = np.zeros(n, dtype=np.float32)
        cached_count = 0
        if evaluate_both_mirrors:
            mirror_flags = np.zeros(n, dtype=bool)
        elif mirror_flags is None:
            mirror_flags = np.zeros(n, dtype=bool)
        elif len(mirror_flags) != n:
            raise ValueError(
                f"mirror_flags length {len(mirror_flags)} does not match population size {n}."
            )
        for i, ind in enumerate(self.population):
            can_reuse = (
                reuse_valid_evaluations
                and bool(getattr(ind, "evaluation_valid", False))
                and not evaluate_both_mirrors
                and not bool(mirror_flags[i])
            )
            if can_reuse:
                cached_count += 1
                if ind.fitness is not None and np.isfinite(float(ind.fitness)):
                    fitnesses[i] = float(ind.fitness)
                else:
                    fitnesses[i] = float(ind.compute_scalar_fitness())
                if verbose:
                    status = self._outcome_status_text(ind.finished, ind.crashes)
                    print(
                        f"{i + 1}/{n} {status} | "
                        f"progress={ind.discrete_progress:.1f}% | "
                        f"dense={ind.dense_progress:.1f}% | "
                        f"time={ind.time:.2f}s | cached"
                    )
                continue
            fitnesses[i] = self.evaluate_individual(
                individual=ind,
                index=i,
                total=n,
                verbose=verbose,
                mirrored=bool(mirror_flags[i]),
                evaluate_both_mirrors=evaluate_both_mirrors,
            )
        self.last_cached_evaluations = cached_count
        return fitnesses

    def _selection_max_time(self) -> float:
        return max(1e-6, float(getattr(self.env, "max_time", 1.0)))

    def _estimated_path_length(self) -> float:
        game_map = getattr(self.env, "map", None)
        if game_map is not None:
            for attr_name in ("estimated_path_length", "estimated_path_lenght"):
                attr = getattr(game_map, attr_name, None)
                if callable(attr):
                    return max(1e-6, float(attr()))
                if attr is not None:
                    return max(1e-6, float(attr))
            path_tiles = getattr(game_map, "path_tiles", None)
            if path_tiles is not None:
                return max(1e-6, float(len(path_tiles)) * 32.0)
        return 1.0

    def _max_episode_distance(self) -> float:
        estimated_path_length = self._estimated_path_length()
        observed_distance = max(
            [float(ind.distance) for ind in self.population] + [estimated_path_length]
        )
        return max(1e-6, observed_distance)

    def _metrics_for_objectives(self, individual: Individual) -> Dict[str, float]:
        return dict(
            finished=int(individual.finished),
            crashes=int(individual.crashes),
            max_crashes=int(getattr(self.env, "max_touches", 1)),
            max_touches=int(getattr(self.env, "max_touches", 1)),
            progress=float(individual.discrete_progress),
            dense_progress=float(individual.dense_progress),
            time=float(individual.time),
            distance=float(individual.distance),
            reward=float(individual.reward),
            fitness=(
                float(individual.compute_scalar_fitness())
                if individual.fitness is None
                else float(individual.fitness)
            ),
        )

    @staticmethod
    def _priority_score(objectives: np.ndarray, priority_indices: Sequence[int]) -> float:
        values = [float(objectives[int(index)]) for index in priority_indices]
        return float(sum((10.0 ** -position) * value for position, value in enumerate(values)))

    def _pareto_objective_matrix(self) -> np.ndarray:
        estimated_path_length = self._estimated_path_length()
        max_episode_distance = self._max_episode_distance()
        rows = []
        for individual in self.population:
            objectives = self._objectives_for_individual(
                individual,
                estimated_path_length=estimated_path_length,
                max_episode_distance=max_episode_distance,
            )
            rows.append(objectives[self.moo_objective_subset_indices])
        return np.vstack(rows).astype(np.float64)

    def _objectives_for_individual(
        self,
        individual: Individual,
        estimated_path_length: Optional[float] = None,
        max_episode_distance: Optional[float] = None,
    ) -> np.ndarray:
        return objectives_from_metrics(
            self._metrics_for_objectives(individual),
            mode=self.moo_objective_mode,
            max_time=self._selection_max_time(),
            estimated_path_length=(
                self._estimated_path_length()
                if estimated_path_length is None
                else float(estimated_path_length)
            ),
            max_episode_distance=(
                self._max_episode_distance()
                if max_episode_distance is None
                else float(max_episode_distance)
            ),
        )

    def _order_population_for_selection(self) -> None:
        if self.selection_mode != "pareto":
            self.population.sort(reverse=True)
            self._last_front0_size = None
            for rank, individual in enumerate(self.population):
                individual.selection_mode = self.selection_mode
                individual.selection_rank = rank
                individual.selection_crowding = 0.0
                individual.selection_objectives = tuple(float(value) for value in individual.ranking_key())
                individual.selection_objective_names = tuple(
                    f"ranking_{idx}" for idx in range(len(individual.selection_objectives))
                )
            return

        objective_matrix = self._pareto_objective_matrix()
        ordering = pareto_order(
            objective_matrix,
            priority_indices=self.moo_priority_indices,
            tiebreak=self.pareto_tiebreak,
            objective_names=self.moo_objective_names,
        )
        original_population = list(self.population)
        self.population = [original_population[index] for index in ordering.order]
        self._last_front0_size = len(ordering.fronts[0]) if ordering.fronts else 0

        for original_index, objectives in enumerate(ordering.objectives):
            individual = original_population[original_index]
            individual.selection_mode = self.selection_mode
            individual.selection_rank = int(ordering.ranks[original_index])
            individual.selection_crowding = float(ordering.crowding[original_index])
            individual.selection_objectives = tuple(float(value) for value in objectives)
            individual.selection_objective_names = tuple(self.moo_objective_names)
            individual.fitness = self._priority_score(objectives, self.moo_priority_indices)

    def _global_best_key(self, individual: Individual) -> Tuple[float, ...]:
        if self.selection_mode == "pareto":
            objectives = individual.selection_objectives
            if objectives is None:
                objectives = tuple(
                    float(value)
                    for value in self._objectives_for_individual(individual)[
                        self.moo_objective_subset_indices
                    ]
                )
            priority_values = tuple(
                float(objectives[int(index)]) for index in self.moo_priority_indices
            )
            return priority_values + tuple(float(value) for value in individual.ranking_key())
        if (
            not Individual.COMPARE_BY_RANKING_KEY
            and individual.fitness is not None
            and np.isfinite(float(individual.fitness))
        ):
            return (float(individual.fitness),)
        return tuple(float(value) for value in individual.ranking_key())

    def _is_better_global_best(self, candidate: Individual, incumbent: Optional[Individual]) -> bool:
        if incumbent is None:
            return True
        return self._global_best_key(candidate) > self._global_best_key(incumbent)

    def next_generation(
        self,
        elite_fraction: float = 0.2,
        mutation_prob: float = 0.1,
        mutation_sigma: float = 0.1,
    ) -> None:
        self._order_population_for_selection()

        elite_count = max(1, int(self.pop_size * elite_fraction))
        parent_pool_size = max(2, self.pop_size // 2)
        parents = self.population[:parent_pool_size]

        new_population: List[Individual] = [
            ind.copy() for ind in self.population[:elite_count]
        ]

        while len(new_population) < self.pop_size:
            shuffled_indices = np.random.permutation(parent_pool_size)
            usable_count = int(shuffled_indices.size - (shuffled_indices.size % 2))
            if usable_count <= 0:
                raise RuntimeError("Parent pool must contain at least two individuals.")

            for pair_start in range(0, usable_count, 2):
                if len(new_population) >= self.pop_size:
                    break
                i1 = int(shuffled_indices[pair_start])
                i2 = int(shuffled_indices[pair_start + 1])
                p1 = parents[i1]
                p2 = parents[i2]
                child = p1.crossover(p2)
                child.mutate(mutation_prob=mutation_prob, sigma=mutation_sigma)
                new_population.append(child)

        self.population = new_population

    def seed_population_from_model(
        self,
        model_path: str,
        exact_copies: int = 1,
        mutation_probs: Sequence[float] = (0.02, 0.05, 0.08),
        mutation_sigmas: Sequence[float] = (0.01, 0.03, 0.05),
        tier_counts: Optional[Sequence[int]] = None,
    ) -> Dict[str, object]:
        if len(mutation_probs) != len(mutation_sigmas):
            raise ValueError("mutation_probs and mutation_sigmas must have the same length.")

        loaded_policy, extra = NeuralPolicy.load(model_path, map_location="cpu")
        loaded_hidden_dims = tuple(int(dim) for dim in loaded_policy.hidden_dims)
        if loaded_policy.obs_dim != self.obs_dim:
            raise ValueError(
                f"Model obs_dim={loaded_policy.obs_dim} does not match trainer obs_dim={self.obs_dim}."
            )
        if loaded_policy.act_dim != self.act_dim:
            raise ValueError(
                f"Model act_dim={loaded_policy.act_dim} does not match trainer act_dim={self.act_dim}."
            )
        if loaded_hidden_dims != self.hidden_dims:
            raise ValueError(
                f"Model hidden_dims={loaded_hidden_dims} do not match trainer hidden_dims={self.hidden_dims}."
            )
        if loaded_policy.action_mode != self.policy_action_mode:
            raise ValueError(
                f"Model action_mode='{loaded_policy.action_mode}' does not match "
                f"trainer action_mode='{self.policy_action_mode}'."
            )
        if loaded_policy.hidden_activations != self.hidden_activations:
            raise ValueError(
                f"Model hidden_activations={list(loaded_policy.hidden_activations)} do not match "
                f"trainer hidden_activations={list(self.hidden_activations)}."
            )

        loaded_scale = loaded_policy.action_scale.detach().cpu().numpy().astype(np.float32)
        if self.policy_action_scale is None:
            self.policy_action_scale = loaded_scale.copy()
        elif not np.allclose(self.policy_action_scale, loaded_scale):
            raise ValueError(
                f"Model action_scale={loaded_scale.tolist()} does not match trainer "
                f"action_scale={self.policy_action_scale.tolist()}."
            )

        base_individual = Individual(
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dims,
            act_dim=self.act_dim,
            genome=loaded_policy.genome.copy(),
            action_scale=self.policy_action_scale.copy(),
            action_mode=self.policy_action_mode,
            hidden_activation=self.hidden_activations,
        )

        exact_copies = int(max(0, exact_copies))
        if exact_copies > self.pop_size:
            raise ValueError(
                f"exact_copies={exact_copies} exceeds population size {self.pop_size}."
            )

        remaining = self.pop_size - exact_copies
        num_tiers = len(mutation_probs)
        if tier_counts is None:
            tier_counts = [remaining // num_tiers for _ in range(num_tiers)]
            for i in range(remaining % num_tiers):
                tier_counts[i] += 1
        else:
            tier_counts = [int(count) for count in tier_counts]
            if len(tier_counts) != num_tiers:
                raise ValueError("tier_counts length must match number of mutation tiers.")
            if sum(tier_counts) != remaining:
                raise ValueError(
                    f"tier_counts sum to {sum(tier_counts)}, expected remaining population size {remaining}."
                )

        seeded_population: List[Individual] = [base_individual.copy() for _ in range(exact_copies)]
        tier_summaries = []
        for count, prob, sigma in zip(tier_counts, mutation_probs, mutation_sigmas):
            count = int(count)
            if count <= 0:
                continue
            for _ in range(count):
                child = base_individual.copy()
                child.mutate(mutation_prob=float(prob), sigma=float(sigma))
                seeded_population.append(child)
            tier_summaries.append(
                dict(
                    count=count,
                    mutation_prob=float(prob),
                    mutation_sigma=float(sigma),
                )
            )

        if len(seeded_population) != self.pop_size:
            raise RuntimeError(
                f"Seeded population has size {len(seeded_population)}, expected {self.pop_size}."
            )

        self.population = seeded_population
        self.best_individual = None
        self.generation = 0
        self._loaded_checkpoint_evaluated = False
        self._pending_initial_downselect = False

        return dict(
            source_model=model_path,
            exact_copies=exact_copies,
            tiers=tier_summaries,
            model_extra=extra,
            hidden_dims=list(self.hidden_dims),
            hidden_activations=list(self.hidden_activations),
            action_mode=self.policy_action_mode,
        )

    def _log_generation(
        self,
        generation: int,
        best_gen: Individual,
        best_so_far: Individual,
        dnf_time_for_plot: float,
        history: Dict[str, List[float]],
    ) -> None:
        if self.logger is None:
            return

        timestamp = TrainingLogger._timestamp_utc()

        individual_rows: List[Dict] = []
        for idx, ind in enumerate(self.population):
            individual_rows.append(
                dict(
                    timestamp_utc=timestamp,
                    generation=generation,
                    individual_index=idx,
                    finished=int(ind.finished),
                    crashes=int(ind.crashes),
                    timeout=int(int(ind.finished) <= 0 and int(ind.crashes) <= 0),
                    progress=float(ind.discrete_progress),
                    discrete_progress=float(ind.discrete_progress),
                    dense_progress=float(ind.dense_progress),
                    ranking_progress=float(ind.ranking_progress()),
                    distance=float(ind.distance),
                    time=float(ind.time),
                    reward=float(ind.reward),
                    fitness=np.nan if ind.fitness is None else float(ind.fitness),
                    ranking_key=json.dumps([float(value) for value in ind.ranking_key()]),
                    selection_mode=self.selection_mode,
                    selection_rank=(
                        "" if ind.selection_rank is None else int(ind.selection_rank)
                    ),
                    selection_crowding=float(ind.selection_crowding),
                    selection_objectives=json.dumps(
                        [] if ind.selection_objectives is None else list(ind.selection_objectives)
                    ),
                    selection_objective_names=json.dumps(
                        list(ind.selection_objective_names)
                    ),
                    evaluation_steps=int(ind.evaluation_steps),
                    evaluation_terminated=int(bool(ind.evaluation_terminated)),
                    evaluation_truncated=int(bool(ind.evaluation_truncated)),
                    evaluation_valid=int(bool(ind.evaluation_valid)),
                )
            )
        self.logger.log_individual_batch(individual_rows)

        discrete_progresses = np.array(
            [float(ind.discrete_progress) for ind in self.population],
            dtype=np.float32,
        )
        finisheds = np.array([int(ind.finished) for ind in self.population], dtype=np.int32)
        crashes = np.array([int(ind.crashes) for ind in self.population], dtype=np.int32)
        dense_progresses = np.array(
            [float(ind.dense_progress) for ind in self.population],
            dtype=np.float32,
        )
        ranking_progresses = np.array(
            [float(ind.ranking_progress()) for ind in self.population],
            dtype=np.float32,
        )
        times = np.array([float(ind.time) for ind in self.population], dtype=np.float32)
        distances = np.array([float(ind.distance) for ind in self.population], dtype=np.float32)
        rewards = np.array([float(ind.reward) for ind in self.population], dtype=np.float32)
        steps = np.array([int(ind.evaluation_steps) for ind in self.population], dtype=np.int32)
        finish_rate = float((finisheds > 0).mean())
        crash_rate = float((crashes > 0).mean())
        timeouts = (finisheds <= 0) & (crashes <= 0)
        timeout_rate = float(timeouts.mean())

        summary_row = dict(
            timestamp_utc=timestamp,
            generation=generation,
            cached_evaluations=int(self.last_cached_evaluations),
            evaluated_count=int(len(self.population) - self.last_cached_evaluations),
            population_size=int(len(self.population)),
            dist_avg=history["dist_avg"][-1],
            dist_best_gen=history["dist_best_gen"][-1],
            dist_best_global=history["dist_best_global"][-1],
            time_avg=history["time_avg"][-1],
            time_best_gen=history["time_best_gen"][-1],
            time_best_global=history["time_best_global"][-1],
            finish_count=int((finisheds > 0).sum()),
            crash_count=int((crashes > 0).sum()),
            timeout_count=int(timeouts.sum()),
            finish_rate=finish_rate,
            crash_rate=crash_rate,
            timeout_rate=timeout_rate,
            best_finished=int(best_gen.finished),
            best_crashes=int(best_gen.crashes),
            best_progress=float(best_gen.discrete_progress),
            best_dense_progress=float(best_gen.dense_progress),
            best_ranking_progress=float(best_gen.ranking_progress()),
            best_ranking_key=json.dumps([float(value) for value in best_gen.ranking_key()]),
            mean_discrete_progress=(
                float(discrete_progresses.mean()) if len(discrete_progresses) else 0.0
            ),
            mean_dense_progress=(
                float(dense_progresses.mean()) if len(dense_progresses) else 0.0
            ),
            mean_ranking_progress=(
                float(ranking_progresses.mean()) if len(ranking_progresses) else 0.0
            ),
            std_dense_progress=(
                float(dense_progresses.std()) if len(dense_progresses) else 0.0
            ),
            std_ranking_progress=(
                float(ranking_progresses.std()) if len(ranking_progresses) else 0.0
            ),
            best_distance=float(best_gen.distance),
            best_time=float(best_gen.time),
            best_reward=float(best_gen.reward),
            best_fitness=np.nan if best_gen.fitness is None else float(best_gen.fitness),
            selection_mode=self.selection_mode,
            front0_size=(
                ""
                if self._last_front0_size is None
                else int(self._last_front0_size)
            ),
            best_selection_rank=(
                "" if best_gen.selection_rank is None else int(best_gen.selection_rank)
            ),
            best_selection_crowding=float(best_gen.selection_crowding),
            best_selection_objectives=json.dumps(
                []
                if best_gen.selection_objectives is None
                else list(best_gen.selection_objectives)
            ),
            selection_objective_names=json.dumps(
                list(best_gen.selection_objective_names)
            ),
            virtual_time_sum=float(times.sum()) if len(times) else 0.0,
            virtual_distance_sum=float(distances.sum()) if len(distances) else 0.0,
            evaluation_steps_sum=int(steps.sum()) if len(steps) else 0,
        )
        self.logger.log_generation_summary(summary_row)

    def run(
        self,
        generations: int,
        elite_fraction: float = 0.2,
        mutation_prob: float = 0.1,
        mutation_sigma: float = 0.1,
        mutation_prob_decay: float = 1.0,
        mutation_prob_min: float = 0.0,
        mutation_sigma_decay: float = 1.0,
        mutation_sigma_min: float = 0.0,
        mirror_episode_prob: float = 0.0,
        evaluate_both_mirrors: bool = True,
        verbose: bool = True,
        dnf_time_for_plot: float = 30.0,
        checkpoint_every: int = 10,
        training_config: Optional[dict] = None,
    ) -> Dict[str, List[float]]:
        history = {
            "dist_avg": [],
            "dist_best_gen": [],
            "dist_best_global": [],
            "time_avg": [],
            "time_best_gen": [],
            "time_best_global": [],
        }

        best_so_far: Optional[Individual] = (
            None if self.best_individual is None else self.best_individual.copy()
        )

        if self.logger is not None:
            cfg = dict(
                obs_dim=self.obs_dim,
                observation_layout=list(self.observation_layout),
                vertical_mode=bool(self.vertical_mode),
                hidden_dim=self.hidden_dim,
                act_dim=self.act_dim,
                pop_size=self.pop_size,
                max_steps=self.max_steps,
                policy_action_mode=self.policy_action_mode,
                hidden_activation=self.hidden_activation,
                hidden_activations=list(self.hidden_activations),
                run_started_utc=TrainingLogger._timestamp_utc(),
                start_generation=self.generation,
                generations_requested=generations,
                elite_fraction=elite_fraction,
                mutation_prob=mutation_prob,
                mutation_sigma=mutation_sigma,
                mutation_prob_decay=mutation_prob_decay,
                mutation_prob_min=mutation_prob_min,
                mutation_sigma_decay=mutation_sigma_decay,
                mutation_sigma_min=mutation_sigma_min,
                mirror_episode_prob=mirror_episode_prob,
                evaluate_both_mirrors=bool(evaluate_both_mirrors),
                checkpoint_every=checkpoint_every,
                dnf_time_for_plot=dnf_time_for_plot,
            )
            if training_config is not None:
                cfg.update(training_config)
            self.logger.write_config(cfg, merge=True)

        current_mutation_prob = float(mutation_prob)
        current_mutation_sigma = float(mutation_sigma)
        mutation_prob_decay = float(mutation_prob_decay)
        mutation_prob_min = float(mutation_prob_min)
        mutation_sigma_decay = float(mutation_sigma_decay)
        mutation_sigma_min = float(mutation_sigma_min)

        restored_mutation_state = (
            self._loaded_checkpoint_evaluated
            and self._checkpoint_current_mutation_prob is not None
            and self._checkpoint_current_mutation_sigma is not None
        )
        if restored_mutation_state:
            current_mutation_prob = float(self._checkpoint_current_mutation_prob)
            current_mutation_sigma = float(self._checkpoint_current_mutation_sigma)
            if verbose:
                print(
                    "Restored mutation state from checkpoint: "
                    f"prob={current_mutation_prob:.6f}, sigma={current_mutation_sigma:.6f}"
                )
        elif self._loaded_checkpoint_evaluated and verbose:
            print(
                "Checkpoint does not contain saved mutation state. "
                "Using mutation values from the current trainer script."
            )

        if self._loaded_checkpoint_evaluated:
            if verbose:
                print(
                    f"Loaded evaluated generation {self.generation}. "
                    "Creating next generation before continuing..."
                )
            self.next_generation(
                elite_fraction=elite_fraction,
                mutation_prob=current_mutation_prob,
                mutation_sigma=current_mutation_sigma,
            )
            self._loaded_checkpoint_evaluated = False
            self._checkpoint_current_mutation_prob = None
            self._checkpoint_current_mutation_sigma = None

        if self._pending_initial_downselect:
            loaded_count = len(self.population)
            if loaded_count > self.pop_size:
                if verbose:
                    print("\n" + "=" * 20)
                    print("Initial TM Screening (Generation 0)")
                    print("=" * 20)
                    print(
                        f"Evaluating all loaded candidates: {loaded_count} -> "
                        f"select top {self.pop_size} for TM generation 1"
                    )

                screening_mirror_flags = self._sample_mirror_flags(
                    loaded_count,
                    mirror_episode_prob=mirror_episode_prob,
                )
                if verbose and evaluate_both_mirrors:
                    print(
                        "Screening mode: evaluating every individual as normal and mirrored, "
                        "then averaging both results."
                    )
                elif verbose and screening_mirror_flags.any():
                    print(
                        f"Screening mirror flags: "
                        f"{int(screening_mirror_flags.sum())}/{loaded_count}"
                    )
                _ = self.evaluate_population(
                    verbose=verbose,
                    mirror_flags=screening_mirror_flags,
                    evaluate_both_mirrors=evaluate_both_mirrors,
                )
                self._order_population_for_selection()
                screened_best = self.population[0].copy()
                best_so_far = screened_best
                self.population = self.population[: self.pop_size]

                if verbose:
                    status = self._outcome_status_text(
                        screened_best.finished,
                        screened_best.crashes,
                    )
                    print(
                        f"Screening best: {status} | "
                        f"progress={screened_best.discrete_progress:.1f}% | "
                        f"dense={screened_best.dense_progress:.1f}% | "
                        f"time={screened_best.time:.2f}s"
                    )
                    print(
                        f"Downselected to {len(self.population)} individuals for TM training."
                    )
            self._pending_initial_downselect = False

        # Ensure global best artifact exists from the start (useful when resuming).
        if self.logger is not None and best_so_far is not None:
            self.logger.save_best_individual(
                best_so_far,
                generation=self.generation,
                observation_layout=self.observation_layout,
                vertical_mode=self.vertical_mode,
            )

        for local_gen in range(generations):
            current_generation = self.generation + 1
            if verbose:
                print("\n" + "=" * 20)
                print(
                    f"Generation {current_generation} | "
                    f"mut_p={current_mutation_prob:.4f}, sigma={current_mutation_sigma:.4f}"
                )
                print("=" * 20)

            gen_mirror_flags = self._sample_mirror_flags(
                len(self.population),
                mirror_episode_prob=mirror_episode_prob,
            )
            if verbose and evaluate_both_mirrors:
                print(
                    "Mirror eval: every individual runs normal + mirrored, "
                    "selection uses the mean of both rollout scores."
                )
            elif verbose and gen_mirror_flags.any():
                print(
                    f"Mirror flags this generation: "
                    f"{int(gen_mirror_flags.sum())}/{len(self.population)}"
                )

            _ = self.evaluate_population(
                verbose=verbose,
                mirror_flags=gen_mirror_flags,
                evaluate_both_mirrors=evaluate_both_mirrors,
            )
            self._order_population_for_selection()

            progresses = np.array(
                [float(ind.ranking_progress()) for ind in self.population],
                dtype=np.float32,
            )
            times_plot = np.array(
                [
                    float(ind.time) if int(ind.finished) > 0 else float(dnf_time_for_plot)
                    for ind in self.population
                ],
                dtype=np.float32,
            )

            dist_avg = float(progresses.mean())
            time_avg = float(times_plot.mean())

            best_gen = self.population[0]
            dist_best_gen = float(best_gen.ranking_progress())
            time_best_gen = float(
                best_gen.time if int(best_gen.finished) > 0 else dnf_time_for_plot
            )

            best_improved = self._is_better_global_best(best_gen, best_so_far)
            if best_improved:
                best_so_far = best_gen.copy()

            dist_best_global = float(best_so_far.ranking_progress())
            time_best_global = float(
                best_so_far.time if int(best_so_far.finished) > 0 else dnf_time_for_plot
            )

            history["dist_avg"].append(dist_avg)
            history["dist_best_gen"].append(dist_best_gen)
            history["dist_best_global"].append(dist_best_global)
            history["time_avg"].append(time_avg)
            history["time_best_gen"].append(time_best_gen)
            history["time_best_global"].append(time_best_global)

            if verbose:
                from_status = self._outcome_status_text(best_gen.finished, best_gen.crashes)
                print(
                    f"Best of generation: {from_status} | "
                    f"progress={dist_best_gen:.1f}% | "
                    f"dense={best_gen.dense_progress:.1f}% | "
                    f"time={best_gen.time:.2f}s"
                )

            self.generation = current_generation
            self._log_generation(
                generation=current_generation,
                best_gen=best_gen,
                best_so_far=best_so_far,
                dnf_time_for_plot=dnf_time_for_plot,
                history=history,
            )

            if self.logger is not None and best_improved and best_so_far is not None:
                global_best_path = self.logger.save_best_individual(
                    best_so_far,
                    generation=current_generation,
                    observation_layout=self.observation_layout,
                    vertical_mode=self.vertical_mode,
                )
                if verbose:
                    print(f"Global best updated: {global_best_path}")

            should_checkpoint = (
                self.logger is not None
                and checkpoint_every > 0
                and (
                    current_generation % checkpoint_every == 0
                    or local_gen == generations - 1
                )
            )
            if should_checkpoint:
                checkpoint_path = self.logger.save_population_checkpoint(
                    population=self.population,
                    generation=current_generation,
                    obs_dim=self.obs_dim,
                    hidden_dim=self.hidden_dim,
                    act_dim=self.act_dim,
                    best_individual=best_so_far,
                    current_mutation_prob=current_mutation_prob,
                    current_mutation_sigma=current_mutation_sigma,
                    observation_layout=self.observation_layout,
                    vertical_mode=self.vertical_mode,
                )
                if verbose:
                    print(f"Checkpoint saved: {checkpoint_path}")

            if local_gen < generations - 1:
                self.next_generation(
                    elite_fraction=elite_fraction,
                    mutation_prob=current_mutation_prob,
                    mutation_sigma=current_mutation_sigma,
                )
                current_mutation_prob = max(
                    mutation_prob_min,
                    current_mutation_prob * mutation_prob_decay,
                )
                current_mutation_sigma = max(
                    mutation_sigma_min,
                    current_mutation_sigma * mutation_sigma_decay,
                )

        self.best_individual = best_so_far

        if self.logger is not None and self.best_individual is not None:
            self.logger.save_best_individual(
                self.best_individual,
                generation=self.generation,
                observation_layout=self.observation_layout,
                vertical_mode=self.vertical_mode,
            )
            self.logger.save_final_population(
                population=self.population,
                generation=self.generation,
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                act_dim=self.act_dim,
                best_individual=self.best_individual,
                observation_layout=self.observation_layout,
                vertical_mode=self.vertical_mode,
            )

        return history

    def load_population_checkpoint(
        self, checkpoint_path: str, assume_evaluated_generation: bool = True
    ) -> int:
        data = np.load(checkpoint_path)
        if "genomes" not in data.files:
            raise ValueError(f"Checkpoint {checkpoint_path} does not contain 'genomes'.")

        genomes = data["genomes"].astype(np.float32)
        if genomes.ndim != 2:
            raise ValueError(f"Expected 2D genomes array, got shape {genomes.shape}.")
        loaded_pop_size = int(genomes.shape[0])
        if loaded_pop_size < self.pop_size:
            raise ValueError(
                f"Checkpoint pop_size={loaded_pop_size}, expected at least {self.pop_size}."
            )
        if loaded_pop_size > self.pop_size and assume_evaluated_generation:
            raise ValueError(
                "Checkpoint population is larger than trainer pop_size and is marked as "
                "already evaluated. Use assume_evaluated_generation=False for initial "
                "TM screening + downselect, or match pop_size exactly."
            )

        expected_genome_size = self.population[0].genome.shape[0]
        if genomes.shape[1] != expected_genome_size:
            raise ValueError(
                f"Checkpoint genome_size={genomes.shape[1]}, expected {expected_genome_size}."
            )

        checkpoint_hidden_dims: Optional[Tuple[int, ...]] = None
        if "hidden_dims" in data.files:
            checkpoint_hidden_dims = tuple(
                int(value) for value in np.asarray(data["hidden_dims"]).reshape(-1)
            )
        elif "hidden_dim" in data.files:
            checkpoint_hidden_dims = (
                int(np.asarray(data["hidden_dim"]).reshape(-1)[0]),
            )
        if checkpoint_hidden_dims is not None and checkpoint_hidden_dims != self.hidden_dims:
            raise ValueError(
                f"Checkpoint hidden_dims={checkpoint_hidden_dims} do not match "
                f"trainer hidden_dims={self.hidden_dims}."
            )

        checkpoint_hidden_activations: Optional[Tuple[str, ...]] = None
        if "hidden_activations" in data.files:
            checkpoint_hidden_activations = normalize_hidden_activations(
                [str(value) for value in np.asarray(data["hidden_activations"]).reshape(-1)],
                len(self.hidden_dims),
            )
        elif "hidden_activation" in data.files:
            raw_hidden_activations = np.asarray(data["hidden_activation"]).reshape(-1)
            if raw_hidden_activations.size > 0:
                if raw_hidden_activations.size == 1:
                    checkpoint_hidden_activations = normalize_hidden_activations(
                        str(raw_hidden_activations[0]),
                        len(self.hidden_dims),
                    )
                else:
                    checkpoint_hidden_activations = normalize_hidden_activations(
                        [str(value) for value in raw_hidden_activations],
                        len(self.hidden_dims),
                    )
        if (
            checkpoint_hidden_activations is not None
            and checkpoint_hidden_activations != self.hidden_activations
        ):
            raise ValueError(
                f"Checkpoint hidden_activations={list(checkpoint_hidden_activations)} do not match "
                f"trainer hidden_activations={list(self.hidden_activations)}."
            )

        progresses = data["progresses"] if "progresses" in data.files else None
        dense_progresses = data["dense_progresses"] if "dense_progresses" in data.files else None
        times = data["times"] if "times" in data.files else None
        finisheds = data["finisheds"] if "finisheds" in data.files else None
        crashes = data["crashes"] if "crashes" in data.files else None
        distances = data["distances"] if "distances" in data.files else None
        rewards = data["rewards"] if "rewards" in data.files else None
        evaluation_steps = data["evaluation_steps"] if "evaluation_steps" in data.files else None
        evaluation_terminated = (
            data["evaluation_terminated"] if "evaluation_terminated" in data.files else None
        )
        evaluation_truncated = (
            data["evaluation_truncated"] if "evaluation_truncated" in data.files else None
        )
        evaluation_valid = data["evaluation_valid"] if "evaluation_valid" in data.files else None
        fitnesses = data["fitnesses"] if "fitnesses" in data.files else None
        checkpoint_current_mutation_prob = (
            float(np.asarray(data["current_mutation_prob"]).reshape(-1)[0])
            if "current_mutation_prob" in data.files
            else None
        )
        checkpoint_current_mutation_sigma = (
            float(np.asarray(data["current_mutation_sigma"]).reshape(-1)[0])
            if "current_mutation_sigma" in data.files
            else None
        )
        restored_metrics_are_valid = bool(
            assume_evaluated_generation
            and progresses is not None
            and dense_progresses is not None
            and times is not None
            and finisheds is not None
            and crashes is not None
            and distances is not None
        )

        restored_population: List[Individual] = []
        for i in range(loaded_pop_size):
            ind = Individual(
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                act_dim=self.act_dim,
                genome=genomes[i],
                action_scale=self.policy_action_scale,
                action_mode=self.policy_action_mode,
                hidden_activation=self.hidden_activations,
            )
            if progresses is not None:
                ind.discrete_progress = float(progresses[i])
            if dense_progresses is not None:
                ind.dense_progress = float(dense_progresses[i])
            else:
                ind.dense_progress = float(ind.discrete_progress)
            if times is not None:
                ind.time = float(times[i])
            if finisheds is not None:
                ind.finished = int(finisheds[i])
            if crashes is not None:
                ind.crashes = int(crashes[i])
            if distances is not None:
                ind.distance = float(distances[i])
            if rewards is not None:
                ind.reward = float(rewards[i])
            if evaluation_steps is not None:
                ind.evaluation_steps = int(evaluation_steps[i])
            if evaluation_terminated is not None:
                ind.evaluation_terminated = bool(int(evaluation_terminated[i]))
            if evaluation_truncated is not None:
                ind.evaluation_truncated = bool(int(evaluation_truncated[i]))
            if fitnesses is not None:
                val = float(fitnesses[i])
                ind.fitness = None if np.isnan(val) else val
            ind.evaluation_valid = (
                bool(int(evaluation_valid[i]))
                if evaluation_valid is not None
                else restored_metrics_are_valid
            )
            restored_population.append(ind)

        self.population = restored_population

        generation = 0
        if "generation" in data.files:
            generation = int(np.asarray(data["generation"]).reshape(-1)[0])
        self.generation = generation
        self._loaded_checkpoint_evaluated = bool(assume_evaluated_generation)
        self._pending_initial_downselect = (
            (loaded_pop_size > self.pop_size) and (not assume_evaluated_generation)
        )
        self._checkpoint_current_mutation_prob = checkpoint_current_mutation_prob
        self._checkpoint_current_mutation_sigma = checkpoint_current_mutation_sigma

        if "best_genome" in data.files:
            best = Individual(
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                act_dim=self.act_dim,
                genome=np.asarray(data["best_genome"], dtype=np.float32),
                action_scale=self.policy_action_scale,
                action_mode=self.policy_action_mode,
                hidden_activation=self.hidden_activations,
            )
            if "best_progress" in data.files:
                best.discrete_progress = float(np.asarray(data["best_progress"]).reshape(-1)[0])
            if "best_dense_progress" in data.files:
                best.dense_progress = float(np.asarray(data["best_dense_progress"]).reshape(-1)[0])
            else:
                best.dense_progress = float(best.discrete_progress)
            if "best_time" in data.files:
                best.time = float(np.asarray(data["best_time"]).reshape(-1)[0])
            if "best_finished" in data.files:
                best.finished = int(np.asarray(data["best_finished"]).reshape(-1)[0])
            if "best_crashes" in data.files:
                best.crashes = int(np.asarray(data["best_crashes"]).reshape(-1)[0])
            if "best_distance" in data.files:
                best.distance = float(np.asarray(data["best_distance"]).reshape(-1)[0])
            if "best_reward" in data.files:
                best.reward = float(np.asarray(data["best_reward"]).reshape(-1)[0])
            if "best_evaluation_steps" in data.files:
                best.evaluation_steps = int(np.asarray(data["best_evaluation_steps"]).reshape(-1)[0])
            if "best_evaluation_terminated" in data.files:
                best.evaluation_terminated = bool(
                    int(np.asarray(data["best_evaluation_terminated"]).reshape(-1)[0])
                )
            if "best_evaluation_truncated" in data.files:
                best.evaluation_truncated = bool(
                    int(np.asarray(data["best_evaluation_truncated"]).reshape(-1)[0])
                )
            if "best_fitness" in data.files:
                bf = float(np.asarray(data["best_fitness"]).reshape(-1)[0])
                best.fitness = None if np.isnan(bf) else bf
            best.evaluation_valid = restored_metrics_are_valid
            self.best_individual = best
        else:
            if self.population:
                self._order_population_for_selection()
                self.best_individual = self.population[0].copy()
            else:
                self.best_individual = None

        return self.generation

    @staticmethod
    def find_latest_checkpoint(base_dir: str = "logs/ga_runs") -> str:
        pattern = os.path.join(base_dir, "**", "checkpoints", "population_gen_*.npz")
        files = glob.glob(pattern, recursive=True)
        if not files:
            raise FileNotFoundError(f"No checkpoints found in {base_dir}.")
        return max(files, key=os.path.getmtime)

    @staticmethod
    def find_latest_supervised_model(base_dir: str = "logs/supervised_runs") -> str:
        pattern = os.path.join(base_dir, "**", "best_model.pt")
        files = glob.glob(pattern, recursive=True)
        if not files:
            raise FileNotFoundError(f"No supervised models found in {base_dir}.")
        return max(files, key=os.path.getmtime)


# Backward-compatible alias for historical scripts. New code should import and
# instantiate GeneticTrainer.
EvolutionTrainer = GeneticTrainer


if __name__ == "__main__":
    from Enviroment import RacingGameEnviroment

    # map dependend constants
    map_name = "AI Training #5"
    env_max_time = 25
    
    # neural network architecture
    hidden_dim = [32, 16]
    hidden_activation = ["relu", "tanh"]
    action_mode = "target"  # target / delta
    vertical_mode = True

    # Evolution
    pop_size = 64
    # Full-risk MOO run: keep the same elite fraction as the TM2D tests.
    elite_fraction = 4 / 32
    generations_to_run = 200
    checkpoint_every = 4

    # Selection metric for the overnight GA experiment.
    #
    # With selection_fitness_mode="ranking", Individual.fitness remains a
    # log-friendly scalar, but population sorting uses Individual.ranking_key().
    # This mirrors the best current TM2D candidate:
    # (finished, progress, -time), where progress resolves to dense_progress.
    #
    # selection_mode="lexicographic" keeps the original GA behavior.
    # selection_mode="pareto" switches only the ordering/downselection step to
    # NSGA-II-style non-dominated sorting while reusing the same evaluation loop.
    selection_mode = "pareto"  # lexicographic / pareto
    selection_fitness_mode = "ranking"  # scalar / ranking
    ranking_mode = "lexicographic"
    ranking_key = "(finished, progress, -time, -crashes)"
    ranking_progress_source = "dense_progress"
    moo_objective_mode = "lexicographic_primitives"
    moo_objective_subset = "finished,progress,neg_time,neg_crashes,neg_distance"
    moo_objective_priority = "finished,progress,neg_time,neg_crashes,neg_distance"
    pareto_tiebreak = "priority"

    mutation_prob = 0.18
    mutation_prob_decay = 1.0
    mutation_prob_min = 0.18

    mutation_sigma = 0.22
    mutation_sigma_decay = 1.0
    mutation_sigma_min = 0.22


    # Fancy updates
    mirror_episode_prob = 0.0
    evaluate_both_mirrors = False
    target_steer_deadzone = 0.00
    max_touches = 1
    

    
    # Other constants
    act_dim = 3
    max_steps = None
    env_dt_ref = 1.0 / 100.0
    env_dt_ratio_clip = 3.0
    surface_probe_height = Car.SURFACE_PROBE_HEIGHT
    surface_ray_lift = Car.SURFACE_RAY_LIFT
    policy_action_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    start_idle_max_time = 2.0
    # Baseline run from scratch: stronger exploration first, then gradual annealing.
    Individual.COMPARE_BY_RANKING_KEY = selection_fitness_mode == "ranking"
    Individual.RANKING_KEY = ranking_key
    Individual.RANKING_PROGRESS_SOURCE = ranking_progress_source
    
    # Train from checkpoint or supervised predtrainded model
    initial_population_source: Optional[str] = None
    # Old v2d population checkpoints are intentionally not used as the default
    # source anymore because the canonical training observation is now v3d.
    # initial_population_source = (
    #     r"logs/ga_runs\20260409_081105_map_AI_Training__5_v2d_h32x16_p32\checkpoints\population_gen_0190.npz"
    # )
    # initial_population_source = r"logs/supervised_runs\20260317_123456_target_supervised\best_model.pt"
    seed_model_exact_copies = 2
    seed_model_mutation_probs = (0.015, 0.04, 0.08)
    seed_model_mutation_sigmas = (0.008, 0.02, 0.04)
    
    # True = TM checkpoint already evaluated in TM -> continue from next generation.
    # False = mini pretrain checkpoint -> evaluate loaded population in TM first.
    resume_assume_evaluated_generation = True

    resume_checkpoint: Optional[str] = None
    seed_model_path: Optional[str] = None
    initial_population_source_kind = "random"
    if initial_population_source:
        source_ext = os.path.splitext(initial_population_source)[1].lower()
        if source_ext == ".pt":
            seed_model_path = initial_population_source
            initial_population_source_kind = "model_seed"
        elif source_ext == ".npz":
            resume_checkpoint = initial_population_source
            initial_population_source_kind = "population_checkpoint"
        else:
            raise ValueError(
                "initial_population_source must point to a .pt model or .npz population checkpoint."
            )

    if seed_model_path:
        seed_policy, _ = NeuralPolicy.load(seed_model_path, map_location="cpu")
        hidden_dim = list(seed_policy.hidden_dims)
        act_dim = seed_policy.act_dim
        action_mode = seed_policy.action_mode
        hidden_activation = list(seed_policy.hidden_activations)
        policy_action_scale = (
            seed_policy.action_scale.detach().cpu().numpy().astype(np.float32)
        )

    env = RacingGameEnviroment(
        map_name=map_name,
        never_quit=False,
        action_mode=action_mode,
        dt_ref=env_dt_ref,
        dt_ratio_clip=env_dt_ratio_clip,
        vertical_mode=vertical_mode,
        surface_probe_height=surface_probe_height,
        surface_ray_lift=surface_ray_lift,
        max_time=env_max_time,
        max_touches=max_touches,
        start_idle_max_time=start_idle_max_time,
    )
    obs, info = env.reset()
    obs_dim = obs.shape[0]

    logger: Optional[TrainingLogger]
    if resume_checkpoint:
        if resume_assume_evaluated_generation:
            checkpoint_dir = os.path.dirname(resume_checkpoint)
            run_dir = os.path.dirname(checkpoint_dir)
            logger = TrainingLogger(run_dir=run_dir)
        else:
            source_checkpoint_name = os.path.splitext(os.path.basename(resume_checkpoint))[0]
            source_run_name = os.path.basename(os.path.dirname(os.path.dirname(resume_checkpoint)))
            run_name = (
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_tm_finetune_map_{map_name}_{'v3d' if vertical_mode else 'v2d'}_h{hidden_dims_tag(hidden_dim)}_p{pop_size}"
                f"_src_{source_run_name}_{source_checkpoint_name}"
            )
            logger = TrainingLogger(base_dir="logs/tm_finetune_runs", run_name=run_name)
    elif seed_model_path:
        source_model_name = os.path.splitext(os.path.basename(seed_model_path))[0]
        run_name = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f"_tm_seed_map_{map_name}_{'v3d' if vertical_mode else 'v2d'}_h{hidden_dims_tag(hidden_dim)}_p{pop_size}"
            f"_src_{source_model_name}"
        )
        logger = TrainingLogger(base_dir="logs/tm_finetune_runs", run_name=run_name)
    else:
        selection_tag = (
            f"{selection_fitness_mode}_{safe_ranking_key_tag(ranking_key)}"
            if selection_mode == "lexicographic"
            else (
                f"pareto_{moo_objective_mode}_"
                f"{safe_objective_tag(moo_objective_subset)}"
            )
        )
        run_name = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f"_map_{map_name}_{'v3d' if vertical_mode else 'v2d'}_h{hidden_dims_tag(hidden_dim)}_p{pop_size}"
            f"_{selection_tag}"
        )
        logger = TrainingLogger(run_name=run_name)

    trainer = GeneticTrainer(
        env=env,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        act_dim=act_dim,
        pop_size=pop_size,
        max_steps=max_steps,
        policy_action_scale=policy_action_scale,
        policy_action_mode=action_mode,
        hidden_activation=hidden_activation,
        target_steer_deadzone=target_steer_deadzone,
        logger=logger,
        observation_layout=env.obs_encoder.feature_names(vertical_mode=env.obs_encoder.vertical_mode),
        selection_mode=selection_mode,
        moo_objective_mode=moo_objective_mode,
        moo_objective_subset=moo_objective_subset,
        moo_objective_priority=moo_objective_priority,
        pareto_tiebreak=pareto_tiebreak,
    )

    try:
        if resume_checkpoint:
            loaded_generation = trainer.load_population_checkpoint(
                resume_checkpoint,
                assume_evaluated_generation=resume_assume_evaluated_generation,
            )
            print(f"Loaded checkpoint from generation {loaded_generation}: {resume_checkpoint}")
            if not resume_assume_evaluated_generation:
                trainer.generation = 0
                print("Reset TM generation counter to 0 for fine-tuning.")
        elif seed_model_path:
            seed_summary = trainer.seed_population_from_model(
                model_path=seed_model_path,
                exact_copies=seed_model_exact_copies,
                mutation_probs=seed_model_mutation_probs,
                mutation_sigmas=seed_model_mutation_sigmas,
            )
            print(f"Seeded population from model: {seed_model_path}")
            print(f"Seed summary: {seed_summary}")

        history = trainer.run(
            generations=generations_to_run,
            elite_fraction=elite_fraction,
            # Exploratory start + annealing toward fine-tuning.
            mutation_prob=mutation_prob,
            mutation_sigma=mutation_sigma,
            mutation_prob_decay=mutation_prob_decay,
            mutation_prob_min=mutation_prob_min,
            mutation_sigma_decay=mutation_sigma_decay,
            mutation_sigma_min=mutation_sigma_min,
            mirror_episode_prob=mirror_episode_prob,
            evaluate_both_mirrors=evaluate_both_mirrors,
            verbose=True,
            dnf_time_for_plot=env_max_time,
            checkpoint_every=checkpoint_every,
            training_config=dict(
                map_name=map_name,
                max_steps=max_steps,
                env_max_time=env_max_time,
                env_dt_ref=env_dt_ref,
                env_dt_ratio_clip=env_dt_ratio_clip,
                vertical_mode=vertical_mode,
                surface_probe_height=surface_probe_height,
                surface_ray_lift=surface_ray_lift,
                action_mode=action_mode,
                policy_action_scale=policy_action_scale.tolist(),
                hidden_activation=trainer.hidden_activation,
                hidden_activations=list(trainer.hidden_activations),
                target_steer_deadzone=target_steer_deadzone,
                evaluate_both_mirrors=bool(evaluate_both_mirrors),
                finetune_from_checkpoint=resume_checkpoint,
                initial_population_source=initial_population_source,
                initial_population_source_kind=initial_population_source_kind,
                seed_model_path=seed_model_path,
                seed_model_exact_copies=seed_model_exact_copies,
                seed_model_mutation_probs=list(seed_model_mutation_probs),
                seed_model_mutation_sigmas=list(seed_model_mutation_sigmas),
                mirror_episode_prob=mirror_episode_prob,
                max_touches=max_touches,
                start_idle_max_time=start_idle_max_time,
                selection_fitness_mode=selection_fitness_mode,
                selection_mode=selection_mode,
                compare_by_ranking_key=bool(Individual.COMPARE_BY_RANKING_KEY),
                ranking_mode=ranking_mode,
                ranking_key=ranking_key,
                ranking_key_expression=canonical_ranking_key_expression(ranking_key),
                ranking_progress_source=ranking_progress_source,
                moo_objective_mode=moo_objective_mode,
                moo_objective_subset=moo_objective_subset,
                moo_objective_priority=moo_objective_priority,
                pareto_tiebreak=pareto_tiebreak,
                moo_objective_names=list(trainer.moo_objective_names),
                mutation_prob_decay=mutation_prob_decay,
                mutation_prob_min=mutation_prob_min,
                mutation_sigma_decay=mutation_sigma_decay,
                mutation_sigma_min=mutation_sigma_min,
            ),
        )

        if trainer.best_individual is not None:
            print(
                f"\nBest overall: finished={trainer.best_individual.finished}, "
                f"crashes={trainer.best_individual.crashes}, "
                f"progress={trainer.best_individual.discrete_progress:.1f}%, "
                f"dense={trainer.best_individual.dense_progress:.1f}%, "
                f"time={trainer.best_individual.time:.2f}s"
            )
        if logger is not None:
            print(f"Run artifacts saved in: {logger.run_dir}")

    finally:
        env.close()
    input("Training finished...")
