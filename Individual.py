import numpy as np
from typing import Optional, Tuple

from NeuralPolicy import HiddenActivations, HiddenDims, NeuralPolicy
from RankingKey import evaluate_ranking_key, finite_or_large


class Individual:
    """
    Genetic algorithm individual.

    Holds a policy network plus evaluation metrics used by lexicographic ranking.
    """
    FINISHED_WEIGHT = 1_000_000_000.0
    PROGRESS_WEIGHT = 1_000_000.0
    TIME_WEIGHT = 10_000.0
    DISTANCE_WEIGHT = 1.0
    COMPARE_BY_RANKING_KEY = False
    RANKING_KEY = "(finished, progress, -time, -crashes, -distance)"
    RANKING_PROGRESS_SOURCE = "dense_progress"
    RANKING_PROGRESS_SOURCES = ("discrete_progress", "dense_progress")

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: HiddenDims,
        act_dim: int,
        genome: Optional[np.ndarray] = None,
        action_scale: Optional[np.ndarray] = None,
        action_mode: str = "delta",
        hidden_activation: HiddenActivations = "tanh",
    ) -> None:
        self.policy = NeuralPolicy(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            act_dim=act_dim,
            genome=genome,
            action_scale=action_scale,
            action_mode=action_mode,
            hidden_activation=hidden_activation,
        )

        self.fitness: Optional[float] = None
        self.discrete_progress: float = 0.0
        self.dense_progress: float = 0.0
        self.time: float = float("inf")
        self.finished: int = 0
        self.crashes: int = 0
        self.distance: float = 0.0
        self.reward: float = 0.0
        self.evaluation_valid: bool = False
        self.evaluation_steps: int = 0
        self.evaluation_terminated: bool = False
        self.evaluation_truncated: bool = False
        self.evaluation_trajectory = None
        self.evaluation_context: str = ""
        self.selection_rank: Optional[int] = None
        self.selection_crowding: float = 0.0
        self.selection_objectives: Optional[Tuple[float, ...]] = None
        self.selection_objective_names: Tuple[str, ...] = ()
        self.selection_mode: str = ""

    @property
    def genome(self) -> np.ndarray:
        return self.policy.genome

    @genome.setter
    def genome(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.shape[0] != self.policy.genome_size:
            raise ValueError(
                f"New genome has size {value.shape[0]}, expected {self.policy.genome_size}."
            )
        self.policy.genome = value
        self.invalidate_evaluation()

    def invalidate_evaluation(self) -> None:
        self.fitness = None
        self.discrete_progress = 0.0
        self.dense_progress = 0.0
        self.time = float("inf")
        self.finished = 0
        self.crashes = 0
        self.distance = 0.0
        self.reward = 0.0
        self.evaluation_valid = False
        self.evaluation_steps = 0
        self.evaluation_terminated = False
        self.evaluation_truncated = False
        self.evaluation_trajectory = None
        self.evaluation_context = ""
        self.selection_rank = None
        self.selection_crowding = 0.0
        self.selection_objectives = None
        self.selection_objective_names = ()
        self.selection_mode = ""

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.policy.act(obs)

    @staticmethod
    def ranking_key_for(
        finished: int,
        crashes: int,
        progress: float,
        time_value: float,
        distance: float,
    ) -> Tuple[float, ...]:
        return Individual.ranking_key_from_values(
            finished=finished,
            crashes=crashes,
            progress=progress,
            time_value=time_value,
            distance=distance,
        )

    @staticmethod
    def ranking_key_from_values(
        finished: int,
        crashes: int,
        progress: float,
        time_value: float,
        distance: float,
        discrete_progress: float | None = None,
        dense_progress: float | None = None,
    ) -> Tuple[float, ...]:
        finished = 1 if int(finished) > 0 else 0
        crashes = max(0, int(crashes))
        progress = float(progress)
        discrete = progress if discrete_progress is None else float(discrete_progress)
        dense = progress if dense_progress is None else float(dense_progress)
        dist = finite_or_large(float(distance))
        t = finite_or_large(float(time_value))
        progress_norm = float(np.clip(progress / 100.0, 0.0, 1.0))
        discrete_norm = float(np.clip(discrete / 100.0, 0.0, 1.0))
        dense_norm = float(np.clip(dense / 100.0, 0.0, 1.0))

        metrics = {
            "finished": float(finished),
            "crashes": float(crashes),
            "progress": progress,
            "progress_norm": progress_norm,
            "ranking_progress": progress,
            "ranking_progress_norm": progress_norm,
            "discrete_progress": discrete,
            "discrete_progress_norm": discrete_norm,
            "dense_progress": dense,
            "dense_progress_norm": dense_norm,
            "time": t,
            "distance": dist,
        }
        return evaluate_ranking_key(Individual.RANKING_KEY, metrics)

    def ranking_progress(self) -> float:
        source = str(Individual.RANKING_PROGRESS_SOURCE)
        if source == "discrete_progress":
            return float(self.discrete_progress)
        if source == "dense_progress":
            return float(self.dense_progress)
        raise ValueError(f"Unknown Individual.RANKING_PROGRESS_SOURCE: {source}")

    def ranking_key(self) -> Tuple[float, ...]:
        return self.ranking_key_from_values(
            finished=self.finished,
            crashes=self.crashes,
            progress=self.ranking_progress(),
            time_value=self.time,
            distance=self.distance,
            discrete_progress=self.discrete_progress,
            dense_progress=self.dense_progress,
        )

    @classmethod
    def compute_scalar_fitness_for(
        cls,
        finished: int,
        crashes: int,
        progress: float,
        time_value: float,
        distance: float,
    ) -> float:
        key = cls.ranking_key_for(
            finished=finished,
            crashes=crashes,
            progress=progress,
            time_value=time_value,
            distance=distance,
        )

        if cls.RANKING_KEY == "(finished, progress, -time, -crashes, -distance)" and len(key) == 5:
            finished_key, progress_key, neg_time_key, neg_crashes_key, neg_dist = key
            time_key = -neg_time_key
            crashes_key = -neg_crashes_key
            dist = -neg_dist

            return (
                finished_key * cls.FINISHED_WEIGHT
                + progress_key * cls.PROGRESS_WEIGHT
                - time_key * cls.TIME_WEIGHT
                - crashes_key * cls.TIME_WEIGHT
                - dist * cls.DISTANCE_WEIGHT
            )

        # Diagnostic scalar only: real GA selection can compare ranking_key()
        # tuples directly via __lt__/__eq__ when fitness is None.
        score = 0.0
        scale = 1.0
        for value in reversed(key):
            score += float(value) * scale
            scale *= 1_000_000.0
        return float(score)

    @staticmethod
    def _is_terminal_score_state(finished: int, crashes: int, time_value: float, max_time: float) -> bool:
        return int(finished) > 0 or int(crashes) > 0 or float(time_value) >= float(max_time)

    @staticmethod
    def _progress_unit_norm(
        path_tile_count: int | None = None,
        progress_bucket: float | None = None,
    ) -> float:
        if path_tile_count is not None and int(path_tile_count) > 1:
            return 1.0 / float(int(path_tile_count) - 1)
        if progress_bucket is not None and float(progress_bucket) > 0.0:
            return float(progress_bucket) / 100.0
        return 0.01

    @classmethod
    def compute_lexicographic_score_for(
        cls,
        finished: int,
        crashes: int,
        progress: float,
        time_value: float,
        distance: float,
        max_time: float,
        path_tile_count: int | None = None,
        progress_bucket: float | None = None,
        estimated_path_length: float | None = None,
        max_episode_distance: float | None = None,
        include_failure_term: bool = False,
        ignore_time_below_progress: float = 0.0,
        distance_mode: str = "all",
    ) -> float:
        """Bounded lexicographic score for GA/RL reward shaping.

        The scale is derived from map/time geometry:
        progress is normalized to `[0, 1]`, one full path-tile progress unit
        dominates the whole time tie-break range, and the time tie-break
        dominates the whole distance tie-break range.
        """

        max_time = max(1e-6, float(max_time))
        progress = float(np.clip(float(progress), 0.0, 100.0))
        time_value = max(0.0, float(time_value))
        distance = max(0.0, float(distance))
        terminal = cls._is_terminal_score_state(finished, crashes, time_value, max_time)
        progress_norm = progress / 100.0
        tile_unit = cls._progress_unit_norm(path_tile_count, progress_bucket)

        score = progress_norm
        if terminal and int(finished) > 0:
            score += 1.0
        elif terminal and include_failure_term:
            score -= 1.0

        if progress >= float(ignore_time_below_progress):
            time_norm = float(np.clip(time_value / max_time, 0.0, 1.0))
            score += tile_unit * (1.0 - time_norm)

            use_distance = distance_mode == "all" or (
                distance_mode == "finish" and terminal and int(finished) > 0
            )
            if use_distance:
                distance_limit_candidates = [
                    value
                    for value in (estimated_path_length, max_episode_distance)
                    if value is not None and float(value) > 0.0
                ]
                if distance_limit_candidates:
                    distance_limit = max(float(value) for value in distance_limit_candidates)
                    distance_norm = float(np.clip(distance / distance_limit, 0.0, 1.0))
                    score += (tile_unit * tile_unit) * (1.0 - distance_norm)

        return float(score)

    @classmethod
    def compute_delta_lexicographic_score_for(
        cls,
        finished: int,
        crashes: int,
        progress: float,
        time_value: float,
        distance: float,
        max_time: float,
        path_tile_count: int | None = None,
        progress_bucket: float | None = None,
        estimated_path_length: float | None = None,
        max_episode_distance: float | None = None,
    ) -> float:
        return cls.compute_lexicographic_score_for(
            finished=finished,
            crashes=crashes,
            progress=progress,
            time_value=time_value,
            distance=distance,
            max_time=max_time,
            path_tile_count=path_tile_count,
            progress_bucket=progress_bucket,
            estimated_path_length=estimated_path_length,
            max_episode_distance=max_episode_distance,
            include_failure_term=False,
            ignore_time_below_progress=0.0,
            distance_mode="all",
        )

    @classmethod
    def compute_terminal_lexicographic_score_for(
        cls,
        finished: int,
        crashes: int,
        progress: float,
        time_value: float,
        distance: float,
        max_time: float,
        path_tile_count: int | None = None,
        progress_bucket: float | None = None,
        estimated_path_length: float | None = None,
        max_episode_distance: float | None = None,
    ) -> float:
        if not cls._is_terminal_score_state(finished, crashes, time_value, max_time):
            return 0.0
        return cls.compute_lexicographic_score_for(
            finished=finished,
            crashes=crashes,
            progress=progress,
            time_value=time_value,
            distance=distance,
            max_time=max_time,
            path_tile_count=path_tile_count,
            progress_bucket=progress_bucket,
            estimated_path_length=estimated_path_length,
            max_episode_distance=max_episode_distance,
            include_failure_term=True,
            ignore_time_below_progress=0.0,
            distance_mode="all",
        )

    @classmethod
    def compute_progress_time_efficiency_score_for(
        cls,
        finished: int,
        crashes: int,
        progress: float,
        time_value: float,
        distance: float,
        max_time: float,
        path_tile_count: int | None = None,
        progress_bucket: float | None = None,
        estimated_path_length: float | None = None,
        max_episode_distance: float | None = None,
        terminal_only: bool = False,
    ) -> float:
        """Progress-primary score with map-derived time and efficiency tie-breaks.

        Human goal: get as far as possible, and for equal progress get there
        sooner. The time term is gated by progress, so a zero-progress crash
        cannot beat a zero-progress attempt by simply ending earlier. Distance
        is only an excess-distance penalty, which avoids rewarding zig-zags.
        """

        if terminal_only and not cls._is_terminal_score_state(finished, crashes, time_value, max_time):
            return 0.0

        max_time = max(1e-6, float(max_time))
        progress = float(np.clip(float(progress), 0.0, 100.0))
        progress_norm = progress / 100.0
        time_value = max(0.0, float(time_value))
        distance = max(0.0, float(distance))
        tile_unit = cls._progress_unit_norm(path_tile_count, progress_bucket)

        score = progress_norm
        if cls._is_terminal_score_state(finished, crashes, time_value, max_time) and int(finished) > 0:
            score += 1.0

        if progress_norm > 0.0:
            time_norm = float(np.clip(time_value / max_time, 0.0, 1.0))
            score += tile_unit * progress_norm * (1.0 - time_norm)

            distance_limit_candidates = [
                value
                for value in (estimated_path_length, max_episode_distance)
                if value is not None and float(value) > 0.0
            ]
            if distance_limit_candidates:
                estimated_length = max(1e-6, float(estimated_path_length or 0.0))
                distance_limit = max(float(value) for value in distance_limit_candidates)
                ideal_distance = progress_norm * estimated_length
                excess_distance = max(0.0, distance - ideal_distance)
                excess_norm = float(np.clip(excess_distance / max(1e-6, distance_limit), 0.0, 1.0))
                score -= (tile_unit * tile_unit) * progress_norm * excess_norm

        return float(score)

    @classmethod
    def compute_delta_progress_time_efficiency_score_for(
        cls,
        finished: int,
        crashes: int,
        progress: float,
        time_value: float,
        distance: float,
        max_time: float,
        path_tile_count: int | None = None,
        progress_bucket: float | None = None,
        estimated_path_length: float | None = None,
        max_episode_distance: float | None = None,
    ) -> float:
        return cls.compute_progress_time_efficiency_score_for(
            finished=finished,
            crashes=crashes,
            progress=progress,
            time_value=time_value,
            distance=distance,
            max_time=max_time,
            path_tile_count=path_tile_count,
            progress_bucket=progress_bucket,
            estimated_path_length=estimated_path_length,
            max_episode_distance=max_episode_distance,
            terminal_only=False,
        )

    @classmethod
    def compute_terminal_progress_time_efficiency_score_for(
        cls,
        finished: int,
        crashes: int,
        progress: float,
        time_value: float,
        distance: float,
        max_time: float,
        path_tile_count: int | None = None,
        progress_bucket: float | None = None,
        estimated_path_length: float | None = None,
        max_episode_distance: float | None = None,
    ) -> float:
        return cls.compute_progress_time_efficiency_score_for(
            finished=finished,
            crashes=crashes,
            progress=progress,
            time_value=time_value,
            distance=distance,
            max_time=max_time,
            path_tile_count=path_tile_count,
            progress_bucket=progress_bucket,
            estimated_path_length=estimated_path_length,
            max_episode_distance=max_episode_distance,
            terminal_only=True,
        )

    @classmethod
    def compute_finished_progress_time_score_for(
        cls,
        finished: int,
        crashes: int,
        progress: float,
        time_value: float,
        distance: float,
        max_time: float,
        path_tile_count: int | None = None,
        progress_bucket: float | None = None,
        terminal_only: bool = False,
    ) -> float:
        """Dense RL score matching the GA tuple `(finished, progress, -time)`.

        The score is intentionally minimal:
        - finish adds one full normalized completion unit;
        - progress is the primary dense curriculum signal;
        - time is only a progress-gated tie-breaker smaller than one map tile.

        Crash/timeout are not given an extra terminal bonus or penalty here.
        They simply stop future progress from accumulating, which avoids the
        old problem where a late crash retroactively punished good driving.
        """

        del distance
        if terminal_only and not cls._is_terminal_score_state(finished, crashes, time_value, max_time):
            return 0.0

        max_time = max(1e-6, float(max_time))
        progress = float(np.clip(float(progress), 0.0, 100.0))
        progress_norm = progress / 100.0
        time_value = max(0.0, float(time_value))
        tile_unit = cls._progress_unit_norm(path_tile_count, progress_bucket)

        score = progress_norm
        if cls._is_terminal_score_state(finished, crashes, time_value, max_time) and int(finished) > 0:
            score += 1.0

        if progress_norm > 0.0:
            time_norm = float(np.clip(time_value / max_time, 0.0, 1.0))
            score += tile_unit * progress_norm * (1.0 - time_norm)

        return float(score)

    @classmethod
    def compute_delta_finished_progress_time_score_for(
        cls,
        finished: int,
        crashes: int,
        progress: float,
        time_value: float,
        distance: float,
        max_time: float,
        path_tile_count: int | None = None,
        progress_bucket: float | None = None,
    ) -> float:
        return cls.compute_finished_progress_time_score_for(
            finished=finished,
            crashes=crashes,
            progress=progress,
            time_value=time_value,
            distance=distance,
            max_time=max_time,
            path_tile_count=path_tile_count,
            progress_bucket=progress_bucket,
            terminal_only=False,
        )

    @classmethod
    def compute_terminal_finished_progress_time_score_for(
        cls,
        finished: int,
        crashes: int,
        progress: float,
        time_value: float,
        distance: float,
        max_time: float,
        path_tile_count: int | None = None,
        progress_bucket: float | None = None,
    ) -> float:
        return cls.compute_finished_progress_time_score_for(
            finished=finished,
            crashes=crashes,
            progress=progress,
            time_value=time_value,
            distance=distance,
            max_time=max_time,
            path_tile_count=path_tile_count,
            progress_bucket=progress_bucket,
            terminal_only=True,
        )

    @classmethod
    def compute_progress_time_safety_score_for(
        cls,
        finished: int,
        crashes: int,
        progress: float,
        time_value: float,
        distance: float,
        max_time: float,
        path_tile_count: int | None = None,
        progress_bucket: float | None = None,
        terminal_only: bool = False,
    ) -> float:
        """SAC-friendly progress score with terminal failure safety.

        The score stays in progress-percent units so SAC gets a clear dense
        signal, but an unsuccessful terminal state subtracts one full track
        completion. That keeps "go farther" as the main objective while making
        both crash and timeout worse than continuing.
        """

        if terminal_only and not cls._is_terminal_score_state(finished, crashes, time_value, max_time):
            return 0.0

        del distance
        max_time = max(1e-6, float(max_time))
        progress = float(np.clip(float(progress), 0.0, 100.0))
        progress_norm = progress / 100.0
        time_value = max(0.0, float(time_value))
        tile_unit = cls._progress_unit_norm(path_tile_count, progress_bucket)
        tile_percent = 100.0 * tile_unit

        score = progress
        if progress_norm > 0.0:
            time_norm = float(np.clip(time_value / max_time, 0.0, 1.0))
            score += tile_percent * progress_norm * (1.0 - time_norm)

        if cls._is_terminal_score_state(finished, crashes, time_value, max_time):
            if int(finished) > 0:
                score += 100.0
            else:
                score -= 100.0

        return float(score)

    @classmethod
    def compute_delta_progress_time_safety_score_for(
        cls,
        finished: int,
        crashes: int,
        progress: float,
        time_value: float,
        distance: float,
        max_time: float,
        path_tile_count: int | None = None,
        progress_bucket: float | None = None,
    ) -> float:
        return cls.compute_progress_time_safety_score_for(
            finished=finished,
            crashes=crashes,
            progress=progress,
            time_value=time_value,
            distance=distance,
            max_time=max_time,
            path_tile_count=path_tile_count,
            progress_bucket=progress_bucket,
            terminal_only=False,
        )

    @classmethod
    def compute_terminal_progress_time_safety_score_for(
        cls,
        finished: int,
        crashes: int,
        progress: float,
        time_value: float,
        distance: float,
        max_time: float,
        path_tile_count: int | None = None,
        progress_bucket: float | None = None,
    ) -> float:
        return cls.compute_progress_time_safety_score_for(
            finished=finished,
            crashes=crashes,
            progress=progress,
            time_value=time_value,
            distance=distance,
            max_time=max_time,
            path_tile_count=path_tile_count,
            progress_bucket=progress_bucket,
            terminal_only=True,
        )

    @classmethod
    def compute_progress_time_block_penalty_score_for(
        cls,
        finished: int,
        crashes: int,
        progress: float,
        time_value: float,
        distance: float,
        max_time: float,
        path_tile_count: int | None = None,
        progress_bucket: float | None = None,
        terminal_only: bool = False,
    ) -> float:
        """Dense progress score with one-block terminal failure penalty.

        This is a softer SAC curriculum version of
        `compute_progress_time_safety_score_for`: a failed episode loses one
        map-derived progress bucket instead of one full track completion.
        """

        if terminal_only and not cls._is_terminal_score_state(finished, crashes, time_value, max_time):
            return 0.0

        del distance
        max_time = max(1e-6, float(max_time))
        progress = float(np.clip(float(progress), 0.0, 100.0))
        progress_norm = progress / 100.0
        time_value = max(0.0, float(time_value))
        tile_unit = cls._progress_unit_norm(path_tile_count, progress_bucket)
        tile_percent = 100.0 * tile_unit

        score = progress
        if progress_norm > 0.0:
            time_norm = float(np.clip(time_value / max_time, 0.0, 1.0))
            score += tile_percent * progress_norm * (1.0 - time_norm)

        if cls._is_terminal_score_state(finished, crashes, time_value, max_time):
            if int(finished) > 0:
                score += 100.0
            else:
                score -= tile_percent

        return float(score)

    @classmethod
    def compute_delta_progress_time_block_penalty_score_for(
        cls,
        finished: int,
        crashes: int,
        progress: float,
        time_value: float,
        distance: float,
        max_time: float,
        path_tile_count: int | None = None,
        progress_bucket: float | None = None,
    ) -> float:
        return cls.compute_progress_time_block_penalty_score_for(
            finished=finished,
            crashes=crashes,
            progress=progress,
            time_value=time_value,
            distance=distance,
            max_time=max_time,
            path_tile_count=path_tile_count,
            progress_bucket=progress_bucket,
            terminal_only=False,
        )

    @classmethod
    def compute_terminal_progress_time_block_penalty_score_for(
        cls,
        finished: int,
        crashes: int,
        progress: float,
        time_value: float,
        distance: float,
        max_time: float,
        path_tile_count: int | None = None,
        progress_bucket: float | None = None,
    ) -> float:
        return cls.compute_progress_time_block_penalty_score_for(
            finished=finished,
            crashes=crashes,
            progress=progress,
            time_value=time_value,
            distance=distance,
            max_time=max_time,
            path_tile_count=path_tile_count,
            progress_bucket=progress_bucket,
            terminal_only=True,
        )

    def compute_scalar_fitness(self) -> float:
        return self._diagnostic_scalar_for_key(self.ranking_key())

    @staticmethod
    def _diagnostic_scalar_for_key(key: Tuple[float, ...]) -> float:
        score = 0.0
        scale = 1.0
        for value in reversed(key):
            score += float(value) * scale
            scale *= 1_000_000.0
        return float(score)

    def __lt__(self, other: "Individual") -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        if (
            not Individual.COMPARE_BY_RANKING_KEY
            and self.fitness is not None
            and other.fitness is not None
            and np.isfinite(self.fitness)
            and np.isfinite(other.fitness)
        ):
            return float(self.fitness) < float(other.fitness)
        return self.ranking_key() < other.ranking_key()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        if (
            not Individual.COMPARE_BY_RANKING_KEY
            and self.fitness is not None
            and other.fitness is not None
            and np.isfinite(self.fitness)
            and np.isfinite(other.fitness)
        ):
            return float(self.fitness) == float(other.fitness)
        return self.ranking_key() == other.ranking_key()

    def __repr__(self) -> str:
        return (
            "Individual("
            f"finished={self.finished}, "
            f"crashes={self.crashes}, "
            f"discrete_progress={self.discrete_progress:.1f}, "
            f"dense_progress={self.dense_progress:.1f}, "
            f"distance={self.distance:.1f}, "
            f"time={self.time:.2f}, "
            f"fitness={self.fitness})"
        )

    def copy(self) -> "Individual":
        new = Individual(
            obs_dim=self.policy.obs_dim,
            hidden_dim=self.policy.hidden_dims,
            act_dim=self.policy.act_dim,
            genome=self.genome.copy(),
            action_scale=self.policy.action_scale.detach().cpu().numpy().copy(),
            action_mode=self.policy.action_mode,
            hidden_activation=self.policy.hidden_activation,
        )
        new.fitness = self.fitness
        new.discrete_progress = self.discrete_progress
        new.dense_progress = self.dense_progress
        new.time = self.time
        new.finished = self.finished
        new.crashes = self.crashes
        new.distance = self.distance
        new.reward = self.reward
        new.evaluation_valid = self.evaluation_valid
        new.evaluation_steps = self.evaluation_steps
        new.evaluation_terminated = self.evaluation_terminated
        new.evaluation_truncated = self.evaluation_truncated
        new.evaluation_trajectory = self.evaluation_trajectory
        new.evaluation_context = self.evaluation_context
        new.selection_rank = self.selection_rank
        new.selection_crowding = self.selection_crowding
        new.selection_objectives = self.selection_objectives
        new.selection_objective_names = self.selection_objective_names
        new.selection_mode = self.selection_mode
        return new

    def mutate(self, mutation_prob: float = 0.1, sigma: float = 0.1) -> None:
        genome = self.genome.copy()
        mask = np.random.rand(genome.size) < mutation_prob
        if np.any(mask):
            genome[mask] += np.random.randn(mask.sum()).astype(np.float32) * sigma
        self.genome = genome

    def crossover(self, other: "Individual") -> "Individual":
        if self.genome.shape != other.genome.shape:
            raise ValueError("Individuals have different genome sizes.")

        child_genome = (
            0.5 * (self.genome.astype(np.float32) + other.genome.astype(np.float32))
        ).astype(np.float32, copy=False)

        return Individual(
            obs_dim=self.policy.obs_dim,
            hidden_dim=self.policy.hidden_dims,
            act_dim=self.policy.act_dim,
            genome=child_genome,
            action_scale=self.policy.action_scale.detach().cpu().numpy().copy(),
            action_mode=self.policy.action_mode,
            hidden_activation=self.policy.hidden_activation,
        )
