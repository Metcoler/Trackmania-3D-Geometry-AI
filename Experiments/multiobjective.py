from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


TRACKMANIA_RACING_OBJECTIVES = [
    "finish",
    "progress",
    "speed_for_progress",
    "safe_progress",
    "path_efficiency",
]


@dataclass(frozen=True)
class ParetoOrdering:
    order: list[int]
    ranks: np.ndarray
    crowding: np.ndarray
    fronts: list[list[int]]
    objectives: np.ndarray
    objective_names: list[str]


def objective_names_for_mode(mode: str) -> list[str]:
    mode = str(mode).strip().lower()
    if mode == "trackmania_racing":
        return list(TRACKMANIA_RACING_OBJECTIVES)
    raise ValueError("objective mode must be trackmania_racing.")


def trackmania_racing_objectives(
    metrics: dict,
    max_time: float,
    estimated_path_length: float,
    max_episode_distance: float,
) -> np.ndarray:
    """Return maximization objectives for racing without hand-tuned weights.

    The objectives deliberately gate speed, safety and path efficiency by
    progress. This keeps a parked or instantly-crashed car from becoming a
    strong Pareto solution just because it is "safe" or quick to terminate.
    """

    finished = 1.0 if int(metrics.get("finished", 0)) > 0 else 0.0
    crashes = max(0.0, float(metrics.get("crashes", 0.0)))
    max_crashes = max(1.0, float(metrics.get("max_crashes", metrics.get("max_touches", 1.0))))
    progress = max(
        float(metrics.get("progress", 0.0)),
        float(metrics.get("dense_progress", metrics.get("progress", 0.0))),
    )
    progress_norm = float(np.clip(progress / 100.0, 0.0, 1.0))
    time_value = max(0.0, float(metrics.get("time", 0.0)))
    time_norm = float(np.clip(time_value / max(1e-6, float(max_time)), 0.0, 1.0))
    distance = max(0.0, float(metrics.get("distance", 0.0)))

    estimated_path_length = max(1e-6, float(estimated_path_length))
    max_episode_distance = max(estimated_path_length, float(max_episode_distance), 1e-6)
    ideal_distance = progress_norm * estimated_path_length
    excess_distance = max(0.0, distance - ideal_distance)
    excess_norm = float(np.clip(excess_distance / max_episode_distance, 0.0, 1.0))

    speed_for_progress = progress_norm * (1.0 - time_norm)
    safe_progress = progress_norm * (1.0 - float(np.clip(crashes / max_crashes, 0.0, 1.0)))
    path_efficiency = progress_norm * (1.0 - excess_norm)

    return np.asarray(
        [
            finished,
            progress_norm,
            speed_for_progress,
            safe_progress,
            path_efficiency,
        ],
        dtype=np.float64,
    )


def objectives_from_metrics(
    metrics: dict,
    mode: str,
    max_time: float,
    estimated_path_length: float,
    max_episode_distance: float,
) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode == "trackmania_racing":
        return trackmania_racing_objectives(
            metrics=metrics,
            max_time=max_time,
            estimated_path_length=estimated_path_length,
            max_episode_distance=max_episode_distance,
        )
    raise ValueError("objective mode must be trackmania_racing.")


def dominates(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> bool:
    """True when left Pareto-dominates right. All objectives are maximized."""

    return bool(np.all(left >= right - eps) and np.any(left > right + eps))


def non_dominated_sort(objectives: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    objectives = np.asarray(objectives, dtype=np.float64)
    population_size = int(objectives.shape[0])
    domination_counts = np.zeros(population_size, dtype=np.int32)
    dominated: list[list[int]] = [[] for _ in range(population_size)]
    fronts: list[list[int]] = [[]]

    for p in range(population_size):
        for q in range(population_size):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                dominated[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                domination_counts[p] += 1
        if domination_counts[p] == 0:
            fronts[0].append(p)

    ranks = np.full(population_size, fill_value=-1, dtype=np.int32)
    current_front = 0
    while current_front < len(fronts) and fronts[current_front]:
        next_front: list[int] = []
        for p in fronts[current_front]:
            ranks[p] = current_front
            for q in dominated[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        current_front += 1
        if next_front:
            fronts.append(next_front)

    return fronts, ranks


def crowding_distance(objectives: np.ndarray, front: Sequence[int]) -> dict[int, float]:
    objectives = np.asarray(objectives, dtype=np.float64)
    front = [int(index) for index in front]
    if not front:
        return {}
    if len(front) <= 2:
        return {index: float("inf") for index in front}

    distances = {index: 0.0 for index in front}
    front_values = objectives[front]
    objective_count = int(front_values.shape[1])

    for objective_idx in range(objective_count):
        values = front_values[:, objective_idx]
        order = np.argsort(values)
        min_value = float(values[order[0]])
        max_value = float(values[order[-1]])
        distances[front[order[0]]] = float("inf")
        distances[front[order[-1]]] = float("inf")
        span = max_value - min_value
        if span <= 1e-12:
            continue
        for pos in range(1, len(front) - 1):
            index = front[order[pos]]
            if np.isinf(distances[index]):
                continue
            prev_value = float(values[order[pos - 1]])
            next_value = float(values[order[pos + 1]])
            distances[index] += (next_value - prev_value) / span

    return distances


def priority_tuple(objectives: np.ndarray, index: int, priority_indices: Sequence[int]) -> tuple[float, ...]:
    return tuple(float(objectives[int(index), int(objective_idx)]) for objective_idx in priority_indices)


def pareto_order(
    objectives: np.ndarray,
    priority_indices: Sequence[int] | None = None,
    tiebreak: str = "priority",
) -> ParetoOrdering:
    """Order individuals with NSGA-II fronts and optional priority tie-break.

    `priority` tie-break is inspired by the paper's modified NSGA-II: Pareto
    fronts preserve multi-objective diversity, while the last/within-front
    ordering can still prefer the project owner's known objective priorities.
    """

    objectives = np.asarray(objectives, dtype=np.float64)
    if objectives.ndim != 2:
        raise ValueError("objectives must be a 2D array.")
    fronts, ranks = non_dominated_sort(objectives)
    priority_indices = list(range(objectives.shape[1])) if priority_indices is None else list(priority_indices)
    tiebreak = str(tiebreak).strip().lower()
    if tiebreak not in {"priority", "crowding"}:
        raise ValueError("tiebreak must be priority or crowding.")

    crowding = np.zeros(objectives.shape[0], dtype=np.float64)
    ordered: list[int] = []
    for front in fronts:
        distances = crowding_distance(objectives, front)
        for index, value in distances.items():
            crowding[index] = value
        if tiebreak == "crowding":
            front_order = sorted(front, key=lambda idx: (distances.get(idx, 0.0),), reverse=True)
        else:
            front_order = sorted(
                front,
                key=lambda idx: (
                    priority_tuple(objectives, idx, priority_indices),
                    distances.get(idx, 0.0),
                ),
                reverse=True,
            )
        ordered.extend(front_order)

    return ParetoOrdering(
        order=ordered,
        ranks=ranks,
        crowding=crowding,
        fronts=fronts,
        objectives=objectives,
        objective_names=list(TRACKMANIA_RACING_OBJECTIVES),
    )
