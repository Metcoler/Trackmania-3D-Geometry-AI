from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


TRAJECTORY_MANIFEST_HEADERS = [
    "generation",
    "rank",
    "original_index",
    "finished",
    "crashes",
    "time",
    "dense_progress",
    "path_file",
]


def should_log_trajectory(
    *,
    mode: str,
    generation: int,
    rank: int,
    finished: int,
    final_generation: int,
    top_k: int,
) -> bool:
    mode = str(mode).strip().lower()
    if mode == "off":
        return False
    if mode == "all":
        return True

    rank = int(rank)
    top_k = max(0, int(top_k))
    is_top = top_k > 0 and rank <= top_k
    if mode == "top":
        return is_top
    if mode == "top-finishers-final":
        return is_top or int(finished) > 0 or int(generation) >= int(final_generation)
    raise ValueError("trajectory_log_mode must be one of: off, top, top-finishers-final, all.")


class TrajectoryLogger:
    """Write compact rollout trajectories as NPZ files plus a small CSV manifest."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.trajectory_dir = self.run_dir / "trajectories"
        self.manifest_path = self.trajectory_dir / "trajectory_manifest.csv"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            with self.manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=TRAJECTORY_MANIFEST_HEADERS)
                writer.writeheader()

    @staticmethod
    def _records_to_arrays(records: Iterable[dict], log_actions: bool) -> dict[str, np.ndarray]:
        rows = list(records)
        count = len(rows)
        arrays: dict[str, np.ndarray] = {
            "step": np.arange(count, dtype=np.float32),
            "time": np.zeros(count, dtype=np.float32),
            "x": np.zeros(count, dtype=np.float32),
            "y": np.zeros(count, dtype=np.float32),
            "z": np.zeros(count, dtype=np.float32),
            "speed": np.zeros(count, dtype=np.float32),
            "dense_progress": np.zeros(count, dtype=np.float32),
        }
        if log_actions:
            arrays["gas"] = np.zeros(count, dtype=np.float32)
            arrays["brake"] = np.zeros(count, dtype=np.float32)
            arrays["steer"] = np.zeros(count, dtype=np.float32)

        for index, row in enumerate(rows):
            arrays["step"][index] = float(row.get("step", index))
            arrays["time"][index] = float(row.get("time", 0.0))
            if "x" in row:
                arrays["x"][index] = float(row.get("x", 0.0))
                arrays["y"][index] = float(row.get("y", 0.0))
                arrays["z"][index] = float(row.get("z", 0.0))
            else:
                position = np.asarray(row.get("position", np.zeros(3)), dtype=np.float32).reshape(-1)
                arrays["x"][index] = float(position[0]) if position.size > 0 else 0.0
                if position.size >= 3:
                    arrays["y"][index] = float(position[1])
                    arrays["z"][index] = float(position[2])
                elif position.size >= 2:
                    arrays["y"][index] = 0.0
                    arrays["z"][index] = float(position[1])
            arrays["speed"][index] = float(row.get("speed", 0.0))
            arrays["dense_progress"][index] = float(row.get("dense_progress", 0.0))
            if log_actions:
                action = np.asarray(row.get("action", np.zeros(3)), dtype=np.float32).reshape(-1)
                arrays["gas"][index] = float(row.get("gas", action[0] if action.size > 0 else 0.0))
                arrays["brake"][index] = float(row.get("brake", action[1] if action.size > 1 else 0.0))
                arrays["steer"][index] = float(row.get("steer", action[2] if action.size > 2 else 0.0))

        return arrays

    def save(
        self,
        *,
        generation: int,
        rank: int,
        original_index: int,
        finished: int,
        crashes: int,
        time_value: float,
        dense_progress: float,
        trajectory: Iterable[dict],
        log_actions: bool = False,
    ) -> Path | None:
        arrays = self._records_to_arrays(trajectory, log_actions=log_actions)
        if arrays["step"].size == 0:
            return None

        file_name = (
            f"gen_{int(generation):04d}_"
            f"rank_{int(rank):03d}_"
            f"idx_{int(original_index):03d}_"
            f"finish_{int(finished)}.npz"
        )
        path = self.trajectory_dir / file_name
        np.savez_compressed(path, **arrays)
        relative_path = path.relative_to(self.run_dir)
        with self.manifest_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRAJECTORY_MANIFEST_HEADERS)
            writer.writerow(
                {
                    "generation": int(generation),
                    "rank": int(rank),
                    "original_index": int(original_index),
                    "finished": int(finished),
                    "crashes": int(crashes),
                    "time": float(time_value),
                    "dense_progress": float(dense_progress),
                    "path_file": str(relative_path).replace("\\", "/"),
                }
            )
        return path
