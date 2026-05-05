from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NeuralPolicy import NeuralPolicy

from Experiments.tm2d_env import TM2DPhysicsConfig, TM2DRewardConfig, TM2DSimEnv
from Experiments.tm_map_plotting import _all_road_heights, add_map_legend, render_map_background
from Map import Map


MAP_NAMES = (
    "single_surface_flat",
    "multi_surface_flat",
    "single_surface_height",
)

AGENT_COLOR = "#7dd3fc"
TEACHER_COLOR = "#94a3b8"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_")


def parse_maps(value: str) -> list[str]:
    maps = [part.strip() for part in str(value).split(",") if part.strip()]
    if not maps:
        raise ValueError("--maps must contain at least one map name")
    return maps


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_latest_dataset(data_root: Path, map_name: str) -> Path:
    candidates: list[Path] = []
    for config_path in data_root.glob("*/config.json"):
        try:
            config = read_json(config_path)
        except Exception:
            continue
        if str(config.get("map_name")) != str(map_name):
            continue
        if not bool(config.get("vertical_mode", False)):
            continue
        if not bool(config.get("multi_surface_mode", False)):
            continue
        if int(config.get("observation_dim", 0)) != 53:
            continue
        candidates.append(config_path.parent)
    if not candidates:
        raise FileNotFoundError(f"No v3d+surface dataset found for map {map_name!r} under {data_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_latest_model(specialist_root: Path, map_name: str) -> Path:
    search_root = specialist_root / map_name
    if not search_root.exists():
        search_root = specialist_root
    models = list(search_root.rglob("best_model.pt"))
    if not models:
        raise FileNotFoundError(f"No best_model.pt found for map {map_name!r} under {search_root}")
    return max(models, key=lambda path: path.stat().st_mtime)


def attempt_files(dataset_dir: Path, limit: int) -> list[Path]:
    files = sorted((dataset_dir / "attempts").glob("attempt_*.npz"))
    if len(files) < int(limit):
        raise ValueError(f"{dataset_dir} has only {len(files)} attempts, expected at least {limit}.")
    return files[: int(limit)]


def load_teacher_path(path: Path) -> dict[str, Any]:
    with np.load(path) as data:
        positions = np.asarray(data["positions"], dtype=np.float32)
        laser_clearances = np.asarray(data.get("laser_clearances", np.empty((0, 0))), dtype=np.float32)
        crashes = np.asarray(data.get("crashes", np.zeros((positions.shape[0],), dtype=np.int32)), dtype=np.int32)
        finished = int(np.asarray(data.get("finish_finished", [0])).reshape(-1)[0])
        finish_time = float(np.asarray(data.get("finish_time", [0.0])).reshape(-1)[0])
        progress = float(np.asarray(data.get("finish_dense_progress", data.get("finish_progress", [0.0]))).reshape(-1)[0])
        distance = float(np.asarray(data.get("finish_distance", [0.0])).reshape(-1)[0])

    min_clearance = float(np.min(laser_clearances)) if laser_clearances.size else float("inf")
    crash_indices = np.flatnonzero(crashes > 0)
    if crash_indices.size == 0 and laser_clearances.size:
        per_frame_min = np.min(laser_clearances, axis=1)
        crash_indices = np.flatnonzero(per_frame_min <= 0.0)
    crash_index = int(crash_indices[0]) if crash_indices.size else -1
    return {
        "path": path,
        "positions": positions,
        "frames": int(positions.shape[0]),
        "finished": finished,
        "finish_time": finish_time,
        "progress": progress,
        "distance": distance,
        "min_clearance": min_clearance,
        "crash_index": crash_index,
    }


def rollout_agent(policy: NeuralPolicy, map_name: str, max_time: float, seed: int) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    env = TM2DSimEnv(
        map_name=map_name,
        max_time=float(max_time),
        reward_config=TM2DRewardConfig(mode="delta_finished_progress_time_crashes"),
        physics_config=TM2DPhysicsConfig().with_tick_profile("fixed100"),
        seed=int(seed),
        collision_mode="lidar",
        vertical_mode=True,
        multi_surface_mode=True,
        binary_gas_brake=True,
    )
    obs, info = env.reset(seed=int(seed))
    records: list[dict[str, float]] = []
    total_reward = 0.0
    terminated = False
    truncated = False
    try:
        for step in range(10000):
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            records.append(
                {
                    "step": float(step + 1),
                    "time": float(info.get("time", env.time)),
                    "x": float(info.get("x", env.position[0])),
                    "y": float(info.get("y", 0.0)),
                    "z": float(info.get("z", env.position[1])),
                    "speed": float(info.get("speed", env.speed)),
                    "progress": float(info.get("progress", info.get("dense_progress", 0.0))),
                    "block_progress": float(info.get("block_progress", info.get("discrete_progress", 0.0))),
                    "min_laser_clearance": float(info.get("min_laser_clearance", float("inf"))),
                    "crashes": float(info.get("crashes", 0.0)),
                    "gas": float(action[0]) if len(action) > 0 else 0.0,
                    "brake": float(action[1]) if len(action) > 1 else 0.0,
                    "steer": float(action[2]) if len(action) > 2 else 0.0,
                }
            )
            if terminated or truncated:
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    finished = int(info.get("finished", 0))
    crashes = int(info.get("crashes", 0))
    if terminated and finished <= 0 and crashes <= 0:
        crashes = 1
    metrics = {
        "finished": finished,
        "crashes": crashes,
        "timeout": int(bool(truncated) and finished <= 0 and crashes <= 0),
        "progress": float(info.get("progress", info.get("dense_progress", 0.0))),
        "block_progress": float(info.get("block_progress", info.get("discrete_progress", 0.0))),
        "time": float(info.get("time", env.time)),
        "distance": float(info.get("distance", env.distance)),
        "reward": float(total_reward),
        "steps": int(len(records)),
        "terminated": int(bool(terminated)),
        "truncated": int(bool(truncated)),
    }
    if not records:
        trajectory = {key: np.asarray([], dtype=np.float32) for key in ["step", "time", "x", "y", "z"]}
    else:
        trajectory = {
            key: np.asarray([record[key] for record in records], dtype=np.float32)
            for key in records[0].keys()
        }
    return metrics, trajectory


def save_agent_trajectory(path: Path, trajectory: dict[str, np.ndarray], metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **trajectory, metrics_json=np.asarray([json.dumps(metrics, sort_keys=True)]))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def plot_paths(
    map_name: str,
    teacher_paths: list[dict[str, Any]],
    agent_trajectory: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    game_map = Map(map_name)
    fig, ax = plt.subplots(figsize=(13.35, 8.1))
    projection = render_map_background(ax, game_map, show_legend=False, alpha=0.92)

    for teacher in teacher_paths:
        positions = np.asarray(teacher["positions"], dtype=np.float32)
        if positions.shape[0] < 2:
            continue
        projected = projection.points(positions[:, [0, 2]])
        ax.plot(
            projected[:, 0],
            projected[:, 1],
            color=TEACHER_COLOR,
            linewidth=0.9,
            alpha=0.34,
            zorder=80,
        )

    if agent_trajectory.get("x", np.asarray([])).size >= 2:
        agent_xz = np.stack([agent_trajectory["x"], agent_trajectory["z"]], axis=1)
        projected_agent = projection.points(agent_xz)
        ax.plot(
            projected_agent[:, 0],
            projected_agent[:, 1],
            color=AGENT_COLOR,
            linewidth=3.2,
            alpha=0.96,
            zorder=100,
            label="trained agent",
        )
        min_clearance = np.asarray(
            agent_trajectory.get("min_laser_clearance", np.full(agent_xz.shape[0], np.inf)),
            dtype=np.float32,
        )
        crashes = np.asarray(
            agent_trajectory.get("crashes", np.zeros(agent_xz.shape[0], dtype=np.float32)),
            dtype=np.float32,
        )
        crash_indices = np.flatnonzero((crashes > 0.0) | (min_clearance <= 0.0))
        if crash_indices.size:
            idx = int(crash_indices[0])
            ax.scatter(
                [projected_agent[idx, 0]],
                [projected_agent[idx, 1]],
                s=190,
                facecolors="none",
                edgecolors=AGENT_COLOR,
                linewidths=2.8,
                zorder=120,
            )

    heights = _all_road_heights(game_map)
    ax.set_title(f"{map_name}: teacher paths and supervised agent rollout", fontsize=13, pad=6)
    fig.subplots_adjust(left=0.055, right=0.835, top=0.94, bottom=0.13)
    add_map_legend(fig, ax, game_map, float(np.min(heights)), float(np.max(heights)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def analyze_map(
    map_name: str,
    data_root: Path,
    specialist_root: Path,
    output_dir: Path,
    teacher_count: int,
    max_time: float,
    seed: int,
) -> dict[str, Any]:
    dataset_dir = find_latest_dataset(data_root, map_name)
    model_path = find_latest_model(specialist_root, map_name)
    policy, extra = NeuralPolicy.load(str(model_path), map_location="cpu")
    if int(policy.obs_dim) != 53:
        raise ValueError(f"{model_path} has obs_dim={policy.obs_dim}, expected 53.")

    teachers = [load_teacher_path(path) for path in attempt_files(dataset_dir, teacher_count)]
    metrics, trajectory = rollout_agent(policy, map_name=map_name, max_time=max_time, seed=seed)

    map_dir = output_dir / safe_name(map_name)
    save_agent_trajectory(map_dir / "agent_trajectory.npz", trajectory, metrics)
    write_csv(
        map_dir / "teacher_paths_summary.csv",
        [
            {
                "attempt_file": str(item["path"]),
                "frames": item["frames"],
                "finished": item["finished"],
                "finish_time": item["finish_time"],
                "progress": item["progress"],
                "distance": item["distance"],
                "min_clearance": item["min_clearance"],
                "crash_index": item["crash_index"],
            }
            for item in teachers
        ],
        [
            "attempt_file",
            "frames",
            "finished",
            "finish_time",
            "progress",
            "distance",
            "min_clearance",
            "crash_index",
        ],
    )
    write_csv(
        map_dir / "agent_rollout_metrics.csv",
        [metrics],
        [
            "finished",
            "crashes",
            "timeout",
            "progress",
            "block_progress",
            "time",
            "distance",
            "reward",
            "steps",
            "terminated",
            "truncated",
        ],
    )
    plot_path = map_dir / f"{safe_name(map_name)}_teacher_agent_paths.png"
    plot_paths(map_name, teachers, trajectory, plot_path)
    return {
        "map_name": map_name,
        "dataset_dir": str(dataset_dir),
        "model_path": str(model_path),
        "plot_path": str(plot_path),
        "teacher_attempts": len(teachers),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze supervised map specialists and plot teacher/agent paths.")
    parser.add_argument("--data-root", default="logs/supervised_data")
    parser.add_argument("--specialist-root", default="logs/supervised_runs_map_specialists_20260505")
    parser.add_argument("--output-dir", default="Experiments/analysis/supervised_map_specialists_20260505")
    parser.add_argument("--maps", default=",".join(MAP_NAMES), help="Comma-separated map names to analyze.")
    parser.add_argument("--teacher-count", type=int, default=10)
    parser.add_argument("--max-time", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=2026050509)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, map_name in enumerate(parse_maps(args.maps)):
        print(f"Analyzing {map_name}...")
        rows.append(
            analyze_map(
                map_name=map_name,
                data_root=Path(args.data_root),
                specialist_root=Path(args.specialist_root),
                output_dir=output_dir,
                teacher_count=int(args.teacher_count),
                max_time=float(args.max_time),
                seed=int(args.seed) + index,
            )
        )
    write_csv(
        output_dir / "summary.csv",
        rows,
        [
            "map_name",
            "dataset_dir",
            "model_path",
            "plot_path",
            "teacher_attempts",
            "finished",
            "crashes",
            "timeout",
            "progress",
            "block_progress",
            "time",
            "distance",
            "reward",
            "steps",
            "terminated",
            "truncated",
        ],
    )
    (output_dir / "REPORT.md").write_text(
        "# Supervised Map Specialists\n\n"
        "Each map-specific supervised model is plotted against the 10 recorded teacher paths. "
        "The thick cyan line is the deterministic TM2D rollout of the trained model.\n",
        encoding="utf-8",
    )
    print(f"Saved supervised map specialist analysis to {output_dir}")


if __name__ == "__main__":
    main()
