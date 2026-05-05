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
from matplotlib import colors as mcolors
from matplotlib import patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NeuralPolicy import NeuralPolicy
from ObservationEncoder import ObservationEncoder

from Experiments.tm2d_env import TM2DPhysicsConfig, TM2DRewardConfig, TM2DSimEnv
from Experiments.tm_map_plotting import _all_road_heights, add_map_legend, render_map_background
from Map import Map


MAP_NAMES = (
    "single_surface_flat",
    "multi_surface_flat",
    "single_surface_height",
)

AGENT_COLOR = "#7dd3fc"
TEACHER_LEGEND_COLOR = "#4b5563"
CRASH_COLOR = "#050505"
WALL_HUG_COLOR = "#fb7185"
SPEED_CMAP = plt.get_cmap("turbo")
PRACTICAL_INFINITY_TOUCHES = 1_000_000
MAP_BOUNDS_MARGIN = 8.0
LAYOUTS = {
    "v2d_asphalt": (False, False, 34),
    "v2d_surface": (False, True, 40),
    "v3d_asphalt": (True, False, 43),
    "v3d_surface": (True, True, 53),
}


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_")


def parse_maps(value: str) -> list[str]:
    maps = [part.strip() for part in str(value).split(",") if part.strip()]
    if not maps:
        raise ValueError("--maps must contain at least one map name")
    return maps


def parse_layout(value: str) -> str:
    layout = str(value).strip().lower()
    if layout == "any":
        return layout
    if layout not in LAYOUTS:
        raise ValueError(f"Unsupported layout {value!r}. Use any or one of {sorted(LAYOUTS)}.")
    return layout


def modes_from_layout(layout: str) -> tuple[bool, bool, int]:
    if layout not in LAYOUTS:
        raise ValueError(f"Unsupported layout {layout!r}.")
    return LAYOUTS[layout]


def parse_max_touches(value: str) -> int:
    text = str(value).strip().lower()
    if text in {"inf", "infinite", "infinity", "nekonecno", "nekonečno"}:
        return PRACTICAL_INFINITY_TOUCHES
    touches = int(text)
    if touches < 1:
        raise ValueError("--max-touches must be >= 1 or 'inf'")
    return touches


def parse_slowdown(value: str) -> float:
    slowdown = float(value)
    if not 0.0 <= slowdown <= 1.0:
        raise ValueError("--collision-slowdown must be in [0, 1]")
    return slowdown


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def map_xz_bounds(game_map: Map) -> tuple[float, float, float, float]:
    points: list[np.ndarray] = []
    for mesh_getter in (getattr(game_map, "get_road_mesh", None), getattr(game_map, "get_sensor_walls_mesh", None)):
        if not callable(mesh_getter):
            continue
        try:
            mesh = mesh_getter()
        except Exception:
            continue
        vertices = np.asarray(getattr(mesh, "vertices", np.empty((0, 3))), dtype=np.float32)
        if vertices.ndim == 2 and vertices.shape[0] and vertices.shape[1] >= 3:
            points.append(vertices[:, [0, 2]])
    if not points:
        return 0.0, 0.0, 1.0, 1.0
    all_points = np.vstack(points)
    min_xz = np.min(all_points, axis=0)
    max_xz = np.max(all_points, axis=0)
    return float(min_xz[0]), float(min_xz[1]), float(max_xz[0]), float(max_xz[1])


def find_latest_dataset(data_root: Path, map_name: str, source_layout: str = "any") -> Path:
    source_layout = parse_layout(source_layout)
    candidates: list[Path] = []
    for config_path in data_root.glob("*/config.json"):
        try:
            config = read_json(config_path)
        except Exception:
            continue
        if str(config.get("map_name")) != str(map_name):
            continue
        if source_layout != "any":
            expected_vertical, expected_multi_surface, expected_obs_dim = modes_from_layout(source_layout)
            if bool(config.get("vertical_mode", False)) != expected_vertical:
                continue
            if bool(config.get("multi_surface_mode", False)) != expected_multi_surface:
                continue
            observed_dim = int(config.get("observation_dim", 0))
            if observed_dim and observed_dim != int(expected_obs_dim):
                continue
        candidates.append(config_path.parent)
    if not candidates:
        raise FileNotFoundError(
            f"No supervised dataset found for map {map_name!r} under {data_root} "
            f"with source layout {source_layout!r}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_latest_model(specialist_root: Path, map_name: str) -> Path:
    search_root = specialist_root / map_name
    if not search_root.exists():
        search_root = specialist_root
    models = list(search_root.rglob("best_model.pt"))
    if not models:
        raise FileNotFoundError(f"No best_model.pt found for map {map_name!r} under {search_root}")
    return max(models, key=lambda path: path.stat().st_mtime)


def infer_policy_modes(policy: NeuralPolicy, extra: dict[str, Any], model_path: Path) -> tuple[bool, bool]:
    inferred = ObservationEncoder.infer_modes_from_dim(int(policy.obs_dim))
    if inferred is not None:
        return bool(inferred[0]), bool(inferred[1])

    if "vertical_mode" in extra and "multi_surface_mode" in extra:
        return bool(extra["vertical_mode"]), bool(extra["multi_surface_mode"])

    summary_path = model_path.with_name("summary.json")
    if summary_path.exists():
        try:
            summary = read_json(summary_path)
            if "vertical_mode" in summary and "multi_surface_mode" in summary:
                return bool(summary["vertical_mode"]), bool(summary["multi_surface_mode"])
        except Exception:
            pass

    raise ValueError(
        f"Cannot infer observation layout for {model_path} with obs_dim={policy.obs_dim}. "
        "Use a model saved by the current SupervisedTraining.py."
    )


def attempt_files(dataset_dir: Path, limit: int) -> list[Path]:
    files = sorted((dataset_dir / "attempts").glob("attempt_*.npz"))
    if len(files) < int(limit):
        raise ValueError(f"{dataset_dir} has only {len(files)} attempts, expected at least {limit}.")
    return files[: int(limit)]


def load_teacher_path(path: Path) -> dict[str, Any]:
    with np.load(path) as data:
        positions = np.asarray(data["positions"], dtype=np.float32)
        speeds = np.asarray(data.get("speeds", np.zeros((positions.shape[0],), dtype=np.float32)), dtype=np.float32)
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
        "speeds": speeds,
        "frames": int(positions.shape[0]),
        "finished": finished,
        "finish_time": finish_time,
        "progress": progress,
        "distance": distance,
        "min_clearance": min_clearance,
        "crash_index": crash_index,
    }


def rollout_agent(
    policy: NeuralPolicy,
    map_name: str,
    vertical_mode: bool,
    multi_surface_mode: bool,
    max_time: float,
    seed: int,
    max_touches: int,
    collision_slowdown: float,
    collision_path_axis_bias: float,
    collision_normal_restitution: float,
    collision_wall_tangent_blend: float,
    collision_reference_window: int,
    collision_skip_recent: int,
    never_stop: bool,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    speed_retention = float(np.clip(1.0 - float(collision_slowdown), 0.0, 1.0))
    env = TM2DSimEnv(
        map_name=map_name,
        max_time=float(max_time),
        reward_config=TM2DRewardConfig(mode="delta_finished_progress_time_crashes"),
        physics_config=TM2DPhysicsConfig().with_tick_profile("fixed100"),
        seed=int(seed),
        collision_mode="lidar",
        vertical_mode=bool(vertical_mode),
        multi_surface_mode=bool(multi_surface_mode),
        binary_gas_brake=True,
        max_touches=int(max_touches),
        collision_bounce_speed_retention=speed_retention,
        collision_bounce_path_axis_bias=float(collision_path_axis_bias),
        collision_bounce_normal_restitution=float(collision_normal_restitution),
        collision_bounce_wall_tangent_blend=float(collision_wall_tangent_blend),
        collision_bounce_reference_window=int(collision_reference_window),
        collision_bounce_skip_recent=int(collision_skip_recent),
        never_stop=bool(never_stop),
    )
    min_x, min_z, max_x, max_z = env.geometry.bounds
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
                    "touch_count": float(info.get("touch_count", info.get("crashes", 0.0))),
                    "collision_event": float(info.get("collision_event", 0.0)),
                    "collision_x": float(info.get("collision_x", info.get("x", env.position[0]))),
                    "collision_y": float(info.get("collision_y", 0.0)),
                    "collision_z": float(info.get("collision_z", info.get("z", env.position[1]))),
                    "wall_hug_active": float(info.get("wall_hug_active", 0.0)),
                    "wall_hug_frames": float(info.get("wall_hug_frames", 0.0)),
                    "bounce_recovered": float(info.get("bounce_recovered", 0.0)),
                    "bounce_clearance": float(info.get("bounce_clearance", float("inf"))),
                    "bounce_reference_samples": float(info.get("bounce_reference_samples", 0.0)),
                    "on_road": float(env.geometry.point_on_road(env.position, margin=0.25)),
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
        "max_touches": int(max_touches),
        "collision_slowdown": float(collision_slowdown),
        "collision_bounce_speed_retention": speed_retention,
        "collision_bounce_path_axis_bias": float(collision_path_axis_bias),
        "collision_bounce_normal_restitution": float(collision_normal_restitution),
        "collision_bounce_wall_tangent_blend": float(collision_wall_tangent_blend),
        "collision_bounce_reference_window": int(collision_reference_window),
        "collision_bounce_skip_recent": int(collision_skip_recent),
        "never_stop": int(bool(never_stop)),
    }
    if not records:
        trajectory = {key: np.asarray([], dtype=np.float32) for key in ["step", "time", "x", "y", "z"]}
    else:
        trajectory = {
            key: np.asarray([record[key] for record in records], dtype=np.float32)
            for key in records[0].keys()
        }
    left_bounds_index = -1
    if trajectory.get("x", np.asarray([])).size:
        x = trajectory["x"]
        z = trajectory["z"]
        inside = (
            (x >= (min_x - MAP_BOUNDS_MARGIN))
            & (x <= (max_x + MAP_BOUNDS_MARGIN))
            & (z >= (min_z - MAP_BOUNDS_MARGIN))
            & (z <= (max_z + MAP_BOUNDS_MARGIN))
        )
        outside_indices = np.flatnonzero(~inside)
        if outside_indices.size:
            left_bounds_index = int(outside_indices[0])
    metrics.update(
        {
            "left_map_bounds": int(left_bounds_index >= 0),
            "left_map_bounds_index": int(left_bounds_index),
            "left_map_bounds_time": (
                float(trajectory["time"][left_bounds_index]) if left_bounds_index >= 0 else -1.0
            ),
        }
    )
    off_road_index = -1
    if trajectory.get("on_road", np.asarray([])).size:
        off_road_indices = np.flatnonzero(trajectory["on_road"] <= 0.0)
        if off_road_indices.size:
            off_road_index = int(off_road_indices[0])
    metrics.update(
        {
            "left_road_surface": int(off_road_index >= 0),
            "left_road_surface_index": int(off_road_index),
            "left_road_surface_time": (
                float(trajectory["time"][off_road_index]) if off_road_index >= 0 else -1.0
            ),
        }
    )
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


def plot_speed_gradient(
    ax,
    projected_points: np.ndarray,
    speed: np.ndarray,
    *,
    linewidth: float = 3.6,
    alpha: float = 0.96,
    zorder: int = 100,
    norm: mcolors.Normalize | None = None,
    segment_mask: np.ndarray | None = None,
) -> None:
    if projected_points.shape[0] < 2:
        return None
    segments = np.stack([projected_points[:-1], projected_points[1:]], axis=1)
    segment_speed = np.asarray(speed[: max(0, projected_points.shape[0] - 1)], dtype=np.float32)
    if segment_mask is not None:
        mask = np.asarray(segment_mask, dtype=bool).reshape(-1)
        mask = mask[: segments.shape[0]]
        if mask.size < segments.shape[0]:
            mask = np.pad(mask, (0, segments.shape[0] - mask.size), constant_values=True)
        segments = segments[mask]
        segment_speed = segment_speed[mask]
        if segments.shape[0] <= 0:
            return None
    collection = LineCollection(
        segments,
        cmap=SPEED_CMAP,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    collection.set_array(segment_speed)
    if norm is None:
        norm = mcolors.Normalize(
            vmin=float(np.nanmin(speed)),
            vmax=max(float(np.nanmax(speed)), float(np.nanmin(speed)) + 1e-6),
        )
    collection.set_norm(norm)
    ax.add_collection(collection)
    return collection


def plot_path_outline(
    ax,
    projected_points: np.ndarray,
    *,
    linewidth: float,
    alpha: float,
    zorder: int,
    segment_mask: np.ndarray | None = None,
) -> None:
    if projected_points.shape[0] < 2:
        return
    if segment_mask is not None:
        segments = np.stack([projected_points[:-1], projected_points[1:]], axis=1)
        mask = np.asarray(segment_mask, dtype=bool).reshape(-1)
        mask = mask[: segments.shape[0]]
        if mask.size < segments.shape[0]:
            mask = np.pad(mask, (0, segments.shape[0] - mask.size), constant_values=True)
        segments = segments[mask]
        if segments.shape[0] <= 0:
            return
        ax.add_collection(LineCollection(
            segments,
            colors="#111827",
            linewidths=linewidth,
            alpha=alpha,
            zorder=zorder,
            capstyle="round",
            joinstyle="round",
        ))
        return
    ax.plot(
        projected_points[:, 0],
        projected_points[:, 1],
        color="#111827",
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
        solid_capstyle="round",
        solid_joinstyle="round",
    )


def add_path_legend(fig, ax, speed_norm: mcolors.Normalize, mappable) -> None:
    agent_handle = Line2D([0], [0], color=AGENT_COLOR, lw=3.9, alpha=0.96, label="Agent rollout")
    agent_handle.set_path_effects([
        pe.Stroke(linewidth=6.1, foreground="#111827", alpha=0.42),
        pe.Normal(),
    ])
    handles = [
        Line2D([0], [0], color=TEACHER_LEGEND_COLOR, lw=2.1, alpha=0.78, label="Teacher paths"),
        agent_handle,
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CRASH_COLOR,
               markeredgecolor=CRASH_COLOR, markersize=7, label="Crash/touch"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.78),
        frameon=False,
        fontsize=10.5,
        borderaxespad=0.0,
    )
    cax = fig.add_axes([0.862, 0.36, 0.014, 0.27])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("Speed", fontsize=10.5, labelpad=8)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.text(-0.85, 0.02, "slow", transform=cbar.ax.transAxes, fontsize=9,
                 ha="right", va="bottom")
    cbar.ax.text(-0.85, 0.98, "fast", transform=cbar.ax.transAxes, fontsize=9,
                 ha="right", va="top")


def plot_paths(
    map_name: str,
    teacher_paths: list[dict[str, Any]],
    agent_trajectory: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    game_map = Map(map_name)
    fig, ax = plt.subplots(figsize=(13.35, 8.1))
    projection = render_map_background(ax, game_map, show_legend=False, alpha=0.92)
    map_xlim = ax.get_xlim()
    map_ylim = ax.get_ylim()
    min_x, min_z, max_x, max_z = map_xz_bounds(game_map)
    ax.set_autoscale_on(False)

    speed_arrays = [
        np.asarray(teacher.get("speeds", np.asarray([], dtype=np.float32)), dtype=np.float32)
        for teacher in teacher_paths
    ]
    if agent_trajectory.get("speed", np.asarray([])).size:
        speed_arrays.append(np.asarray(agent_trajectory["speed"], dtype=np.float32))
    valid_speeds = np.concatenate([values[np.isfinite(values)] for values in speed_arrays if values.size])
    if valid_speeds.size:
        speed_norm = mcolors.Normalize(
            vmin=float(np.nanmin(valid_speeds)),
            vmax=max(float(np.nanmax(valid_speeds)), float(np.nanmin(valid_speeds)) + 1e-6),
        )
    else:
        speed_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    speed_mappable = None

    for teacher in teacher_paths:
        positions = np.asarray(teacher["positions"], dtype=np.float32)
        if positions.shape[0] < 2:
            continue
        projected = projection.points(positions[:, [0, 2]])
        teacher_speed = np.asarray(teacher.get("speeds", np.zeros(positions.shape[0])), dtype=np.float32)
        speed_mappable = plot_speed_gradient(
            ax,
            projected,
            teacher_speed,
            linewidth=1.75,
            alpha=0.50,
            zorder=80,
            norm=speed_norm,
        )

    if agent_trajectory.get("x", np.asarray([])).size >= 2:
        agent_xz = np.stack([agent_trajectory["x"], agent_trajectory["z"]], axis=1)
        projected_agent = projection.points(agent_xz)
        inside = (
            (agent_xz[:, 0] >= (min_x - MAP_BOUNDS_MARGIN))
            & (agent_xz[:, 0] <= (max_x + MAP_BOUNDS_MARGIN))
            & (agent_xz[:, 1] >= (min_z - MAP_BOUNDS_MARGIN))
            & (agent_xz[:, 1] <= (max_z + MAP_BOUNDS_MARGIN))
        )
        outside_indices = np.flatnonzero(~inside)
        off_road_indices = np.flatnonzero(
            np.asarray(agent_trajectory.get("on_road", np.ones(agent_xz.shape[0])), dtype=np.float32) <= 0.0
        )
        stop_indices = []
        if outside_indices.size:
            stop_indices.append(int(outside_indices[0]))
        if off_road_indices.size:
            stop_indices.append(int(off_road_indices[0]))
        plot_until = min(stop_indices) if stop_indices else int(projected_agent.shape[0])
        plot_until = max(2, min(plot_until, int(projected_agent.shape[0])))
        projected_agent_plot = projected_agent[:plot_until]
        speed = np.asarray(agent_trajectory.get("speed", np.zeros(agent_xz.shape[0])), dtype=np.float32)
        collision_event = np.asarray(
            agent_trajectory.get("collision_event", np.zeros(agent_xz.shape[0], dtype=np.float32)),
            dtype=np.float32,
        )
        draw_segment_mask = np.ones(max(0, plot_until - 1), dtype=bool)
        if collision_event.size:
            # The sample at a collision event is already the recovered pose after
            # bounce correction.  Do not draw the artificial pre-crash -> recovered
            # connector as part of the driven path.
            draw_segment_mask &= collision_event[1:plot_until] <= 0.0
        plot_path_outline(
            ax,
            projected_agent_plot,
            linewidth=6.2,
            alpha=0.42,
            zorder=99,
            segment_mask=draw_segment_mask,
        )
        speed_mappable = plot_speed_gradient(
            ax,
            projected_agent_plot,
            speed[:plot_until],
            linewidth=3.8,
            alpha=0.98,
            zorder=100,
            norm=speed_norm,
            segment_mask=draw_segment_mask,
        )
        min_clearance = np.asarray(
            agent_trajectory.get("min_laser_clearance", np.full(agent_xz.shape[0], np.inf)),
            dtype=np.float32,
        )
        touch_count = np.asarray(
            agent_trajectory.get(
                "touch_count",
                agent_trajectory.get("crashes", np.zeros(agent_xz.shape[0], dtype=np.float32)),
            ),
            dtype=np.float32,
        )
        crash_indices = np.flatnonzero(np.diff(np.concatenate([[0.0], touch_count])) > 0.0)
        if crash_indices.size == 0:
            contact = min_clearance <= 0.0
            crash_indices = np.flatnonzero(contact & ~np.concatenate([[False], contact[:-1]]))
        if crash_indices.size:
            crash_indices = crash_indices[crash_indices < plot_until]
        if crash_indices.size:
            collision_x = np.asarray(
                agent_trajectory.get("collision_x", agent_trajectory["x"]),
                dtype=np.float32,
            )
            collision_z = np.asarray(
                agent_trajectory.get("collision_z", agent_trajectory["z"]),
                dtype=np.float32,
            )
            collision_xz = np.stack([collision_x, collision_z], axis=1)
            projected_crashes = projection.points(collision_xz)
            ax.scatter(
                projected_crashes[crash_indices, 0],
                projected_crashes[crash_indices, 1],
                s=48,
                facecolors=CRASH_COLOR,
                edgecolors=CRASH_COLOR,
                linewidths=0.0,
                zorder=120,
            )
        wall_hug = np.asarray(
            agent_trajectory.get("wall_hug_active", np.zeros(agent_xz.shape[0], dtype=np.float32)),
            dtype=np.float32,
        )
        wall_hug_indices = np.flatnonzero(wall_hug[:plot_until] > 0.0)
        if wall_hug_indices.size:
            hug_points = projected_agent[:plot_until].copy()
            # Split into short contiguous line chunks so non-hug gaps are not connected.
            chunks = np.split(wall_hug_indices, np.where(np.diff(wall_hug_indices) > 1)[0] + 1)
            for chunk in chunks:
                if chunk.size < 2:
                    continue
                ax.plot(
                    hug_points[chunk, 0],
                    hug_points[chunk, 1],
                    color=WALL_HUG_COLOR,
                    linewidth=2.0,
                    alpha=0.95,
                    zorder=112,
                )
        if stop_indices:
            idx = max(0, int(min(stop_indices)) - 1)
            ax.scatter(
                [projected_agent[idx, 0]],
                [projected_agent[idx, 1]],
                marker="x",
                s=125,
                color=AGENT_COLOR,
                linewidths=2.4,
                zorder=121,
            )

    heights = _all_road_heights(game_map)
    ax.set_title(f"{map_name}: teacher paths and supervised agent rollout", fontsize=13, pad=6)
    fig.subplots_adjust(left=0.055, right=0.835, top=0.94, bottom=0.13)
    add_map_legend(fig, ax, game_map, float(np.min(heights)), float(np.max(heights)))
    if speed_mappable is not None:
        add_path_legend(fig, ax, speed_norm, speed_mappable)
    ax.set_xlim(map_xlim)
    ax.set_ylim(map_ylim)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def analyze_map(
    map_name: str,
    data_root: Path,
    specialist_root: Path,
    output_dir: Path,
    source_layout: str,
    teacher_count: int,
    max_time: float,
    seed: int,
    max_touches: int,
    collision_slowdown: float,
    collision_path_axis_bias: float,
    collision_normal_restitution: float,
    collision_wall_tangent_blend: float,
    collision_reference_window: int,
    collision_skip_recent: int,
    never_stop: bool,
) -> dict[str, Any]:
    dataset_dir = find_latest_dataset(data_root, map_name, source_layout=source_layout)
    model_path = find_latest_model(specialist_root, map_name)
    policy, extra = NeuralPolicy.load(str(model_path), map_location="cpu")
    model_vertical_mode, model_multi_surface_mode = infer_policy_modes(policy, extra, model_path)

    teachers = [load_teacher_path(path) for path in attempt_files(dataset_dir, teacher_count)]
    metrics, trajectory = rollout_agent(
        policy,
        map_name=map_name,
        vertical_mode=model_vertical_mode,
        multi_surface_mode=model_multi_surface_mode,
        max_time=max_time,
        seed=seed,
        max_touches=max_touches,
        collision_slowdown=collision_slowdown,
        collision_path_axis_bias=collision_path_axis_bias,
        collision_normal_restitution=collision_normal_restitution,
        collision_wall_tangent_blend=collision_wall_tangent_blend,
        collision_reference_window=collision_reference_window,
        collision_skip_recent=collision_skip_recent,
        never_stop=never_stop,
    )

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
            "max_touches",
            "collision_slowdown",
            "collision_bounce_speed_retention",
            "collision_bounce_path_axis_bias",
            "collision_bounce_normal_restitution",
            "collision_bounce_wall_tangent_blend",
            "collision_bounce_reference_window",
            "collision_bounce_skip_recent",
            "never_stop",
            "left_map_bounds",
            "left_map_bounds_index",
            "left_map_bounds_time",
            "left_road_surface",
            "left_road_surface_index",
            "left_road_surface_time",
        ],
    )
    plot_path = map_dir / f"{safe_name(map_name)}_teacher_agent_paths.png"
    plot_paths(map_name, teachers, trajectory, plot_path)
    return {
        "map_name": map_name,
        "dataset_dir": str(dataset_dir),
        "model_path": str(model_path),
        "model_obs_dim": int(policy.obs_dim),
        "model_vertical_mode": bool(model_vertical_mode),
        "model_multi_surface_mode": bool(model_multi_surface_mode),
        "source_layout": source_layout,
        "plot_path": str(plot_path),
        "teacher_attempts": len(teachers),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze supervised map specialists and plot teacher/agent paths.")
    parser.add_argument("--data-root", default="logs/supervised_data")
    parser.add_argument("--specialist-root", default="logs/supervised_runs_map_specialists_v2d_asphalt_20260505")
    parser.add_argument("--output-dir", default="Experiments/analysis/supervised_map_specialists_v2d_asphalt_20260505")
    parser.add_argument("--maps", default=",".join(MAP_NAMES), help="Comma-separated map names to analyze.")
    parser.add_argument(
        "--source-layout",
        default="any",
        choices=["any", *LAYOUTS.keys()],
        help="Dataset layout filter for teacher paths. The agent rollout layout is inferred from the model.",
    )
    parser.add_argument("--teacher-count", type=int, default=10)
    parser.add_argument("--max-time", type=float, default=60.0)
    parser.add_argument(
        "--max-touches",
        default="inf",
        help="TM2D rollout touches before termination. Use 'inf' to keep replaying after crashes.",
    )
    parser.add_argument(
        "--collision-slowdown",
        default="0.30",
        help="Fraction of speed removed after a non-terminal replay collision. 0.30 keeps 70%% speed.",
    )
    parser.add_argument(
        "--collision-path-axis-bias",
        type=float,
        default=0.0,
        help="Blend reflected replay bounce direction toward the local route axis. 0 keeps pure reflection.",
    )
    parser.add_argument(
        "--collision-normal-restitution",
        type=float,
        default=1.0,
        help="Fallback wall-normal reflection factor when wall-tangent blend is disabled.",
    )
    parser.add_argument(
        "--collision-wall-tangent-blend",
        type=float,
        default=0.30,
        help="Blend reflected bounce direction toward the actual 2D wall segment direction.",
    )
    parser.add_argument(
        "--collision-reference-window",
        type=int,
        default=4,
        help="Number of earlier safe velocity samples used to smooth bounce direction.",
    )
    parser.add_argument(
        "--collision-skip-recent",
        type=int,
        default=1,
        help="How many most recent safe samples to ignore before averaging bounce direction.",
    )
    parser.add_argument(
        "--never-stop",
        action="store_true",
        default=True,
        help="Diagnostic replay continues through stuck/start-idle failure states until finish or max-time.",
    )
    parser.add_argument(
        "--allow-stuck-stop",
        dest="never_stop",
        action="store_false",
        help="Restore normal TM2D stuck/start-idle termination for replay debugging.",
    )
    parser.add_argument("--seed", type=int, default=2026050509)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    max_touches = parse_max_touches(args.max_touches)
    collision_slowdown = parse_slowdown(args.collision_slowdown)
    collision_path_axis_bias = float(np.clip(args.collision_path_axis_bias, 0.0, 1.0))
    collision_normal_restitution = float(np.clip(args.collision_normal_restitution, 0.0, 1.0))
    collision_wall_tangent_blend = float(np.clip(args.collision_wall_tangent_blend, 0.0, 1.0))
    collision_reference_window = max(1, int(args.collision_reference_window))
    collision_skip_recent = max(0, int(args.collision_skip_recent))
    for index, map_name in enumerate(parse_maps(args.maps)):
        print(f"Analyzing {map_name}...")
        rows.append(
            analyze_map(
                map_name=map_name,
                data_root=Path(args.data_root),
                specialist_root=Path(args.specialist_root),
                output_dir=output_dir,
                source_layout=str(args.source_layout),
                teacher_count=int(args.teacher_count),
                max_time=float(args.max_time),
                seed=int(args.seed) + index,
                max_touches=max_touches,
                collision_slowdown=collision_slowdown,
                collision_path_axis_bias=collision_path_axis_bias,
                collision_normal_restitution=collision_normal_restitution,
                collision_wall_tangent_blend=collision_wall_tangent_blend,
                collision_reference_window=collision_reference_window,
                collision_skip_recent=collision_skip_recent,
                never_stop=bool(args.never_stop),
            )
        )
    write_csv(
        output_dir / "summary.csv",
        rows,
        [
            "map_name",
            "dataset_dir",
            "model_path",
            "model_obs_dim",
            "model_vertical_mode",
            "model_multi_surface_mode",
            "source_layout",
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
            "max_touches",
            "collision_slowdown",
            "collision_bounce_speed_retention",
            "collision_bounce_path_axis_bias",
            "collision_bounce_normal_restitution",
            "collision_bounce_wall_tangent_blend",
            "collision_bounce_reference_window",
            "collision_bounce_skip_recent",
            "never_stop",
            "left_map_bounds",
            "left_map_bounds_index",
            "left_map_bounds_time",
            "left_road_surface",
            "left_road_surface_index",
            "left_road_surface_time",
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
