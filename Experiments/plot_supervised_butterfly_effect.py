from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NeuralPolicy import NeuralPolicy

from Experiments.analyze_supervised_map_specialists import (
    PRACTICAL_INFINITY_TOUCHES,
    find_latest_model,
    infer_policy_modes,
    map_xz_bounds,
    parse_max_touches,
    parse_slowdown,
    safe_name,
)
from Experiments.tm2d_env import TM2DPhysicsConfig, TM2DRewardConfig, TM2DSimEnv
from Experiments.tm_map_plotting import _all_road_heights, add_map_legend, render_map_background
from Map import Map


BASELINE_COLOR = "#38bdf8"
PERTURBED_COLOR = "#f97316"
PERTURB_MARKER_COLOR = "#dc2626"
CRASH_COLOR = "#050505"


def rollout_policy(
    policy: NeuralPolicy,
    *,
    map_name: str,
    vertical_mode: bool,
    multi_surface_mode: bool,
    seed: int,
    max_time: float,
    max_touches: int,
    collision_slowdown: float,
    perturb: bool,
    trigger_path_index: int,
    perturb_duration_steps: int,
    perturb_steer: float | None,
    perturb_gas: float | None,
    perturb_brake: float | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
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
        collision_bounce_speed_retention=float(np.clip(1.0 - collision_slowdown, 0.0, 1.0)),
        collision_bounce_wall_tangent_blend=0.30,
        collision_bounce_reference_window=4,
        collision_bounce_skip_recent=1,
        never_stop=True,
    )
    obs, info = env.reset(seed=int(seed))
    records: list[dict[str, float]] = []
    perturb_start_step = -1
    perturb_end_step = -1
    perturb_remaining = 0
    total_reward = 0.0
    terminated = False
    truncated = False

    try:
        for step in range(10000):
            action = policy.act(obs)
            path_index = int(info.get("path_index", env.path_index))
            if perturb and perturb_start_step < 0 and path_index >= int(trigger_path_index):
                perturb_start_step = step
                perturb_remaining = max(1, int(perturb_duration_steps))
                perturb_end_step = perturb_start_step + perturb_remaining - 1
            perturb_active = perturb_remaining > 0
            if perturb_active:
                action = np.asarray(action, dtype=np.float32).copy()
                if perturb_gas is not None:
                    action[0] = float(np.clip(perturb_gas, 0.0, 1.0))
                if perturb_brake is not None:
                    action[1] = float(np.clip(perturb_brake, 0.0, 1.0))
                if perturb_steer is not None:
                    action[2] = float(np.clip(perturb_steer, -1.0, 1.0))
                perturb_remaining -= 1

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
                    "path_index": float(info.get("path_index", env.path_index)),
                    "crashes": float(info.get("crashes", 0.0)),
                    "touch_count": float(info.get("touch_count", info.get("crashes", 0.0))),
                    "collision_event": float(info.get("collision_event", 0.0)),
                    "collision_x": float(info.get("collision_x", info.get("x", env.position[0]))),
                    "collision_z": float(info.get("collision_z", info.get("z", env.position[1]))),
                    "perturb_active": float(1.0 if perturb_active else 0.0),
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

    trajectory = (
        {key: np.asarray([record[key] for record in records], dtype=np.float32) for key in records[0].keys()}
        if records
        else {key: np.asarray([], dtype=np.float32) for key in ("step", "time", "x", "z")}
    )
    metrics = {
        "variant": "perturbed" if perturb else "baseline",
        "finished": int(info.get("finished", 0)),
        "crashes": int(info.get("crashes", 0)),
        "timeout": int(bool(truncated) and int(info.get("finished", 0)) <= 0),
        "progress": float(info.get("progress", info.get("dense_progress", 0.0))),
        "block_progress": float(info.get("block_progress", info.get("discrete_progress", 0.0))),
        "time": float(info.get("time", env.time)),
        "distance": float(info.get("distance", env.distance)),
        "reward": float(total_reward),
        "steps": int(len(records)),
        "perturb_start_step": int(perturb_start_step),
        "perturb_end_step": int(perturb_end_step),
        "perturb_duration_steps": int(perturb_duration_steps if perturb else 0),
        "perturb_steer": "" if perturb_steer is None else float(perturb_steer),
        "perturb_gas": "" if perturb_gas is None else float(perturb_gas),
        "perturb_brake": "" if perturb_brake is None else float(perturb_brake),
    }
    return metrics, trajectory


def metrics_from_trajectory(trajectory: dict[str, np.ndarray]) -> dict[str, Any]:
    if "metrics_json" in trajectory:
        try:
            import json

            metrics_text = str(np.asarray(trajectory["metrics_json"]).reshape(-1)[0])
            parsed = json.loads(metrics_text)
            return {
                "variant": "baseline",
                "finished": int(parsed.get("finished", 0)),
                "crashes": int(parsed.get("crashes", parsed.get("touch_count", 0))),
                "timeout": int(parsed.get("timeout", 0)),
                "progress": float(parsed.get("progress", 0.0)),
                "block_progress": float(parsed.get("block_progress", parsed.get("progress", 0.0))),
                "time": float(parsed.get("time", 0.0)),
                "distance": float(parsed.get("distance", 0.0)),
                "reward": float(parsed.get("reward", 0.0)),
                "steps": int(parsed.get("steps", 0)),
                "perturb_start_step": -1,
                "perturb_end_step": -1,
                "perturb_duration_steps": 0,
                "perturb_steer": "",
                "perturb_gas": "",
                "perturb_brake": "",
            }
        except Exception:
            pass

    steps = int(np.asarray(trajectory.get("x", np.asarray([]))).size)
    touch_count = np.asarray(trajectory.get("touch_count", trajectory.get("crashes", np.asarray([0]))), dtype=np.float32)
    return {
        "variant": "baseline",
        "finished": int(float(np.asarray(trajectory.get("progress", np.asarray([0.0]))).reshape(-1)[-1]) >= 100.0),
        "crashes": int(float(touch_count[-1])) if touch_count.size else 0,
        "timeout": 0,
        "progress": float(np.asarray(trajectory.get("progress", np.asarray([0.0]))).reshape(-1)[-1]),
        "block_progress": float(np.asarray(trajectory.get("block_progress", np.asarray([0.0]))).reshape(-1)[-1]),
        "time": float(np.asarray(trajectory.get("time", np.asarray([0.0]))).reshape(-1)[-1]),
        "distance": 0.0,
        "reward": 0.0,
        "steps": steps,
        "perturb_start_step": -1,
        "perturb_end_step": -1,
        "perturb_duration_steps": 0,
        "perturb_steer": "",
        "perturb_gas": "",
        "perturb_brake": "",
    }


def rollout_recorded_actions(
    reference_trajectory: dict[str, np.ndarray],
    *,
    map_name: str,
    seed: int,
    max_time: float,
    max_touches: int,
    collision_slowdown: float,
    trigger_path_index: int,
    perturb_duration_steps: int,
    perturb_steer: float | None,
    perturb_gas: float | None,
    perturb_brake: float | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    actions = np.stack(
        [
            np.asarray(reference_trajectory["gas"], dtype=np.float32),
            np.asarray(reference_trajectory["brake"], dtype=np.float32),
            np.asarray(reference_trajectory["steer"], dtype=np.float32),
        ],
        axis=1,
    )
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
        max_touches=int(max_touches),
        collision_bounce_speed_retention=float(np.clip(1.0 - collision_slowdown, 0.0, 1.0)),
        collision_bounce_wall_tangent_blend=0.30,
        collision_bounce_reference_window=4,
        collision_bounce_skip_recent=1,
        never_stop=True,
    )
    _obs, info = env.reset(seed=int(seed))
    records: list[dict[str, float]] = []
    perturb_start_step = -1
    perturb_end_step = -1
    perturb_remaining = 0
    total_reward = 0.0
    terminated = False
    truncated = False
    try:
        for step, recorded_action in enumerate(actions):
            action = np.asarray(recorded_action, dtype=np.float32).copy()
            path_index = int(info.get("path_index", env.path_index))
            if perturb_start_step < 0 and path_index >= int(trigger_path_index):
                perturb_start_step = step
                perturb_remaining = max(1, int(perturb_duration_steps))
                perturb_end_step = perturb_start_step + perturb_remaining - 1
            perturb_active = perturb_remaining > 0
            if perturb_active:
                if perturb_gas is not None:
                    action[0] = float(np.clip(perturb_gas, 0.0, 1.0))
                if perturb_brake is not None:
                    action[1] = float(np.clip(perturb_brake, 0.0, 1.0))
                if perturb_steer is not None:
                    action[2] = float(np.clip(perturb_steer, -1.0, 1.0))
                perturb_remaining -= 1

            _obs, reward, terminated, truncated, info = env.step(action)
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
                    "path_index": float(info.get("path_index", env.path_index)),
                    "crashes": float(info.get("crashes", 0.0)),
                    "touch_count": float(info.get("touch_count", info.get("crashes", 0.0))),
                    "collision_event": float(info.get("collision_event", 0.0)),
                    "collision_x": float(info.get("collision_x", info.get("x", env.position[0]))),
                    "collision_z": float(info.get("collision_z", info.get("z", env.position[1]))),
                    "perturb_active": float(1.0 if perturb_active else 0.0),
                    "gas": float(action[0]),
                    "brake": float(action[1]),
                    "steer": float(action[2]),
                }
            )
            if terminated or truncated:
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    trajectory = (
        {key: np.asarray([record[key] for record in records], dtype=np.float32) for key in records[0].keys()}
        if records
        else {key: np.asarray([], dtype=np.float32) for key in ("step", "time", "x", "z")}
    )
    metrics = {
        "variant": "action_replay_perturbed",
        "finished": int(info.get("finished", 0)),
        "crashes": int(info.get("crashes", 0)),
        "timeout": int(bool(truncated) and int(info.get("finished", 0)) <= 0),
        "progress": float(info.get("progress", info.get("dense_progress", 0.0))),
        "block_progress": float(info.get("block_progress", info.get("discrete_progress", 0.0))),
        "time": float(info.get("time", env.time)),
        "distance": float(info.get("distance", env.distance)),
        "reward": float(total_reward),
        "steps": int(len(records)),
        "perturb_start_step": int(perturb_start_step),
        "perturb_end_step": int(perturb_end_step),
        "perturb_duration_steps": int(perturb_duration_steps),
        "perturb_steer": "" if perturb_steer is None else float(perturb_steer),
        "perturb_gas": "" if perturb_gas is None else float(perturb_gas),
        "perturb_brake": "" if perturb_brake is None else float(perturb_brake),
    }
    return metrics, trajectory


def _draw_path(ax, projected: np.ndarray, *, color: str, linewidth: float, alpha: float, zorder: int, label: str) -> None:
    if projected.shape[0] < 2:
        return
    ax.plot(
        projected[:, 0],
        projected[:, 1],
        color="#111827",
        linewidth=linewidth + 2.2,
        alpha=0.36,
        zorder=zorder - 1,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    ax.plot(
        projected[:, 0],
        projected[:, 1],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
        solid_capstyle="round",
        solid_joinstyle="round",
        label=label,
    )


def plot_butterfly(
    *,
    map_name: str,
    baseline: dict[str, np.ndarray],
    perturbed: dict[str, np.ndarray],
    baseline_metrics: dict[str, Any],
    perturbed_metrics: dict[str, Any],
    perturb_label: str,
    perturb_marker_label: str,
    title: str,
    output_path: Path,
) -> None:
    game_map = Map(map_name)
    fig, ax = plt.subplots(figsize=(13.35, 8.1))
    projection = render_map_background(ax, game_map, show_legend=False, alpha=0.92)
    map_xlim = ax.get_xlim()
    map_ylim = ax.get_ylim()
    ax.set_autoscale_on(False)

    baseline_xz = np.stack([baseline["x"], baseline["z"]], axis=1)
    perturbed_xz = np.stack([perturbed["x"], perturbed["z"]], axis=1)
    baseline_projected = projection.points(baseline_xz)
    perturbed_projected = projection.points(perturbed_xz)

    _draw_path(
        ax,
        baseline_projected,
        color=BASELINE_COLOR,
        linewidth=3.2,
        alpha=0.92,
        zorder=95,
        label="Baseline rollout",
    )
    _draw_path(
        ax,
        perturbed_projected,
        color=PERTURBED_COLOR,
        linewidth=3.4,
        alpha=0.95,
        zorder=100,
        label=perturb_label,
    )

    active = np.flatnonzero(np.asarray(perturbed.get("perturb_active", []), dtype=np.float32) > 0.0)
    if active.size:
        start = int(active[0])
        end = int(active[-1])
        ax.scatter(
            perturbed_projected[start, 0],
            perturbed_projected[start, 1],
            marker="D",
            s=78,
            facecolors=PERTURB_MARKER_COLOR,
            edgecolors="#111827",
            linewidths=0.8,
            zorder=130,
        )
        ax.plot(
            perturbed_projected[start : end + 1, 0],
            perturbed_projected[start : end + 1, 1],
            color=PERTURB_MARKER_COLOR,
            linewidth=5.2,
            alpha=0.82,
            zorder=125,
            solid_capstyle="round",
        )

    touch_count = np.asarray(perturbed.get("touch_count", np.asarray([])), dtype=np.float32)
    crash_indices = np.flatnonzero(np.diff(np.concatenate([[0.0], touch_count])) > 0.0)
    if crash_indices.size:
        collision_x = np.asarray(perturbed.get("collision_x", perturbed["x"]), dtype=np.float32)
        collision_z = np.asarray(perturbed.get("collision_z", perturbed["z"]), dtype=np.float32)
        crash_projected = projection.points(np.stack([collision_x, collision_z], axis=1))
        ax.scatter(
            crash_projected[crash_indices, 0],
            crash_projected[crash_indices, 1],
            marker="o",
            s=36,
            facecolors=CRASH_COLOR,
            edgecolors=CRASH_COLOR,
            linewidths=0.0,
            zorder=132,
        )

    handles = [
        Line2D([0], [0], color=BASELINE_COLOR, lw=3.2, label="Baseline rollout"),
        Line2D([0], [0], color=PERTURBED_COLOR, lw=3.4, label=perturb_label),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=PERTURB_MARKER_COLOR,
            markeredgecolor="#111827",
            markersize=7,
            label=perturb_marker_label,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CRASH_COLOR,
            markeredgecolor=CRASH_COLOR,
            markersize=6,
            label="Perturbed crash/touch",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.79),
        frameon=False,
        fontsize=10.5,
        borderaxespad=0.0,
    )

    ax.text(
        1.01,
        0.42,
        (
            "Metrics\n"
            f"baseline: {baseline_metrics['time']:.2f}s, "
            f"{int(baseline_metrics['crashes'])} touches\n"
            f"perturbed: {perturbed_metrics['time']:.2f}s, "
            f"{int(perturbed_metrics['crashes'])} touches"
        ),
        transform=ax.transAxes,
        fontsize=9.5,
        va="top",
        ha="left",
    )

    heights = _all_road_heights(game_map)
    add_map_legend(fig, ax, game_map, float(np.min(heights)), float(np.max(heights)))
    ax.set_title(title, fontsize=13, pad=6)
    fig.subplots_adjust(left=0.055, right=0.835, top=0.94, bottom=0.13)
    ax.set_xlim(map_xlim)
    ax.set_ylim(map_ylim)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "variant",
        "finished",
        "crashes",
        "timeout",
        "progress",
        "block_progress",
        "time",
        "distance",
        "reward",
        "steps",
        "perturb_start_step",
        "perturb_end_step",
        "perturb_duration_steps",
        "perturb_steer",
        "perturb_gas",
        "perturb_brake",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a supervised specialist butterfly-effect trajectory demo.")
    parser.add_argument("--map-name", default="single_surface_flat")
    parser.add_argument("--specialist-root", default="logs/supervised_runs_map_specialists_v2d_asphalt_20260505")
    parser.add_argument("--model-path", default="", help="Optional direct path to best_model.pt.")
    parser.add_argument(
        "--action-replay-trajectory",
        default="",
        help=(
            "Optional trajectory .npz containing gas/brake/steer. "
            "When set, the baseline is the saved trajectory and the perturbed "
            "run replays its recorded actions instead of using a model."
        ),
    )
    parser.add_argument("--output-dir", default="Experiments/analysis/supervised_map_specialists_v2d_asphalt_20260505")
    parser.add_argument("--seed", type=int, default=2026050505)
    parser.add_argument("--max-time", type=float, default=60.0)
    parser.add_argument("--max-touches", default="inf")
    parser.add_argument("--collision-slowdown", default="0.30")
    parser.add_argument("--trigger-path-index", type=int, default=1)
    parser.add_argument("--perturb-duration-steps", type=int, default=35)
    parser.add_argument("--perturb-steer", type=float, default=None)
    parser.add_argument("--perturb-gas", type=float, default=None)
    parser.add_argument("--perturb-brake", type=float, default=None)
    parser.add_argument(
        "--perturb-label",
        default="",
        help="Optional legend label for the perturbed rollout.",
    )
    parser.add_argument("--output-tag", default="", help="Optional suffix for output filenames.")
    args = parser.parse_args()

    max_touches = parse_max_touches(args.max_touches)
    collision_slowdown = parse_slowdown(args.collision_slowdown)
    perturb_steer = args.perturb_steer
    perturb_gas = args.perturb_gas
    perturb_brake = args.perturb_brake
    if perturb_steer is None and perturb_gas is None and perturb_brake is None:
        # Backward-compatible default: old script meant a hard-left steer perturbation.
        perturb_steer = 1.0

    if args.action_replay_trajectory:
        with np.load(args.action_replay_trajectory) as data:
            baseline_trajectory = {key: np.asarray(data[key]) for key in data.files}
        baseline_metrics = metrics_from_trajectory(baseline_trajectory)
        perturbed_metrics, perturbed_trajectory = rollout_recorded_actions(
            baseline_trajectory,
            map_name=args.map_name,
            seed=int(args.seed),
            max_time=float(args.max_time),
            max_touches=max_touches,
            collision_slowdown=collision_slowdown,
            trigger_path_index=int(args.trigger_path_index),
            perturb_duration_steps=int(args.perturb_duration_steps),
            perturb_steer=perturb_steer,
            perturb_gas=perturb_gas,
            perturb_brake=perturb_brake,
        )
    else:
        model_path = Path(args.model_path) if args.model_path else find_latest_model(Path(args.specialist_root), args.map_name)
        policy, extra = NeuralPolicy.load(str(model_path), map_location="cpu")
        vertical_mode, multi_surface_mode = infer_policy_modes(policy, extra, model_path)
        baseline_metrics, baseline_trajectory = rollout_policy(
            policy,
            map_name=args.map_name,
            vertical_mode=vertical_mode,
            multi_surface_mode=multi_surface_mode,
            seed=int(args.seed),
            max_time=float(args.max_time),
            max_touches=max_touches,
            collision_slowdown=collision_slowdown,
            perturb=False,
            trigger_path_index=int(args.trigger_path_index),
            perturb_duration_steps=int(args.perturb_duration_steps),
            perturb_steer=perturb_steer,
            perturb_gas=perturb_gas,
            perturb_brake=perturb_brake,
        )
        perturbed_metrics, perturbed_trajectory = rollout_policy(
            policy,
            map_name=args.map_name,
            vertical_mode=vertical_mode,
            multi_surface_mode=multi_surface_mode,
            seed=int(args.seed),
            max_time=float(args.max_time),
            max_touches=max_touches,
            collision_slowdown=collision_slowdown,
            perturb=True,
            trigger_path_index=int(args.trigger_path_index),
            perturb_duration_steps=int(args.perturb_duration_steps),
            perturb_steer=perturb_steer,
            perturb_gas=perturb_gas,
            perturb_brake=perturb_brake,
        )

    map_dir = Path(args.output_dir) / safe_name(args.map_name)
    map_dir.mkdir(parents=True, exist_ok=True)
    if args.output_tag:
        tag = safe_name(args.output_tag)
    else:
        if perturb_steer is not None:
            tag = "left_steer" if float(perturb_steer) >= 0.0 else "right_steer"
        elif perturb_brake is not None and float(perturb_brake) > 0.5:
            tag = "brake"
        elif perturb_gas is not None and float(perturb_gas) < 0.5:
            tag = "gas_cut"
        else:
            tag = "gas"
    if args.perturb_label:
        perturb_label = str(args.perturb_label)
        perturb_marker_label = "Perturbation starts"
    elif perturb_steer is not None:
        direction_name = "left" if float(perturb_steer) >= 0.0 else "right"
        perturb_label = f"Hard-{direction_name} perturbed rollout"
        perturb_marker_label = f"Forced {direction_name} steer starts"
    elif perturb_brake is not None and float(perturb_brake) > 0.5:
        perturb_label = "Brake-pulse perturbed rollout"
        perturb_marker_label = "Forced brake starts"
    elif perturb_gas is not None and float(perturb_gas) < 0.5:
        perturb_label = "Gas-cut perturbed rollout"
        perturb_marker_label = "Forced gas cut starts"
    else:
        perturb_label = "Gas-pulse perturbed rollout"
        perturb_marker_label = "Forced gas starts"
    np.savez(map_dir / f"butterfly_baseline_{tag}_trajectory.npz", **baseline_trajectory)
    np.savez(map_dir / f"butterfly_perturbed_{tag}_trajectory.npz", **perturbed_trajectory)
    write_metrics_csv(map_dir / f"butterfly_metrics_{tag}.csv", [baseline_metrics, perturbed_metrics])
    output_path = map_dir / f"{safe_name(args.map_name)}_butterfly_{tag}.png"
    plot_butterfly(
        map_name=args.map_name,
        baseline=baseline_trajectory,
        perturbed=perturbed_trajectory,
        baseline_metrics=baseline_metrics,
        perturbed_metrics=perturbed_metrics,
        perturb_label=perturb_label,
        perturb_marker_label=perturb_marker_label,
        title=f"{args.map_name}: short action perturbation changes the trajectory",
        output_path=output_path,
    )
    print(f"Saved butterfly plot to {output_path}")
    print("Baseline:", baseline_metrics)
    print("Perturbed:", perturbed_metrics)


if __name__ == "__main__":
    main()
