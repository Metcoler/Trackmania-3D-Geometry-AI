from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Map import Map
from Experiments.tm_map_plotting import MapProjection, render_map_background


def load_manifest(run_dir: Path) -> list[dict]:
    manifest_path = run_dir / "trajectories" / "trajectory_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Trajectory manifest not found: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_path(run_dir: Path, row: dict) -> dict[str, np.ndarray]:
    path = run_dir / str(row["path_file"])
    data = np.load(path)
    return {key: data[key] for key in data.files}


def infer_map_name(run_dir: Path, explicit_map_name: str | None = None) -> str | None:
    if explicit_map_name:
        return explicit_map_name
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    map_name = config.get("map_name")
    return str(map_name) if map_name else None


def load_map_projection(map_name: str | None) -> tuple[Map | None, MapProjection | None]:
    if not map_name:
        return None, None
    try:
        game_map = Map(map_name)
    except Exception as exc:
        print(f"Warning: could not load map background '{map_name}': {exc}")
        return None, None
    return game_map, None


def project_path(x: np.ndarray, z: np.ndarray, projection: MapProjection | None) -> tuple[np.ndarray, np.ndarray]:
    points = np.column_stack([x, z])
    if projection is not None:
        points = projection.points(points)
    return points[:, 0], points[:, 1]


def setup_axis_with_map(ax, game_map: Map | None, projection: MapProjection | None) -> MapProjection | None:
    if game_map is None:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.grid(True, alpha=0.2)
        return projection
    projection = render_map_background(ax, game_map, projection=projection, alpha=0.86)
    return projection


def plot_overview(
    run_dir: Path,
    rows: list[dict],
    output_dir: Path,
    game_map: Map | None = None,
    projection: MapProjection | None = None,
) -> MapProjection | None:
    fig, ax = plt.subplots(figsize=(12, 7) if game_map is not None else (9, 9))
    projection = setup_axis_with_map(ax, game_map, projection)
    for row in rows:
        data = load_path(run_dir, row)
        x = np.asarray(data["x"], dtype=np.float32)
        z = np.asarray(data["z"], dtype=np.float32)
        if x.size < 2:
            continue
        px, pz = project_path(x, z, projection if game_map is not None else None)
        finished = int(row.get("finished", 0))
        rank = int(row.get("rank", 999999))
        if finished > 0:
            ax.plot(px, pz, color="#0b6e4f", alpha=0.78, linewidth=1.9, zorder=90)
        elif rank == 1:
            ax.plot(px, pz, color="#2454a6", alpha=0.48, linewidth=1.25, zorder=85)
        else:
            ax.plot(px, pz, color="#111111", alpha=0.12, linewidth=0.7, zorder=80)
    ax.set_title("Logged trajectories: top paths and finishers")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_overview.png", dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return projection


def plot_best_speed(
    run_dir: Path,
    rows: list[dict],
    output_dir: Path,
    game_map: Map | None = None,
    projection: MapProjection | None = None,
) -> MapProjection | None:
    if not rows:
        return projection
    ranked = sorted(
        rows,
        key=lambda row: (
            -int(row.get("finished", 0)),
            int(row.get("rank", 999999)),
            -float(row.get("dense_progress", 0.0)),
            float(row.get("time", 1e9)),
        ),
    )
    data = load_path(run_dir, ranked[0])
    x = np.asarray(data["x"], dtype=np.float32)
    z = np.asarray(data["z"], dtype=np.float32)
    speed = np.asarray(data["speed"], dtype=np.float32)
    if x.size < 2:
        return projection
    px, pz = project_path(x, z, projection if game_map is not None else None)
    points = np.column_stack([px, pz]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    speed_segments = speed[:-1] if speed.size >= x.size else np.zeros(x.size - 1, dtype=np.float32)
    collection = LineCollection(segments, cmap="turbo_r", linewidth=2.0)
    collection.set_array(speed_segments)

    fig, ax = plt.subplots(figsize=(12, 7) if game_map is not None else (9, 9))
    projection = setup_axis_with_map(ax, game_map, projection)
    ax.add_collection(collection)
    ax.autoscale()
    ax.set_title("Best logged trajectory colored by speed")
    fig.colorbar(collection, ax=ax, label="speed")
    fig.tight_layout()
    fig.savefig(output_dir / "best_trajectory_speed.png", dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return projection


def plot_heatmap(
    run_dir: Path,
    rows: list[dict],
    output_dir: Path,
    game_map: Map | None = None,
    projection: MapProjection | None = None,
) -> MapProjection | None:
    xs: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    for row in rows:
        data = load_path(run_dir, row)
        xs.append(np.asarray(data["x"], dtype=np.float32))
        zs.append(np.asarray(data["z"], dtype=np.float32))
    if not xs:
        return
    x = np.concatenate(xs)
    z = np.concatenate(zs)
    if x.size == 0:
        return projection
    px, pz = project_path(x, z, projection if game_map is not None else None)
    fig, ax = plt.subplots(figsize=(12, 7) if game_map is not None else (9, 9))
    projection = setup_axis_with_map(ax, game_map, projection)
    heatmap = ax.hist2d(px, pz, bins=120, cmap="magma", alpha=0.72, zorder=95)
    ax.set_title("Trajectory visitation heatmap")
    fig.colorbar(heatmap[3], ax=ax, label="sample count")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_heatmap.png", dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return projection


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GA trajectory logs.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing trajectories/trajectory_manifest.csv.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--map-name", default=None, help="Optional map background. Defaults to config.json map_name when present.")
    parser.add_argument("--no-map", action="store_true", help="Disable map background even if config.json has map_name.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "trajectory_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(run_dir)
    map_name = None if args.no_map else infer_map_name(run_dir, args.map_name)
    game_map, projection = load_map_projection(map_name)
    projection = plot_overview(run_dir, rows, output_dir, game_map, projection)
    projection = plot_best_speed(run_dir, rows, output_dir, game_map, projection)
    plot_heatmap(run_dir, rows, output_dir, game_map, projection)
    print(f"Loaded trajectories: {len(rows)}")
    if game_map is not None:
        print(f"Map background: {map_name}")
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
