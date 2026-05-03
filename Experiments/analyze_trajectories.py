from __future__ import annotations

import argparse
import csv
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


def plot_overview(run_dir: Path, rows: list[dict], output_dir: Path) -> None:
    plt.figure(figsize=(9, 9))
    for row in rows:
        data = load_path(run_dir, row)
        x = np.asarray(data["x"], dtype=np.float32)
        z = np.asarray(data["z"], dtype=np.float32)
        if x.size < 2:
            continue
        finished = int(row.get("finished", 0))
        rank = int(row.get("rank", 999999))
        if finished > 0:
            plt.plot(x, z, color="#0b6e4f", alpha=0.65, linewidth=1.8)
        elif rank == 1:
            plt.plot(x, z, color="#2454a6", alpha=0.35, linewidth=1.2)
        else:
            plt.plot(x, z, color="#222222", alpha=0.08, linewidth=0.7)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Logged trajectories: top paths and finishers")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_overview.png", dpi=180)
    plt.close()


def plot_best_speed(run_dir: Path, rows: list[dict], output_dir: Path) -> None:
    if not rows:
        return
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
        return
    points = np.column_stack([x, z]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    speed_segments = speed[:-1] if speed.size >= x.size else np.zeros(x.size - 1, dtype=np.float32)
    collection = LineCollection(segments, cmap="turbo", linewidth=2.0)
    collection.set_array(speed_segments)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.add_collection(collection)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("Best logged trajectory colored by speed")
    fig.colorbar(collection, ax=ax, label="speed")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "best_trajectory_speed.png", dpi=180)
    plt.close(fig)


def plot_heatmap(run_dir: Path, rows: list[dict], output_dir: Path) -> None:
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
        return
    plt.figure(figsize=(9, 9))
    plt.hist2d(x, z, bins=120, cmap="magma")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Trajectory visitation heatmap")
    plt.colorbar(label="sample count")
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_heatmap.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GA trajectory logs.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing trajectories/trajectory_manifest.csv.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "trajectory_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(run_dir)
    plot_overview(run_dir, rows, output_dir)
    plot_best_speed(run_dir, rows, output_dir)
    plot_heatmap(run_dir, rows, output_dir)
    print(f"Loaded trajectories: {len(rows)}")
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
