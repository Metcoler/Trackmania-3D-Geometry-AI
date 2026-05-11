from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from Experiments.tm_map_plotting import render_map_background
from Map import Map
from prepare_thesis_figures import IMAGE_DIR, RUNS, _load_run, _project_points


OUTPUT_NAMES = {
    "single_surface_flat": "final_ga_training_trajectory_single_surface_flat",
    "single_surface_height": "final_ga_training_trajectory_single_surface_height",
    "multi_surface_flat": "final_ga_training_trajectory_multi_surface_flat",
}

TITLE_LABELS = {
    "single_surface_flat": "Flat track",
    "single_surface_height": "Height track",
    "multi_surface_flat": "Different surfaces",
}


def fmt(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def _draw_block_legend(fig: plt.Figure) -> None:
    block_legend = [
        Patch(facecolor="#b9b9b9", edgecolor="#555555", label="asphalt"),
        Patch(facecolor="#b7d984", edgecolor="#6b8f3a", label="grass"),
        Patch(facecolor="#a2642f", edgecolor="#6b3d1e", label="dirt"),
        Patch(facecolor="#16a34a", edgecolor="#0f6f32", label="start"),
        Patch(facecolor="#dc2626", edgecolor="#8f1d1d", label="finish"),
    ]
    fig.legend(
        handles=block_legend,
        loc="center right",
        bbox_to_anchor=(0.995, 0.54),
        frameon=True,
        fontsize=8,
        title="Blocks",
        title_fontsize=9,
        borderpad=0.55,
        labelspacing=0.45,
        handlelength=1.4,
    )


def _height_range(game_map: Map) -> tuple[float, float]:
    heights: list[np.ndarray] = []
    for block in game_map.blocks.values():
        mesh = block.get_road_mesh()
        triangles = np.asarray(mesh.triangles, dtype=np.float64)
        if triangles.size:
            heights.append(triangles[..., 1].mean(axis=1))
    if not heights:
        return 0.0, 0.0
    values = np.concatenate(heights)
    return float(np.nanmin(values)), float(np.nanmax(values))


def _draw_height_legend(fig: plt.Figure, height_min: float, height_max: float) -> None:
    cax = fig.add_axes([0.905, 0.30, 0.024, 0.42])
    gradient = np.linspace(0.0, 1.0, 256).reshape(-1, 1)
    cmap = LinearSegmentedColormap.from_list("tm_height_gray", ["#3e3e3e", "#d8d8d8"])
    cax.imshow(gradient, aspect="auto", cmap=cmap, origin="lower")
    cax.set_xticks([])
    cax.set_yticks([0, 255])
    cax.set_yticklabels(
        [
            f"{height_min:.0f}",
            f"{height_max:.0f}",
        ],
        fontsize=8,
    )
    cax.set_title("height", fontsize=8, pad=4)
    cax.set_ylabel("map units", fontsize=8, labelpad=3)
    for spine in cax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor("#555555")


def plot_single_trajectory(run_data) -> Path:
    data = run_data.best_trajectory
    speed_kmh = np.asarray(data["speed"], dtype=np.float64) * 3.6
    vmax = max(180.0, float(np.nanpercentile(speed_kmh, 98)))
    norm = Normalize(vmin=100.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8.9, 6.15))
    game_map = Map(str(run_data.config["map_name"]))
    projection = render_map_background(ax, game_map, alpha=0.88)

    x = np.asarray(data["x"], dtype=np.float64)
    z = np.asarray(data["z"], dtype=np.float64)
    px, pz = _project_points(projection, x, z)

    points = np.column_stack([px, pz]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    collection = LineCollection(segments, cmap="turbo_r", norm=norm, linewidth=2.35, zorder=100)
    collection.set_array(speed_kmh[:-1] if speed_kmh.size > 1 else np.asarray([0.0]))
    ax.add_collection(collection)

    time_text = fmt(float(run_data.summary["best_trajectory_time_s"]))
    progress_text = f"{float(run_data.summary['best_trajectory_progress_pct']):.1f}"
    title_label = TITLE_LABELS.get(run_data.key, run_data.label)
    ax.set_title(f"{title_label}: best run, time {time_text} s, progress {progress_text} %")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    has_surface_variants = run_data.key == "multi_surface_flat"
    has_height_variants = run_data.key == "single_surface_height"
    if has_surface_variants:
        _draw_block_legend(fig)
    if has_height_variants:
        height_min, height_max = _height_range(game_map)
        _draw_height_legend(fig, height_min, height_max)
    colorbar = fig.colorbar(collection, ax=ax, orientation="horizontal", fraction=0.052, pad=0.055)
    colorbar.set_label("Speed [km/h]  (blue = faster, red = slower)", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)

    right_margin = 0.88 if (has_surface_variants or has_height_variants) else 0.985
    fig.subplots_adjust(left=0.015, right=right_margin, top=0.92, bottom=0.12)
    output = IMAGE_DIR / f"{OUTPUT_NAMES[run_data.key]}.pdf"
    fig.savefig(output, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(output.with_suffix(".png"), dpi=210, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    return output


def main() -> int:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for spec in RUNS:
        outputs.append(plot_single_trajectory(_load_run(spec)))
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
