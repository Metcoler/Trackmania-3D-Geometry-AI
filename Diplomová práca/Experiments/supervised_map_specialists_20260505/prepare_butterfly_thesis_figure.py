from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = REPO_ROOT / "Experiments" / "supervised_map_specialists_20260505" / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from Map import Map  # noqa: E402
from tm_map_plotting import render_map_background  # noqa: E402


INPUT_DIR = (
    REPO_ROOT
    / "Experiments"
    / "analysis"
    / "supervised_butterfly_sweep_20260507"
    / "single_surface_flat"
)
OUTPUT_DIR = REPO_ROOT / "Masters thesis" / "Latex" / "images" / "training_policy"
FIGURE_STEM = "supervised_behavior_cloning_butterfly_p04_left_120"
MAP_NAME = "single_surface_flat"

BASELINE_COLOR = "#38bdf8"
PERTURBED_COLOR = "#f97316"
PERTURB_MARKER_COLOR = "#dc2626"
CRASH_COLOR = "#050505"


def read_metrics(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0], rows[1]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def fmt_seconds(value: Any) -> str:
    return f"{float(value):.2f}".replace(".", ",")


def draw_path(ax, points: np.ndarray, *, color: str, linewidth: float, alpha: float, zorder: int) -> None:
    if points.shape[0] < 2:
        return
    ax.plot(
        points[:, 0],
        points[:, 1],
        color="#111827",
        linewidth=linewidth + 2.3,
        alpha=0.35,
        zorder=zorder - 1,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
        solid_capstyle="round",
        solid_joinstyle="round",
    )


def add_slovak_map_legend(fig) -> None:
    handles = [
        Patch(facecolor="#159947", edgecolor="none", label="start"),
        Patch(facecolor="#dc2626", edgecolor="none", label="finish"),
        Patch(facecolor="#b7b7b7", edgecolor="none", label="road"),
        Patch(facecolor="#2f2f2f", edgecolor="none", label="track boundary"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.395, 0.025),
        ncol=4,
        frameon=False,
        fontsize=9.8,
        handlelength=1.2,
        handleheight=0.9,
        columnspacing=1.3,
    )


def build_figure() -> None:
    baseline = load_npz(INPUT_DIR / "butterfly_baseline_p04_left_120_trajectory.npz")
    perturbed = load_npz(INPUT_DIR / "butterfly_perturbed_p04_left_120_trajectory.npz")
    baseline_metrics, perturbed_metrics = read_metrics(INPUT_DIR / "butterfly_metrics_p04_left_120.csv")

    game_map = Map(MAP_NAME)
    fig, ax = plt.subplots(figsize=(12.0, 7.2))
    projection = render_map_background(ax, game_map, show_legend=False, alpha=0.92)
    map_xlim = ax.get_xlim()
    map_ylim = ax.get_ylim()
    ax.set_autoscale_on(False)

    baseline_points = projection.points(np.stack([baseline["x"], baseline["z"]], axis=1))
    perturbed_points = projection.points(np.stack([perturbed["x"], perturbed["z"]], axis=1))

    draw_path(ax, baseline_points, color=BASELINE_COLOR, linewidth=3.15, alpha=0.93, zorder=95)
    draw_path(ax, perturbed_points, color=PERTURBED_COLOR, linewidth=3.45, alpha=0.96, zorder=100)

    active = np.flatnonzero(np.asarray(perturbed.get("perturb_active", []), dtype=np.float32) > 0.0)
    if active.size:
        start = int(active[0])
        end = int(active[-1])
        ax.plot(
            perturbed_points[start : end + 1, 0],
            perturbed_points[start : end + 1, 1],
            color=PERTURB_MARKER_COLOR,
            linewidth=5.4,
            alpha=0.82,
            zorder=125,
            solid_capstyle="round",
        )
        ax.scatter(
            perturbed_points[start, 0],
            perturbed_points[start, 1],
            marker="D",
            s=82,
            facecolors=PERTURB_MARKER_COLOR,
            edgecolors="#111827",
            linewidths=0.8,
            zorder=130,
        )

    touch_count = np.asarray(perturbed.get("touch_count", np.asarray([])), dtype=np.float32)
    crash_indices = np.flatnonzero(np.diff(np.concatenate([[0.0], touch_count])) > 0.0)
    if crash_indices.size:
        collision_x = np.asarray(perturbed.get("collision_x", perturbed["x"]), dtype=np.float32)
        collision_z = np.asarray(perturbed.get("collision_z", perturbed["z"]), dtype=np.float32)
        crash_points = projection.points(np.stack([collision_x, collision_z], axis=1))
        ax.scatter(
            crash_points[crash_indices, 0],
            crash_points[crash_indices, 1],
            marker="o",
            s=38,
            facecolors=CRASH_COLOR,
            edgecolors=CRASH_COLOR,
            linewidths=0.0,
            zorder=132,
        )

    handles = [
        Line2D([0], [0], color=BASELINE_COLOR, lw=3.15, label="baseline run"),
        Line2D([0], [0], color=PERTURBED_COLOR, lw=3.45, label="run after forced action change"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=PERTURB_MARKER_COLOR,
            markeredgecolor="#111827",
            markersize=7,
            label="perturbation start",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CRASH_COLOR,
            markeredgecolor=CRASH_COLOR,
            markersize=6,
            label="wall contact",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.015, 0.79),
        frameon=False,
        fontsize=9.8,
        borderaxespad=0.0,
    )

    fig.text(
        0.803,
        0.43,
        "run metrics\n"
        "forced action: left, 120 steps\n"
        f"baseline: {fmt_seconds(baseline_metrics['time'])} s, {int(float(baseline_metrics['crashes']))} contacts\n"
        f"after perturbation: {fmt_seconds(perturbed_metrics['time'])} s, {int(float(perturbed_metrics['crashes']))} contacts",
        ha="left",
        va="top",
        fontsize=9.4,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.34",
            "facecolor": "white",
            "edgecolor": "#d1d5db",
            "linewidth": 0.8,
            "alpha": 0.94,
        },
    )

    add_slovak_map_legend(fig)
    ax.set_title("")
    ax.set_axis_off()
    ax.set_xlim(map_xlim)
    ax.set_ylim(map_ylim)
    fig.subplots_adjust(left=0.02, right=0.78, top=0.98, bottom=0.11)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(OUTPUT_DIR / f"{FIGURE_STEM}{suffix}", dpi=220, bbox_inches="tight", pad_inches=0.13)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
