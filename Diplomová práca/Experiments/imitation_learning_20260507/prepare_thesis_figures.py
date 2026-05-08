from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Experiments.tm_map_plotting import render_map_background  # noqa: E402
from Map import Map  # noqa: E402


MAP_NAME = "single_surface_flat"
TEACHER_ANALYSIS_DIR = (
    REPO_ROOT
    / "Experiments"
    / "supervised_map_specialists_20260505"
    / "analysis"
    / MAP_NAME
)
IMITATION_ATTEMPT = (
    REPO_ROOT
    / "logs"
    / "imitation_data"
    / "20260507_214750_map_single_surface_flat_v2d_asphalt_target_dataset"
    / "attempts"
    / "attempt_0003.npz"
)
OUTPUT_DIR = REPO_ROOT / "Diplomová práca" / "Latex" / "images" / "training_policy"
FIGURE_STEM = "imitation_learning_final_agent_path"

SPEED_TO_KMH = 3.6
SPEED_CMAP = plt.get_cmap("turbo_r")
TEACHER_COLOR = "#4b5563"
AGENT_COLOR = "#38bdf8"
CONTACT_COLOR = "#050505"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_teacher_path(path: Path) -> dict[str, Any]:
    with np.load(path) as data:
        positions = np.asarray(data["positions"], dtype=np.float32)
        speeds = np.asarray(data.get("speeds", np.zeros((positions.shape[0],), dtype=np.float32)), dtype=np.float32)
    return {"positions": positions, "speeds": speeds}


def load_imitation_attempt(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def fmt_seconds(value: float) -> str:
    return f"{float(value):.2f}".replace(".", ",")


def plot_speed_gradient(
    ax,
    points: np.ndarray,
    speed: np.ndarray,
    *,
    norm: mcolors.Normalize,
    linewidth: float,
    alpha: float,
    zorder: int,
) -> LineCollection | None:
    if points.shape[0] < 2:
        return None
    segments = np.stack([points[:-1], points[1:]], axis=1)
    segment_speed = np.asarray(speed[: points.shape[0] - 1], dtype=np.float32)
    collection = LineCollection(
        segments,
        cmap=SPEED_CMAP,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    collection.set_array(segment_speed)
    collection.set_norm(norm)
    ax.add_collection(collection)
    return collection


def contact_start_indices(agent: dict[str, np.ndarray]) -> np.ndarray:
    crashes = np.asarray(agent.get("crashes", np.asarray([], dtype=np.float32)), dtype=np.float32)
    if crashes.size:
        crash_starts = np.flatnonzero(np.diff(np.concatenate([[0.0], crashes])) > 0.0)
        if crash_starts.size:
            return crash_starts

    laser_clearances = np.asarray(agent.get("laser_clearances", np.asarray([], dtype=np.float32)), dtype=np.float32)
    if laser_clearances.ndim == 2 and laser_clearances.size:
        contact = np.min(laser_clearances, axis=1) <= 0.0
        return np.flatnonzero(contact & ~np.concatenate([[False], contact[:-1]]))
    return np.asarray([], dtype=np.int64)


def build_figure() -> None:
    teacher_summary = read_csv_rows(TEACHER_ANALYSIS_DIR / "teacher_paths_summary.csv")
    teacher_paths = [
        load_teacher_path(REPO_ROOT / row["attempt_file"])
        for row in teacher_summary
        if row.get("attempt_file")
    ]
    teacher_times = [
        float(row["finish_time"])
        for row in teacher_summary
        if row.get("finished") == "1" and row.get("finish_time")
    ]
    agent = load_imitation_attempt(IMITATION_ATTEMPT)

    finish_time = float(np.asarray(agent.get("finish_time", [np.nan])).reshape(-1)[0])
    finished = int(np.asarray(agent.get("finish_finished", [0])).reshape(-1)[0])

    game_map = Map(MAP_NAME)
    fig, ax = plt.subplots(figsize=(12.4, 7.2))
    projection = render_map_background(ax, game_map, show_legend=False, alpha=0.92)
    map_xlim = ax.get_xlim()
    map_ylim = ax.get_ylim()
    ax.set_autoscale_on(False)

    speed_values: list[np.ndarray] = []
    for teacher in teacher_paths:
        if teacher["speeds"].size:
            speed_values.append(np.asarray(teacher["speeds"], dtype=np.float32) * SPEED_TO_KMH)
    if agent.get("speeds", np.asarray([])).size:
        speed_values.append(np.asarray(agent["speeds"], dtype=np.float32) * SPEED_TO_KMH)
    valid = np.concatenate([values[np.isfinite(values)] for values in speed_values if values.size])
    speed_norm = mcolors.Normalize(
        vmin=float(np.nanmin(valid)) if valid.size else 0.0,
        vmax=max(float(np.nanmax(valid)) if valid.size else 1.0, 1.0),
    )

    mappable = None
    for teacher in teacher_paths:
        positions = np.asarray(teacher["positions"], dtype=np.float32)
        projected = projection.points(positions[:, [0, 2]])
        mappable = plot_speed_gradient(
            ax,
            projected,
            np.asarray(teacher["speeds"], dtype=np.float32) * SPEED_TO_KMH,
            norm=speed_norm,
            linewidth=1.65,
            alpha=0.42,
            zorder=80,
        )

    positions = np.asarray(agent["positions"], dtype=np.float32)
    projected_agent = projection.points(positions[:, [0, 2]])
    agent_speed = np.asarray(agent.get("speeds", np.zeros(positions.shape[0])), dtype=np.float32) * SPEED_TO_KMH
    ax.plot(
        projected_agent[:, 0],
        projected_agent[:, 1],
        color="#111827",
        linewidth=6.0,
        alpha=0.38,
        zorder=98,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    mappable = plot_speed_gradient(
        ax,
        projected_agent,
        agent_speed,
        norm=speed_norm,
        linewidth=3.8,
        alpha=0.98,
        zorder=100,
    )

    contacts = contact_start_indices(agent)
    if contacts.size:
        ax.scatter(
            projected_agent[contacts, 0],
            projected_agent[contacts, 1],
            s=46,
            facecolors=CONTACT_COLOR,
            edgecolors=CONTACT_COLOR,
            linewidths=0,
            zorder=120,
        )

    ax.set_xlim(map_xlim)
    ax.set_ylim(map_ylim)
    ax.set_axis_off()

    path_handles = [
        Line2D([0], [0], color=TEACHER_COLOR, lw=2.0, alpha=0.78, label="jazdy učiteľa"),
        Line2D([0], [0], color=AGENT_COLOR, lw=3.8, alpha=0.98, label="agent po učení"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CONTACT_COLOR,
            markeredgecolor=CONTACT_COLOR,
            markersize=6.5,
            label="kontakt s okrajom",
        ),
    ]
    map_handles = [
        Patch(facecolor="#16a34a", edgecolor="none", label="štart"),
        Patch(facecolor="#dc2626", edgecolor="none", label="cieľ"),
        Patch(facecolor="#b9b9b9", edgecolor="none", label="cesta"),
        Patch(facecolor="#262626", edgecolor="none", label="okraj trate"),
    ]
    legend1 = ax.legend(
        handles=path_handles,
        loc="upper left",
        bbox_to_anchor=(0.945, 0.82),
        frameon=False,
        fontsize=9.2,
        borderaxespad=0.0,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=map_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, -0.05),
        ncol=4,
        frameon=False,
        fontsize=9.2,
        handlelength=1.1,
        columnspacing=1.0,
    )

    if teacher_times and np.isfinite(finish_time):
        status = "dokončené" if finished else "nedokončené"
        fig.text(
            0.805,
            0.235,
            "čas prejazdu\n"
            f"agent: {fmt_seconds(finish_time)} s\n"
            f"učiteľ: {fmt_seconds(min(teacher_times))}-{fmt_seconds(max(teacher_times))} s\n"
            f"stav: {status}",
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

    if mappable is not None:
        cax = fig.add_axes([0.848, 0.36, 0.014, 0.28])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label("rýchlosť [km/h]", fontsize=9.4, labelpad=7)
        cbar.ax.tick_params(labelsize=8.4)
        cbar.ax.text(-0.82, 0.02, "pomaly", transform=cbar.ax.transAxes, fontsize=8.4, ha="right", va="bottom")
        cbar.ax.text(-0.82, 0.98, "rýchlo", transform=cbar.ax.transAxes, fontsize=8.4, ha="right", va="top")

    fig.subplots_adjust(left=0.02, right=0.765, top=0.98, bottom=0.10)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(OUTPUT_DIR / f"{FIGURE_STEM}{suffix}", dpi=220, bbox_inches="tight", pad_inches=0.24)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
