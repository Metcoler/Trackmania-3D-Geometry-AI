from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiments.tm_map_plotting import render_map_background
from Map import Map


PACKAGE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "Diplomová práca" / "Latex" / "images" / "evaluation"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

RUN_DIR = (
    ROOT
    / "logs"
    / "tm_finetune_runs"
    / "20260507_235715_tm_finetune_map_small_map_v2d_asphalt_h48x24_p48_src_population_gen_0080"
)
MANIFEST_PATH = RUN_DIR / "trajectories" / "trajectory_manifest.csv"


def fmt(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def load_best_row() -> dict[str, str]:
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    finishers = [row for row in rows if int(float(row["finished"])) == 1]
    if not finishers:
        raise RuntimeError("No finished small_map trajectory found.")
    return min(finishers, key=lambda row: float(row["time"]))


def project_points(projection: object, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.column_stack([x, z])
    if projection is not None:
        points = projection.points(points)
    return points[:, 0], points[:, 1]


def plot_small_map_trajectory() -> None:
    row = load_best_row()
    trajectory_path = RUN_DIR / row["path_file"]
    data = np.load(trajectory_path)

    x = np.asarray(data["x"], dtype=np.float64)
    z = np.asarray(data["z"], dtype=np.float64)
    speed_kmh = np.asarray(data["speed"], dtype=np.float64) * 3.6
    finish_time = float(row["time"])

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    projection = render_map_background(
        ax,
        Map("small_map"),
        alpha=0.88,
        checkpoint_as_straight=True,
    )
    px, pz = project_points(projection, x, z)

    points = np.column_stack([px, pz]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    vmax = max(140.0, float(np.nanpercentile(speed_kmh, 98)))
    collection = LineCollection(
        segments,
        cmap="turbo_r",
        norm=Normalize(vmin=80.0, vmax=vmax),
        linewidth=2.4,
        zorder=100,
    )
    collection.set_array(speed_kmh[:-1])
    ax.add_collection(collection)

    ax.scatter(px[0], pz[0], s=42, color="#16a34a", edgecolor="white", linewidth=0.8, zorder=110)
    ax.scatter(px[-1], pz[-1], s=42, color="#dc2626", edgecolor="white", linewidth=0.8, zorder=110)
    ax.annotate(
        "štart",
        xy=(px[0], pz[0]),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=9,
        color="#166534",
        arrowprops={"arrowstyle": "-", "color": "#166534", "linewidth": 0.8},
    )
    ax.annotate(
        "cieľ",
        xy=(px[-1], pz[-1]),
        xytext=(10, -2),
        textcoords="offset points",
        fontsize=9,
        color="#991b1b",
        arrowprops={"arrowstyle": "-", "color": "#991b1b", "linewidth": 0.8},
    )

    ax.set_title(f"Trajektória agenta na mape small_map, čas {fmt(finish_time)} s", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    colorbar = fig.colorbar(collection, ax=ax, orientation="horizontal", fraction=0.05, pad=0.055)
    colorbar.set_label("Rýchlosť [km/h]  (modrá = rýchlejšie, červená = pomalšie)")

    fig.tight_layout()
    output = IMAGE_DIR / "evaluation_small_map_trajectory.pdf"
    fig.savefig(output, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)

    report = [
        "# small_map trajectory",
        "",
        f"- Source run: `{RUN_DIR.relative_to(ROOT)}`",
        f"- Trajectory: `{row['path_file']}`",
        f"- Finished: `{row['finished']}`",
        f"- Time: `{fmt(finish_time)} s`",
        f"- Dense progress: `{fmt(float(row['dense_progress']))} %`",
    ]
    (PACKAGE_DIR / "small_map_trajectory.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    plot_small_map_trajectory()
    print("Wrote small_map trajectory figure.")


if __name__ == "__main__":
    main()
