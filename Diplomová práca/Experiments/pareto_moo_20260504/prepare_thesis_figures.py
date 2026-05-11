from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "Masters thesis" / "Latex" / "images" / "training_policy"

DIRECT_RUN = ROOT / (
    "Experiments/runs_ga_moo/"
    "moo_lex_sweep_aabb_lidar_fixed100_seed_2026050406/"
    "finished_progress_time_crashes/"
    "20260504_220703_tm2d_ga_moo_AI_Training_5_lexicographic_primitives"
)
ADJUSTED_RUN = ROOT / (
    "Experiments/runs_ga_moo/"
    "moo_trackmania_racing_aabb_lidar_fixed100_seed_2026050406/"
    "20260505_004133_tm2d_ga_moo_AI_Training_5_trackmania_racing"
)

RUNS = [
    {
        "label": "direct metrics",
        "generation": DIRECT_RUN / "generation_metrics.csv",
        "individual": DIRECT_RUN / "individual_metrics.csv",
        "color": "#ef4444",
    },
    {
        "label": "progress-weighted metrics",
        "generation": ADJUSTED_RUN / "generation_metrics.csv",
        "individual": ADJUSTED_RUN / "individual_metrics.csv",
        "color": "#2563eb",
    },
]


def load_generation(run: dict) -> pd.DataFrame:
    df = pd.read_csv(run["generation"])
    df["label"] = run["label"]
    return df


def load_final_population(run: dict) -> pd.DataFrame:
    df = pd.read_csv(run["individual"])
    final_generation = int(df["generation"].max())
    final = df[df["generation"] == final_generation].copy()
    final["label"] = run["label"]
    return final


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(OUTPUT_DIR / f"{stem}{suffix}", dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def draw_progress_comparison() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 6.2), sharex=True)

    for run in RUNS:
        df = load_generation(run)
        x = df["generation"]
        axes[0].plot(x, df["best_progress"], color=run["color"], linewidth=2.2, label=run["label"])
        if {"progress_p25", "progress_p75"}.issubset(df.columns):
            axes[1].fill_between(
                x,
                df["progress_p25"],
                df["progress_p75"],
                color=run["color"],
                alpha=0.12,
                linewidth=0.0,
                zorder=1,
            )
        axes[1].plot(x, df["mean_progress"], color=run["color"], linewidth=2.0, label=run["label"])

    axes[0].set_ylabel("Best progress [%]")
    axes[1].set_ylabel("Mean progress [%]")
    axes[1].set_xlabel("Generation")

    for ax in axes:
        ax.set_ylim(-2, 104)
        ax.grid(True, alpha=0.24)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(loc="upper left", frameon=True, framealpha=0.94, fontsize=10)
    fig.suptitle("Pareto comparison: progress over generations", fontsize=14, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_figure(fig, "pareto_moo_progress_comparison")


def draw_front_diagnostic() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=True)

    for ax, run in zip(axes, RUNS):
        df = load_final_population(run)
        front = df[df["front0"].astype(int) > 0].copy()
        other = df[df["front0"].astype(int) <= 0].copy()

        ax.scatter(
            other["dense_progress"],
            other["time"],
            s=24,
            color="#d1d5db",
            edgecolor="none",
            alpha=0.75,
            zorder=10,
        )

        if not front.empty:
            front_finished = front[front["finished"].astype(int) > 0]
            front_crashed = front[(front["finished"].astype(int) <= 0) & (front["crashes"].astype(int) > 0)]
            front_other = front[(front["finished"].astype(int) <= 0) & (front["crashes"].astype(int) <= 0)]

            ax.scatter(
                front_crashed["dense_progress"],
                front_crashed["time"],
                s=58,
                marker="X",
                color="#ef4444",
                edgecolor="#7f1d1d",
                linewidth=0.8,
                zorder=40,
            )
            ax.scatter(
                front_other["dense_progress"],
                front_other["time"],
                s=58,
                marker="o",
                color="#f59e0b",
                edgecolor="#92400e",
                linewidth=0.8,
                zorder=35,
            )
            ax.scatter(
                front_finished["dense_progress"],
                front_finished["time"],
                s=70,
                marker="o",
                color="#16a34a",
                edgecolor="#14532d",
                linewidth=0.9,
                zorder=45,
            )

        ax.set_title(run["label"], fontsize=12)
        ax.set_xlabel("Progress [%]")
        ax.set_xlim(0, 104)
        ax.set_ylim(0, 31)
        ax.grid(True, alpha=0.24)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Driving time [s]")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d1d5db", markersize=7, label="outside Pareto front"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#ef4444", markeredgecolor="#7f1d1d", markersize=8, label="Pareto front: crash"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#16a34a", markeredgecolor="#14532d", markersize=8, label="Pareto front: finish"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
        fontsize=10,
    )
    fig.suptitle("Final Pareto front after 300 generations", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    save_figure(fig, "pareto_moo_front_diagnostic")


def main() -> None:
    draw_progress_comparison()
    draw_front_diagnostic()


if __name__ == "__main__":
    main()
