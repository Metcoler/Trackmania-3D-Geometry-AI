from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PACKAGE_ROOT / "runs" / "fixed100_lidar_finished_progress_time_crashes_seed_2026050314"
SUMMARY_CSV = (
    PACKAGE_ROOT
    / "analysis"
    / "architecture_ablation_debug_20260504"
    / "architecture_ablation_debug_summary.csv"
)
OUT_DIR = ROOT / "Masters thesis" / "Latex" / "images" / "training_policy"


VARIANTS = [
    {
        "dir": "arch_32x16_relu_relu",
        "run": "20260503_234912_tm2d_ga_AI_Training_5",
        "summary_label": "32x16 relu,relu",
        "label": r"$32{\times}16$ ReLU,ReLU",
        "short": "32x16\nReLU,ReLU",
        "color": "#356EA8",
        "linestyle": "--",
    },
    {
        "dir": "arch_32x16_relu_tanh",
        "run": "20260503_234758_tm2d_ga_AI_Training_5",
        "summary_label": "32x16 relu,tanh",
        "label": r"$32{\times}16$ ReLU,$\tanh$",
        "short": "32x16\nReLU,tanh",
        "color": "#356EA8",
        "linestyle": "-",
    },
    {
        "dir": "arch_48x24_relu_relu",
        "run": "20260504_000459_tm2d_ga_AI_Training_5",
        "summary_label": "48x24 relu,relu",
        "label": r"$48{\times}24$ ReLU,ReLU",
        "short": "48x24\nReLU,ReLU",
        "color": "#D9802E",
        "linestyle": "--",
    },
    {
        "dir": "arch_48x24_relu_tanh",
        "run": "20260504_000348_tm2d_ga_AI_Training_5",
        "summary_label": "48x24 relu,tanh",
        "label": r"$48{\times}24$ ReLU,$\tanh$",
        "short": "48x24\nReLU,tanh",
        "color": "#D9802E",
        "linestyle": "-",
    },
]


def load_generation_curves() -> dict[str, pd.DataFrame]:
    curves: dict[str, pd.DataFrame] = {}
    for variant in VARIANTS:
        path = RUN_ROOT / variant["dir"] / variant["run"] / "generation_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        curves[variant["summary_label"]] = pd.read_csv(path)
    return curves


def load_summary() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(SUMMARY_CSV)
    summary = pd.read_csv(SUMMARY_CSV).set_index("label")
    missing = {variant["summary_label"] for variant in VARIANTS} - set(summary.index)
    if missing:
        raise ValueError(f"Summary is missing variants: {sorted(missing)}")
    return summary


def draw_activation_comparison() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curves = load_generation_curves()
    summary = load_summary()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.2,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "figure.dpi": 150,
        }
    )

    fig = plt.figure(figsize=(12.4, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0])
    ax_progress = fig.add_subplot(gs[0, 0])
    ax_summary = fig.add_subplot(gs[0, 1])

    for variant in VARIANTS:
        df = curves[variant["summary_label"]]
        ax_progress.plot(
            df["generation"],
            df["best_dense_progress"].cummax().clip(upper=100),
            color=variant["color"],
            linestyle=variant["linestyle"],
            linewidth=2.4,
            label=variant["label"],
        )

    ax_progress.set_title("Best progress found")
    ax_progress.set_xlabel("Generation")
    ax_progress.set_ylabel("Progress [%]")
    ax_progress.set_ylim(-2, 104)
    ax_progress.set_xlim(0, 200)
    ax_progress.grid(True, color="#b0b0b0", alpha=0.25)
    ax_progress.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#dddddd",
        framealpha=0.92,
    )

    labels = [variant["short"] for variant in VARIANTS]
    finish_counts = [summary.loc[variant["summary_label"], "total_finish_individuals"] for variant in VARIANTS]
    best_times = [summary.loc[variant["summary_label"], "best_finish_time"] for variant in VARIANTS]
    colors = [variant["color"] for variant in VARIANTS]
    x_positions = range(len(VARIANTS))

    bars = ax_summary.bar(
        x_positions,
        finish_counts,
        color=colors,
        alpha=0.72,
        label="Finishing individuals",
    )
    ax_summary.set_title("Quality of finished runs")
    ax_summary.set_ylabel("Finishing individuals")
    ax_summary.set_xticks(list(x_positions), labels)
    ax_summary.set_ylim(0, max(finish_counts) * 1.18)
    ax_summary.grid(True, axis="y", color="#b0b0b0", alpha=0.25)
    for bar, value in zip(bars, finish_counts):
        ax_summary.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(finish_counts) * 0.025,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )

    ax_time = ax_summary.twinx()
    ax_time.plot(
        list(x_positions),
        best_times,
        color="#1C1C1C",
        marker="o",
        linewidth=2.2,
        label="Best time",
    )
    ax_time.set_ylabel("Best time [s]")
    ax_time.set_ylim(17.0, 25.5)
    for x, value in zip(x_positions, best_times):
        ax_time.text(
            x,
            value - 0.42,
            f"{value:.2f} s".replace(".", ","),
            ha="center",
            va="top",
            fontsize=8.6,
            color="#1C1C1C",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    summary_handles, summary_labels = ax_summary.get_legend_handles_labels()
    time_handles, time_labels = ax_time.get_legend_handles_labels()
    ax_summary.legend(
        summary_handles + time_handles,
        summary_labels + time_labels,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#dddddd",
        framealpha=0.92,
    )

    for extension in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"architecture_closed_loop_activation_comparison.{extension}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw_activation_comparison()
