from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "Experiments" / "training_improvements_20260505"
ANALYSIS = PACKAGE / "analysis" / "latest_training_results_20260505"
SUMMARY = ANALYSIS / "summary.csv"
OUT_DIR = ROOT / "Diplomová práca" / "Latex" / "images" / "training_policy"


VARIANTS = [
    ("exp00_base_fixed100", "Základ", "#425466"),
    ("exp01_mutation_decay_fixed100", "Pokles\nod začiatku", "#8c6bb1"),
    ("exp01b_mutation_decay_first_finish_fixed100", "Pokles\npo prvom cieli", "#5b8dd6"),
    ("exp02_both_mirrors_fixed100", "Obojstranné\nzrkadlenie", "#2f9e8f"),
    ("exp03_mirror_prob_050_fixed100", "Náhodné\nzrkadlenie", "#d28f2d"),
    ("exp05a_variable_tick_no_cache_control", "Premenlivé\nbez pamäte", "#c55a5a"),
    ("exp05b_variable_tick_elite_cache", "Premenlivé\ns pamäťou elity", "#2f8f5b"),
]

FOCUS_DECAY = [
    ("exp00_base_fixed100", "Základ", "#425466"),
    ("exp01b_mutation_decay_first_finish_fixed100", "Pokles po prvom cieli", "#5b8dd6"),
]

FOCUS_CACHE = [
    ("exp05a_variable_tick_no_cache_control", "Bez pamäte elity", "#c55a5a"),
    ("exp05b_variable_tick_elite_cache", "S pamäťou elity", "#2f8f5b"),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(OUT_DIR / f"{name}{suffix}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY)
    return df.set_index("experiment")


def load_generation(summary: pd.DataFrame, experiment: str) -> pd.DataFrame:
    run_dir = Path(summary.loc[experiment, "run_dir"])
    return pd.read_csv(run_dir / "generation_metrics.csv")


def plot_summary(summary: pd.DataFrame) -> None:
    rows = [summary.loc[key] for key, _, _ in VARIANTS]
    labels = [label for _, label, _ in VARIANTS]
    colors = [color for _, _, color in VARIANTS]
    x = np.arange(len(labels))

    best_time = np.array([row["best_time"] for row in rows], dtype=float)
    total_finish = np.array([row["total_finish"] for row in rows], dtype=float)
    late_finish = np.array([row["last50_finish_rate"] for row in rows], dtype=float) * 100.0

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), constrained_layout=True)

    axes[0].bar(x, np.nan_to_num(best_time, nan=0.0), color=colors, width=0.72)
    axes[0].set_title("Najlepší dokončený čas")
    axes[0].set_ylabel("Čas [s]")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    for i, value in enumerate(best_time):
        if math.isfinite(value):
            axes[0].text(i, value + 0.25, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        else:
            axes[0].text(i, 0.45, "bez\ncieľa", ha="center", va="bottom", fontsize=8)
    finite_times = best_time[np.isfinite(best_time)]
    axes[0].set_ylim(0, max(finite_times) + 4.0)

    axes[1].bar(x, total_finish, color=colors, width=0.72)
    axes[1].set_title("Dokončujúci jedinci")
    axes[1].set_ylabel("Počet")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    for i, value in enumerate(total_finish):
        axes[1].text(i, value + max(total_finish) * 0.025, f"{int(value)}", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, max(total_finish) * 1.14)

    axes[2].bar(x, late_finish, color=colors, width=0.72)
    axes[2].set_title("Stabilita v závere")
    axes[2].set_ylabel("Dokončenia\nv závere [%]")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=35, ha="right")
    for i, value in enumerate(late_finish):
        axes[2].text(i, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    axes[2].set_ylim(0, max(late_finish) + 8.0)

    save(fig, "ga_training_improvements_summary")


def plot_focus(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.3), sharex="col", constrained_layout=True)

    def draw_column(col: int, experiments: list[tuple[str, str, str]], title: str) -> None:
        progress_ax = axes[0, col]
        finish_ax = axes[1, col]
        for key, label, color in experiments:
            gen = load_generation(summary, key)
            x = gen["generation"].astype(float)
            if "best_dense_progress" in gen:
                best_progress = gen["best_dense_progress"].astype(float).cummax().clip(0, 100)
            else:
                best_progress = gen["best_progress"].astype(float).cummax().clip(0, 100)
            progress_ax.plot(x, best_progress, color=color, linewidth=2.2, label=label)
            finish_ax.plot(x, gen["finish_count"].astype(float), color=color, linewidth=1.8, label=label)

        progress_ax.set_title(title)
        progress_ax.set_ylabel("Najlepší progres [%]")
        progress_ax.set_ylim(-3, 103)
        finish_ax.set_ylabel("Dokončení jedinci")
        finish_ax.set_xlabel("Generácia")

    draw_column(0, FOCUS_DECAY, "Pokles mutácie")
    draw_column(1, FOCUS_CACHE, "Vyrovnávacia pamäť elity")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, loc="lower right", frameon=True)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    axes[0, 1].legend(handles, labels, loc="lower right", frameon=True)

    save(fig, "ga_training_improvements_focus")


def main() -> None:
    configure_style()
    summary = load_summary()
    plot_summary(summary)
    plot_focus(summary)
    print(f"Saved thesis figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
