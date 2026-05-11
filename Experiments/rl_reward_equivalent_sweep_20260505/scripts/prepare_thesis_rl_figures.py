from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis" / "rl_reward_equivalent_sweep_20260505"
THESIS_IMAGE_DIR = ROOT.parents[1] / "Masters thesis" / "Latex" / "images" / "training_policy"


RUN_ORDER = [
    "ppo_fixed100",
    "ppo_supervised_v2d",
    "sac_fixed100",
    "sac_supervised_v2d",
    "td3_fixed100",
    "td3_supervised_v2d",
]

ALGORITHM_ORDER = ["PPO", "SAC", "TD3"]

ALGORITHM_COLORS = {
    "PPO": "#1f77b4",
    "SAC": "#2ca02c",
    "TD3": "#d62728",
}


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#d9dee7", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#677384")
    ax.spines["bottom"].set_color("#677384")


def save_figure(fig: plt.Figure, name: str) -> None:
    THESIS_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(THESIS_IMAGE_DIR / f"{name}{suffix}", bbox_inches="tight", dpi=220)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.read_csv(ANALYSIS_DIR / "combined_episode_metrics.csv")
    summary = pd.read_csv(ANALYSIS_DIR / "summary.csv")
    combined = combined[combined["run_short"].isin(RUN_ORDER)].copy()
    summary = summary[summary["run_short"].isin(RUN_ORDER)].copy()
    combined["run_short"] = pd.Categorical(combined["run_short"], RUN_ORDER, ordered=True)
    summary["run_short"] = pd.Categorical(summary["run_short"], RUN_ORDER, ordered=True)
    return combined.sort_values(["run_short", "episode"]), summary.sort_values("run_short")


def plot_progress(combined: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.1), sharex=True)

    for algorithm in ALGORITHM_ORDER:
        algorithm_group = combined[combined["algorithm"].astype(str).str.upper() == algorithm].copy()
        if algorithm_group.empty:
            continue
        color = ALGORITHM_COLORS[algorithm]

        per_run = []
        for _, run_group in algorithm_group.groupby("run_short", observed=True):
            run_group = run_group.sort_values("episode").copy()
            run_group["best_progress"] = run_group["progress"].astype(float).cummax()
            run_group["rolling_progress"] = (
                run_group["progress"].astype(float).rolling(50, min_periods=1).mean()
            )
            per_run.append(run_group[["episode", "best_progress", "rolling_progress"]])

        merged = pd.concat(per_run, ignore_index=True)
        aggregated = (
            merged.groupby("episode", as_index=False)
            .agg(best_progress=("best_progress", "max"), rolling_progress=("rolling_progress", "mean"))
            .sort_values("episode")
        )

        axes[0].plot(
            aggregated["episode"],
            aggregated["best_progress"],
            label=algorithm,
            color=color,
            linewidth=2.3,
        )
        axes[1].plot(
            aggregated["episode"],
            aggregated["rolling_progress"],
            label=algorithm,
            color=color,
            linewidth=2.3,
        )

    axes[0].set_title("Best achieved progress")
    axes[0].set_ylabel("Progress [%]")
    axes[0].set_ylim(-2, 104)
    axes[0].set_xlabel("Episode")

    axes[1].set_title("Rolling mean progress")
    axes[1].set_ylabel("Progress [%]")
    axes[1].set_ylim(-2, 104)
    axes[1].set_xlabel("Episode")

    for ax in axes:
        style_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=10,
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.22)
    save_figure(fig, "rl_reward_progress_over_episodes")
    plt.close(fig)


def plot_results_summary(summary: pd.DataFrame) -> None:
    rows = []
    for algorithm in ALGORITHM_ORDER:
        group = summary[summary["algorithm"].astype(str).str.upper() == algorithm]
        if group.empty:
            continue
        best_times = group["best_finish_time"].astype(float).to_numpy()
        finite_times = best_times[np.isfinite(best_times)]
        rows.append(
            {
                "algorithm": algorithm,
                "finish_count": float(group["finish_count"].astype(float).sum()),
                "best_finish_time": float(np.min(finite_times)) if len(finite_times) else np.nan,
            }
        )
    aggregated = pd.DataFrame(rows)
    labels = aggregated["algorithm"].tolist()
    colors = [ALGORITHM_COLORS[label] for label in labels]
    x = np.arange(len(aggregated))

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.1))

    finish_counts = aggregated["finish_count"].astype(float).to_numpy()
    bars = axes[0].bar(x, finish_counts, color=colors, edgecolor="#303642", linewidth=0.8)
    axes[0].set_title("Number of finishes")
    axes[0].set_ylabel("Episode count")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=24, ha="right")
    axes[0].set_ylim(0, max(50, finish_counts.max() * 1.22))
    for bar, value in zip(bars, finish_counts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    best_times = aggregated["best_finish_time"].astype(float).to_numpy()
    finite_times = np.isfinite(best_times)
    plotted_times = np.where(finite_times, best_times, 0.0)
    bars = axes[1].bar(x, plotted_times, color=colors, edgecolor="#303642", linewidth=0.8)
    axes[1].set_title("Best finished time")
    axes[1].set_ylabel("Time [s]")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=24, ha="right")
    axes[1].set_ylim(0, 31)
    for bar, value, is_finite in zip(bars, best_times, finite_times):
        cx = bar.get_x() + bar.get_width() / 2
        if is_finite:
            axes[1].text(cx, bar.get_height() + 0.45, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
        else:
            bar.set_facecolor("#d8dbe2")
            bar.set_hatch("//")
            axes[1].text(cx, 1.2, "no\nfinish", ha="center", va="bottom", fontsize=8, color="#333333")

    for ax in axes:
        style_axes(ax)

    fig.subplots_adjust(bottom=0.28, wspace=0.22)
    save_figure(fig, "rl_reward_results_summary")
    plt.close(fig)


def plot_episode_outcomes(summary: pd.DataFrame) -> None:
    rows = []
    for algorithm in ALGORITHM_ORDER:
        group = summary[summary["algorithm"].astype(str).str.upper() == algorithm]
        if group.empty:
            continue
        rows.append(
            {
                "algorithm": algorithm,
                "finished": float(group["finish_count"].astype(float).sum()),
                "crashes": float(group["crash_count"].astype(float).sum()),
                "timeouts": float(group["timeout_count"].astype(float).sum()),
            }
        )
    aggregated = pd.DataFrame(rows)
    labels = aggregated["algorithm"].tolist()
    x = np.arange(len(aggregated))
    finished = aggregated["finished"].to_numpy()
    crashes = aggregated["crashes"].to_numpy()
    timeouts = aggregated["timeouts"].to_numpy()

    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    ax.bar(x, finished, label="finish", color="#2ca25f", edgecolor="#303642", linewidth=0.6)
    ax.bar(x, crashes, bottom=finished, label="crash", color="#de5b49", edgecolor="#303642", linewidth=0.6)
    ax.bar(x, timeouts, bottom=finished + crashes, label="timeout", color="#8da0cb", edgecolor="#303642", linewidth=0.6)
    ax.set_ylabel("Episode count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_title("Episode outcomes")
    style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.28)
    save_figure(fig, "rl_reward_episode_outcomes")
    plt.close(fig)


def main() -> None:
    combined, summary = load_data()
    plot_progress(combined)
    plot_results_summary(summary)
    plot_episode_outcomes(summary)
    print(f"Saved thesis RL figures to {THESIS_IMAGE_DIR}")


if __name__ == "__main__":
    main()
