from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
THESIS_IMAGES = REPO_ROOT / "Masters thesis" / "Latex" / "images" / "training_policy"

ANALYSIS_ID = "lex_sweep_aabb_lidar_fixed100_20260503"


VARIANTS = [
    {
        "variant": "finished_progress",
        "ranking_key": "(finished, progress)",
        "label": "(finish, progress)",
        "color": "#4c6ef5",
    },
    {
        "variant": "finished_progress_time",
        "ranking_key": "(finished, progress, -time)",
        "label": "(finish, progress, -time)",
        "color": "#12b886",
    },
    {
        "variant": "finished_progress_time_crashes",
        "ranking_key": "(finished, progress, -time, -crashes)",
        "label": "(finish, progress, -time, -crashes)",
        "color": "#f08c00",
    },
    {
        "variant": "finished_progress_crashes_time",
        "ranking_key": "(finished, progress, -crashes, -time)",
        "label": "(finish, progress, -crashes, -time)",
        "color": "#e03131",
    },
]


TOP_DETAIL_VARIANTS = [
    {
        "variant": "finished_progress_time_crashes",
        "ranking_key": "(finished, progress, -time, -crashes)",
        "label": "(finish, progress, -time, -crashes)",
        "short": "lex_reward_top1",
        "color": "#f08c00",
    },
    {
        "variant": "finished_progress_crashes_time",
        "ranking_key": "(finished, progress, -crashes, -time)",
        "label": "(finish, progress, -crashes, -time)",
        "short": "lex_reward_top2",
        "color": "#e03131",
    },
]


OUTPUT_NAMES = [
    "lex_reward_progress_best_mean",
    "lex_reward_finish_count",
    "lex_reward_best_time",
    "lex_reward_top1_progress_detail",
    "lex_reward_top1_time_detail",
    "lex_reward_top2_progress_detail",
    "lex_reward_top2_time_detail",
]


def find_run_dir(variant: str) -> Path:
    root = PACKAGE_ROOT / "runs" / "lex_sweep_aabb_lidar_fixed100_seed_2026050306" / variant
    candidates = sorted(
        path.parent
        for path in root.rglob("config.json")
        if (path.parent / "generation_metrics.csv").exists()
        and (path.parent / "individual_metrics.csv").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"Missing complete run for {variant!r} under {root}")
    return candidates[0]


def load_variant(info: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    run_dir = find_run_dir(info["variant"])
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    generation = pd.read_csv(run_dir / "generation_metrics.csv")
    individual = pd.read_csv(run_dir / "individual_metrics.csv")
    generation["variant"] = info["variant"]
    individual["variant"] = info["variant"]
    return config, generation, individual


def as_num(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float64)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(output_dir / f"{name}{suffix}", bbox_inches="tight", pad_inches=0.04, dpi=220)
    plt.close(fig)


def legend_below(ax: plt.Axes, *, ncol: int = 2, y: float = -0.24) -> None:
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=True,
        borderaxespad=0.0,
        columnspacing=1.4,
        handlelength=2.6,
    )


def copy_to_latex(output_dir: Path) -> None:
    THESIS_IMAGES.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        for suffix in (".pdf", ".png"):
            src = output_dir / f"{name}{suffix}"
            if src.exists():
                shutil.copy2(src, THESIS_IMAGES / src.name)


def common_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 13.0,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_global_progress(data: dict[str, tuple[dict, pd.DataFrame, pd.DataFrame]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.9))
    for info in VARIANTS:
        _, generation, _ = data[info["variant"]]
        gen = generation.sort_values("generation")
        x = gen["generation"].to_numpy()
        ax.plot(
            x,
            gen["best_dense_progress"],
            color=info["color"],
            linewidth=2.1,
            label=f"{info['label']} - best",
        )
        ax.plot(
            x,
            gen["mean_dense_progress"],
            color=info["color"],
            linewidth=1.5,
            linestyle="--",
            alpha=0.62,
            label=f"{info['label']} - population mean",
        )
    ax.set_title("Population progress for lexicographic ranking variants")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Progress [%]")
    ax.set_ylim(-2, 103)
    legend_below(ax, ncol=2, y=-0.20)
    save_figure(fig, output_dir, "lex_reward_progress_best_mean")


def plot_finish_count(data: dict[str, tuple[dict, pd.DataFrame, pd.DataFrame]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    window = 15
    for info in VARIANTS:
        _, generation, _ = data[info["variant"]]
        gen = generation.sort_values("generation").copy()
        x = gen["generation"].to_numpy()
        finish = gen["finish_count"].rolling(window=window, min_periods=1).mean()
        ax.plot(x, finish, color=info["color"], linewidth=2.0, label=info["label"])
    ax.set_title(f"Finished runs per generation, {window}-generation moving average")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Number of finishing individuals")
    ax.set_ylim(bottom=0)
    legend_below(ax, ncol=2, y=-0.20)
    save_figure(fig, output_dir, "lex_reward_finish_count")


def plot_best_time(data: dict[str, tuple[dict, pd.DataFrame, pd.DataFrame]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    for info in VARIANTS:
        _, generation, _ = data[info["variant"]]
        gen = generation.sort_values("generation").copy()
        best_time = pd.to_numeric(gen["best_time"], errors="coerce")
        best_time = best_time.where(gen["finish_count"] > 0)
        best_so_far = best_time.cummin()
        ax.plot(
            gen["generation"],
            best_so_far,
            color=info["color"],
            linewidth=2.1,
            label=info["label"],
        )
    ax.set_title("Best finished time found during training")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Time [s]")
    ax.set_ylim(16.5, 31.0)
    legend_below(ax, ncol=2, y=-0.20)
    save_figure(fig, output_dir, "lex_reward_best_time")


def group_stats(individual: pd.DataFrame, max_time: float) -> pd.DataFrame:
    df = individual.copy()
    df["dense_progress"] = as_num(df, "dense_progress")
    df["time"] = as_num(df, "time", default=max_time)
    df["finished_bool"] = as_num(df, "finished") > 0
    df["penalized_time"] = np.where(df["finished_bool"], df["time"], float(max_time))
    df["is_parent_bool"] = as_num(df, "is_parent") > 0
    df["is_elite_bool"] = as_num(df, "is_elite") > 0

    rows: list[dict] = []
    for generation, group in df.groupby("generation", sort=True):
        parents = group[group["is_parent_bool"]]
        elites = group[group["is_elite_bool"]]
        finished = group[group["finished_bool"]]
        progress = group["dense_progress"].to_numpy(dtype=np.float64)
        ptime = group["penalized_time"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "generation": int(generation),
                "progress_p10": float(np.percentile(progress, 10)),
                "progress_p25": float(np.percentile(progress, 25)),
                "progress_p75": float(np.percentile(progress, 75)),
                "progress_p90": float(np.percentile(progress, 90)),
                "progress_mean_all": float(group["dense_progress"].mean()),
                "progress_mean_parent": float(parents["dense_progress"].mean()),
                "progress_mean_elite": float(elites["dense_progress"].mean()),
                "progress_best": float(group["dense_progress"].max()),
                "finish_count": int(len(finished)),
                "best_finished_time": float(finished["time"].min()) if not finished.empty else np.nan,
                "penalized_p25": float(np.percentile(ptime, 25)),
                "penalized_p75": float(np.percentile(ptime, 75)),
                "penalized_mean_all": float(group["penalized_time"].mean()),
                "penalized_mean_parent": float(parents["penalized_time"].mean()),
                "penalized_mean_elite": float(elites["penalized_time"].mean()),
            }
        )
    stats = pd.DataFrame(rows).sort_values("generation")
    stats["best_finished_time_so_far"] = stats["best_finished_time"].cummin()
    return stats


def plot_detail_progress(info: dict, stats: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.7))
    x = stats["generation"].to_numpy()
    ax.fill_between(
        x,
        stats["progress_p10"],
        stats["progress_p90"],
        color="#8ecae6",
        alpha=0.16,
        label="p10-p90 of the population",
    )
    ax.fill_between(
        x,
        stats["progress_p25"],
        stats["progress_p75"],
        color="#219ebc",
        alpha=0.22,
        label="p25-p75 of the population",
    )
    ax.plot(x, stats["progress_best"], color="#c92a2a", linewidth=2.1, label="best individual")
    ax.plot(
        x,
        stats["progress_mean_all"],
        color="#868e96",
        linewidth=1.3,
        linestyle="--",
        label="population mean",
    )
    ax.plot(
        x,
        stats["progress_mean_parent"],
        color="#0b7285",
        linewidth=2.0,
        label="parent mean",
    )
    ax.plot(
        x,
        stats["progress_mean_elite"],
        color=info["color"],
        linewidth=2.0,
        label="elite mean",
    )
    ax.set_title(f"Progress detail: {info['label']}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Progress [%]")
    ax.set_ylim(-2, 103)
    legend_below(ax, ncol=3, y=-0.20)
    save_figure(fig, output_dir, f"{info['short']}_progress_detail")


def plot_detail_time(info: dict, stats: pd.DataFrame, output_dir: Path, max_time: float) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.7))
    x = stats["generation"].to_numpy()
    ax.fill_between(
        x,
        stats["penalized_p25"],
        stats["penalized_p75"],
        color="#ffd43b",
        alpha=0.20,
        label="p25-p75 of population penalized time",
    )
    ax.plot(
        x,
        stats["penalized_mean_all"],
        color="#868e96",
        linestyle="--",
        linewidth=1.35,
        label="population mean",
    )
    ax.plot(
        x,
        stats["penalized_mean_parent"],
        color="#e67700",
        linewidth=2.0,
        label="parent mean",
    )
    ax.plot(
        x,
        stats["penalized_mean_elite"],
        color="#9c36b5",
        linewidth=2.0,
        label="elite mean",
    )
    ax.scatter(
        x,
        stats["best_finished_time"],
        color="#1c7ed6",
        s=13,
        alpha=0.45,
        label="best finished time in generation",
    )
    ax.plot(
        x,
        stats["best_finished_time_so_far"],
        color="#1864ab",
        linewidth=2.25,
        label="best finished time so far",
    )
    ax.axhline(float(max_time), color="#495057", linestyle=":", linewidth=1.1, label=f"max. episode time {max_time:.0f} s")
    finite_values = stats[
        [
            "penalized_p25",
            "penalized_p75",
            "penalized_mean_all",
            "penalized_mean_parent",
            "penalized_mean_elite",
            "best_finished_time",
            "best_finished_time_so_far",
        ]
    ].to_numpy(dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size:
        ax.set_ylim(max(0.0, float(np.min(finite_values)) - 1.0), max_time + 0.7)
    ax.set_title(f"Population time: {info['label']}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Time [s]")
    legend_below(ax, ncol=2, y=-0.22)
    save_figure(fig, output_dir, f"{info['short']}_time_detail")


def write_summary(
    output_dir: Path,
    data: dict[str, tuple[dict, pd.DataFrame, pd.DataFrame]],
    detail_stats: dict[str, pd.DataFrame],
) -> None:
    rows: list[dict] = []
    for info in VARIANTS:
        config, generation, individual = data[info["variant"]]
        finished = individual[as_num(individual, "finished") > 0]
        first_finish = generation.loc[generation["finish_count"] > 0, "generation"]
        last50 = generation[generation["generation"] >= int(generation["generation"].max()) - 49]
        rows.append(
            {
                "variant": info["variant"],
                "ranking_key": info["ranking_key"],
                "max_time": float(config.get("max_time", 30.0)),
                "population_size": int(config.get("population_size", 0)),
                "elite_count": int(config.get("elite_count", 0)),
                "parent_count": int(config.get("parent_count", 0)),
                "mutation_prob": float(config.get("mutation_prob", np.nan)),
                "mutation_sigma": float(config.get("mutation_sigma", np.nan)),
                "first_finish_generation": int(first_finish.iloc[0]) if not first_finish.empty else np.nan,
                "total_finish_individuals": int(len(finished)),
                "best_finish_time": float(finished["time"].min()) if not finished.empty else np.nan,
                "last50_mean_finish_count": float(last50["finish_count"].mean()),
                "last50_mean_dense_progress": float(last50["mean_dense_progress"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "thesis_lex_reward_summary.csv", index=False)
    for variant, stats in detail_stats.items():
        stats.to_csv(output_dir / f"{variant}_parent_elite_stats.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Slovak thesis figures for the lexicographic reward sweep.")
    parser.add_argument(
        "--output-dir",
        default=str(PACKAGE_ROOT / "analysis" / ANALYSIS_ID / "thesis_figures"),
    )
    parser.add_argument("--copy-to-latex", action="store_true")
    args = parser.parse_args()

    common_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, tuple[dict, pd.DataFrame, pd.DataFrame]] = {}
    for info in VARIANTS:
        data[info["variant"]] = load_variant(info)

    plot_global_progress(data, output_dir)
    plot_finish_count(data, output_dir)
    plot_best_time(data, output_dir)

    detail_stats: dict[str, pd.DataFrame] = {}
    for info in TOP_DETAIL_VARIANTS:
        config, _, individual = data[info["variant"]]
        max_time = float(config.get("max_time", 30.0))
        stats = group_stats(individual, max_time=max_time)
        detail_stats[info["variant"]] = stats
        plot_detail_progress(info, stats, output_dir)
        plot_detail_time(info, stats, output_dir, max_time=max_time)

    write_summary(output_dir, data, detail_stats)
    if args.copy_to_latex:
        copy_to_latex(output_dir)

    summary_path = output_dir / "thesis_lex_reward_summary.csv"
    print(f"Wrote thesis lex reward figures to {output_dir}")
    print(pd.read_csv(summary_path).to_string(index=False))


if __name__ == "__main__":
    main()
