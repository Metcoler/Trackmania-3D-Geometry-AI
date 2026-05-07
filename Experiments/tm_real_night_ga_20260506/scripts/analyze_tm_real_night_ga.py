from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_DIR = Path(
    "logs/tm_finetune_runs/"
    "20260506_004011_tm_seed_map_single_surface_flat_v2d_asphalt_h48x24_p48_src_best_model"
)
PACKAGE_DIR = Path("Diplomová práca/Experiments/tm_real_night_ga_20260506")
ANALYSIS_DIR = PACKAGE_DIR / "analysis" / "tm_real_night_ga_20260506"
LATEX_IMAGE_DIR = Path("Diplomová práca/Latex/images/training_policy")


def _rolling_mean(values: pd.Series, window: int = 5) -> pd.Series:
    return values.rolling(window=window, min_periods=1).mean()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    generation = pd.read_csv(RUN_DIR / "generation_summary.csv")
    individuals = pd.read_csv(RUN_DIR / "individual_metrics.csv")
    with open(RUN_DIR / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return generation, individuals, config


def summarize(generation: pd.DataFrame, individuals: pd.DataFrame, config: dict) -> dict:
    final_generation = int(generation["generation"].max())
    checkpoint_generations = []
    for path in (RUN_DIR / "checkpoints").glob("population_gen_*.npz"):
        try:
            checkpoint_generations.append(int(path.stem.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    latest_checkpoint_generation = max(checkpoint_generations) if checkpoint_generations else None
    first_finish_rows = generation[generation["finish_count"] > 0]
    first_finish_generation = (
        int(first_finish_rows["generation"].iloc[0]) if not first_finish_rows.empty else None
    )
    best_finish_rows = generation[generation["best_finished"] > 0]
    best_finish_time = (
        float(best_finish_rows["time_best_global"].min()) if not best_finish_rows.empty else None
    )
    best_finish_generation = (
        int(best_finish_rows.loc[best_finish_rows["time_best_global"].idxmin(), "generation"])
        if not best_finish_rows.empty
        else None
    )
    final_population = individuals[individuals["generation"] == final_generation].copy()
    checkpoint_population = (
        individuals[individuals["generation"] == latest_checkpoint_generation].copy()
        if latest_checkpoint_generation is not None
        else pd.DataFrame()
    )
    final_finished = int(final_population["finished"].sum())
    final_timeouts = int(final_population["timeout"].sum())
    final_crashes = int(final_population["crashes"].sum())
    final_best_time = (
        float(final_population.loc[final_population["finished"] > 0, "time"].min())
        if final_finished > 0
        else None
    )
    checkpoint_finished = int(checkpoint_population["finished"].sum()) if not checkpoint_population.empty else None
    checkpoint_best_time = (
        float(checkpoint_population.loc[checkpoint_population["finished"] > 0, "time"].min())
        if checkpoint_finished
        else None
    )

    return {
        "source_run_dir": str(RUN_DIR),
        "map_name": config.get("map_name"),
        "population_size": int(config.get("pop_size", final_population.shape[0])),
        "generations_logged": int(len(generation)),
        "final_generation": final_generation,
        "latest_checkpoint_generation": latest_checkpoint_generation,
        "first_finish_generation": first_finish_generation,
        "total_finish_individuals": int(generation["finish_count"].sum()),
        "max_finish_count_per_generation": int(generation["finish_count"].max()),
        "best_finish_time": best_finish_time,
        "best_finish_generation": best_finish_generation,
        "final_finish_count": final_finished,
        "final_timeout_count": final_timeouts,
        "final_crash_count": final_crashes,
        "final_best_time": final_best_time,
        "final_mean_progress": float(final_population["progress"].mean()),
        "final_best_progress": float(final_population["progress"].max()),
        "checkpoint_finish_count": checkpoint_finished,
        "checkpoint_best_time": checkpoint_best_time,
        "checkpoint_mean_progress": (
            float(checkpoint_population["progress"].mean()) if not checkpoint_population.empty else None
        ),
        "ranking_key": config.get("ranking_key"),
        "seed_model_path": config.get("seed_model_path"),
        "env_max_time": float(config.get("env_max_time", np.nan)),
        "mutation_prob_start": float(config.get("mutation_prob", np.nan)),
        "mutation_sigma_start": float(config.get("mutation_sigma", np.nan)),
        "mutation_prob_final": float(generation["current_mutation_prob"].iloc[-1]),
        "mutation_sigma_final": float(generation["current_mutation_sigma"].iloc[-1]),
    }


def plot_training_overview(generation: pd.DataFrame, summary: dict, output_path: Path) -> None:
    g = generation.sort_values("generation").copy()
    x = g["generation"].to_numpy()
    progress_mean = g["mean_progress"].to_numpy(dtype=float)
    progress_std = g["std_progress"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.0), sharex=True)
    ax_progress, ax_outcomes, ax_time, ax_mutation = axes.flatten()

    ax_progress.fill_between(
        x,
        np.clip(progress_mean - progress_std, 0, 100),
        np.clip(progress_mean + progress_std, 0, 100),
        color="#91a7ff",
        alpha=0.22,
        label="Priemer +/- std populacie",
    )
    ax_progress.plot(x, g["mean_progress"], color="#364fc7", linewidth=2.0, label="Priemerny progress")
    ax_progress.plot(x, g["best_progress"], color="#0b7285", linewidth=1.8, label="Najlepsi progress v generacii")
    ax_progress.plot(
        x,
        g["best_progress"].cummax(),
        color="#087f5b",
        linewidth=2.2,
        linestyle="--",
        label="Najlepsi progress doteraz",
    )
    ax_progress.set_ylabel("Progress [%]")
    ax_progress.set_ylim(0, 103)
    ax_progress.grid(True, alpha=0.25)
    ax_progress.legend(fontsize=8, loc="lower right")

    ax_outcomes.plot(x, g["finish_count"], color="#2b8a3e", linewidth=2.2, label="Finish")
    ax_outcomes.plot(x, g["timeout_count"], color="#f08c00", linewidth=1.8, label="Timeout")
    ax_outcomes.plot(x, g["crash_count"], color="#c92a2a", linewidth=1.8, label="Crash")
    ax_outcomes.set_ylabel("Pocet jedincov")
    ax_outcomes.set_ylim(0, max(48, float(g[["finish_count", "timeout_count", "crash_count"]].max().max())) + 2)
    ax_outcomes.grid(True, alpha=0.25)
    ax_outcomes.legend(fontsize=8, loc="upper left")

    finish_times = g["best_time"].where(g["finish_count"] > 0)
    global_finish = g["time_best_global"].where(g["time_best_global"] < summary["env_max_time"])
    ax_time.scatter(x, finish_times, color="#1c7ed6", s=18, alpha=0.5, label="Najlepsi finish v generacii")
    ax_time.plot(x, global_finish, color="#1864ab", linewidth=2.2, label="Najlepsi finish doteraz")
    ax_time.axhline(summary["env_max_time"], color="#495057", linestyle=":", linewidth=1.1, label="Max cas epizody")
    y_min_candidates = [float(v) for v in global_finish.dropna().to_list() + finish_times.dropna().to_list()]
    y_min = max(0.0, min(y_min_candidates) - 1.0) if y_min_candidates else 0.0
    ax_time.set_ylim(y_min, summary["env_max_time"] + 0.8)
    ax_time.set_ylabel("Cas [s]")
    ax_time.grid(True, alpha=0.25)
    ax_time.legend(fontsize=8, loc="upper right")

    ax_mutation.plot(x, g["current_mutation_prob"], color="#5f3dc4", linewidth=2.0, label="mutation_prob")
    ax_mutation_b = ax_mutation.twinx()
    ax_mutation_b.plot(x, g["current_mutation_sigma"], color="#d6336c", linewidth=2.0, label="mutation_sigma")
    ax_mutation.set_ylabel("mutation_prob")
    ax_mutation_b.set_ylabel("mutation_sigma")
    ax_mutation.set_xlabel("Generacia")
    ax_mutation.grid(True, alpha=0.25)
    lines_a, labels_a = ax_mutation.get_legend_handles_labels()
    lines_b, labels_b = ax_mutation_b.get_legend_handles_labels()
    ax_mutation.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8, loc="upper right")

    ax_progress.set_title("Progress populacie")
    ax_outcomes.set_title("Vysledky evaluacii")
    ax_time.set_title("Finish time")
    ax_mutation.set_title("Pokles mutacie")
    for ax in axes[-1, :]:
        ax.set_xlabel("Generacia")
    fig.suptitle("Live Trackmania GA nocny run: single_surface_flat", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_final_population(individuals: pd.DataFrame, summary: dict, output_path: Path) -> None:
    final = individuals[individuals["generation"] == summary["final_generation"]].copy()
    final = final.sort_values(["finished", "progress", "time"], ascending=[False, False, True])
    ranks = np.arange(1, len(final) + 1)

    colors = np.where(
        final["finished"].to_numpy(dtype=int) > 0,
        "#2b8a3e",
        np.where(final["timeout"].to_numpy(dtype=int) > 0, "#f08c00", "#c92a2a"),
    )

    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.bar(ranks, final["progress"], color=colors, alpha=0.82)
    ax.set_ylim(0, 103)
    ax.set_xlabel("Poradie jedinca vo finalnej populacii")
    ax.set_ylabel("Progress [%]")
    ax.set_title("Finalna populacia live GA runu")
    ax.grid(True, axis="y", alpha=0.25)

    from matplotlib.patches import Patch

    legend = [
        Patch(facecolor="#2b8a3e", label="Finish"),
        Patch(facecolor="#f08c00", label="Timeout"),
        Patch(facecolor="#c92a2a", label="Crash"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def write_report(summary: dict, output_path: Path) -> None:
    lines = [
        "# Live Trackmania GA Night Run 20260506",
        "",
        "## Summary",
        f"- Source run: `{summary['source_run_dir']}`",
        f"- Map: `{summary['map_name']}`",
        f"- Ranking key: `{summary['ranking_key']}`",
        f"- Population size: `{summary['population_size']}`",
        f"- Logged generations: `{summary['generations_logged']}` (final generation `{summary['final_generation']}`)",
        f"- Latest replayable population checkpoint: generation `{summary['latest_checkpoint_generation']}`",
        f"- First finish generation: `{summary['first_finish_generation']}`",
        f"- Best finish time: `{summary['best_finish_time']:.2f} s` in generation `{summary['best_finish_generation']}`",
        f"- Total finishing individuals across logged generations: `{summary['total_finish_individuals']}`",
        f"- Max finishers in one generation: `{summary['max_finish_count_per_generation']}`",
        "",
        "## Final Population",
        f"- Finishers: `{summary['final_finish_count']}` / `{summary['population_size']}`",
        f"- Timeouts: `{summary['final_timeout_count']}` / `{summary['population_size']}`",
        f"- Crashes: `{summary['final_crash_count']}` / `{summary['population_size']}`",
        f"- Final best time: `{summary['final_best_time']:.2f} s`",
        f"- Final mean progress: `{summary['final_mean_progress']:.2f} %`",
        f"- Final best progress: `{summary['final_best_progress']:.2f} %`",
        "",
        "## Latest Replayable Checkpoint",
        f"- Checkpoint generation: `{summary['latest_checkpoint_generation']}`",
        f"- Finishers: `{summary['checkpoint_finish_count']}` / `{summary['population_size']}`",
        f"- Best time: `{summary['checkpoint_best_time']:.2f} s`",
        f"- Mean progress: `{summary['checkpoint_mean_progress']:.2f} %`",
        "",
        "## Interpretation",
        "- This is a positive live-Trackmania sanity result: the GA run did not only improve progress, it produced real finishers on `single_surface_flat`.",
        "- The run is still noisy: most final-population individuals crash, while a smaller elite group finishes or reaches high progress.",
        "- The best time is slower than the clean 2D sandbox results, which is expected because this run evaluates policies through the live game loop.",
        "- The result supports continuing with replay/evaluation of the final population, but it should not yet be presented as a final robust agent.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    generation, individuals, config = load_inputs()
    summary = summarize(generation, individuals, config)

    summary_path = ANALYSIS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame([summary]).to_csv(ANALYSIS_DIR / "summary.csv", index=False)

    overview_path = ANALYSIS_DIR / "tm_real_night_ga_training_overview.png"
    final_population_path = ANALYSIS_DIR / "tm_real_night_ga_final_population.png"
    plot_training_overview(generation, summary, overview_path)
    plot_final_population(individuals, summary, final_population_path)
    write_report(summary, ANALYSIS_DIR / "REPORT.md")

    for path in (overview_path, final_population_path):
        shutil.copy2(path, LATEX_IMAGE_DIR / path.name)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote analysis to: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
