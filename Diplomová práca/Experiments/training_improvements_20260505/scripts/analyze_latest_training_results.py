from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training_focus_plots import (
    FocusRun,
    canonicalize_generation_metrics,
    canonicalize_individual_metrics,
    plot_focus_progress,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "Experiments" / "analysis" / "latest_training_results_20260505"
TRAINING_ROOT = ROOT / "Experiments" / "runs_ga_training_improvements" / "seed_2026050407"
MOO_ROOT = ROOT / "Experiments" / "runs_ga_moo"
GENERALIZATION_ROOT = ROOT / "Experiments" / "runs_ga_generalization"
LIVE_TM_RUN = (
    ROOT
    / "logs"
    / "ga_runs"
    / "20260505_012752_map_AI_Training__5_v2d_asphalt_h32x16_p32_ranking_finished_progress_neg_time_neg_crashes"
)


@dataclass
class RunData:
    label: str
    family: str
    run_dir: Path
    config: dict
    generation: pd.DataFrame
    individual: pd.DataFrame
    generalization: pd.DataFrame


def safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def first_existing_csv(run_dir: Path) -> pd.DataFrame:
    generation = safe_read_csv(run_dir / "generation_metrics.csv")
    if not generation.empty:
        return generation
    return safe_read_csv(run_dir / "generation_summary.csv")


def infer_training_label(run_dir: Path) -> str:
    parts = list(run_dir.parts)
    if "seed_2026050407" in parts:
        index = parts.index("seed_2026050407")
        if index + 1 < len(parts):
            return parts[index + 1]
    return run_dir.parent.name


def infer_moo_label(run_dir: Path, config: dict) -> str:
    if "trackmania_racing" in str(run_dir):
        return "moo_trackmania_racing"
    names = config.get("objective_subset") or config.get("objective_names") or []
    if isinstance(names, list) and names:
        return "moo_" + "_".join(str(name) for name in names)
    return "moo_" + run_dir.parent.name


def infer_generalization_label(run_dir: Path) -> str:
    return "both_mirrors_holdout_single_surface_flat"


def load_run(run_dir: Path, family: str, label: str) -> RunData | None:
    config = safe_read_json(run_dir / "config.json")
    generation = first_existing_csv(run_dir)
    if generation.empty:
        return None
    individual = safe_read_csv(run_dir / "individual_metrics.csv")
    generalization = safe_read_csv(run_dir / "generalization_metrics.csv")
    generation = canonicalize_generation_metrics(generation)
    generation["run_label"] = label
    generation["family"] = family
    generation["run_dir"] = str(run_dir)
    if not individual.empty:
        individual = canonicalize_individual_metrics(individual)
        individual["run_label"] = label
        individual["family"] = family
        individual["run_dir"] = str(run_dir)
    if not generalization.empty:
        generalization = generalization.copy()
        generalization["run_label"] = label
        generalization["family"] = family
        generalization["run_dir"] = str(run_dir)
    return RunData(label, family, run_dir, config, generation, individual, generalization)


def completed_score(run: RunData) -> tuple[int, int, float]:
    target = int(run.config.get("generations", run.config.get("generations_requested", 0)) or 0)
    max_generation = int(run.generation["generation"].max())
    complete = int(target <= 0 or max_generation >= target)
    mtime = (run.run_dir / "generation_metrics.csv").stat().st_mtime if (run.run_dir / "generation_metrics.csv").exists() else 0.0
    return complete, max_generation, mtime


def dedupe_runs(runs: list[RunData]) -> list[RunData]:
    by_key: dict[tuple[str, str], list[RunData]] = {}
    for run in runs:
        by_key.setdefault((run.family, run.label), []).append(run)
    selected: list[RunData] = []
    for candidates in by_key.values():
        selected.append(sorted(candidates, key=completed_score)[-1])
    return sorted(selected, key=lambda run: (run.family, run.label))


def discover_runs() -> list[RunData]:
    runs: list[RunData] = []

    if TRAINING_ROOT.exists():
        for config_path in sorted(TRAINING_ROOT.rglob("config.json")):
            run_dir = config_path.parent
            loaded = load_run(run_dir, "training_improvements", infer_training_label(run_dir))
            if loaded is not None:
                runs.append(loaded)

    if MOO_ROOT.exists():
        for config_path in sorted(MOO_ROOT.rglob("config.json")):
            run_dir = config_path.parent
            config = safe_read_json(config_path)
            loaded = load_run(run_dir, "moo", infer_moo_label(run_dir, config))
            if loaded is not None:
                runs.append(loaded)

    if GENERALIZATION_ROOT.exists():
        for config_path in sorted(GENERALIZATION_ROOT.rglob("config.json")):
            run_dir = config_path.parent
            loaded = load_run(run_dir, "generalization", infer_generalization_label(run_dir))
            if loaded is not None:
                runs.append(loaded)

    if LIVE_TM_RUN.exists():
        loaded = load_run(LIVE_TM_RUN, "live_tm", "live_tm_overnight_20260505")
        if loaded is not None:
            runs.append(loaded)

    return dedupe_runs(runs)


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def population_size(run: RunData) -> int:
    for key in ("population_size", "pop_size"):
        if key in run.config:
            return int(run.config[key])
    if "population_size" in run.generation:
        return int(run.generation["population_size"].dropna().iloc[0])
    return 1


def best_time_from_run(run: RunData) -> float:
    if not run.individual.empty and "finished" in run.individual and "time" in run.individual:
        finished = run.individual[run.individual["finished"].astype(float) > 0]
        if not finished.empty:
            return float(finished["time"].min())
    if "best_finished" in run.generation and "best_time" in run.generation:
        finished_generations = run.generation[run.generation["best_finished"].astype(float) > 0]
        if not finished_generations.empty:
            return float(finished_generations["best_time"].min())
    return math.nan


def summarize_run(run: RunData) -> dict:
    generation = run.generation.sort_values("generation")
    pop = population_size(run)
    finish_col = "finish_count" if "finish_count" in generation else None
    crash_col = "crash_count" if "crash_count" in generation else None
    timeout_col = "timeout_count" if "timeout_count" in generation else None

    first_finish_generation = math.nan
    if finish_col:
        finish_rows = generation[generation[finish_col].astype(float) > 0]
        if not finish_rows.empty:
            first_finish_generation = int(finish_rows["generation"].iloc[0])

    if not run.individual.empty and "finished" in run.individual:
        total_finish = int((run.individual["finished"].astype(float) > 0).sum())
    elif finish_col:
        total_finish = int(generation[finish_col].sum())
    else:
        total_finish = 0

    last50 = generation.tail(min(50, len(generation)))

    def rate(col: str | None) -> float:
        if not col or pop <= 0 or last50.empty:
            return math.nan
        return float(last50[col].astype(float).mean() / pop)

    mean_progress_col = "mean_progress" if "mean_progress" in generation else "progress_mean"
    last50_progress = float(last50[mean_progress_col].mean()) if mean_progress_col in last50 else math.nan

    return {
        "experiment": run.label,
        "family": run.family,
        "run_dir": str(run.run_dir),
        "generations": int(generation["generation"].max()),
        "population_size": pop,
        "first_finish_generation": first_finish_generation,
        "total_finish": total_finish,
        "best_time": best_time_from_run(run),
        "last50_finish_rate": rate(finish_col),
        "last50_mean_progress": last50_progress,
        "last50_crash_rate": rate(crash_col),
        "last50_timeout_rate": rate(timeout_col),
        "cached_evaluations": int(generation["cached_evaluations"].sum()) if "cached_evaluations" in generation else 0,
    }


def line_plot(runs: list[RunData], columns: list[tuple[str, str]], output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(len(columns), 1, figsize=(12, 4 * len(columns)), sharex=True)
    if len(columns) == 1:
        axes = [axes]
    for axis, (column, ylabel) in zip(axes, columns):
        for run in runs:
            if column not in run.generation:
                continue
            df = run.generation.sort_values("generation").copy()
            y = df[column].astype(float)
            if column == "best_time" and "best_finished" in df:
                y = y.mask(df["best_finished"].astype(float) <= 0)
            axis.plot(df["generation"], y, label=run.label, linewidth=1.8)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Generation")
    axes[0].set_title(title)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def focus_plot_for_runs(runs: list[RunData], output_path: Path, title: str, *, density: str = "auto") -> None:
    focus_runs = [
        FocusRun(
            run_dir=run.run_dir,
            label=run.label,
            generation=run.generation,
            individual=run.individual,
        )
        for run in runs
    ]
    plot_focus_progress(focus_runs, output_path, title=title, density=density)


def count_plot(runs: list[RunData], output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    columns = [("finish_count", "Finish count"), ("crash_count", "Crash count"), ("timeout_count", "Timeout count")]
    for axis, (column, ylabel) in zip(axes, columns):
        for run in runs:
            if column not in run.generation:
                continue
            df = run.generation.sort_values("generation")
            axis.plot(df["generation"], df[column], label=run.label, linewidth=1.6)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Generation")
    axes[0].set_title(title)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_mutation_and_cache(runs: list[RunData], output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    columns = [
        ("current_mutation_prob", "Mutation prob"),
        ("current_mutation_sigma", "Mutation sigma"),
        ("cached_evaluations", "Cached evaluations"),
    ]
    for axis, (column, ylabel) in zip(axes, columns):
        for run in runs:
            if column not in run.generation:
                continue
            df = run.generation.sort_values("generation")
            values = df[column].astype(float)
            if column == "cached_evaluations" and values.sum() <= 0:
                continue
            axis.plot(df["generation"], values, label=run.label, linewidth=1.8)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Generation")
    axes[0].set_title(title)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_summary_bars(summary: pd.DataFrame, output_path: Path) -> None:
    display = summary.copy()
    display["finish_rate_pct"] = display["last50_finish_rate"] * 100.0
    display["crash_rate_pct"] = display["last50_crash_rate"] * 100.0
    labels = display["experiment"].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    axes[0].bar(x, display["finish_rate_pct"].fillna(0.0), color="#2f8f5b")
    axes[0].set_ylabel("Last50 finish rate [%]")
    axes[1].bar(x, display["last50_mean_progress"].fillna(0.0), color="#3777b7")
    axes[1].set_ylabel("Last50 mean progress")
    axes[2].bar(x, display["best_time"].fillna(0.0), color="#c9822b")
    axes[2].set_ylabel("Best finish time [s]")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_title("Latest experiment summary")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_generalization(run: RunData, output_dir: Path) -> None:
    if run.generalization.empty:
        return
    gen = run.generalization.sort_values("generation")
    train = run.generation.sort_values("generation")
    train_progress_col = "best_progress" if "best_progress" in train else "best_dense_progress"
    test_progress_col = "test_progress" if "test_progress" in gen else "test_dense_progress"
    merged = pd.merge(
        train[["generation", train_progress_col, "finish_count"]].rename(columns={train_progress_col: "train_progress"}),
        gen[["generation", test_progress_col, "test_finished", "test_crashes", "test_time"]].rename(
            columns={test_progress_col: "test_progress"}
        ),
        on="generation",
        how="inner",
    )
    if merged.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(merged["generation"], merged["train_progress"], label="Train best progress", linewidth=1.9)
    axes[0].plot(merged["generation"], merged["test_progress"], label="Holdout progress", linewidth=1.9)
    axes[0].set_ylabel("Progress")
    axes[0].legend()
    axes[1].plot(merged["generation"], merged["finish_count"], label="Train finish count")
    axes[1].plot(merged["generation"], merged["test_finished"], label="Holdout finished top-1")
    axes[1].set_ylabel("Finish signal")
    axes[1].legend()
    axes[2].plot(merged["generation"], merged["test_crashes"], label="Holdout crashes")
    axes[2].plot(merged["generation"], merged["test_time"], label="Holdout time")
    axes[2].set_ylabel("Holdout crash/time")
    axes[2].legend()
    for axis in axes:
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Generation")
    fig.suptitle("Both-mirror holdout generalization: train vs single_surface_flat")
    fig.tight_layout()
    fig.savefig(output_dir / "generalization_holdout_train_vs_test.png", dpi=170)
    plt.close(fig)


def run_trajectory_analysis(output_dir: Path) -> None:
    manifest = LIVE_TM_RUN / "trajectories" / "trajectory_manifest.csv"
    if not manifest.exists():
        return
    trajectory_output = output_dir / "live_tm_trajectories"
    command = [
        sys.executable,
        str(ROOT / "Experiments" / "analyze_trajectories.py"),
        "--run-dir",
        str(LIVE_TM_RUN),
        "--output-dir",
        str(trajectory_output),
    ]
    subprocess.run(command, check=False)


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    def row(name: str) -> pd.Series | None:
        match = summary[summary["experiment"] == name]
        return match.iloc[0] if not match.empty else None

    exp05a = row("exp05a_variable_tick_no_cache_control")
    exp05b = row("exp05b_variable_tick_elite_cache")
    exp06 = row("exp06_supervised_seeded_fixed100")
    decay = row("exp01b_mutation_decay_first_finish_fixed100")
    max_touches = row("exp04_max_touches_3_fixed100")
    moo_racing = row("moo_trackmania_racing")
    gen = row("both_mirrors_holdout_single_surface_flat")

    lines = [
        "# Latest Training Results 20260505",
        "",
        "This package summarizes the latest runs received from the second PC.",
        "",
        "## Main takeaways",
        "",
    ]
    if exp05a is not None and exp05b is not None:
        lines.append(
            "- Variable physics tick with elite cache is the strongest positive result: "
            f"no-cache had {int(exp05a['total_finish'])} total finishes, cache had "
            f"{int(exp05b['total_finish'])} total finishes and best time {exp05b['best_time']:.2f}s."
        )
    if exp06 is not None:
        best_time_text = (
            f"{exp06['best_time']:.2f}s"
            if math.isfinite(float(exp06["best_time"]))
            else "no finish"
        )
        first_finish_text = (
            str(int(exp06["first_finish_generation"]))
            if math.isfinite(float(exp06["first_finish_generation"]))
            else "none"
        )
        lines.append(
            "- Supervised-seeded GA is a hybrid BC + GA fine-tuning experiment, not a pure GA baseline: "
            f"first finish generation {first_finish_text}, best time {best_time_text}, "
            f"last50 mean progress {exp06['last50_mean_progress']:.1f}."
        )
    if decay is not None:
        lines.append(
            "- First-finish mutation decay is useful as a tradeoff experiment: "
            f"best time {decay['best_time']:.2f}s, but last50 finish rate "
            f"{100.0 * decay['last50_finish_rate']:.1f}% is not better than baseline stability."
        )
    if max_touches is not None:
        lines.append(
            "- Max touches 3 is diagnostic/mid: progress is high, but the crash profile remains too noisy "
            f"({100.0 * max_touches['last50_crash_rate']:.1f}% last50 crash rate)."
        )
    if moo_racing is not None:
        lines.append(
            "- MOO trackmania_racing is promising but not the new default: "
            f"best time {moo_racing['best_time']:.2f}s, first finish generation "
            f"{int(moo_racing['first_finish_generation'])}, last50 finish rate "
            f"{100.0 * moo_racing['last50_finish_rate']:.1f}%."
        )
    if gen is not None:
        lines.append(
            "- Both-mirror holdout evaluation did not yet prove generalization: "
            f"train total finishes {int(gen['total_finish'])}, holdout top-1 finishes 0."
        )

    lines.extend(
        [
            "",
            "## Recommended thesis usage",
            "",
            "- Include elite-cache vs no-cache under variable physics as a thesis-grade positive result.",
            "- Include first-finish decay as an optimization idea with mixed results.",
            "- Keep max-touches and mirror holdout as diagnostic evidence, not as final improvements.",
            "- Mention MOO trackmania_racing as a promising revised MOO formulation, while lexicographic ranking remains the safer baseline.",
            "- Treat supervised-seeded GA separately from pure GA improvements because it uses extra human demonstrations.",
            "",
            "## Generated files",
            "",
        ]
    )
    for path in sorted(output_dir.glob("*.png")):
        lines.append(f"- `{path.name}`")
    if (output_dir / "live_tm_trajectories").exists():
        for path in sorted((output_dir / "live_tm_trajectories").glob("*.png")):
            lines.append(f"- `live_tm_trajectories/{path.name}`")

    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_csv(summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary[
        [
            "experiment",
            "family",
            "first_finish_generation",
            "total_finish",
            "best_time",
            "last50_finish_rate",
            "last50_mean_progress",
            "last50_crash_rate",
            "last50_timeout_rate",
            "cached_evaluations",
            "run_dir",
        ]
    ].copy()
    ordered.to_csv(output_dir / "summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze latest TM2D/live training results.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs()
    if not runs:
        raise SystemExit("No runs found.")

    summaries = pd.DataFrame([summarize_run(run) for run in runs])
    write_summary_csv(summaries, output_dir)

    training = [run for run in runs if run.family == "training_improvements"]
    moo = [run for run in runs if run.family == "moo"]
    generalization = [run for run in runs if run.family == "generalization"]
    live = [run for run in runs if run.family == "live_tm"]

    if training:
        focus_plot_for_runs(
            training,
            output_dir / "training_improvements_focus_progress.png",
            "Training improvements: focus progress",
            density="off",
        )
        cache_ab = [
            run
            for run in training
            if run.label in {"exp05a_variable_tick_no_cache_control", "exp05b_variable_tick_elite_cache"}
        ]
        if len(cache_ab) == 2:
            focus_plot_for_runs(
                cache_ab,
                output_dir / "variable_tick_cache_focus_progress.png",
                "Variable physics tick: no cache vs elite cache",
                density="auto",
            )
        decay_ab = [
            run
            for run in training
            if run.label in {"exp00_base_fixed100", "exp01b_mutation_decay_first_finish_fixed100"}
        ]
        if len(decay_ab) == 2:
            focus_plot_for_runs(
                decay_ab,
                output_dir / "base_vs_first_finish_decay_focus_progress.png",
                "Fixed100 base vs first-finish mutation decay",
                density="auto",
            )
        count_plot(training, output_dir / "training_improvements_outcomes.png", "Training improvements: outcomes")
        line_plot(
            training,
            [("best_time", "Best finish time [s]")],
            output_dir / "training_improvements_best_time.png",
            "Training improvements: best finish time",
        )
        plot_mutation_and_cache(
            training,
            output_dir / "training_improvements_mutation_and_cache.png",
            "Training improvements: mutation schedule and elite cache",
        )

    if moo:
        line_plot(
            moo,
            [("best_progress", "Best progress"), ("mean_progress", "Mean progress")],
            output_dir / "moo_progress.png",
            "MOO variants: progress over generations",
        )
        count_plot(moo, output_dir / "moo_outcomes.png", "MOO variants: outcomes")
        columns = [("best_time", "Best finish time [s]")]
        if any("front0_size" in run.generation for run in moo):
            columns.append(("front0_size", "Pareto front 0 size"))
        if any("best_scalar_fitness" in run.generation for run in moo):
            columns.append(("best_scalar_fitness", "Best scalar/priority fitness"))
        line_plot(moo, columns, output_dir / "moo_best_time_front.png", "MOO variants: time and front diagnostics")

    for run in generalization:
        plot_generalization(run, output_dir)

    if live:
        focus_plot_for_runs(
            live,
            output_dir / "live_tm_focus_progress.png",
            "Live TM overnight run: focus progress",
            density="auto",
        )
        count_plot(live, output_dir / "live_tm_outcomes.png", "Live TM overnight run: outcomes")
        plot_mutation_and_cache(
            live,
            output_dir / "live_tm_mutation_and_cache.png",
            "Live TM overnight run: mutation trigger and elite cache",
        )
        run_trajectory_analysis(output_dir)

    plot_summary_bars(summaries, output_dir / "summary_bars.png")
    write_report(summaries, output_dir)

    print(f"Analyzed runs: {len(runs)}")
    print(f"Saved summary: {output_dir / 'summary.csv'}")
    print(f"Saved report:  {output_dir / 'REPORT.md'}")
    print("Saved plots:")
    for path in sorted(output_dir.glob("*.png")):
        print(f"  {path}")
    trajectory_dir = output_dir / "live_tm_trajectories"
    if trajectory_dir.exists():
        for path in sorted(trajectory_dir.glob("*.png")):
            print(f"  {path}")


if __name__ == "__main__":
    main()
