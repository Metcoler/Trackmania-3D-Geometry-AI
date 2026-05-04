from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


VARIANT_ORDER = [
    "(finished, progress)",
    "(finished, progress, -time)",
    "(finished, progress, -time, -crashes)",
    "(finished, progress, -crashes, -time)",
]

VARIANT_LABELS = {
    "(finished, progress)": "finished_progress",
    "(finished, progress, -time)": "finished_progress_time",
    "(finished, progress, -time, -crashes)": "finished_progress_time_crashes",
    "(finished, progress, -crashes, -time)": "finished_progress_crashes_time",
}


def fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "" if value is None else str(value)
    if not np.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def frame_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def discover_runs(roots: list[Path]) -> list[Path]:
    run_dirs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for config_path in sorted(root.rglob("config.json")):
            run_dir = config_path.parent
            if (run_dir / "generation_metrics.csv").exists():
                run_dirs.append(run_dir)
    return run_dirs


def load_run(run_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    generation = pd.read_csv(run_dir / "generation_metrics.csv")
    individual_path = run_dir / "individual_metrics.csv"
    individual = pd.read_csv(individual_path) if individual_path.exists() else pd.DataFrame()
    return config, generation, individual


def summarize_run(run_dir: Path, config: dict, generation: pd.DataFrame, individual: pd.DataFrame) -> dict:
    ranking_key = str(config.get("ranking_key", ""))
    variant = VARIANT_LABELS.get(ranking_key, ranking_key)
    seed = int(config.get("seed", infer_seed_from_run_dir(run_dir)))
    expected_generations = int(config.get("generations", 0))
    final_generation = int(generation["generation"].max()) if not generation.empty else 0
    final_row = generation.sort_values("generation").iloc[-1] if not generation.empty else pd.Series(dtype=object)

    first_finish_rows = generation[generation.get("finish_count", 0) > 0]
    first_finish_generation = (
        int(first_finish_rows["generation"].iloc[0])
        if not first_finish_rows.empty
        else np.nan
    )

    finished_individuals = individual[individual.get("finished", 0) > 0] if not individual.empty else pd.DataFrame()
    if not finished_individuals.empty:
        best_finish_row = finished_individuals.sort_values("time").iloc[0]
        best_finish_time = float(best_finish_row["time"])
        best_finish_generation = int(best_finish_row["generation"])
    else:
        best_finish_time = np.nan
        best_finish_generation = np.nan

    if not individual.empty and final_generation > 0:
        final_population = individual[individual["generation"] == final_generation].copy()
    else:
        final_population = pd.DataFrame()

    if not final_population.empty:
        final_finish_count = int((final_population["finished"] > 0).sum())
        final_crash_count = int((final_population["crashes"] > 0).sum())
        final_timeout_count = int((final_population["timeout"] > 0).sum())
        final_mean_dense_progress = float(final_population["dense_progress"].mean())
        final_std_dense_progress = float(final_population["dense_progress"].std(ddof=0))
        final_best_dense_progress = float(final_population["dense_progress"].max())
        final_finished = final_population[final_population["finished"] > 0]
        final_mean_finish_time = (
            float(final_finished["time"].mean()) if not final_finished.empty else np.nan
        )
    else:
        final_finish_count = int(final_row.get("finish_count", 0)) if not final_row.empty else 0
        final_crash_count = int(final_row.get("crash_count", 0)) if not final_row.empty else 0
        final_timeout_count = int(final_row.get("timeout_count", 0)) if not final_row.empty else 0
        final_mean_dense_progress = float(final_row.get("mean_dense_progress", np.nan))
        final_std_dense_progress = float(final_row.get("dense_progress_std", np.nan))
        final_best_dense_progress = float(final_row.get("best_dense_progress", np.nan))
        final_mean_finish_time = np.nan

    population_size = int(config.get("population_size", final_row.get("population_size", 0)))
    total_cached = int(generation.get("cached_evaluations", pd.Series(dtype=int)).sum())
    max_cached = int(generation.get("cached_evaluations", pd.Series(dtype=int)).max()) if not generation.empty else 0

    return {
        "variant": variant,
        "ranking_key": ranking_key,
        "seed": seed,
        "run_dir": str(run_dir),
        "completed": int(final_generation >= expected_generations),
        "expected_generations": expected_generations,
        "final_generation": final_generation,
        "population_size": population_size,
        "elite_count": int(config.get("elite_count", 0)),
        "parent_count": int(config.get("parent_count", 0)),
        "mutation_prob": float(config.get("mutation_prob", np.nan)),
        "mutation_sigma": float(config.get("mutation_sigma", np.nan)),
        "fixed_fps": float(config.get("fixed_fps", np.nan)),
        "collision_mode": str(config.get("collision_mode", "")),
        "collision_distance_threshold": float(config.get("collision_distance_threshold", np.nan)),
        "ranking_progress_source": str(config.get("ranking_progress_source", "")),
        "binary_gas_brake": bool(config.get("binary_gas_brake", False)),
        "elite_cache_enabled": bool(config.get("elite_cache_enabled", True)),
        "total_cached_evaluations": total_cached,
        "max_cached_evaluations": max_cached,
        "first_finish_generation": first_finish_generation,
        "best_finish_time": best_finish_time,
        "best_finish_generation": best_finish_generation,
        "final_finish_count": final_finish_count,
        "final_finish_rate": final_finish_count / max(1, population_size),
        "final_crash_count": final_crash_count,
        "final_crash_rate": final_crash_count / max(1, population_size),
        "final_timeout_count": final_timeout_count,
        "final_timeout_rate": final_timeout_count / max(1, population_size),
        "final_best_dense_progress": final_best_dense_progress,
        "final_mean_dense_progress": final_mean_dense_progress,
        "final_std_dense_progress": final_std_dense_progress,
        "final_mean_finish_time": final_mean_finish_time,
        "cumulative_virtual_time_hours": float(final_row.get("cumulative_virtual_time", np.nan)) / 3600.0,
        "cumulative_wall_minutes": float(final_row.get("cumulative_wall_seconds", np.nan)) / 60.0,
    }


def add_run_columns(df: pd.DataFrame, config: dict, run_dir: Path) -> pd.DataFrame:
    result = df.copy()
    ranking_key = str(config.get("ranking_key", ""))
    result["variant"] = VARIANT_LABELS.get(ranking_key, ranking_key)
    result["ranking_key_config"] = ranking_key
    result["seed"] = int(config.get("seed", infer_seed_from_run_dir(run_dir)))
    result["run_dir"] = str(run_dir)
    return result


def infer_seed_from_run_dir(run_dir: Path) -> int:
    for part in run_dir.parts:
        if part.startswith("lex_sweep_seed_"):
            try:
                return int(part.replace("lex_sweep_seed_", ""))
            except ValueError:
                return -1
    return -1


def sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    order = {VARIANT_LABELS[key]: idx for idx, key in enumerate(VARIANT_ORDER)}
    result = summary.copy()
    result["_variant_order"] = result["variant"].map(order).fillna(999).astype(int)
    result = result.sort_values(["_variant_order", "seed", "run_dir"]).drop(columns=["_variant_order"])
    return result


def build_stability(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    grouped = summary.groupby(["variant", "ranking_key"], dropna=False)
    rows: list[dict] = []
    for (variant, ranking_key), group in grouped:
        rows.append(
            {
                "variant": variant,
                "ranking_key": ranking_key,
                "runs": int(len(group)),
                "completed_runs": int(group["completed"].sum()),
                "first_finish_mean": float(group["first_finish_generation"].mean()),
                "first_finish_std": float(group["first_finish_generation"].std(ddof=0)),
                "best_finish_time_min": float(group["best_finish_time"].min()),
                "best_finish_time_mean": float(group["best_finish_time"].mean()),
                "best_finish_time_std": float(group["best_finish_time"].std(ddof=0)),
                "final_finish_rate_mean": float(group["final_finish_rate"].mean()),
                "final_finish_rate_std": float(group["final_finish_rate"].std(ddof=0)),
                "final_crash_rate_mean": float(group["final_crash_rate"].mean()),
                "final_timeout_rate_mean": float(group["final_timeout_rate"].mean()),
                "final_mean_dense_progress_mean": float(group["final_mean_dense_progress"].mean()),
                "max_cached_evaluations": int(group["max_cached_evaluations"].max()),
            }
        )
    stability = pd.DataFrame(rows)
    order = {VARIANT_LABELS[key]: idx for idx, key in enumerate(VARIANT_ORDER)}
    stability["_variant_order"] = stability["variant"].map(order).fillna(999).astype(int)
    return stability.sort_values("_variant_order").drop(columns=["_variant_order"])


def plot_best_dense_progress(generation: pd.DataFrame, output_dir: Path) -> None:
    if generation.empty:
        return
    plt.figure(figsize=(11, 6))
    for (variant, seed), group in generation.groupby(["variant", "seed"]):
        group = group.sort_values("generation")
        plt.plot(group["generation"], group["best_dense_progress"], label=f"{variant} | seed {seed}", alpha=0.9)
    plt.xlabel("Generation")
    plt.ylabel("Best dense progress [%]")
    plt.title("Best dense progress over generations")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "01_best_dense_progress.png", dpi=160)
    plt.close()


def plot_finish_count(generation: pd.DataFrame, output_dir: Path) -> None:
    if generation.empty:
        return
    plt.figure(figsize=(11, 6))
    for (variant, seed), group in generation.groupby(["variant", "seed"]):
        group = group.sort_values("generation")
        plt.plot(group["generation"], group["finish_count"], label=f"{variant} | seed {seed}", alpha=0.9)
    plt.xlabel("Generation")
    plt.ylabel("Finish count in population")
    plt.title("Population finish count")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "02_finish_count.png", dpi=160)
    plt.close()


def plot_best_finish_time(generation: pd.DataFrame, output_dir: Path) -> None:
    if generation.empty or "best_finished" not in generation.columns:
        return
    plt.figure(figsize=(11, 6))
    for (variant, seed), group in generation.groupby(["variant", "seed"]):
        group = group.sort_values("generation").copy()
        group["finish_time"] = np.where(group["best_finished"] > 0, group["best_time"], np.nan)
        group["best_so_far"] = group["finish_time"].cummin()
        plt.plot(group["generation"], group["best_so_far"], label=f"{variant} | seed {seed}", alpha=0.9)
    plt.xlabel("Generation")
    plt.ylabel("Best finish time so far [s]")
    plt.title("Best finish time found during training")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "03_best_finish_time.png", dpi=160)
    plt.close()


def plot_outcome_rates(generation: pd.DataFrame, output_dir: Path) -> None:
    if generation.empty:
        return
    plt.figure(figsize=(11, 6))
    for (variant, seed), group in generation.groupby(["variant", "seed"]):
        group = group.sort_values("generation").copy()
        population = group["population_size"].replace(0, np.nan)
        finish_rate = group["finish_count"] / population
        crash_rate = group["crash_count"] / population
        plt.plot(group["generation"], finish_rate, label=f"finish | {variant} | {seed}", alpha=0.9)
        plt.plot(group["generation"], crash_rate, linestyle="--", label=f"crash | {variant} | {seed}", alpha=0.55)
    plt.xlabel("Generation")
    plt.ylabel("Population rate")
    plt.title("Finish and crash rates")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "04_finish_crash_rates.png", dpi=160)
    plt.close()


def plot_final_population(individual: pd.DataFrame, output_dir: Path) -> None:
    if individual.empty:
        return
    final_parts: list[pd.DataFrame] = []
    for _, group in individual.groupby(["variant", "seed", "run_dir"]):
        final_generation = group["generation"].max()
        final_parts.append(group[group["generation"] == final_generation])
    if not final_parts:
        return
    final = pd.concat(final_parts, ignore_index=True)
    final["outcome"] = np.where(
        final["finished"] > 0,
        "finish",
        np.where(final["crashes"] > 0, "crash", "timeout"),
    )
    colors = {"finish": "#2ca02c", "crash": "#d62728", "timeout": "#7f7f7f"}
    plt.figure(figsize=(11, 6))
    for outcome, group in final.groupby("outcome"):
        plt.scatter(
            group["dense_progress"],
            group["time"],
            s=22,
            alpha=0.65,
            c=colors.get(outcome, "#1f77b4"),
            label=outcome,
        )
    plt.xlabel("Dense progress [%]")
    plt.ylabel("Time [s]")
    plt.title("Final populations: progress vs time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "05_final_population_scatter.png", dpi=160)
    plt.close()


def write_report(output_dir: Path, summary: pd.DataFrame, stability: pd.DataFrame, roots: list[Path]) -> None:
    compact_cols = [
        "variant",
        "seed",
        "completed",
        "first_finish_generation",
        "best_finish_time",
        "final_finish_rate",
        "final_crash_rate",
        "final_timeout_rate",
        "final_mean_dense_progress",
        "max_cached_evaluations",
    ]
    compact = summary[[col for col in compact_cols if col in summary.columns]].copy()
    for col in compact.columns:
        if col not in {"variant", "seed", "completed", "max_cached_evaluations"}:
            compact[col] = compact[col].map(lambda value: fmt(value, 3))

    stability_compact = stability.copy()
    for col in stability_compact.columns:
        if col not in {"variant", "ranking_key", "runs", "completed_runs", "max_cached_evaluations"}:
            stability_compact[col] = stability_compact[col].map(lambda value: fmt(value, 3))

    cache_warning = ""
    if not summary.empty and int(summary["max_cached_evaluations"].max()) > 0:
        cache_warning = (
            "\n\nWARNING: At least one run has cached evaluations. "
            "For the planned fair sweep this should be zero in every generation.\n"
        )

    report = f"""# Lexicographic GA Reward Sweep Analysis

## Input Roots

{chr(10).join(f"- `{root}`" for root in roots)}

## Summary

This report compares lexicographic GA ranking functions under the fixed fair-sweep setup:
fixed 60 FPS, laser collision threshold 2.0, binary gas/brake, v2d asphalt observation,
population 64, elite count 8, parent count 32, fixed mutation probability 0.2,
fixed mutation sigma 0.2, max episode time 30 seconds, 300 generations, and disabled elite cache.

{cache_warning}

## Per-Run Results

{frame_to_markdown(compact)}

## Cross-Seed Stability

{frame_to_markdown(stability_compact)}

## Generated Plots

- `01_best_dense_progress.png`
- `02_finish_count.png`
- `03_best_finish_time.png`
- `04_finish_crash_rates.png`
- `05_final_population_scatter.png`

## Interpretation Notes

- `(finished, progress)` should show whether progress alone can reliably discover finish, but it has no incentive to become faster after a comparable progress value.
- `(finished, progress, -time)` adds speed pressure and is expected to be the strongest simple baseline, but may prefer faster risky endings when progress is equal.
- `(finished, progress, -time, -crashes)` tests whether crash safety matters when it is only a late tie-breaker after time.
- `(finished, progress, -crashes, -time)` tests the safer ordering where crash avoidance beats time among equally far policies.
- The key thesis signal is not only the best run, but cross-seed stability: first finish generation, final finish rate, crash/timeout rates, and whether the best finish time reproduces.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze fair lexicographic GA sweep runs.")
    parser.add_argument(
        "--run-roots",
        nargs="+",
        default=[
            "Experiments/runs_ga/lex_sweep_seed_2026050201",
            "Experiments/runs_ga/lex_sweep_seed_2026050202",
        ],
        help="One or more sweep root folders containing train_ga.py run directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="Experiments/analysis/lexicographic_sweep",
        help="Directory where combined CSVs, plots and REPORT.md will be written.",
    )
    args = parser.parse_args()

    roots = [Path(value) for value in args.run_roots]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_runs(roots)
    if not run_dirs:
        raise SystemExit(f"No train_ga.py runs found under: {', '.join(str(root) for root in roots)}")

    summary_rows: list[dict] = []
    generation_parts: list[pd.DataFrame] = []
    individual_parts: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        config, generation, individual = load_run(run_dir)
        summary_rows.append(summarize_run(run_dir, config, generation, individual))
        generation_parts.append(add_run_columns(generation, config, run_dir))
        if not individual.empty:
            individual_parts.append(add_run_columns(individual, config, run_dir))

    summary = sort_summary(pd.DataFrame(summary_rows))
    generation = pd.concat(generation_parts, ignore_index=True) if generation_parts else pd.DataFrame()
    individual = pd.concat(individual_parts, ignore_index=True) if individual_parts else pd.DataFrame()
    stability = build_stability(summary)

    summary.to_csv(output_dir / "summary.csv", index=False)
    stability.to_csv(output_dir / "stability_summary.csv", index=False)
    generation.to_csv(output_dir / "combined_generation_metrics.csv", index=False)
    if not individual.empty:
        individual.to_csv(output_dir / "combined_individual_metrics.csv", index=False)

    plot_best_dense_progress(generation, output_dir)
    plot_finish_count(generation, output_dir)
    plot_best_finish_time(generation, output_dir)
    plot_outcome_rates(generation, output_dir)
    plot_final_population(individual, output_dir)
    write_report(output_dir, summary, stability, roots)

    print(f"Analyzed {len(run_dirs)} runs.")
    print(f"Saved report to {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
