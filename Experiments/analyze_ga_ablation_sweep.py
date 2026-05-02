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


ABLATION_ORDER = [
    "collision_corners",
    "variable_fps",
    "max_time_45",
    "elite_cache",
    "continuous_gas_brake",
]


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


def infer_seed(run_dir: Path) -> int:
    for part in run_dir.parts:
        if "_seed_" in part:
            try:
                return int(part.rsplit("_seed_", 1)[1])
            except ValueError:
                return -1
    return -1


def infer_ablation_tag(run_dir: Path) -> str:
    for part in reversed(run_dir.parts):
        if part in ABLATION_ORDER:
            return part
    parent = run_dir.parent.name
    return parent if parent else "unknown"


def infer_pc_label(run_dir: Path) -> str:
    for part in run_dir.parts:
        if part.startswith("pc1_"):
            return "PC1"
        if part.startswith("pc2_"):
            return "PC2"
    return "unknown"


def load_run(run_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    generation = pd.read_csv(run_dir / "generation_metrics.csv")
    individual_path = run_dir / "individual_metrics.csv"
    individual = pd.read_csv(individual_path) if individual_path.exists() else pd.DataFrame()
    return config, generation, individual


def summarize_run(run_dir: Path, config: dict, generation: pd.DataFrame, individual: pd.DataFrame) -> dict:
    final_generation = int(generation["generation"].max()) if not generation.empty else 0
    expected_generations = int(config.get("generations", 0))
    final_row = generation.sort_values("generation").iloc[-1] if not generation.empty else pd.Series(dtype=object)
    max_best_row = generation.loc[generation["best_dense_progress"].idxmax()] if not generation.empty else pd.Series(dtype=object)
    max_mean_row = generation.loc[generation["mean_dense_progress"].idxmax()] if not generation.empty else pd.Series(dtype=object)
    first_half = generation[generation["generation"] <= 150] if not generation.empty else pd.DataFrame()
    second_half = generation[generation["generation"] > 150] if not generation.empty else pd.DataFrame()

    first_finish_rows = generation[generation.get("finish_count", 0) > 0] if not generation.empty else pd.DataFrame()
    first_finish_generation = int(first_finish_rows["generation"].iloc[0]) if not first_finish_rows.empty else np.nan

    finished_individuals = individual[individual.get("finished", 0) > 0] if not individual.empty else pd.DataFrame()
    if not finished_individuals.empty:
        best_finish_row = finished_individuals.sort_values("time").iloc[0]
        best_finish_time = float(best_finish_row["time"])
        best_finish_generation = int(best_finish_row["generation"])
    else:
        best_finish_time = np.nan
        best_finish_generation = np.nan

    max_best_dense = float(generation["best_dense_progress"].max()) if not generation.empty else np.nan
    early_best = float(first_half["best_dense_progress"].max()) if not first_half.empty else np.nan
    late_best = float(second_half["best_dense_progress"].max()) if not second_half.empty else np.nan
    late_gain = late_best - early_best if np.isfinite(early_best) and np.isfinite(late_best) else np.nan

    if not individual.empty and final_generation > 0:
        final_population = individual[individual["generation"] == final_generation].copy()
        final_mean_dense = float(final_population["dense_progress"].mean())
        final_std_dense = float(final_population["dense_progress"].std(ddof=0))
        final_best_dense = float(final_population["dense_progress"].max())
        final_finish_count = int((final_population["finished"] > 0).sum())
        final_crash_count = int((final_population["crashes"] > 0).sum())
        final_timeout_count = int((final_population["timeout"] > 0).sum())
    else:
        final_mean_dense = float(final_row.get("mean_dense_progress", np.nan))
        final_std_dense = float(final_row.get("dense_progress_std", np.nan))
        final_best_dense = float(final_row.get("best_dense_progress", np.nan))
        final_finish_count = int(final_row.get("finish_count", 0)) if not final_row.empty else 0
        final_crash_count = int(final_row.get("crash_count", 0)) if not final_row.empty else 0
        final_timeout_count = int(final_row.get("timeout_count", 0)) if not final_row.empty else 0

    population_size = int(config.get("population_size", final_row.get("population_size", 0)))
    total_cached = int(generation.get("cached_evaluations", pd.Series(dtype=int)).sum())
    max_cached = int(generation.get("cached_evaluations", pd.Series(dtype=int)).max()) if not generation.empty else 0

    return {
        "pc": infer_pc_label(run_dir),
        "ablation": infer_ablation_tag(run_dir),
        "ranking_key": str(config.get("ranking_key", "")),
        "seed": int(config.get("seed", infer_seed(run_dir))),
        "run_dir": str(run_dir),
        "completed": int(final_generation >= expected_generations),
        "expected_generations": expected_generations,
        "final_generation": final_generation,
        "population_size": population_size,
        "elite_count": int(config.get("elite_count", 0)),
        "parent_count": int(config.get("parent_count", 0)),
        "mutation_prob": float(config.get("mutation_prob", np.nan)),
        "mutation_sigma": float(config.get("mutation_sigma", np.nan)),
        "fixed_fps": config.get("fixed_fps", None),
        "collision_mode": str(config.get("collision_mode", "")),
        "collision_distance_threshold": float(config.get("collision_distance_threshold", np.nan)),
        "binary_gas_brake": bool(config.get("binary_gas_brake", False)),
        "elite_cache_enabled": bool(config.get("elite_cache_enabled", True)),
        "total_cached_evaluations": total_cached,
        "max_cached_evaluations": max_cached,
        "first_finish_generation": first_finish_generation,
        "best_finish_time": best_finish_time,
        "best_finish_generation": best_finish_generation,
        "max_best_dense_progress": max_best_dense,
        "gen_max_best_dense_progress": int(max_best_row.get("generation", 0)) if not max_best_row.empty else np.nan,
        "max_mean_dense_progress": float(max_mean_row.get("mean_dense_progress", np.nan)) if not max_mean_row.empty else np.nan,
        "gen_max_mean_dense_progress": int(max_mean_row.get("generation", 0)) if not max_mean_row.empty else np.nan,
        "early_best_dense_progress": early_best,
        "late_best_dense_progress": late_best,
        "late_gain_dense_progress": late_gain,
        "final_best_dense_progress": final_best_dense,
        "final_mean_dense_progress": final_mean_dense,
        "final_std_dense_progress": final_std_dense,
        "final_finish_count": final_finish_count,
        "final_finish_rate": final_finish_count / max(1, population_size),
        "final_crash_count": final_crash_count,
        "final_crash_rate": final_crash_count / max(1, population_size),
        "final_timeout_count": final_timeout_count,
        "final_timeout_rate": final_timeout_count / max(1, population_size),
        "max_finish_count": int(generation.get("finish_count", pd.Series(dtype=int)).max()) if not generation.empty else 0,
        "cumulative_virtual_time_hours": float(final_row.get("cumulative_virtual_time", np.nan)) / 3600.0,
        "cumulative_wall_minutes": float(final_row.get("cumulative_wall_seconds", np.nan)) / 60.0,
    }


def add_run_columns(df: pd.DataFrame, config: dict, run_dir: Path) -> pd.DataFrame:
    result = df.copy()
    result["pc"] = infer_pc_label(run_dir)
    result["ablation"] = infer_ablation_tag(run_dir)
    result["ranking_key_config"] = str(config.get("ranking_key", ""))
    result["seed"] = int(config.get("seed", infer_seed(run_dir)))
    result["run_dir"] = str(run_dir)
    return result


def load_baseline(baseline_dir: Path | None) -> pd.DataFrame:
    if baseline_dir is None:
        return pd.DataFrame()
    gen_path = baseline_dir / "combined_generation_metrics.csv"
    if not gen_path.exists():
        return pd.DataFrame()
    generation = pd.read_csv(gen_path)
    key_column = "ranking_key_config" if "ranking_key_config" in generation.columns else "ranking_key"
    rows: list[dict] = []
    for ranking_key, group in generation.groupby(key_column):
        final_parts = []
        for _, run in group.groupby("run_dir"):
            last = run.sort_values("generation").iloc[-1]
            final_parts.append(last)
        final = pd.DataFrame(final_parts)
        rows.append(
            {
                "ranking_key": ranking_key,
                "baseline_runs": int(group["run_dir"].nunique()),
                "baseline_max_best_dense_mean": float(group.groupby("run_dir")["best_dense_progress"].max().mean()),
                "baseline_final_mean_dense_mean": float(final["mean_dense_progress"].mean()),
                "baseline_final_finish_rate_mean": float(
                    (final["finish_count"] / final["population_size"].replace(0, np.nan)).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def attach_baseline(summary: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if baseline.empty:
        return summary
    result = summary.merge(baseline, on="ranking_key", how="left")
    result["delta_max_best_dense_vs_baseline"] = (
        result["max_best_dense_progress"] - result["baseline_max_best_dense_mean"]
    )
    result["delta_final_mean_dense_vs_baseline"] = (
        result["final_mean_dense_progress"] - result["baseline_final_mean_dense_mean"]
    )
    return result


def sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    order = {name: index for index, name in enumerate(ABLATION_ORDER)}
    result = summary.copy()
    result["_order"] = result["ablation"].map(order).fillna(999).astype(int)
    return result.sort_values(["_order", "pc", "ranking_key"]).drop(columns=["_order"])


def build_pair_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for ablation, group in summary.groupby("ablation"):
        pc1 = group[group["pc"] == "PC1"]
        pc2 = group[group["pc"] == "PC2"]
        if pc1.empty or pc2.empty:
            continue
        a = pc1.iloc[0]
        b = pc2.iloc[0]
        rows.append(
            {
                "ablation": ablation,
                "pc1_key": a["ranking_key"],
                "pc2_key": b["ranking_key"],
                "pc1_max_best_dense": float(a["max_best_dense_progress"]),
                "pc2_max_best_dense": float(b["max_best_dense_progress"]),
                "pc1_minus_pc2_max_best_dense": float(a["max_best_dense_progress"] - b["max_best_dense_progress"]),
                "pc1_final_mean_dense": float(a["final_mean_dense_progress"]),
                "pc2_final_mean_dense": float(b["final_mean_dense_progress"]),
                "pc1_minus_pc2_final_mean_dense": float(a["final_mean_dense_progress"] - b["final_mean_dense_progress"]),
                "pc1_final_finish_rate": float(a["final_finish_rate"]),
                "pc2_final_finish_rate": float(b["final_finish_rate"]),
            }
        )
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison
    order = {name: index for index, name in enumerate(ABLATION_ORDER)}
    comparison["_order"] = comparison["ablation"].map(order).fillna(999).astype(int)
    return comparison.sort_values("_order").drop(columns=["_order"])


def plot_best_dense(generation: pd.DataFrame, output_dir: Path) -> None:
    if generation.empty:
        return
    plt.figure(figsize=(12, 7))
    for (pc, ablation), group in generation.groupby(["pc", "ablation"]):
        group = group.sort_values("generation")
        plt.plot(group["generation"], group["best_dense_progress"], label=f"{pc} {ablation}", alpha=0.85)
    plt.xlabel("Generation")
    plt.ylabel("Best dense progress [%]")
    plt.title("Ablation sweep: best dense progress")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "01_best_dense_progress.png", dpi=160)
    plt.close()


def plot_summary_bars(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    data = sort_summary(summary)
    labels = [f"{row.pc}\n{row.ablation}" for row in data.itertuples()]
    x = np.arange(len(data))
    width = 0.38
    plt.figure(figsize=(13, 6))
    plt.bar(x - width / 2, data["max_best_dense_progress"], width, label="max best dense")
    plt.bar(x + width / 2, data["final_mean_dense_progress"], width, label="final mean dense")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Dense progress [%]")
    plt.title("Ablation potential summary")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "02_summary_bars.png", dpi=160)
    plt.close()


def plot_pair_delta(pair: pd.DataFrame, output_dir: Path) -> None:
    if pair.empty:
        return
    plt.figure(figsize=(10, 5))
    x = np.arange(len(pair))
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.bar(x, pair["pc1_minus_pc2_max_best_dense"])
    plt.xticks(x, pair["ablation"], rotation=25, ha="right")
    plt.ylabel("PC1 - PC2 max best dense progress")
    plt.title("Ranking tuple difference under same ablation")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "03_pc1_minus_pc2.png", dpi=160)
    plt.close()


def write_report(output_dir: Path, summary: pd.DataFrame, pair: pd.DataFrame, roots: list[Path], baseline_dir: Path | None) -> None:
    compact_cols = [
        "pc",
        "ablation",
        "ranking_key",
        "completed",
        "max_best_dense_progress",
        "gen_max_best_dense_progress",
        "final_mean_dense_progress",
        "final_std_dense_progress",
        "late_gain_dense_progress",
        "final_finish_rate",
        "final_crash_rate",
        "max_cached_evaluations",
        "delta_max_best_dense_vs_baseline",
        "delta_final_mean_dense_vs_baseline",
    ]
    compact = summary[[col for col in compact_cols if col in summary.columns]].copy()
    for col in compact.columns:
        if col not in {"pc", "ablation", "ranking_key", "completed", "gen_max_best_dense_progress", "max_cached_evaluations"}:
            compact[col] = compact[col].map(lambda value: fmt(value, 3))

    pair_compact = pair.copy()
    for col in pair_compact.columns:
        if col not in {"ablation", "pc1_key", "pc2_key"}:
            pair_compact[col] = pair_compact[col].map(lambda value: fmt(value, 3))

    baseline_line = f"`{baseline_dir}`" if baseline_dir is not None else "_not provided_"
    report = f"""# GA Ablation Sweep Analysis

## Input Roots

{chr(10).join(f"- `{root}`" for root in roots)}

Baseline analysis: {baseline_line}

## Summary

This is an internal diagnostic sweep, not a thesis-final comparison. Each run changes exactly one setting relative to the hard lexicographic baseline: collision mode, FPS mode, max episode time, elite caching, or gas/brake continuity.

PC1 uses `(finished, progress, -crashes, -time)`.
PC2 uses `(finished, progress, -time)`.

## Per-Run Results

{frame_to_markdown(compact)}

## PC1 vs PC2 Under Same Ablation

{frame_to_markdown(pair_compact)}

## Generated Plots

- `01_best_dense_progress.png`
- `02_summary_bars.png`
- `03_pc1_minus_pc2.png`

## Reading Guide

- Positive `delta_*_vs_baseline` means the ablation improved over the previous hard sweep average for the same ranking tuple.
- Positive `PC1 - PC2` means crash-before-time ranking did better than time-only ranking in that specific ablation.
- Treat finish rate as the strongest signal if any ablation reaches finish; otherwise use max best dense progress, final mean dense progress, and late gain as diagnostic potential signals.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze internal GA ablation sweep runs.")
    parser.add_argument(
        "--run-roots",
        nargs="+",
        default=[
            "Experiments/runs_ga_ablation/pc1_finished_progress_crashes_time_seed_2026050203",
            "Experiments/runs_ga_ablation/pc2_finished_progress_time_seed_2026050203",
        ],
    )
    parser.add_argument(
        "--baseline-analysis",
        default="Experiments/analysis/lexicographic_sweep_20260502",
        help="Optional previous hard-sweep analysis folder used as baseline.",
    )
    parser.add_argument(
        "--output-dir",
        default="Experiments/analysis/ga_ablation_20260502",
    )
    args = parser.parse_args()

    roots = [Path(value) for value in args.run_roots]
    baseline_dir = Path(args.baseline_analysis) if args.baseline_analysis else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_runs(roots)
    if not run_dirs:
        raise SystemExit(f"No train_ga.py runs found under: {', '.join(str(root) for root in roots)}")

    summaries: list[dict] = []
    generation_parts: list[pd.DataFrame] = []
    individual_parts: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        config, generation, individual = load_run(run_dir)
        summaries.append(summarize_run(run_dir, config, generation, individual))
        generation_parts.append(add_run_columns(generation, config, run_dir))
        if not individual.empty:
            individual_parts.append(add_run_columns(individual, config, run_dir))

    summary = sort_summary(pd.DataFrame(summaries))
    baseline = load_baseline(baseline_dir)
    summary = attach_baseline(summary, baseline)
    pair = build_pair_comparison(summary)
    generation = pd.concat(generation_parts, ignore_index=True) if generation_parts else pd.DataFrame()
    individual = pd.concat(individual_parts, ignore_index=True) if individual_parts else pd.DataFrame()

    summary.to_csv(output_dir / "summary.csv", index=False)
    pair.to_csv(output_dir / "pc_pair_comparison.csv", index=False)
    generation.to_csv(output_dir / "combined_generation_metrics.csv", index=False)
    if not individual.empty:
        individual.to_csv(output_dir / "combined_individual_metrics.csv", index=False)

    plot_best_dense(generation, output_dir)
    plot_summary_bars(summary, output_dir)
    plot_pair_delta(pair, output_dir)
    write_report(output_dir, summary, pair, roots, baseline_dir)

    print(f"Analyzed {len(run_dirs)} runs.")
    print(f"Saved report to {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
