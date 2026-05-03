from __future__ import annotations

import argparse
import json
import re
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


HEATMAP_METRICS = [
    ("first_finish_generation_plot", "First finish generation", "lower"),
    ("best_finish_time_plot", "Best finish time", "lower"),
    ("last50_finish_rate", "Last 50 finish rate", "higher"),
    ("last50_mean_dense_progress", "Last 50 mean dense progress", "higher"),
    ("last50_crash_rate", "Last 50 crash rate", "lower"),
    ("last50_timeout_rate", "Last 50 timeout rate", "lower"),
    ("last50_penalized_mean_time", "Last 50 penalized mean time", "lower"),
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


def infer_grid(run_dir: Path) -> str:
    parts = [part.lower() for part in run_dir.parts]
    if any("mutation_grid" in part for part in parts):
        return "mutation"
    if any("selection_grid" in part for part in parts):
        return "selection"
    return "unknown"


def parse_ratio_tag(name: str, prefix: str) -> float | None:
    match = re.search(prefix + r"_(\d{3})", name)
    if not match:
        return None
    return float(match.group(1)) / 100.0


def infer_grid_coordinates(run_dir: Path, config: dict) -> tuple[float, float]:
    grid = infer_grid(run_dir)
    path_text = "\\".join(run_dir.parts)
    if grid == "mutation":
        return float(config.get("mutation_prob", np.nan)), float(config.get("mutation_sigma", np.nan))
    if grid == "selection":
        parent_ratio = parse_ratio_tag(path_text, "parents_ratio")
        elite_ratio = parse_ratio_tag(path_text, "elites_ratio")
        population_size = max(1, int(config.get("population_size", 1)))
        parent_count = max(1, int(config.get("parent_count", 1)))
        if parent_ratio is None:
            parent_ratio = parent_count / population_size
        if elite_ratio is None:
            elite_ratio = int(config.get("elite_count", 0)) / parent_count
        return float(parent_ratio), float(elite_ratio)
    return np.nan, np.nan


def load_run(run_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    generation = pd.read_csv(run_dir / "generation_metrics.csv")
    individual_path = run_dir / "individual_metrics.csv"
    individual = pd.read_csv(individual_path) if individual_path.exists() else pd.DataFrame()
    return config, generation, individual


def add_run_columns(df: pd.DataFrame, config: dict, run_dir: Path) -> pd.DataFrame:
    result = df.copy()
    grid_x, grid_y = infer_grid_coordinates(run_dir, config)
    result["grid"] = infer_grid(run_dir)
    result["grid_x"] = grid_x
    result["grid_y"] = grid_y
    result["run_tag"] = run_dir.parent.name
    result["run_dir"] = str(run_dir)
    result["ranking_key_config"] = str(config.get("ranking_key", ""))
    result["seed"] = int(config.get("seed", -1))
    result["mutation_prob_config"] = float(config.get("mutation_prob", np.nan))
    result["mutation_sigma_config"] = float(config.get("mutation_sigma", np.nan))
    result["population_size_config"] = int(config.get("population_size", 0))
    result["elite_count_config"] = int(config.get("elite_count", 0))
    result["parent_count_config"] = int(config.get("parent_count", 0))
    return result


def summarize_run(run_dir: Path, config: dict, generation: pd.DataFrame, individual: pd.DataFrame) -> dict:
    generation = generation.sort_values("generation").copy()
    final_generation = int(generation["generation"].max()) if not generation.empty else 0
    expected_generations = int(config.get("generations", 0))
    final_row = generation.iloc[-1] if not generation.empty else pd.Series(dtype=object)
    population_size = int(config.get("population_size", final_row.get("population_size", 0)))
    max_time = float(config.get("max_time", 30.0))
    grid = infer_grid(run_dir)
    grid_x, grid_y = infer_grid_coordinates(run_dir, config)

    finish_rows = generation[generation.get("finish_count", 0) > 0] if not generation.empty else pd.DataFrame()
    first_finish_generation = int(finish_rows["generation"].iloc[0]) if not finish_rows.empty else np.nan
    first_finish_generation_plot = (
        first_finish_generation if np.isfinite(first_finish_generation) else expected_generations + 1
    )

    if not individual.empty:
        individual = individual.copy()
        individual["penalized_time"] = np.where(individual["finished"] > 0, individual["time"], max_time)
        finished_individuals = individual[individual["finished"] > 0]
        if not finished_individuals.empty:
            best_finish_row = finished_individuals.sort_values("time").iloc[0]
            best_finish_time = float(best_finish_row["time"])
            best_finish_generation = int(best_finish_row["generation"])
        else:
            best_finish_time = np.nan
            best_finish_generation = np.nan
        total_finish_individuals = int((individual["finished"] > 0).sum())
        total_crash_individuals = int((individual["crashes"] > 0).sum())
        total_timeout_individuals = int((individual["timeout"] > 0).sum())
        last50_threshold = max(1, final_generation - 49)
        last50_individual = individual[individual["generation"] >= last50_threshold]
        if not last50_individual.empty:
            last50_finish_rate = float((last50_individual["finished"] > 0).mean())
            last50_crash_rate = float((last50_individual["crashes"] > 0).mean())
            last50_timeout_rate = float((last50_individual["timeout"] > 0).mean())
            last50_penalized_mean_time = float(last50_individual["penalized_time"].mean())
        else:
            last50_finish_rate = np.nan
            last50_crash_rate = np.nan
            last50_timeout_rate = np.nan
            last50_penalized_mean_time = np.nan
        final_population = individual[individual["generation"] == final_generation]
        final_penalized_mean_time = float(final_population["penalized_time"].mean()) if not final_population.empty else np.nan
    else:
        best_finish_time = np.nan
        best_finish_generation = np.nan
        total_finish_individuals = int(generation.get("finish_count", pd.Series(dtype=float)).sum())
        total_crash_individuals = int(generation.get("crash_count", pd.Series(dtype=float)).sum())
        total_timeout_individuals = int(generation.get("timeout_count", pd.Series(dtype=float)).sum())
        last50 = generation[generation["generation"] >= max(1, final_generation - 49)]
        denominator = max(1, int(len(last50)) * max(1, population_size))
        last50_finish_rate = float(last50.get("finish_count", pd.Series(dtype=float)).sum() / denominator)
        last50_crash_rate = float(last50.get("crash_count", pd.Series(dtype=float)).sum() / denominator)
        last50_timeout_rate = float(last50.get("timeout_count", pd.Series(dtype=float)).sum() / denominator)
        last50_penalized_mean_time = np.nan
        final_penalized_mean_time = np.nan

    last50_generation = generation[generation["generation"] >= max(1, final_generation - 49)] if not generation.empty else pd.DataFrame()
    best_finish_time_plot = best_finish_time if np.isfinite(best_finish_time) else max_time
    max_best_dense = float(generation["best_dense_progress"].max()) if not generation.empty else np.nan
    max_mean_dense = float(generation["mean_dense_progress"].max()) if not generation.empty else np.nan
    final_mean_dense = float(final_row.get("mean_dense_progress", np.nan)) if not final_row.empty else np.nan
    final_std_dense = float(final_row.get("dense_progress_std", np.nan)) if not final_row.empty else np.nan
    last50_mean_dense = (
        float(last50_generation["mean_dense_progress"].mean()) if not last50_generation.empty else np.nan
    )
    last50_best_dense = (
        float(last50_generation["best_dense_progress"].max()) if not last50_generation.empty else np.nan
    )
    total_cached = int(generation.get("cached_evaluations", pd.Series(dtype=int)).sum())
    max_cached = int(generation.get("cached_evaluations", pd.Series(dtype=int)).max()) if not generation.empty else 0

    return {
        "grid": grid,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "run_tag": run_dir.parent.name,
        "run_dir": str(run_dir),
        "completed": int(final_generation >= expected_generations),
        "expected_generations": expected_generations,
        "final_generation": final_generation,
        "seed": int(config.get("seed", -1)),
        "ranking_key": str(config.get("ranking_key", "")),
        "population_size": population_size,
        "elite_count": int(config.get("elite_count", 0)),
        "parent_count": int(config.get("parent_count", 0)),
        "parent_population_ratio_actual": int(config.get("parent_count", 0)) / max(1, population_size),
        "elite_parent_ratio_actual": int(config.get("elite_count", 0)) / max(1, int(config.get("parent_count", 1))),
        "mutation_prob": float(config.get("mutation_prob", np.nan)),
        "mutation_sigma": float(config.get("mutation_sigma", np.nan)),
        "fixed_fps": config.get("fixed_fps", None),
        "collision_mode": str(config.get("collision_mode", "")),
        "lidar_mode": str(config.get("lidar_mode", "")),
        "binary_gas_brake": bool(config.get("binary_gas_brake", False)),
        "elite_cache_enabled": bool(config.get("elite_cache_enabled", True)),
        "total_cached_evaluations": total_cached,
        "max_cached_evaluations": max_cached,
        "first_finish_generation": first_finish_generation,
        "first_finish_generation_plot": float(first_finish_generation_plot),
        "best_finish_time": best_finish_time,
        "best_finish_time_plot": float(best_finish_time_plot),
        "best_finish_generation": best_finish_generation,
        "total_finish_individuals": total_finish_individuals,
        "total_crash_individuals": total_crash_individuals,
        "total_timeout_individuals": total_timeout_individuals,
        "max_finish_count": int(generation.get("finish_count", pd.Series(dtype=int)).max()) if not generation.empty else 0,
        "max_best_dense_progress": max_best_dense,
        "max_mean_dense_progress": max_mean_dense,
        "final_mean_dense_progress": final_mean_dense,
        "final_std_dense_progress": final_std_dense,
        "last50_best_dense_progress": last50_best_dense,
        "last50_mean_dense_progress": last50_mean_dense,
        "last50_finish_rate": last50_finish_rate,
        "last50_crash_rate": last50_crash_rate,
        "last50_timeout_rate": last50_timeout_rate,
        "last50_penalized_mean_time": last50_penalized_mean_time,
        "final_penalized_mean_time": final_penalized_mean_time,
        "cumulative_virtual_time_hours": float(final_row.get("cumulative_virtual_time", np.nan)) / 3600.0,
        "cumulative_wall_minutes": float(final_row.get("cumulative_wall_seconds", np.nan)) / 60.0,
    }


def sort_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["_first_finish_sort"] = result["first_finish_generation"].fillna(result["expected_generations"] + 1)
    result["_best_time_sort"] = result["best_finish_time"].fillna(result.get("max_time", pd.Series(30.0, index=result.index)))
    return result.sort_values(
        [
            "last50_finish_rate",
            "_first_finish_sort",
            "_best_time_sort",
            "last50_mean_dense_progress",
            "last50_crash_rate",
            "last50_timeout_rate",
        ],
        ascending=[False, True, True, False, True, True],
    ).drop(columns=["_first_finish_sort", "_best_time_sort"])


def add_compromise_columns(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["compromise_score"] = (
        result["last50_finish_rate"].fillna(0.0)
        + result["last50_mean_dense_progress"].fillna(0.0) / 100.0
        - result["last50_crash_rate"].fillna(1.0)
        - 0.25 * result["last50_timeout_rate"].fillna(1.0)
    )
    return result


def sort_compromise_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["_first_finish_sort"] = result["first_finish_generation"].fillna(result["expected_generations"] + 1)
    result["_best_time_sort"] = result["best_finish_time"].fillna(result.get("max_time", pd.Series(30.0, index=result.index)))
    return result.sort_values(
        [
            "compromise_score",
            "last50_finish_rate",
            "last50_mean_dense_progress",
            "_first_finish_sort",
            "_best_time_sort",
        ],
        ascending=[False, False, False, True, True],
    ).drop(columns=["_first_finish_sort", "_best_time_sort"])


def edge_warning_lines(summary: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if summary.empty:
        return lines
    for grid, group in summary.groupby("grid"):
        if grid == "unknown" or group.empty:
            continue
        x_values = group["grid_x"].dropna().unique()
        y_values = group["grid_y"].dropna().unique()
        if len(x_values) == 0 or len(y_values) == 0:
            continue
        x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
        y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
        leaders = [
            ("stability ranking", sort_candidates(group).iloc[0]),
            ("compromise score", sort_compromise_candidates(group).iloc[0]),
        ]
        for label, row in leaders:
            x = float(row["grid_x"])
            y = float(row["grid_y"])
            edge_parts: list[str] = []
            if np.isclose(x, x_min):
                edge_parts.append("minimum x")
            if np.isclose(x, x_max):
                edge_parts.append("maximum x")
            if np.isclose(y, y_min):
                edge_parts.append("minimum y")
            if np.isclose(y, y_max):
                edge_parts.append("maximum y")
            if edge_parts:
                lines.append(
                    f"- `{grid}` best by {label} is on grid edge ({', '.join(edge_parts)}): "
                    f"`{row['run_tag']}`."
                )
            else:
                lines.append(f"- `{grid}` best by {label} is inside the tested grid: `{row['run_tag']}`.")
    return lines


def pivot_for_heatmap(summary: pd.DataFrame, grid: str, metric: str) -> pd.DataFrame:
    data = summary[summary["grid"] == grid].copy()
    if data.empty:
        return pd.DataFrame()
    return data.pivot_table(index="grid_y", columns="grid_x", values=metric, aggfunc="mean").sort_index(ascending=True)


def plot_heatmap(summary: pd.DataFrame, grid: str, metric: str, title: str, direction: str, output_dir: Path) -> None:
    pivot = pivot_for_heatmap(summary, grid, metric)
    if pivot.empty:
        return
    values = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    plt.figure(figsize=(8.5, 6.5))
    image = plt.imshow(masked, origin="lower", aspect="auto", cmap=cmap)
    plt.colorbar(image, label=title)
    plt.xticks(np.arange(len(pivot.columns)), [fmt(value, 2) for value in pivot.columns])
    plt.yticks(np.arange(len(pivot.index)), [fmt(value, 2) for value in pivot.index])
    if grid == "mutation":
        plt.xlabel("mutation_prob")
        plt.ylabel("mutation_sigma")
    else:
        plt.xlabel("parents / population target ratio")
        plt.ylabel("elites / parents target ratio")
    plt.title(f"{grid.title()} grid: {title} ({direction} is better)")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                digits = 1 if "generation" in metric or "time" in metric else 3
                plt.text(col_idx, row_idx, fmt(value, digits), ha="center", va="center", fontsize=8, color="white")
    plt.tight_layout()
    safe_metric = metric.replace("_plot", "")
    plt.savefig(output_dir / f"heatmap_{grid}_{safe_metric}.png", dpi=160)
    plt.close()


def plot_all_heatmaps(summary: pd.DataFrame, output_dir: Path) -> None:
    for grid in ("mutation", "selection"):
        for metric, title, direction in HEATMAP_METRICS:
            plot_heatmap(summary, grid, metric, title, direction, output_dir)


def plot_candidate_scatter(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    plt.figure(figsize=(9, 6))
    for grid, group in summary.groupby("grid"):
        plt.scatter(
            group["last50_finish_rate"],
            group["best_finish_time_plot"],
            s=70 + 3.0 * group["last50_mean_dense_progress"].fillna(0.0),
            alpha=0.75,
            label=grid,
        )
    plt.xlabel("Last 50 finish rate")
    plt.ylabel("Best finish time, max_time if no finish")
    plt.title("Hyperparameter candidates: stability vs speed")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "candidate_stability_vs_speed.png", dpi=160)
    plt.close()


def compact_table(df: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    columns = [
        "grid",
        "grid_x",
        "grid_y",
        "compromise_score",
        "mutation_prob",
        "mutation_sigma",
        "parent_count",
        "elite_count",
        "first_finish_generation",
        "best_finish_time",
        "last50_finish_rate",
        "last50_mean_dense_progress",
        "last50_crash_rate",
        "last50_timeout_rate",
        "last50_penalized_mean_time",
        "max_cached_evaluations",
    ]
    result = df[[col for col in columns if col in df.columns]].head(limit).copy()
    for col in result.columns:
        if col not in {"grid", "parent_count", "elite_count", "max_cached_evaluations"}:
            result[col] = result[col].map(lambda value: fmt(value, 3))
    return result


def write_report(output_dir: Path, summary: pd.DataFrame, roots: list[Path]) -> None:
    ranked = sort_candidates(summary)
    compromise_ranked = sort_compromise_candidates(summary)
    mutation_ranked = sort_candidates(summary[summary["grid"] == "mutation"])
    selection_ranked = sort_candidates(summary[summary["grid"] == "selection"])
    mutation_compromise = sort_compromise_candidates(summary[summary["grid"] == "mutation"])
    selection_compromise = sort_compromise_candidates(summary[summary["grid"] == "selection"])
    incomplete = summary[summary["completed"] <= 0]
    cached = summary[summary["max_cached_evaluations"] > 0]
    edge_lines = edge_warning_lines(summary)
    plot_lines = [f"- `{path.name}`" for path in sorted(output_dir.glob("*.png"))]

    report = f"""# GA Hyperparameter Sweep Analysis

## Input Roots

{chr(10).join(f"- `{root}`" for root in roots)}

## Summary

This analysis compares GA hyperparameters for the fixed reward tuple `(finished, progress, -time, -crashes)`.
The environment baseline is fixed FPS 100, AABB-clearance lidar, binary gas/brake, max time 30, dense progress, and disabled elite cache.

Loaded runs: `{len(summary)}`.
Incomplete runs: `{len(incomplete)}`.
Runs with cached elite evaluations: `{len(cached)}`.

This is a screening experiment. The tables below identify promising regions; they are not a thesis-final proof until the best candidates are repeated with another seed.

## Best Overall Candidates

{frame_to_markdown(compact_table(ranked, limit=10))}

## Best Compromise Candidates

Compromise score is `last50_finish_rate + last50_mean_dense_progress / 100 - last50_crash_rate - 0.25 * last50_timeout_rate`.
It is a screening helper, not a final fitness value.

{frame_to_markdown(compact_table(compromise_ranked, limit=10))}

## Mutation Grid Candidates

{frame_to_markdown(compact_table(mutation_ranked, limit=8))}

## Mutation Grid Compromise Candidates

{frame_to_markdown(compact_table(mutation_compromise, limit=8))}

## Selection Pressure Grid Candidates

{frame_to_markdown(compact_table(selection_ranked, limit=8))}

## Selection Pressure Compromise Candidates

{frame_to_markdown(compact_table(selection_compromise, limit=8))}

## Edge Check

{chr(10).join(edge_lines) if edge_lines else "_No edge warnings._"}

## Generated Plots

{chr(10).join(plot_lines) if plot_lines else "_No plots generated._"}

## Reading Guide

- `first_finish_generation` measures how quickly a configuration discovers a complete lap.
- `last50_finish_rate` measures late training stability, not only one lucky finisher.
- `best_finish_time` measures speed, but should not override stability by itself.
- `last50_penalized_mean_time` treats unfinished individuals as `max_time`, so it combines finish quality and failure rate.
- If the best value lies on a grid edge, the next experiment should be a smaller refinement grid around that edge.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TM2D GA hyperparameter sweep runs.")
    parser.add_argument(
        "--run-roots",
        nargs="+",
        default=[
            "Experiments/runs_ga_hyperparam/pc1_mutation_grid_seed_2026050311",
            "Experiments/runs_ga_hyperparam/pc2_selection_grid_seed_2026050311",
        ],
    )
    parser.add_argument(
        "--output-dir",
        default="Experiments/analysis/ga_hyperparam_sweep_20260503",
    )
    args = parser.parse_args()

    roots = [Path(value) for value in args.run_roots]
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

    summary = add_compromise_columns(pd.DataFrame(summaries)).sort_values(["grid", "grid_x", "grid_y", "run_tag"])
    generation = pd.concat(generation_parts, ignore_index=True) if generation_parts else pd.DataFrame()
    individual = pd.concat(individual_parts, ignore_index=True) if individual_parts else pd.DataFrame()

    summary.to_csv(output_dir / "summary.csv", index=False)
    generation.to_csv(output_dir / "combined_generation_metrics.csv", index=False)
    if not individual.empty:
        individual.to_csv(output_dir / "combined_individual_metrics.csv", index=False)

    plot_all_heatmaps(summary, output_dir)
    plot_candidate_scatter(summary, output_dir)
    write_report(output_dir, summary, roots)

    print(f"Analyzed {len(run_dirs)} runs.")
    print(f"Saved summary to {output_dir / 'summary.csv'}")
    print(f"Saved report to {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
