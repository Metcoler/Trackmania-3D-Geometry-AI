from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NeuralPolicy import NeuralPolicy
from Experiments.tm2d_env import TM2DRewardConfig, TM2DSimEnv
from Experiments.train_ga import NumpyPolicyView


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_generation_files(root: Path, include: str = "", exclude: str = "") -> list[Path]:
    if root.is_file() and root.name == "generation_metrics.csv":
        return [root]
    files = sorted(root.rglob("generation_metrics.csv"))
    if include:
        files = [path for path in files if include in str(path)]
    if exclude:
        excluded = [part for part in exclude.split(",") if part]
        files = [
            path for path in files
            if not any(part in str(path) for part in excluded)
        ]
    return files


def load_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def run_label(run_dir: Path, root: Path, config: dict) -> str:
    subset = config.get("objective_subset") or config.get("objective_names") or []
    if isinstance(subset, list) and subset:
        return "MOO | " + ",".join(str(item) for item in subset)
    if root in run_dir.parents or root == run_dir:
        return run_dir.relative_to(root).as_posix()
    return run_dir.name


def load_run(csv_path: Path, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    run_dir = csv_path.parent
    config = load_config(run_dir)
    label = run_label(run_dir, root, config)
    generation_df = pd.read_csv(csv_path)
    generation_df["run_dir"] = str(run_dir)
    generation_df["run_label"] = label
    generation_df["objective_subset"] = ",".join(config.get("objective_subset", config.get("objective_names", [])))
    individual_path = run_dir / "individual_metrics.csv"
    if individual_path.exists():
        individual_df = pd.read_csv(individual_path)
        individual_df["run_dir"] = str(run_dir)
        individual_df["run_label"] = label
    else:
        individual_df = pd.DataFrame()
    return generation_df, individual_df, config


def summarize_run(generation_df: pd.DataFrame, individual_df: pd.DataFrame, config: dict) -> dict:
    final = generation_df.sort_values("generation").iloc[-1]
    finish_generations = generation_df[generation_df["finish_count"] > 0]
    best_finish_generations = generation_df[generation_df["best_finished"] > 0]
    finished_best = generation_df[generation_df["best_finished"] > 0]
    last50 = generation_df.tail(min(50, len(generation_df)))

    final_generation = int(final["generation"])
    final_individuals = individual_df[individual_df["generation"] == final_generation] if not individual_df.empty else pd.DataFrame()
    if not final_individuals.empty:
        final_finish_rate = float((final_individuals["finished"] > 0).mean())
        final_crash_rate = float((final_individuals["crashes"] > 0).mean())
        final_timeout_rate = float((final_individuals["timeout"] > 0).mean())
        final_median_dense = float(final_individuals["dense_progress"].median())
        final_p90_dense = float(final_individuals["dense_progress"].quantile(0.9))
    else:
        population_size = max(1.0, float(config.get("population_size", 1)))
        final_finish_rate = float(final["finish_count"]) / population_size
        final_crash_rate = float(final["crash_count"]) / population_size
        final_timeout_rate = float(final["timeout_count"]) / population_size
        final_median_dense = float(final.get("dense_progress_median", np.nan))
        final_p90_dense = float(final.get("dense_progress_p90", np.nan))

    best_finish_time = (
        float(finished_best["best_time"].min())
        if not finished_best.empty
        else float("nan")
    )
    best_finish_generation = (
        int(finished_best.loc[finished_best["best_time"].idxmin(), "generation"])
        if not finished_best.empty
        else -1
    )

    return {
        "run_label": str(generation_df["run_label"].iloc[0]),
        "run_dir": str(generation_df["run_dir"].iloc[0]),
        "objective_subset": str(generation_df["objective_subset"].iloc[0]),
        "generations": int(final_generation),
        "population_size": int(config.get("population_size", 0)),
        "first_generation_with_any_finish": int(finish_generations["generation"].iloc[0]) if not finish_generations.empty else -1,
        "first_generation_where_best_finished": int(best_finish_generations["generation"].iloc[0]) if not best_finish_generations.empty else -1,
        "best_finish_time": best_finish_time,
        "best_finish_generation": best_finish_generation,
        "median_best_finish_time_last50": float(last50[last50["best_finished"] > 0]["best_time"].median()) if (last50["best_finished"] > 0).any() else float("nan"),
        "max_finish_count": int(generation_df["finish_count"].max()),
        "final_finish_count": int(final["finish_count"]),
        "final_crash_count": int(final["crash_count"]),
        "final_timeout_count": int(final["timeout_count"]),
        "final_finish_rate": final_finish_rate,
        "final_crash_rate": final_crash_rate,
        "final_timeout_rate": final_timeout_rate,
        "max_best_dense_progress": float(generation_df["best_dense_progress"].max()),
        "final_best_dense_progress": float(final["best_dense_progress"]),
        "final_mean_dense_progress": float(final["mean_dense_progress"]),
        "final_median_dense_progress": final_median_dense,
        "final_p90_dense_progress": final_p90_dense,
        "final_front0_size": int(final.get("front0_size", 0)),
        "mean_front0_size": float(generation_df["front0_size"].mean()) if "front0_size" in generation_df else float("nan"),
        "cumulative_virtual_time_hours": float(final["cumulative_virtual_time"]) / 3600.0,
        "cumulative_wall_minutes": float(final["cumulative_wall_seconds"]) / 60.0,
    }


def evaluate_policy(run_dir: Path, config: dict, episodes: int, seed: int) -> pd.DataFrame:
    model_path = run_dir / "best_policy.pt"
    if not model_path.exists():
        return pd.DataFrame()
    policy, _ = NeuralPolicy.load(str(model_path), map_location="cpu")
    reward_mode = str(config.get("reward_mode", "terminal_progress_time_efficiency"))
    map_name = str(config.get("map_name", "AI Training #5"))
    max_time = float(config.get("max_time", 45.0))
    collision_mode = str(config.get("collision_mode", "corners"))
    rows: list[dict] = []
    for episode in range(1, int(episodes) + 1):
        env = TM2DSimEnv(
            map_name=map_name,
            max_time=max_time,
            reward_config=TM2DRewardConfig(mode=reward_mode),
            seed=int(seed) + episode,
            collision_mode=collision_mode,
        )
        metrics = env.rollout_policy(NumpyPolicyView(policy))
        rows.append(
            {
                "episode": episode,
                "finished": int(metrics.get("finished", 0)),
                "crashes": int(metrics.get("crashes", 0)),
                "timeout": int(metrics.get("timeout", 0)),
                "progress": float(metrics.get("progress", 0.0)),
                "dense_progress": float(metrics.get("dense_progress", 0.0)),
                "time": float(metrics.get("time", 0.0)),
                "distance": float(metrics.get("distance", 0.0)),
                "reward": float(metrics.get("reward", 0.0)),
                "steps": int(metrics.get("steps", 0)),
            }
        )
    return pd.DataFrame(rows)


def summarize_validation(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    finished = df[df["finished"] > 0]
    return {
        "validation_episodes": int(len(df)),
        "validation_finish_count": int((df["finished"] > 0).sum()),
        "validation_finish_rate": float((df["finished"] > 0).mean()),
        "validation_crash_rate": float((df["crashes"] > 0).mean()),
        "validation_timeout_rate": float((df["timeout"] > 0).mean()),
        "validation_mean_dense_progress": float(df["dense_progress"].mean()),
        "validation_max_dense_progress": float(df["dense_progress"].max()),
        "validation_best_finish_time": float(finished["time"].min()) if not finished.empty else float("nan"),
        "validation_mean_finish_time": float(finished["time"].mean()) if not finished.empty else float("nan"),
    }


def plot_lines(combined: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        ("best_dense_progress", "01_best_dense_progress.png", "Best dense progress [%]", False),
        ("mean_dense_progress", "02_mean_dense_progress.png", "Mean dense progress [%]", False),
        ("finish_count", "03_finish_count.png", "Finish count", False),
        ("front0_size", "04_front0_size.png", "Pareto front 0 size", False),
        ("best_time", "05_best_time_finished_only.png", "Best time [s]", True),
    ]
    for column, filename, ylabel, finished_only in plots:
        plt.figure(figsize=(12, 7))
        for label, group in combined.groupby("run_label", sort=False):
            group = group.sort_values("generation")
            values = group[column].astype(float).copy()
            if finished_only and "best_finished" in group:
                values[group["best_finished"].astype(int) <= 0] = np.nan
            plt.plot(group["generation"], values, label=label, linewidth=1.9)
        plt.xlabel("generation")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()


def plot_final_scatter(individuals: pd.DataFrame, output_dir: Path) -> None:
    if individuals.empty:
        return
    final_parts = []
    for label, group in individuals.groupby("run_label", sort=False):
        final_generation = int(group["generation"].max())
        final = group[group["generation"] == final_generation].copy()
        final_parts.append(final)
    final_df = pd.concat(final_parts, ignore_index=True)

    plt.figure(figsize=(12, 7))
    for label, group in final_df.groupby("run_label", sort=False):
        colors = np.where(group["finished"].astype(int) > 0, "tab:green", np.where(group["crashes"].astype(int) > 0, "tab:red", "tab:gray"))
        plt.scatter(group["dense_progress"], group["time"], label=label, c=colors, alpha=0.65, s=35)
    plt.xlabel("dense progress [%]")
    plt.ylabel("time [s]")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "06_final_population_progress_time_scatter.png", dpi=160)
    plt.close()


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    labels = summary["run_label"].tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    metrics = [
        ("best_finish_time", "Best finish time [s]"),
        ("final_finish_rate", "Final finish rate"),
        ("final_mean_dense_progress", "Final mean dense progress"),
        ("validation_finish_rate", "Validation finish rate"),
    ]
    for ax, (column, title) in zip(axes.ravel(), metrics):
        values = summary[column] if column in summary else np.zeros(len(summary))
        ax.bar(x, values)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "07_summary_bars.png", dpi=160)
    plt.close()


def frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    headers = [str(column) for column in frame.columns]
    rows = []
    for row in frame.itertuples(index=False, name=None):
        rows.append([
            f"{value:.3f}" if isinstance(value, float) and np.isfinite(value) else ("" if isinstance(value, float) else str(value))
            for value in row
        ])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, output_path: Path) -> None:
    lines = ["# GA MOO Run Comparison", ""]
    if summary.empty:
        lines.append("No MOO runs were found.")
    else:
        solved = summary[summary["max_best_dense_progress"] >= 100.0]
        if solved.empty:
            lines.append("- No MOO run reached finish in training.")
        else:
            fastest = solved.sort_values("best_finish_time").iloc[0]
            lines.append(
                f"- Fastest training finish: `{fastest['run_label']}` "
                f"with `{fastest['best_finish_time']:.3f}s`."
            )
            most_finishes = summary.sort_values("final_finish_count", ascending=False).iloc[0]
            lines.append(
                f"- Most final-generation finishes: `{most_finishes['run_label']}` "
                f"with `{int(most_finishes['final_finish_count'])}` / "
                f"`{int(most_finishes['population_size'])}`."
            )
        if "validation_finish_rate" in summary:
            best_validation = summary.sort_values(
                ["validation_finish_rate", "validation_mean_dense_progress"],
                ascending=[False, False],
            ).iloc[0]
            lines.append(
                f"- Best validation finish rate: `{best_validation['run_label']}` "
                f"with `{best_validation['validation_finish_rate']:.2%}`."
            )
        lines.extend(["", "## Summary", ""])
        selected = [
            "run_label",
            "objective_subset",
            "first_generation_with_any_finish",
            "best_finish_time",
            "max_finish_count",
            "final_finish_count",
            "final_finish_rate",
            "final_mean_dense_progress",
            "final_front0_size",
            "cumulative_virtual_time_hours",
            "cumulative_wall_minutes",
            "validation_finish_rate",
            "validation_mean_dense_progress",
            "validation_best_finish_time",
        ]
        selected = [column for column in selected if column in summary.columns]
        lines.append(frame_to_markdown(summary[selected]))
        lines.extend(
            [
                "",
                "## Generated Graphs",
                "",
                "- `01_best_dense_progress.png`",
                "- `02_mean_dense_progress.png`",
                "- `03_finish_count.png`",
                "- `04_front0_size.png`",
                "- `05_best_time_finished_only.png`",
                "- `06_final_population_progress_time_scatter.png`",
                "- `07_summary_bars.png`",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze local TM2D GA MOO runs.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--include", default="moo_")
    parser.add_argument("--exclude", default="_smoke")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--validate-best", action="store_true")
    parser.add_argument("--validation-episodes", type=int, default=30)
    parser.add_argument("--validation-seed", type=int, default=9100)
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else Path("Experiments/analysis") / f"ga_moo_comparison_{timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_frames: list[pd.DataFrame] = []
    individual_frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    configs: list[dict] = []
    validations: list[pd.DataFrame] = []
    for csv_path in find_generation_files(root, include=args.include, exclude=args.exclude):
        generation_df, individual_df, config = load_run(csv_path, root)
        if generation_df.empty:
            continue
        generation_frames.append(generation_df)
        if not individual_df.empty:
            individual_frames.append(individual_df)
        summary = summarize_run(generation_df, individual_df, config)
        if args.validate_best:
            validation_df = evaluate_policy(
                Path(summary["run_dir"]),
                config,
                episodes=args.validation_episodes,
                seed=args.validation_seed,
            )
            if not validation_df.empty:
                validation_df["run_label"] = summary["run_label"]
                validation_df["run_dir"] = summary["run_dir"]
                validations.append(validation_df)
                summary.update(summarize_validation(validation_df))
        summaries.append(summary)
        configs.append(config)

    combined_generation = pd.concat(generation_frames, ignore_index=True) if generation_frames else pd.DataFrame()
    combined_individual = pd.concat(individual_frames, ignore_index=True) if individual_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    validation_df = pd.concat(validations, ignore_index=True) if validations else pd.DataFrame()

    combined_generation.to_csv(output_dir / "combined_generation_metrics.csv", index=False)
    combined_individual.to_csv(output_dir / "combined_individual_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    validation_df.to_csv(output_dir / "validation_metrics.csv", index=False)
    (output_dir / "configs.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")

    if not combined_generation.empty:
        plot_lines(combined_generation, output_dir)
    if not combined_individual.empty:
        plot_final_scatter(combined_individual, output_dir)
    if not summary_df.empty:
        plot_summary(summary_df, output_dir)
    write_report(summary_df, output_dir / "REPORT.md")
    print(f"Saved MOO analysis to {output_dir}")


if __name__ == "__main__":
    main()
