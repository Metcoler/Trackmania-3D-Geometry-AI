from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
THESIS_ROOT = PACKAGE_ROOT.parent.parent
REPO_ROOT = THESIS_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TOP_VARIANTS = [
    {
        "variant": "finished_progress_time_crashes",
        "ranking_key": "(finished, progress, -time, -crashes)",
        "label": r"$(finished, progress, -time, -crashes)$",
        "short": "lex_top1",
    },
    {
        "variant": "finished_progress_crashes_time",
        "ranking_key": "(finished, progress, -crashes, -time)",
        "label": r"$(finished, progress, -crashes, -time)$",
        "short": "lex_top2",
    },
]


def frame_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        values: list[str] = []
        for column in df.columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("" if not np.isfinite(float(value)) else f"{float(value):.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def find_run_dir(variant: str) -> Path:
    root = PACKAGE_ROOT / "runs" / "lex_sweep_aabb_lidar_fixed100_seed_2026050306" / variant
    candidates = sorted(path.parent for path in root.rglob("config.json"))
    candidates = [
        path
        for path in candidates
        if (path / "generation_metrics.csv").exists() and (path / "individual_metrics.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No complete run directory found for variant {variant!r} under {root}")
    return candidates[0]


def load_run(run_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    generation = pd.read_csv(run_dir / "generation_metrics.csv")
    individual = pd.read_csv(run_dir / "individual_metrics.csv")
    return config, generation, individual


def numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float64)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def per_generation_time_stats(individual: pd.DataFrame, max_time: float) -> pd.DataFrame:
    df = individual.copy()
    df["finished_bool"] = numeric(df, "finished") > 0
    df["time_numeric"] = numeric(df, "time", default=max_time)
    df["penalized_time"] = np.where(df["finished_bool"], df["time_numeric"], float(max_time))

    rows: list[dict] = []
    for generation, group in df.groupby("generation", sort=True):
        penalized = group["penalized_time"].to_numpy(dtype=np.float64)
        finished = group[group["finished_bool"]]
        rows.append(
            {
                "generation": int(generation),
                "finish_count": int(len(finished)),
                "best_finish_time": float(finished["time_numeric"].min()) if not finished.empty else np.nan,
                "mean_penalized_time": float(np.mean(penalized)),
                "std_penalized_time": float(np.std(penalized, ddof=0)),
                "p10_penalized_time": float(np.percentile(penalized, 10)),
                "p25_penalized_time": float(np.percentile(penalized, 25)),
                "median_penalized_time": float(np.percentile(penalized, 50)),
                "p75_penalized_time": float(np.percentile(penalized, 75)),
                "p90_penalized_time": float(np.percentile(penalized, 90)),
            }
        )
    stats = pd.DataFrame(rows).sort_values("generation")
    stats["best_finish_time_so_far"] = stats["best_finish_time"].cummin()
    return stats


def summarize_variant(
    variant_info: dict,
    config: dict,
    generation: pd.DataFrame,
    individual: pd.DataFrame,
    time_stats: pd.DataFrame,
) -> dict:
    max_time = float(config.get("max_time", 30.0))
    first_finish = generation.loc[generation["finish_count"] > 0, "generation"]
    finished = individual[numeric(individual, "finished") > 0].copy()
    best_finish_time = float(finished["time"].min()) if not finished.empty else np.nan
    best_finish_generation = (
        int(finished.loc[finished["time"].idxmin(), "generation"]) if not finished.empty else np.nan
    )
    last_generation = int(generation["generation"].max())
    last50_start = max(1, last_generation - 49)
    last50_generation = generation[generation["generation"] >= last50_start]
    last50_time = time_stats[time_stats["generation"] >= last50_start]
    return {
        "variant": variant_info["variant"],
        "ranking_key": variant_info["ranking_key"],
        "max_time": max_time,
        "first_finish_generation": int(first_finish.iloc[0]) if not first_finish.empty else np.nan,
        "total_finish_individuals": int((numeric(individual, "finished") > 0).sum()),
        "best_finish_time": best_finish_time,
        "best_finish_generation": best_finish_generation,
        "last50_finish_per_generation": float(last50_generation["finish_count"].mean()),
        "last50_mean_dense_progress": float(last50_generation["mean_dense_progress"].mean()),
        "last50_mean_penalized_time": float(last50_time["mean_penalized_time"].mean()),
        "last50_best_finish_time_so_far": float(last50_time["best_finish_time_so_far"].dropna().min())
        if last50_time["best_finish_time_so_far"].notna().any()
        else np.nan,
    }


def plot_training_detail(
    variant_info: dict,
    generation: pd.DataFrame,
    output_path: Path,
) -> None:
    gen = generation.sort_values("generation").copy()
    x = gen["generation"].to_numpy(dtype=np.float64)
    p10 = gen["dense_progress_p10"].to_numpy(dtype=np.float64)
    p25 = gen["dense_progress_p25"].to_numpy(dtype=np.float64)
    p75 = gen["dense_progress_p75"].to_numpy(dtype=np.float64)
    p90 = gen["dense_progress_p90"].to_numpy(dtype=np.float64)
    mean = gen["mean_dense_progress"].to_numpy(dtype=np.float64)
    best = gen["best_dense_progress"].to_numpy(dtype=np.float64)

    fig, (ax_progress, ax_outcomes) = plt.subplots(
        2,
        1,
        figsize=(11.6, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.15]},
    )

    ax_progress.fill_between(x, p10, p90, color="#8ecae6", alpha=0.18, label="p10-p90 populácie")
    ax_progress.fill_between(x, p25, p75, color="#219ebc", alpha=0.24, label="p25-p75 populácie")
    ax_progress.plot(x, mean, color="#0b7285", linewidth=2.0, label="Priemerný progress")
    ax_progress.plot(x, best, color="#c92a2a", linewidth=2.4, label="Najlepší progress")
    ax_progress.set_ylabel("Progress [%]")
    ax_progress.set_title(f"Detailný priebeh tréningu: {variant_info['label']}")
    ax_progress.grid(True, alpha=0.25)
    ax_progress.legend(loc="lower right", fontsize=8)

    ax_outcomes.plot(x, gen["finish_count"], color="#2b8a3e", linewidth=2.0, label="Finish count")
    ax_outcomes.plot(x, gen["crash_count"], color="#e03131", linewidth=1.5, alpha=0.72, label="Crash count")
    ax_outcomes.plot(x, gen["timeout_count"], color="#868e96", linewidth=1.5, alpha=0.72, label="Timeout count")
    ax_outcomes.set_xlabel("Generácia")
    ax_outcomes.set_ylabel("Počet jedincov")
    ax_outcomes.grid(True, alpha=0.25)
    ax_outcomes.legend(loc="upper left", fontsize=8, ncol=3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_population_time_detail(
    variant_info: dict,
    time_stats: pd.DataFrame,
    output_path: Path,
    max_time: float,
) -> None:
    stats = time_stats.sort_values("generation").copy()
    x = stats["generation"].to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(11.6, 5.8))
    ax.fill_between(
        x,
        stats["p10_penalized_time"],
        stats["p90_penalized_time"],
        color="#ffd43b",
        alpha=0.18,
        label="p10-p90 penalizovaného času",
    )
    ax.fill_between(
        x,
        stats["p25_penalized_time"],
        stats["p75_penalized_time"],
        color="#fab005",
        alpha=0.24,
        label="p25-p75 penalizovaného času",
    )
    ax.plot(
        x,
        stats["mean_penalized_time"],
        color="#e67700",
        linewidth=2.0,
        label="Priemerný čas populácie (neúspech = max time)",
    )
    ax.plot(
        x,
        stats["best_finish_time_so_far"],
        color="#1864ab",
        linewidth=2.3,
        label="Najlepší finish time doteraz",
    )
    ax.scatter(
        x,
        stats["best_finish_time"],
        color="#1c7ed6",
        s=13,
        alpha=0.45,
        label="Najlepší finish time v generácii",
    )
    ax.axhline(float(max_time), color="#495057", linestyle=":", linewidth=1.2, label=f"Max time = {max_time:.0f}s")
    ax.set_title(f"Čas populácie: {variant_info['label']}")
    ax.set_xlabel("Generácia")
    ax.set_ylabel("Čas [s]")
    y_columns = [
        "p10_penalized_time",
        "p25_penalized_time",
        "p75_penalized_time",
        "p90_penalized_time",
        "mean_penalized_time",
        "best_finish_time",
        "best_finish_time_so_far",
    ]
    y_values = []
    for column in y_columns:
        values = pd.to_numeric(stats[column], errors="coerce").to_numpy(dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        if len(finite_values) > 0:
            y_values.append(finite_values)

    if y_values:
        all_values = np.concatenate(y_values)
        visible_min = float(np.nanmin(all_values))
        visible_max = float(np.nanmax(all_values))
    else:
        visible_min = 0.0
        visible_max = float(max_time)

    ax.set_ylim(max(0.0, visible_min - 1.0), max(float(max_time) + 0.5, visible_max + 0.5))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def maybe_generate_trajectory(
    run_dir: Path,
    output_dir: Path,
    *,
    copy_to_latex: bool,
) -> dict:
    status: dict = {
        "attempted": True,
        "run_dir": str(run_dir.relative_to(PACKAGE_ROOT)),
        "copied_to_latex": False,
        "reason": "",
    }
    try:
        from Map import Map
        from NeuralPolicy import NeuralPolicy
        from Experiments.tm2d_env import TM2DRewardConfig, TM2DSimEnv
        from Experiments.tm_map_plotting import render_map_background
        from Experiments.train_ga import NumpyPolicyView, make_physics_config
    except Exception as exc:  # pragma: no cover - diagnostic path
        status["reason"] = f"imports failed: {exc}"
        return status

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    policy, _ = NeuralPolicy.load(str(run_dir / "best_policy.pt"), map_location="cpu")
    with contextlib.redirect_stdout(io.StringIO()):
        env = TM2DSimEnv(
            map_name=str(config.get("map_name", "AI Training #5")),
            max_time=float(config.get("max_time", 30.0)),
            reward_config=TM2DRewardConfig(mode=str(config.get("reward_mode", "terminal_fitness"))),
            physics_config=make_physics_config(
                physics_tick_profile=str(config.get("physics_tick_profile", "supervised_v2d")),
                physics_tick_probs=config.get("physics_tick_probs"),
                fixed_fps=config.get("fixed_fps"),
            ),
            seed=int(config.get("seed", 0)),
            collision_mode=str(config.get("collision_mode", "lidar")),
            collision_distance_threshold=float(config.get("collision_distance_threshold", 2.0)),
            vertical_mode=bool(config.get("vertical_mode", False)),
            multi_surface_mode=bool(config.get("multi_surface_mode", False)),
            binary_gas_brake=bool(config.get("binary_gas_brake", True)),
            max_touches=int(config.get("max_touches", 1)),
            collision_bounce_speed_retention=float(config.get("collision_bounce_speed_retention", 0.40)),
            collision_bounce_backoff=float(config.get("collision_bounce_backoff", 0.05)),
            touch_release_clearance_threshold=float(config.get("touch_release_clearance_threshold", 0.50)),
            mask_physics_delay_observation=bool(config.get("mask_physics_delay_observation", False)),
        )
        metrics = env.rollout_policy(
            NumpyPolicyView(policy),
            collect_trajectory=True,
            trajectory_log_actions=True,
            reset_seed=int(config.get("seed", 0)),
        )

    trajectory = pd.DataFrame(metrics.get("trajectory", []))
    metrics_for_json = {key: value for key, value in metrics.items() if key != "trajectory"}
    status.update(metrics_for_json)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_csv = output_dir / "best2d_trajectory.csv"
    metrics_json = output_dir / "best2d_trajectory_metrics.json"
    plot_path = output_dir / "best2d_trajectory_diagnostic.png"
    trajectory.to_csv(trajectory_csv, index=False)
    metrics_json.write_text(json.dumps(metrics_for_json, indent=2, ensure_ascii=False), encoding="utf-8")

    if not trajectory.empty and {"x", "z"}.issubset(trajectory.columns):
        x = trajectory["x"].to_numpy(dtype=np.float64)
        z = trajectory["z"].to_numpy(dtype=np.float64)
        speed = trajectory["speed"].to_numpy(dtype=np.float64) if "speed" in trajectory.columns else np.zeros_like(x)

        fig, ax = plt.subplots(figsize=(12, 7))
        projection = None
        try:
            game_map = Map(str(config.get("map_name", "AI Training #5")))
            projection = render_map_background(ax, game_map, alpha=0.86)
        except Exception:
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.2)
            ax.set_xlabel("x")
            ax.set_ylabel("z")
        points = np.column_stack([x, z])
        if projection is not None:
            points = projection.points(points)
        if len(points) >= 2:
            segments = np.stack([points[:-1], points[1:]], axis=1)
            collection = LineCollection(segments, cmap="turbo_r", linewidth=2.2, zorder=120)
            collection.set_array(speed[:-1] if len(speed) >= len(points) else np.zeros(len(points) - 1))
            ax.add_collection(collection)
            ax.autoscale()
            fig.colorbar(collection, ax=ax, label="rýchlosť")
        ax.set_title("Diagnostická trajektória best policy z lexikografického sweep-u")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)

    valid = bool(int(metrics_for_json.get("finished", 0)) > 0 or float(metrics_for_json.get("progress", 0.0)) >= 95.0)
    status["valid_for_thesis"] = valid
    if valid and copy_to_latex and plot_path.exists():
        latex_path = THESIS_ROOT / "Latex" / "images" / "training_policy" / "lex_best2d_trajectory.png"
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plot_path, latex_path)
        status["copied_to_latex"] = True
        status["latex_path"] = str(latex_path.relative_to(THESIS_ROOT))
    elif not valid:
        status["reason"] = (
            "rollout did not reproduce a usable final trajectory; keeping only diagnostic outputs"
        )
    return status


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    trajectory_status: dict | None,
) -> None:
    lines = [
        "# Top-2 lexicographic reward detail",
        "",
        "This analysis adds per-population detail for the two strongest ranking tuples.",
        "The shaded bands are within-population spread in a single seed, not cross-seed variance.",
        "",
        "## Summary",
        "",
        frame_to_markdown(summary),
        "",
        "## Generated plots",
        "",
        "- `lex_top1_training_detail.png`",
        "- `lex_top1_population_finish_time.png`",
        "- `lex_top2_training_detail.png`",
        "- `lex_top2_population_finish_time.png`",
        "",
    ]
    if trajectory_status is not None:
        lines.extend(
            [
                "## Trajectory replay",
                "",
                "The script attempted to replay the best policy from the selected top ranking.",
                f"- finished: `{trajectory_status.get('finished', '')}`",
                f"- progress: `{trajectory_status.get('progress', '')}`",
                f"- time: `{trajectory_status.get('time', '')}`",
                f"- copied to LaTeX: `{trajectory_status.get('copied_to_latex', False)}`",
                f"- note: {trajectory_status.get('reason', '') or 'usable trajectory generated'}",
                "",
            ]
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate top-2 detailed lexicographic reward plots.")
    parser.add_argument(
        "--output-dir",
        default=str(PACKAGE_ROOT / "analysis" / "lex_sweep_aabb_lidar_fixed100_20260503" / "top2_detail"),
    )
    parser.add_argument("--copy-to-latex", action="store_true")
    parser.add_argument("--trajectory", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    run_dirs: dict[str, Path] = {}
    for variant_info in TOP_VARIANTS:
        run_dir = find_run_dir(variant_info["variant"])
        run_dirs[variant_info["variant"]] = run_dir
        config, generation, individual = load_run(run_dir)
        max_time = float(config.get("max_time", 30.0))
        time_stats = per_generation_time_stats(individual, max_time=max_time)
        summary_rows.append(summarize_variant(variant_info, config, generation, individual, time_stats))

        training_name = f"{variant_info['short']}_training_detail.png"
        time_name = f"{variant_info['short']}_population_finish_time.png"
        plot_training_detail(variant_info, generation, output_dir / training_name)
        plot_population_time_detail(variant_info, time_stats, output_dir / time_name, max_time=max_time)
        time_stats.to_csv(output_dir / f"{variant_info['short']}_population_time_stats.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "top2_detail_summary.csv", index=False)

    trajectory_status = None
    if args.trajectory:
        trajectory_status = maybe_generate_trajectory(
            run_dirs["finished_progress_time_crashes"],
            output_dir,
            copy_to_latex=bool(args.copy_to_latex),
        )
        (output_dir / "trajectory_status.json").write_text(
            json.dumps(trajectory_status, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if args.copy_to_latex:
        latex_dir = THESIS_ROOT / "Latex" / "images" / "training_policy"
        latex_dir.mkdir(parents=True, exist_ok=True)
        for name in [
            "lex_top1_training_detail.png",
            "lex_top1_population_finish_time.png",
            "lex_top2_training_detail.png",
            "lex_top2_population_finish_time.png",
        ]:
            shutil.copy2(output_dir / name, latex_dir / name)

    write_report(output_dir, summary, trajectory_status)
    print(f"Wrote top-2 detail analysis to {output_dir}")
    print(summary.to_string(index=False))
    if trajectory_status is not None:
        print("Trajectory status:")
        print(json.dumps(trajectory_status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
