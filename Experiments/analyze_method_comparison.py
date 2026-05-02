from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiments.tm2d_env import TM2DRewardConfig, TM2DSimEnv
from Experiments.train_ga import NumpyPolicyView
from NeuralPolicy import NeuralPolicy


def fmt(value: float | int | str | None, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def frame_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def find_run_with_config(root: Path, key_name: str, key_value: str) -> Path | None:
    for config_path in sorted(root.rglob("config.json")):
        if "_smoke" in str(config_path):
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if str(config.get(key_name, "")) == key_value:
            model_path = config_path.parent / "best_policy.pt"
            if model_path.exists():
                return config_path.parent
    return None


def evaluate_policy(run_dir: Path, episodes: int, seed: int) -> pd.DataFrame:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    policy, _ = NeuralPolicy.load(str(run_dir / "best_policy.pt"), map_location="cpu")
    rows: list[dict] = []
    for episode in range(1, episodes + 1):
        with contextlib.redirect_stdout(io.StringIO()):
            env = TM2DSimEnv(
                map_name=str(config.get("map_name", "AI Training #5")),
                max_time=float(config.get("max_time", 45.0)),
                reward_config=TM2DRewardConfig(mode=str(config.get("reward_mode", "progress_primary_delta"))),
                seed=seed + episode,
                collision_mode=str(config.get("collision_mode", "corners")),
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
                "steps": int(metrics.get("steps", 0)),
            }
        )
    return pd.DataFrame(rows)


def validation_summary(df: pd.DataFrame) -> dict:
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


def load_ga_rows(path: Path, runs_root: Path, validate_episodes: int, seed: int) -> tuple[list[dict], pd.DataFrame]:
    if not path.exists():
        return [], pd.DataFrame()
    df = pd.read_csv(path)
    rows: list[dict] = []
    validation_parts: list[pd.DataFrame] = []
    for _, row in df.iterrows():
        key = str(row["ranking_key"])
        validation = {}
        run_dir = find_run_with_config(runs_root, "ranking_key", key)
        if run_dir and validate_episodes > 0:
            eval_df = evaluate_policy(run_dir, validate_episodes, seed)
            eval_df["method"] = "GA Lexicographic"
            eval_df["variant"] = key
            eval_df["run_dir"] = str(run_dir)
            validation_parts.append(eval_df)
            validation = validation_summary(eval_df)
        rows.append(
            {
                "method": "GA Lexicographic",
                "variant": key,
                "training_units": int(row["generations"]),
                "first_success_unit": int(row["first_generation_with_any_finish"]),
                "best_finish_time": float(row["best_finish_time"]) if pd.notna(row["best_finish_time"]) else float("nan"),
                "best_finish_unit": int(row["best_finish_generation"]),
                "final_finish_rate": float(row["final_finished_rate"]),
                "final_finish_count": int(row["final_finish_count"]),
                "max_finish_count": int(row["max_finish_count"]),
                "final_mean_dense_progress": float(row.get("final_best_dense_progress", np.nan)),
                "wall_minutes": float(row["cumulative_wall_minutes"]),
                "virtual_time_hours": float(row["cumulative_virtual_time_hours"]),
                **validation,
            }
        )
    validation_df = pd.concat(validation_parts, ignore_index=True) if validation_parts else pd.DataFrame()
    return rows, validation_df


def load_moo_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows: list[dict] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "method": "GA MOO",
                "variant": str(row["objective_subset"]),
                "training_units": int(row["generations"]),
                "first_success_unit": int(row["first_generation_with_any_finish"]),
                "best_finish_time": float(row["best_finish_time"]) if pd.notna(row["best_finish_time"]) else float("nan"),
                "best_finish_unit": int(row["best_finish_generation"]),
                "final_finish_rate": float(row["final_finish_rate"]),
                "final_finish_count": int(row["final_finish_count"]),
                "max_finish_count": int(row["max_finish_count"]),
                "final_mean_dense_progress": float(row["final_mean_dense_progress"]),
                "wall_minutes": float(row["cumulative_wall_minutes"]),
                "virtual_time_hours": float(row["cumulative_virtual_time_hours"]),
                "validation_episodes": int(row.get("validation_episodes", 0)) if pd.notna(row.get("validation_episodes", np.nan)) else 0,
                "validation_finish_count": int(row.get("validation_finish_count", 0)) if pd.notna(row.get("validation_finish_count", np.nan)) else 0,
                "validation_finish_rate": float(row.get("validation_finish_rate", np.nan)),
                "validation_crash_rate": float(row.get("validation_crash_rate", np.nan)),
                "validation_timeout_rate": float(row.get("validation_timeout_rate", np.nan)),
                "validation_mean_dense_progress": float(row.get("validation_mean_dense_progress", np.nan)),
                "validation_max_dense_progress": float(row.get("validation_max_dense_progress", np.nan)),
                "validation_best_finish_time": float(row.get("validation_best_finish_time", np.nan)),
                "validation_mean_finish_time": float(row.get("validation_mean_finish_time", np.nan)),
            }
        )
    return rows


def load_rl_rows(summary_paths: list[Path], eval_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in summary_paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            label = str(row["run_label"])
            eval_path = None
            if label.startswith("PPO |") and "gas_brake" not in label:
                eval_path = eval_root / "ppo_best_deterministic_eval.csv"
            elif label.startswith("SAC |"):
                eval_path = eval_root / "sac_best_deterministic_eval.csv"
            elif label.startswith("TD3 |"):
                eval_path = eval_root / "td3_best_deterministic_eval.csv"
            eval_summary = {}
            if eval_path and eval_path.exists():
                eval_summary = validation_summary(pd.read_csv(eval_path))
            rows.append(
                {
                    "method": "RL",
                    "variant": label,
                    "training_units": int(row["episodes"]),
                    "first_success_unit": int(row["first_finish_episode"]),
                    "best_finish_time": float(row["best_finish_time"]) if pd.notna(row["best_finish_time"]) else float("nan"),
                    "best_finish_unit": int(row["first_finish_episode"]),
                    "final_finish_rate": float(row["finish_count"]) / max(1.0, float(row["episodes"])),
                    "final_finish_count": int(row["finish_count"]),
                    "max_finish_count": int(row["finish_count"]),
                    "final_mean_dense_progress": float(row["last100_mean_dense"]),
                    "wall_minutes": float("nan"),
                    "virtual_time_hours": float("nan"),
                    **eval_summary,
                }
            )
    return rows


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    plot_df = summary.copy()
    plot_df["short_label"] = plot_df["method"] + "\n" + plot_df["variant"].str.replace(",", ", ", regex=False)
    plot_df["validation_or_final_finish_rate"] = plot_df["validation_finish_rate"].fillna(plot_df["final_finish_rate"])
    plots = [
        ("best_finish_time", "01_best_finish_time.png", "Best finish time [s]", True),
        ("first_success_unit", "02_first_success_unit.png", "First finish generation/episode", True),
        ("validation_or_final_finish_rate", "03_finish_rate.png", "Validation finish rate if available, else final rate", False),
        ("validation_mean_dense_progress", "04_validation_dense_progress.png", "Validation mean dense progress [%]", False),
    ]
    for column, filename, ylabel, lower_is_better in plots:
        data = plot_df[np.isfinite(plot_df[column].astype(float))]
        if data.empty:
            continue
        data = data.sort_values(column, ascending=lower_is_better)
        colors = data["method"].map({"GA Lexicographic": "#4c78a8", "GA MOO": "#f58518", "RL": "#54a24b"}).fillna("#888888")
        plt.figure(figsize=(13, 7))
        plt.bar(range(len(data)), data[column].astype(float), color=colors)
        plt.xticks(range(len(data)), data["short_label"], rotation=35, ha="right", fontsize=8)
        plt.ylabel(ylabel)
        plt.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    display = summary.copy()
    columns = [
        "method",
        "variant",
        "first_success_unit",
        "best_finish_time",
        "final_finish_count",
        "final_finish_rate",
        "validation_finish_count",
        "validation_finish_rate",
        "validation_best_finish_time",
        "validation_mean_dense_progress",
        "wall_minutes",
        "virtual_time_hours",
    ]
    display = display[[col for col in columns if col in display.columns]].copy()
    for col in display.columns:
        if col not in {"method", "variant"}:
            display[col] = display[col].map(lambda value: fmt(value))

    best_training = summary[np.isfinite(summary["best_finish_time"].astype(float))].sort_values("best_finish_time").head(1)
    best_validation = summary[np.isfinite(summary["validation_finish_rate"].astype(float))].sort_values(
        ["validation_finish_rate", "validation_best_finish_time"],
        ascending=[False, True],
    ).head(1)
    lines = [
        "# GA vs RL vs GA MOO Comparison",
        "",
        "This report compares local TM2D experiments on `AI Training #5` using the dense-progress racing setup.",
        "",
    ]
    if not best_training.empty:
        row = best_training.iloc[0]
        lines.append(
            f"- Fastest training finish: `{row['method']} | {row['variant']}` with `{fmt(row['best_finish_time'])}s`."
        )
    if not best_validation.empty:
        row = best_validation.iloc[0]
        lines.append(
            f"- Best validation finish rate: `{row['method']} | {row['variant']}` with `{fmt(100.0 * row['validation_finish_rate'], 1)}%`."
        )
    lines.extend(
        [
            "",
            "## Summary Table",
            "",
            frame_to_markdown(display),
            "",
            "## Interpretation",
            "",
            "- `GA Lexicographic` with `(finished, progress, -time)` remains the fastest simple tuple during training, but the 30-episode validation shows that adding crash awareness improves reproducibility.",
            "- `GA Lexicographic` with `(finished, progress, -time, -crashes)` and `(finished, progress, -crashes, -time)` validated more reliably, trading a little lap-time pressure for safer policies.",
            "- `GA MOO` is competitive only when the objective set includes `neg_distance`. Without distance, it can discover progress but validates poorly, which points to unstable/risky trajectories.",
            "- `PPO` with `gas_steer` is the only RL setup that solved the map robustly in deterministic validation, but it is slower and more sample-expensive than GA.",
            "- `SAC` did not leave the early local minimum in this setup. `TD3` found a finish once during training but did not validate as a stable policy.",
            "- `PPO` with `gas_brake_steer` was a negative control in this batch: adding brake enlarged the action problem and the short run collapsed near the start.",
            "",
            "## Generated Graphs",
            "",
            "- `01_best_finish_time.png`",
            "- `02_first_success_unit.png`",
            "- `03_finish_rate.png`",
            "- `04_validation_dense_progress.png`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare local TM2D GA, RL and GA MOO experiments.")
    parser.add_argument("--output-dir", default="Experiments/analysis/ga_rl_moo_comparison_20260501")
    parser.add_argument("--ga-summary", default="Experiments/analysis/ga_lexicographic_reward_comparison_20260501/summary.csv")
    parser.add_argument("--moo-summary", default="Experiments/analysis/ga_moo_comparison_20260501_main_only/summary.csv")
    parser.add_argument("--rl-summary", default="Experiments/analysis/rl_comparison_20260501_202036/summary.csv")
    parser.add_argument("--rl-brake-summary", default="Experiments/analysis/ppo_gas_brake_steer_delta_aitraining5_20260501_204641_2h/summary.csv")
    parser.add_argument("--rl-eval-root", default="Experiments/analysis/rl_comparison_20260501_202036")
    parser.add_argument("--runs-root", default="Experiments/runs")
    parser.add_argument("--validate-ga-episodes", type=int, default=30)
    parser.add_argument("--validation-seed", type=int, default=12300)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    ga_rows, ga_validation = load_ga_rows(Path(args.ga_summary), Path(args.runs_root), args.validate_ga_episodes, args.validation_seed)
    rows.extend(ga_rows)
    rows.extend(load_moo_rows(Path(args.moo_summary)))
    rows.extend(load_rl_rows([Path(args.rl_summary), Path(args.rl_brake_summary)], Path(args.rl_eval_root)))
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "method_summary.csv", index=False)
    if not ga_validation.empty:
        ga_validation.to_csv(output_dir / "ga_best_policy_validation.csv", index=False)
    plot_summary(summary, output_dir)
    write_report(summary, output_dir)
    print(f"Saved method comparison to {output_dir}")


if __name__ == "__main__":
    main()
