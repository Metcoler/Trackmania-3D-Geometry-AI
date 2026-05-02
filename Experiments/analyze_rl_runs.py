from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_episode_files(root: Path) -> list[Path]:
    if root.is_file() and root.name == "episode_metrics.csv":
        return [root]
    return sorted(root.rglob("episode_metrics.csv"))


def load_run(csv_path: Path, root: Path) -> tuple[pd.DataFrame, dict]:
    run_dir = csv_path.parent
    config_path = run_dir / "config.json"
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    label = run_dir.relative_to(root).as_posix() if root in run_dir.parents or run_dir == root else run_dir.name
    algorithm = str(config.get("algorithm", "RL")).upper()
    reward_mode = str(config.get("reward_mode", "unknown"))
    action_layout = str(config.get("action_layout", "unknown"))
    run_label = f"{algorithm} | {reward_mode} | {action_layout}"
    df = pd.read_csv(csv_path)
    df["run_dir"] = str(run_dir)
    df["run_label"] = run_label
    df["run_short"] = label
    df["algorithm"] = algorithm
    df["reward_mode"] = reward_mode
    df["action_layout"] = action_layout
    return df, config


def summarize_run(df: pd.DataFrame) -> dict:
    finished = df[df["finished"] > 0] if "finished" in df else df.iloc[0:0]
    last100 = df.tail(min(100, len(df)))
    return {
        "run_label": str(df["run_label"].iloc[0]),
        "run_short": str(df["run_short"].iloc[0]),
        "episodes": int(len(df)),
        "max_dense_progress": float(df["dense_progress"].max()),
        "max_progress": float(df["progress"].max()),
        "last100_mean_dense": float(last100["dense_progress"].mean()) if len(last100) else 0.0,
        "last100_mean_reward": float(last100["episode_reward"].mean()) if len(last100) else 0.0,
        "finish_count": int((df["finished"] > 0).sum()) if "finished" in df else 0,
        "crash_count": int((df["crashes"] > 0).sum()) if "crashes" in df else 0,
        "timeout_count": int((df["timeout"] > 0).sum()) if "timeout" in df else 0,
        "first_finish_episode": int(finished["episode"].iloc[0]) if len(finished) else -1,
        "best_finish_time": float(finished["time"].min()) if len(finished) else float("nan"),
        "best_dense_episode": int(df.loc[df["dense_progress"].idxmax(), "episode"]),
        "best_dense_time": float(df.loc[df["dense_progress"].idxmax(), "time"]),
        "source_csv": str(Path(df["run_dir"].iloc[0]) / "episode_metrics.csv"),
    }


def plot_lines(
    combined: pd.DataFrame,
    output_path: Path,
    y: str,
    title: str,
    ylabel: str,
    rolling: int | None = None,
    cumulative_best: bool = False,
) -> None:
    plt.figure(figsize=(12, 7))
    for label, group in combined.groupby("run_label", sort=False):
        group = group.sort_values("episode")
        values = group[y].astype(float)
        if cumulative_best:
            values = values.cummax()
        if rolling is not None and rolling > 1:
            values = values.rolling(rolling, min_periods=1).mean()
        plt.plot(group["episode"], values, label=label, linewidth=1.8)
    plt.title(title)
    plt.xlabel("episode")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    labels = summary["run_label"].tolist()
    x = range(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    metrics = [
        ("max_dense_progress", "Max dense progress"),
        ("last100_mean_dense", "Last 100 mean dense progress"),
        ("finish_count", "Finish count"),
        ("best_finish_time", "Best finish time"),
    ]
    for ax, (column, title) in zip(axes, metrics):
        ax.bar(x, summary[column])
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_report(summary: pd.DataFrame, output_path: Path) -> None:
    def frame_to_markdown(frame: pd.DataFrame) -> str:
        if frame.empty:
            return ""
        headers = [str(column) for column in frame.columns]
        rows = [
            [
                f"{value:.3f}" if isinstance(value, float) else str(value)
                for value in row
            ]
            for row in frame.itertuples(index=False, name=None)
        ]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    lines = ["# RL Run Comparison", ""]
    if summary.empty:
        lines.append("No `episode_metrics.csv` files were found.")
    else:
        best_dense = summary.sort_values("max_dense_progress", ascending=False).iloc[0]
        lines.append(
            f"- Best max dense progress: `{best_dense['run_label']}` "
            f"with {best_dense['max_dense_progress']:.2f}%."
        )
        finished = summary[summary["finish_count"] > 0]
        if not finished.empty:
            best_time = finished.sort_values("best_finish_time", ascending=True).iloc[0]
            lines.append(
                f"- Best finish time: `{best_time['run_label']}` "
                f"with {best_time['best_finish_time']:.3f}s."
            )
        else:
            lines.append("- No compared run reached finish yet.")
        lines.extend(["", "## Summary", ""])
        lines.append(frame_to_markdown(summary))
        lines.extend(
            [
                "",
                "## Generated Graphs",
                "",
                "- `01_cumulative_best_dense_progress.png`",
                "- `02_rolling_dense_progress.png`",
                "- `03_rolling_episode_reward.png`",
                "- `04_summary_bars.png`",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze local TM2D RL runs.")
    parser.add_argument("--root", required=True, help="Run root containing one or more episode_metrics.csv files.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)
    output_dir = Path(args.output_dir) if args.output_dir else Path("Experiments/analysis") / f"rl_comparison_{timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    configs = []
    for csv_path in find_episode_files(root):
        df, config = load_run(csv_path, root)
        if len(df) == 0:
            continue
        frames.append(df)
        configs.append(config)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        summary = pd.DataFrame([summarize_run(frame) for frame in frames])
    else:
        combined = pd.DataFrame()
        summary = pd.DataFrame()

    combined.to_csv(output_dir / "combined_episode_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "configs.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")

    if not combined.empty:
        plot_lines(
            combined,
            output_dir / "01_cumulative_best_dense_progress.png",
            y="dense_progress",
            title="Cumulative Best Dense Progress",
            ylabel="dense progress [%]",
            cumulative_best=True,
        )
        plot_lines(
            combined,
            output_dir / "02_rolling_dense_progress.png",
            y="dense_progress",
            title="Dense Progress Rolling Mean",
            ylabel="dense progress [%]",
            rolling=50,
        )
        plot_lines(
            combined,
            output_dir / "03_rolling_episode_reward.png",
            y="episode_reward",
            title="Episode Reward Rolling Mean",
            ylabel="episode reward",
            rolling=50,
        )
        plot_summary(summary, output_dir / "04_summary_bars.png")

    write_report(summary, output_dir / "REPORT.md")
    print(f"Saved RL analysis to {output_dir}")


if __name__ == "__main__":
    main()
