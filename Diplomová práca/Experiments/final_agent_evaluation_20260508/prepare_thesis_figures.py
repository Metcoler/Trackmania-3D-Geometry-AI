from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "Diplomová práca" / "Latex" / "images" / "evaluation"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

SMALL_MAP_RUN = (
    ROOT
    / "logs"
    / "tm_finetune_runs"
    / "20260507_235715_tm_finetune_map_small_map_v2d_asphalt_h48x24_p48_src_population_gen_0080"
)

FINAL_RUNS = [
    {
        "key": "single_surface_flat",
        "title": "Rovinná trať",
        "run_dir": ROOT
        / "logs"
        / "tm_finetune_runs"
        / "20260506_004011_tm_seed_map_single_surface_flat_v2d_asphalt_h48x24_p48_src_best_model",
    },
    {
        "key": "single_surface_height",
        "title": "Výškové zmeny",
        "run_dir": ROOT
        / "logs"
        / "tm_finetune_runs"
        / "20260506_160030_tm_seed_map_single_surface_height_v3d_asphalt_h48x24_p48_src_best_model",
    },
    {
        "key": "multi_surface_flat",
        "title": "Rôzne povrchy",
        "run_dir": ROOT
        / "logs"
        / "tm_finetune_runs"
        / "20260507_090226_tm_seed_map_multi_surface_flat_v2d_surface_h48x24_p48_src_best_model",
    },
]

COMPARISON_TIMES = [
    ("Ľudský hráč", 17.89, "#2f9e44"),
    ("Náš agent", 19.68000030517578, "#c92a2a"),
    ("Bakalársky agent", 23.064, "#5c677d"),
]


def fmt(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(IMAGE_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(IMAGE_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_finishers(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "individual_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    finishers = df[df["finished"].astype(int) == 1].copy()
    finishers["time"] = finishers["time"].astype(float)
    finishers["crashes"] = finishers["crashes"].astype(int)
    return finishers.sort_values("time").reset_index(drop=True)


def plot_small_map_comparison() -> dict[str, float]:
    labels = [item[0] for item in COMPARISON_TIMES]
    times = [item[1] for item in COMPARISON_TIMES]
    colors = [item[2] for item in COMPARISON_TIMES]

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    bars = ax.barh(labels, times, color=colors, height=0.58)
    ax.invert_yaxis()
    ax.set_xlabel("Čas prejazdu [s]")
    ax.set_title("Porovnanie na novej mape small_map")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, time_value in zip(bars, times):
        ax.text(
            bar.get_width() + 0.18,
            bar.get_y() + bar.get_height() / 2,
            f"{fmt(time_value)} s",
            va="center",
            fontsize=10,
        )

    save_figure(fig, "evaluation_small_map_comparison")
    return {
        "human_time_s": COMPARISON_TIMES[0][1],
        "agent_time_s": COMPARISON_TIMES[1][1],
        "bachelor_agent_time_s": COMPARISON_TIMES[2][1],
        "agent_vs_bachelor_improvement_s": COMPARISON_TIMES[2][1] - COMPARISON_TIMES[1][1],
        "agent_gap_to_human_s": COMPARISON_TIMES[1][1] - COMPARISON_TIMES[0][1],
    }


def plot_ranked_final_times() -> list[dict[str, object]]:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=False)
    summaries: list[dict[str, object]] = []

    for ax, spec in zip(axes, FINAL_RUNS):
        finishers = load_finishers(spec["run_dir"])
        y = finishers["time"].to_numpy()
        x = range(1, len(y) + 1)

        ax.scatter(x, y, s=14, color="#2f6fbb", alpha=0.72, label="dokončujúci jedinec")
        if len(y) > 0:
            ax.scatter([1], [y[0]], s=58, color="#c92a2a", zorder=5, label="vybraný agent")
            ax.text(
                0.98,
                0.05,
                f"najlepší čas {fmt(float(y[0]))} s",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color="#c92a2a",
            )

        ax.set_title(spec["title"])
        ax.set_xlabel("Poradie podľa času")
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        summaries.append(
            {
                "key": spec["key"],
                "run_dir": str(spec["run_dir"].relative_to(ROOT)),
                "finishers": int(len(finishers)),
                "best_time_s": float(y[0]) if len(y) else None,
                "worst_finished_time_s": float(y[-1]) if len(y) else None,
                "median_finished_time_s": float(finishers["time"].median()) if len(y) else None,
            }
        )

    axes[0].set_ylabel("Čas prejazdu [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Časy dokončujúcich agentov počas finálnych tréningov", y=1.02)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, "evaluation_final_training_ranked_times")
    return summaries


def write_summary(comparison: dict[str, float], ranked: list[dict[str, object]]) -> None:
    with (PACKAGE_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "kind",
                "key",
                "time_s",
                "finishers",
                "worst_finished_time_s",
                "median_finished_time_s",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "kind": "small_map_comparison",
                "key": "human_player",
                "time_s": comparison["human_time_s"],
                "note": "human reference time supplied for small_map",
            }
        )
        writer.writerow(
            {
                "kind": "small_map_comparison",
                "key": "diploma_agent",
                "time_s": comparison["agent_time_s"],
                "note": "best individual from transferred single_surface_flat generation",
            }
        )
        writer.writerow(
            {
                "kind": "small_map_comparison",
                "key": "bachelor_agent",
                "time_s": comparison["bachelor_agent_time_s"],
                "note": "bachelor implementation reference time supplied for small_map",
            }
        )
        for row in ranked:
            writer.writerow(
                {
                    "kind": "ranked_final_training_times",
                    "key": row["key"],
                    "time_s": row["best_time_s"],
                    "finishers": row["finishers"],
                    "worst_finished_time_s": row["worst_finished_time_s"],
                    "median_finished_time_s": row["median_finished_time_s"],
                    "note": row["run_dir"],
                }
            )

    metadata = {
        "small_map_run": str(SMALL_MAP_RUN.relative_to(ROOT)),
        "comparison": comparison,
        "ranked_final_training_times": ranked,
        "figures": [
            "Diplomová práca/Latex/images/evaluation/evaluation_small_map_comparison.pdf",
            "Diplomová práca/Latex/images/evaluation/evaluation_final_training_ranked_times.pdf",
        ],
    }
    (PACKAGE_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# Final Agent Evaluation 20260508",
        "",
        "This package prepares thesis figures for chapter 8.",
        "",
        "## small_map transfer comparison",
        "",
        f"- Human reference: {fmt(comparison['human_time_s'])} s",
        f"- Diploma agent: {fmt(comparison['agent_time_s'])} s",
        f"- Bachelor agent: {fmt(comparison['bachelor_agent_time_s'])} s",
        f"- Improvement over bachelor agent: {fmt(comparison['agent_vs_bachelor_improvement_s'])} s",
        f"- Gap to human reference: {fmt(comparison['agent_gap_to_human_s'])} s",
        "",
        "## Ranked final training finishers",
        "",
    ]
    for row in ranked:
        report_lines.append(
            f"- `{row['key']}`: {row['finishers']} finishers, best {fmt(float(row['best_time_s']))} s"
        )
    (PACKAGE_DIR / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    comparison = plot_small_map_comparison()
    ranked = plot_ranked_final_times()
    write_summary(comparison, ranked)
    print(f"Wrote figures to {IMAGE_DIR}")
    print(f"Wrote summary to {PACKAGE_DIR}")


if __name__ == "__main__":
    main()
