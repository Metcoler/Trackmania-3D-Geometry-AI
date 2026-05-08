from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "Diplomová práca" / "Latex" / "images" / "evaluation"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

PLAYER_TIMES = ROOT / "Maps" / "GameFiles" / "casy_single_surface_flat.csv"
FINAL_EVALUATION_SUMMARY = (
    ROOT / "Diplomová práca" / "Experiments" / "final_agent_evaluation_20260508" / "summary.csv"
)


def fmt(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def load_agent_time() -> float:
    summary = pd.read_csv(FINAL_EVALUATION_SUMMARY)
    row = summary[
        (summary["kind"] == "ranked_final_training_times")
        & (summary["key"] == "single_surface_flat")
    ].iloc[0]
    return float(row["time_s"])


def plot_human_vs_agent_distribution() -> None:
    times = pd.read_csv(PLAYER_TIMES)
    human_times = times["time_sec"].astype(float).to_numpy()
    agent_time = load_agent_time()

    # Half-second buckets keep the histogram readable while preserving the shape of the sample.
    start = np.floor(human_times.min() * 2) / 2
    end = np.ceil(max(human_times.max(), agent_time) * 2) / 2 + 0.5
    bins = np.arange(start, end + 0.001, 0.5)

    median_time = float(np.median(human_times))
    faster_count = int(np.sum(human_times < agent_time))
    slower_or_equal_count = int(np.sum(human_times >= agent_time))

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.hist(
        human_times,
        bins=bins,
        color="#3c78b5",
        edgecolor="white",
        linewidth=1.0,
        alpha=0.88,
        label="hráčske časy",
    )
    ax.axvline(
        median_time,
        color="#2b2b2b",
        linestyle="--",
        linewidth=1.6,
        label=f"medián hráčov: {fmt(median_time)} s",
    )
    ax.axvline(
        agent_time,
        color="#c92a2a",
        linewidth=2.5,
        label=f"agent: {fmt(agent_time)} s",
    )

    ax.set_title("Poloha agenta v rozdelení hráčskych časov")
    ax.set_xlabel("Čas prejazdu [s]")
    ax.set_ylabel("Počet hráčskych časov")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=9)
    fig.tight_layout()

    fig.savefig(IMAGE_DIR / "evaluation_human_vs_agent_single_surface_flat.pdf", bbox_inches="tight")
    fig.savefig(
        IMAGE_DIR / "evaluation_human_vs_agent_single_surface_flat.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    out_csv = PACKAGE_DIR / "human_vs_agent_single_surface_flat.csv"
    pd.DataFrame(
        {
            "kind": ["agent"],
            "time_s": [agent_time],
            "human_sample_size": [len(human_times)],
            "human_min_s": [float(human_times.min())],
            "human_median_s": [median_time],
            "human_max_s": [float(human_times.max())],
            "humans_faster": [faster_count],
            "humans_slower_or_equal": [slower_or_equal_count],
        }
    ).to_csv(out_csv, index=False, encoding="utf-8")

    report = [
        "# Human vs agent comparison on single_surface_flat",
        "",
        "The player times come from Maps/GameFiles/casy_single_surface_flat.csv.",
        "The figure shows the full local sample of player times, not only the best attempts.",
        "",
        f"- Number of player times: {len(human_times)}",
        f"- Best player time: {fmt(float(human_times.min()))} s",
        f"- Median player time: {fmt(median_time)} s",
        f"- Slowest player time: {fmt(float(human_times.max()))} s",
        f"- Agent time: {fmt(agent_time)} s",
        f"- Player times faster than agent: {faster_count}",
        f"- Player times slower than or equal to agent: {slower_or_equal_count}",
    ]
    (PACKAGE_DIR / "human_vs_agent_single_surface_flat.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )


def main() -> None:
    plot_human_vs_agent_distribution()
    print("Wrote human-vs-agent distribution figure.")


if __name__ == "__main__":
    main()
