from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("Experiments/runs_ga_training_improvements/seed_2026050407")
DEFAULT_OUTPUT = Path("Experiments/analysis/ga_supervised_seeded_20260505")

FOCUS_EXPERIMENTS = [
    "exp00_base_fixed100",
    "exp06_supervised_seeded_fixed100",
    "exp06b_supervised_seeded_dense_fixed100",
]

LABELS = {
    "exp00_base_fixed100": "baseline random",
    "exp06_supervised_seeded_fixed100": "supervised seeded sparse",
    "exp06b_supervised_seeded_dense_fixed100": "supervised seeded dense",
}

COLORS = {
    "exp00_base_fixed100": "#4c78a8",
    "exp06_supervised_seeded_fixed100": "#f58518",
    "exp06b_supervised_seeded_dense_fixed100": "#54a24b",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def first_present(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def canonicalize_generation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    progress_aliases = {
        "best_progress": ("best_progress", "best_dense_progress"),
        "mean_progress": ("mean_progress", "mean_dense_progress"),
        "progress_p10": ("progress_p10", "dense_progress_p10"),
        "progress_p25": ("progress_p25", "dense_progress_p25"),
        "progress_median": ("progress_median", "dense_progress_median"),
        "progress_p75": ("progress_p75", "dense_progress_p75"),
        "progress_p90": ("progress_p90", "dense_progress_p90"),
    }
    for target, aliases in progress_aliases.items():
        source = first_present(out, aliases)
        if source is not None:
            out[target] = pd.to_numeric(out[source], errors="coerce")
    return out


def canonicalize_individual(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    source = first_present(out, ("progress", "dense_progress"))
    if source is not None:
        out["progress"] = pd.to_numeric(out[source], errors="coerce")
    return out


def find_run_dir(exp_dir: Path) -> Path | None:
    candidates = [p for p in exp_dir.rglob("generation_metrics.csv") if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].parent


def load_runs(root: Path) -> dict[str, dict[str, object]]:
    runs: dict[str, dict[str, object]] = {}
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        run_dir = find_run_dir(exp_dir)
        if run_dir is None:
            continue
        generation = canonicalize_generation(read_csv(run_dir / "generation_metrics.csv"))
        individual = canonicalize_individual(read_csv(run_dir / "individual_metrics.csv"))
        config = {}
        config_path = run_dir / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                config = {}
        runs[exp_dir.name] = {
            "run_dir": run_dir,
            "generation": generation,
            "individual": individual,
            "config": config,
        }
    return runs


def summarize_run(name: str, run: dict[str, object]) -> dict[str, object]:
    generation = run["generation"]
    individual = run["individual"]
    config = run["config"]
    assert isinstance(generation, pd.DataFrame)
    assert isinstance(individual, pd.DataFrame)
    assert isinstance(config, dict)

    pop = float(generation["population_size"].iloc[0]) if "population_size" in generation and len(generation) else 0.0
    last50 = generation.tail(min(50, len(generation)))
    finish_generations = generation[generation.get("finish_count", 0) > 0]
    finished_individuals = individual[individual.get("finished", 0) == 1] if not individual.empty else pd.DataFrame()
    best_time = float(finished_individuals["time"].min()) if not finished_individuals.empty else math.nan
    best_time_generation = math.nan
    if not finished_individuals.empty and "generation" in finished_individuals:
        best_time_generation = int(finished_individuals.loc[finished_individuals["time"].idxmin(), "generation"])

    initial_summary = config.get("initial_population_summary") or {}
    tiers = initial_summary.get("tiers") or []
    tier_text = "; ".join(
        f"{tier.get('count')}x p={tier.get('mutation_prob')} s={tier.get('mutation_sigma')}"
        for tier in tiers
    )

    total_evals = float(len(individual)) if len(individual) else float(len(generation) * pop)
    return {
        "experiment": name,
        "label": LABELS.get(name, name),
        "generations": int(len(generation)),
        "population_size": int(pop) if pop else math.nan,
        "first_finish_generation": int(finish_generations["generation"].iloc[0]) if len(finish_generations) else math.nan,
        "total_finish": int(generation["finish_count"].sum()) if "finish_count" in generation else 0,
        "finish_rate_all": float(generation["finish_count"].sum() / total_evals) if total_evals else math.nan,
        "best_time": best_time,
        "best_time_generation": best_time_generation,
        "last50_finish_rate": float(last50["finish_count"].sum() / (len(last50) * pop)) if pop and len(last50) else math.nan,
        "last50_mean_progress": float(last50["mean_progress"].mean()) if "mean_progress" in last50 else math.nan,
        "last50_best_progress": float(last50["best_progress"].mean()) if "best_progress" in last50 else math.nan,
        "last50_crash_rate": float(last50["crash_count"].sum() / (len(last50) * pop)) if pop and "crash_count" in last50 else math.nan,
        "last50_timeout_rate": float(last50["timeout_count"].sum() / (len(last50) * pop)) if pop and "timeout_count" in last50 else math.nan,
        "gen1_best_progress": float(generation["best_progress"].iloc[0]) if "best_progress" in generation and len(generation) else math.nan,
        "gen1_mean_progress": float(generation["mean_progress"].iloc[0]) if "mean_progress" in generation and len(generation) else math.nan,
        "max_best_progress": float(generation["best_progress"].max()) if "best_progress" in generation else math.nan,
        "cached_evaluations": int(generation["cached_evaluations"].sum()) if "cached_evaluations" in generation else 0,
        "initial_population_source": config.get("initial_population_source", "random"),
        "initial_noise_mode": config.get("initial_model_noise_mode", "sparse_or_default"),
        "initial_tiers": tier_text,
        "run_dir": str(run["run_dir"]),
    }


def summarize_phase(name: str, run: dict[str, object]) -> list[dict[str, object]]:
    generation = run["generation"]
    individual = run["individual"]
    assert isinstance(generation, pd.DataFrame)
    assert isinstance(individual, pd.DataFrame)
    phases = [(1, 50), (51, 100), (101, 150), (151, 200)]
    rows: list[dict[str, object]] = []
    pop = float(generation["population_size"].iloc[0]) if "population_size" in generation and len(generation) else 0.0
    for start, end in phases:
        g = generation[(generation["generation"] >= start) & (generation["generation"] <= end)]
        ind = individual[(individual["generation"] >= start) & (individual["generation"] <= end)] if not individual.empty else pd.DataFrame()
        finished = ind[ind.get("finished", 0) == 1] if not ind.empty else pd.DataFrame()
        rows.append(
            {
                "experiment": name,
                "phase": f"{start}-{end}",
                "mean_progress": float(g["mean_progress"].mean()) if "mean_progress" in g and len(g) else math.nan,
                "best_progress": float(g["best_progress"].max()) if "best_progress" in g and len(g) else math.nan,
                "finish_rate": float(g["finish_count"].sum() / (len(g) * pop)) if pop and len(g) else math.nan,
                "crash_rate": float(g["crash_count"].sum() / (len(g) * pop)) if pop and len(g) else math.nan,
                "timeout_rate": float(g["timeout_count"].sum() / (len(g) * pop)) if pop and len(g) else math.nan,
                "best_time": float(finished["time"].min()) if not finished.empty else math.nan,
            }
        )
    return rows


def summarize_thresholds(name: str, run: dict[str, object], thresholds: Iterable[float]) -> list[dict[str, object]]:
    generation = run["generation"]
    assert isinstance(generation, pd.DataFrame)
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        hit = generation[generation["best_progress"] >= threshold]
        rows.append(
            {
                "experiment": name,
                "threshold_progress": threshold,
                "first_generation": int(hit["generation"].iloc[0]) if len(hit) else math.nan,
            }
        )
    return rows


def save_progress_plot(runs: dict[str, dict[str, object]], names: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    for name in names:
        generation = runs[name]["generation"]
        assert isinstance(generation, pd.DataFrame)
        color = COLORS.get(name)
        label = LABELS.get(name, name)
        ax.plot(generation["generation"], generation["best_progress"], color=color, linewidth=2.3, label=f"{label} best")
        ax.plot(generation["generation"], generation["mean_progress"], color=color, linewidth=1.5, linestyle="--", alpha=0.9, label=f"{label} mean")
        if {"progress_p10", "progress_p90"}.issubset(generation.columns):
            ax.fill_between(
                generation["generation"],
                generation["progress_p10"],
                generation["progress_p90"],
                color=color,
                alpha=0.10,
                linewidth=0,
            )
    ax.set_title("Supervised-seeded GA: progress over generations")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Progress [%]")
    ax.set_ylim(0, 103)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_finish_plot(runs: dict[str, dict[str, object]], names: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    for name in names:
        generation = runs[name]["generation"]
        assert isinstance(generation, pd.DataFrame)
        pop = generation["population_size"].iloc[0]
        rate = generation["finish_count"].rolling(10, min_periods=1).mean() / pop
        ax.plot(generation["generation"], rate, color=COLORS.get(name), linewidth=2.3, label=LABELS.get(name, name))
    ax.set_title("Rolling finish rate, 10 generation window")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Finish rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_best_time_plot(runs: dict[str, dict[str, object]], names: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    for name in names:
        individual = runs[name]["individual"]
        generation = runs[name]["generation"]
        assert isinstance(individual, pd.DataFrame)
        assert isinstance(generation, pd.DataFrame)
        finished = individual[individual.get("finished", 0) == 1].copy()
        if finished.empty:
            continue
        per_gen = finished.groupby("generation")["time"].min().reindex(generation["generation"])
        best_so_far = per_gen.cummin()
        ax.plot(generation["generation"], best_so_far, color=COLORS.get(name), linewidth=2.3, label=LABELS.get(name, name))
    ax.set_title("Best finish time so far")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Time [s]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_initial_hist(runs: dict[str, dict[str, object]], names: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    bins = np.linspace(0, 15, 31)
    for name in names:
        individual = runs[name]["individual"]
        assert isinstance(individual, pd.DataFrame)
        gen1 = individual[individual["generation"] == 1]
        ax.hist(
            gen1["progress"],
            bins=bins,
            histtype="stepfilled",
            alpha=0.28,
            color=COLORS.get(name),
            label=LABELS.get(name, name),
        )
        ax.hist(gen1["progress"], bins=bins, histtype="step", linewidth=1.8, color=COLORS.get(name))
    ax.set_title("Initial population progress distribution")
    ax.set_xlabel("Generation 1 progress [%]")
    ax.set_ylabel("Individuals")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_outcome_bars(summary: pd.DataFrame, names: list[str], out: Path) -> None:
    subset = summary.set_index("experiment").loc[names].reset_index()
    labels = [LABELS.get(name, name) for name in subset["experiment"]]
    x = np.arange(len(subset))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    ax.bar(x - width, subset["last50_finish_rate"], width, label="finish", color="#54a24b")
    ax.bar(x, subset["last50_crash_rate"], width, label="crash", color="#e45756")
    ax.bar(x + width, subset["last50_timeout_rate"], width, label="timeout", color="#4c78a8")
    ax.set_title("Last 50 generations: outcome rates")
    ax.set_ylabel("Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_phase_heatmap(phase: pd.DataFrame, names: list[str], out: Path) -> None:
    pivot = phase[phase["experiment"].isin(names)].pivot(index="experiment", columns="phase", values="mean_progress")
    pivot = pivot.loc[names]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    image = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=0, vmax=100)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([LABELS.get(name, name) for name in pivot.index])
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="white" if value < 55 else "black", fontsize=9)
    ax.set_title("Mean progress by training phase")
    fig.colorbar(image, ax=ax, label="Mean progress [%]")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_context_bars(summary: pd.DataFrame, out: Path) -> None:
    context = summary[summary["experiment"].str.startswith("exp")].copy()
    context = context.sort_values("best_time", na_position="last")
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
    colors = ["#54a24b" if "supervised_seeded_dense" in name else "#f58518" if "supervised_seeded" in name else "#9ecae9" for name in context["experiment"]]
    values = context["best_time"].fillna(35)
    ax.barh(context["experiment"], values, color=colors)
    ax.axvline(30, color="black", linestyle=":", linewidth=1)
    ax.set_title("Training-improvements context: best finish time")
    ax.set_xlabel("Best finish time [s], missing shown as 35s")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def write_report(output_dir: Path, summary: pd.DataFrame, phase: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    indexed = summary.set_index("experiment")
    base = indexed.loc["exp00_base_fixed100"]
    sparse = indexed.loc["exp06_supervised_seeded_fixed100"]
    dense = indexed.loc["exp06b_supervised_seeded_dense_fixed100"]
    report = f"""# GA Supervised-Seeded Deep Analysis 20260505

This analysis compares the random GA baseline with two behavior-cloning initialized populations:

- `exp00_base_fixed100`: random initial population.
- `exp06_supervised_seeded_fixed100`: one exact BC copy plus sparse mutation tiers.
- `exp06b_supervised_seeded_dense_fixed100`: one exact BC copy plus dense weight-noise tiers.

## Verdict

Dense-noise supervised seeding is a strong positive result. It found the first finisher at generation {dense['first_finish_generation']:.0f}, slightly earlier than the random baseline at generation {base['first_finish_generation']:.0f}, and reached the best time in this comparison: {dense['best_time']:.2f}s at generation {dense['best_time_generation']:.0f}. The random baseline remained a little more stable in the last 50 generations, but its best time was slower at {base['best_time']:.2f}s.

The original sparse supervised seeding is a negative result. Although generation 1 starts with higher mean progress ({sparse['gen1_mean_progress']:.2f}% vs baseline {base['gen1_mean_progress']:.2f}%), the population stays too concentrated around the BC policy and only finds finishers very late, at generation {sparse['first_finish_generation']:.0f}. Its total finish count is only {sparse['total_finish']:.0f}.

## Key Numbers

| Experiment | First finish gen | Total finishes | Best time | Last50 finish rate | Last50 mean progress | Last50 crash rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline random | {base['first_finish_generation']:.0f} | {base['total_finish']:.0f} | {base['best_time']:.2f}s | {base['last50_finish_rate']:.3f} | {base['last50_mean_progress']:.2f}% | {base['last50_crash_rate']:.3f} |
| Supervised sparse | {sparse['first_finish_generation']:.0f} | {sparse['total_finish']:.0f} | {sparse['best_time']:.2f}s | {sparse['last50_finish_rate']:.3f} | {sparse['last50_mean_progress']:.2f}% | {sparse['last50_crash_rate']:.3f} |
| Supervised dense | {dense['first_finish_generation']:.0f} | {dense['total_finish']:.0f} | {dense['best_time']:.2f}s | {dense['last50_finish_rate']:.3f} | {dense['last50_mean_progress']:.2f}% | {dense['last50_crash_rate']:.3f} |

## Interpretation

- BC initialization alone is not enough. The BC policy starts above random at generation 1, but it is still far from a robust lap-completing solution.
- Sparse mutation around the BC model appears too conservative. It preserves the prior, but does not create enough behavioral diversity to escape local failure modes.
- Dense mutation is the useful version. Mutating every weight with several sigma tiers creates a population that keeps some BC structure while still exploring broadly enough for GA selection to improve it.
- For thesis use, present this as a hybrid method: behavior cloning initialization + evolutionary fine-tuning. It uses extra human demonstration data, so it should not be framed as a pure GA improvement.

## Recommended Thesis Status

- `exp06b_supervised_seeded_dense_fixed100`: thesis-grade positive hybrid experiment.
- `exp06_supervised_seeded_fixed100`: useful negative control showing why naive/sparse BC seeding is not enough.
- `exp00_base_fixed100`: comparison baseline from the same training-improvements sweep.

## Generated Files

- `summary.csv`
- `phase_summary.csv`
- `threshold_summary.csv`
- `01_progress_comparison.png`
- `02_finish_rate_rolling.png`
- `03_best_finish_time_so_far.png`
- `04_last50_outcome_rates.png`
- `05_initial_population_progress_distribution.png`
- `06_phase_mean_progress_heatmap.png`
- `07_training_improvements_best_time_context.png`
"""
    output_dir.joinpath("DEEP_REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep analysis for supervised-seeded TM2D GA experiments.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(root)
    missing = [name for name in FOCUS_EXPERIMENTS if name not in runs]
    if missing:
        raise FileNotFoundError(f"Missing expected experiments under {root}: {missing}")

    summary = pd.DataFrame([summarize_run(name, run) for name, run in runs.items()])
    summary = summary.sort_values(["experiment"]).reset_index(drop=True)
    phase = pd.DataFrame([row for name, run in runs.items() for row in summarize_phase(name, run)])
    thresholds = pd.DataFrame(
        [row for name, run in runs.items() for row in summarize_thresholds(name, run, [5, 10, 25, 50, 75, 90, 100])]
    )

    summary.to_csv(output_dir / "summary.csv", index=False)
    phase.to_csv(output_dir / "phase_summary.csv", index=False)
    thresholds.to_csv(output_dir / "threshold_summary.csv", index=False)

    focus = FOCUS_EXPERIMENTS
    save_progress_plot(runs, focus, output_dir / "01_progress_comparison.png")
    save_finish_plot(runs, focus, output_dir / "02_finish_rate_rolling.png")
    save_best_time_plot(runs, focus, output_dir / "03_best_finish_time_so_far.png")
    save_outcome_bars(summary, focus, output_dir / "04_last50_outcome_rates.png")
    save_initial_hist(runs, focus, output_dir / "05_initial_population_progress_distribution.png")
    save_phase_heatmap(phase, focus, output_dir / "06_phase_mean_progress_heatmap.png")
    save_context_bars(summary, output_dir / "07_training_improvements_best_time_context.png")
    write_report(output_dir, summary, phase, thresholds)

    print(f"Wrote {output_dir / 'DEEP_REPORT.md'}")
    print(f"Wrote {output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
