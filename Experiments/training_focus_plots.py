from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class FocusRun:
    run_dir: Path
    label: str
    generation: pd.DataFrame
    individual: pd.DataFrame


PROGRESS_STAT_SUFFIXES = ("min", "p10", "p25", "median", "mean", "std", "p75", "p90", "max")


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _first_present(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _copy_if_present(target: pd.DataFrame, source: pd.DataFrame, target_name: str, source_names: Iterable[str]) -> None:
    source_name = _first_present(source, source_names)
    if source_name is not None:
        target[target_name] = pd.to_numeric(source[source_name], errors="coerce")


def canonicalize_generation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Map old dense/block naming to the current public terminology.

    Current convention:
    - progress = dense/continuous progress
    - block_progress = old discrete checkpoint/block progress
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    _copy_if_present(out, df, "best_progress", ("best_progress", "best_dense_progress"))
    _copy_if_present(out, df, "mean_progress", ("mean_progress", "mean_dense_progress", "dense_progress_mean"))
    _copy_if_present(out, df, "best_block_progress", ("best_block_progress",))
    _copy_if_present(out, df, "mean_block_progress", ("mean_block_progress", "mean_discrete_progress"))

    if "best_dense_progress" in df.columns:
        out["best_progress"] = pd.to_numeric(df["best_dense_progress"], errors="coerce")
        if "best_progress" in df.columns:
            out["best_block_progress"] = pd.to_numeric(df["best_progress"], errors="coerce")
    if "mean_dense_progress" in df.columns:
        out["mean_progress"] = pd.to_numeric(df["mean_dense_progress"], errors="coerce")
        if "mean_progress" in df.columns:
            out["mean_block_progress"] = pd.to_numeric(df["mean_progress"], errors="coerce")

    for suffix in PROGRESS_STAT_SUFFIXES:
        dense_name = f"dense_progress_{suffix}"
        progress_name = f"progress_{suffix}"
        block_name = f"block_progress_{suffix}"
        if dense_name in df.columns:
            out[progress_name] = pd.to_numeric(df[dense_name], errors="coerce")
            if progress_name in df.columns:
                out[block_name] = pd.to_numeric(df[progress_name], errors="coerce")
        elif progress_name in df.columns:
            out[progress_name] = pd.to_numeric(df[progress_name], errors="coerce")
        if block_name in df.columns:
            out[block_name] = pd.to_numeric(df[block_name], errors="coerce")

    return out


def canonicalize_individual_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "dense_progress" in df.columns:
        out["progress"] = pd.to_numeric(df["dense_progress"], errors="coerce")
        if "progress" in df.columns:
            out["block_progress"] = pd.to_numeric(df["progress"], errors="coerce")
    else:
        _copy_if_present(out, df, "progress", ("progress",))
        _copy_if_present(out, df, "block_progress", ("block_progress", "discrete_progress"))
    if "block_progress" in df.columns:
        out["block_progress"] = pd.to_numeric(df["block_progress"], errors="coerce")
    return out


def infer_label(run_dir: Path) -> str:
    if run_dir.name.startswith("20") and run_dir.parent.name:
        return run_dir.parent.name
    return run_dir.name


def load_focus_run(run_dir: str | Path, label: str | None = None) -> FocusRun:
    path = Path(run_dir)
    generation = safe_read_csv(path / "generation_metrics.csv")
    if generation.empty:
        generation = safe_read_csv(path / "generation_summary.csv")
    if generation.empty:
        raise FileNotFoundError(f"No generation_metrics.csv or generation_summary.csv in {path}")
    individual = safe_read_csv(path / "individual_metrics.csv")
    return FocusRun(
        run_dir=path,
        label=str(label or infer_label(path)),
        generation=canonicalize_generation_metrics(generation),
        individual=canonicalize_individual_metrics(individual),
    )


def _run_color(index: int) -> tuple[float, float, float, float]:
    colors = plt.get_cmap("tab10").colors
    color = colors[index % len(colors)]
    return (float(color[0]), float(color[1]), float(color[2]), 1.0)


def _plot_density(axis, run: FocusRun, color, *, alpha: float) -> None:
    df = run.individual
    if df.empty or "generation" not in df.columns or "progress" not in df.columns:
        return
    clean = df[["generation", "progress"]].dropna()
    if clean.empty:
        return

    generations = np.sort(clean["generation"].astype(int).unique())
    if generations.size == 0:
        return
    gen_to_col = {int(gen): idx for idx, gen in enumerate(generations)}
    bins = np.linspace(0.0, 100.0, 101)
    heat = np.zeros((len(bins) - 1, len(generations)), dtype=np.float32)
    for gen, group in clean.groupby(clean["generation"].astype(int)):
        col = gen_to_col.get(int(gen))
        if col is None:
            continue
        hist, _ = np.histogram(group["progress"].astype(float).to_numpy(), bins=bins)
        if hist.max() > 0:
            heat[:, col] = hist.astype(np.float32) / float(hist.max())
    if not np.any(heat > 0):
        return

    rgba = np.zeros((heat.shape[0], heat.shape[1], 4), dtype=np.float32)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = heat * float(alpha)
    axis.imshow(
        rgba,
        origin="lower",
        aspect="auto",
        extent=[float(generations.min()), float(generations.max()), 0.0, 100.0],
        interpolation="nearest",
        zorder=0,
    )


def plot_focus_progress(
    runs: list[FocusRun],
    output_path: str | Path,
    *,
    title: str | None = None,
    density: str = "auto",
) -> Path:
    if not runs:
        raise ValueError("At least one run is required.")
    density = str(density).lower()
    if density not in {"auto", "always", "off"}:
        raise ValueError("density must be auto, always, or off.")

    fig, axis = plt.subplots(figsize=(13.5, 7.2))
    show_density = density == "always" or (density == "auto" and len(runs) <= 2)
    density_alpha = 0.13 if len(runs) == 1 else 0.075

    for idx, run in enumerate(runs):
        color = _run_color(idx)
        df = run.generation.sort_values("generation").copy()
        if "generation" not in df.columns:
            continue
        x = pd.to_numeric(df["generation"], errors="coerce").to_numpy(dtype=np.float64)
        best_col = _first_present(df, ("best_progress", "progress_max"))
        mean_col = _first_present(df, ("mean_progress", "progress_mean"))
        if best_col is None and mean_col is None:
            continue

        if show_density:
            _plot_density(axis, run, color, alpha=density_alpha)

        p10 = pd.to_numeric(df.get("progress_p10"), errors="coerce") if "progress_p10" in df else None
        p25 = pd.to_numeric(df.get("progress_p25"), errors="coerce") if "progress_p25" in df else None
        p75 = pd.to_numeric(df.get("progress_p75"), errors="coerce") if "progress_p75" in df else None
        p90 = pd.to_numeric(df.get("progress_p90"), errors="coerce") if "progress_p90" in df else None
        if p10 is not None and p90 is not None:
            axis.fill_between(x, p10, p90, color=color, alpha=0.10, linewidth=0, zorder=1)
        if p25 is not None and p75 is not None:
            axis.fill_between(x, p25, p75, color=color, alpha=0.18, linewidth=0, zorder=2)

        if best_col is not None:
            y_best = pd.to_numeric(df[best_col], errors="coerce").to_numpy(dtype=np.float64)
            axis.plot(x, y_best, color=color, linewidth=2.25, label=f"{run.label} best progress", zorder=4)
        if mean_col is not None:
            y_mean = pd.to_numeric(df[mean_col], errors="coerce").to_numpy(dtype=np.float64)
            axis.plot(
                x,
                y_mean,
                color=color,
                linewidth=1.8,
                linestyle="--",
                label=f"{run.label} mean progress",
                zorder=5,
            )

    axis.set_ylim(0.0, 102.0)
    axis.set_xlabel("Generation")
    axis.set_ylabel("Progress [%]")
    axis.set_title(title or "Training Progress Focus")
    axis.grid(True, alpha=0.22)
    axis.legend(fontsize=8, ncol=2)
    if show_density:
        axis.text(
            0.995,
            0.015,
            "soft color background = population density",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def summarize_focus_run(run: FocusRun) -> dict[str, object]:
    df = run.generation.sort_values("generation")
    last = df.tail(min(50, len(df)))
    best_col = _first_present(df, ("best_progress", "progress_max"))
    mean_col = _first_present(df, ("mean_progress", "progress_mean"))
    first_finish = math.nan
    if "finish_count" in df.columns:
        finish_rows = df[pd.to_numeric(df["finish_count"], errors="coerce").fillna(0) > 0]
        if not finish_rows.empty:
            first_finish = int(finish_rows["generation"].iloc[0])
    return {
        "label": run.label,
        "run_dir": str(run.run_dir),
        "generations": int(pd.to_numeric(df["generation"], errors="coerce").max()),
        "first_finish_generation": first_finish,
        "best_progress_max": float(pd.to_numeric(df[best_col], errors="coerce").max()) if best_col else math.nan,
        "last50_mean_progress": float(pd.to_numeric(last[mean_col], errors="coerce").mean()) if mean_col else math.nan,
        "final_best_progress": float(pd.to_numeric(df[best_col], errors="coerce").iloc[-1]) if best_col else math.nan,
        "final_mean_progress": float(pd.to_numeric(df[mean_col], errors="coerce").iloc[-1]) if mean_col else math.nan,
    }


def write_focus_summary(runs: list[FocusRun], output_path: str | Path) -> Path:
    rows = [summarize_focus_run(run) for run in runs]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["label"])
        writer.writeheader()
        writer.writerows(rows)
    return output
