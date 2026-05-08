from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = PACKAGE_ROOT / "analysis" / "ga_hyperparam_refinement_20260504" / "summary.csv"
OUT_DIR = ROOT / "Diplomová práca" / "Latex" / "images" / "training_policy"


def _format_decimal(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def _highlight_cell(ax, xs: list[float], ys: list[float], x_value: float, y_value: float, *, color: str, dashed: bool = False):
    x_idx = xs.index(x_value)
    y_idx = ys.index(y_value)
    patch = Rectangle(
        (x_idx - 0.5, y_idx - 0.5),
        1,
        1,
        fill=False,
        linewidth=2.8,
        edgecolor=color,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)


def _draw_heatmap(
    ax,
    data: pd.DataFrame,
    xs: list[float],
    ys: list[float],
    *,
    title: str,
    cmap: str,
    vmin=None,
    vmax=None,
    fmt: str = "{:.0f}",
    annotation_data: pd.DataFrame | None = None,
):
    matrix = data.reindex(index=ys, columns=xs).to_numpy(dtype=float)
    cmap_obj = mpl.colormaps[cmap]
    norm = mpl.colors.Normalize(
        vmin=float(np.nanmin(matrix)) if vmin is None else vmin,
        vmax=float(np.nanmax(matrix)) if vmax is None else vmax,
    )
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap_obj, norm=norm)
    annotation_matrix = (
        annotation_data.reindex(index=ys, columns=xs).to_numpy(dtype=float)
        if annotation_data is not None
        else matrix
    )
    ax.set_title(title)
    ax.set_xticks(np.arange(len(xs)), [str(int(x)) if float(x).is_integer() else _format_decimal(x, 3) for x in xs])
    ax.set_yticks(np.arange(len(ys)), [str(int(y)) if float(y).is_integer() else _format_decimal(y, 3) for y in ys])
    ax.tick_params(length=0)
    for y_idx, y in enumerate(ys):
        for x_idx, x in enumerate(xs):
            value = matrix[y_idx, x_idx]
            label_value = annotation_matrix[y_idx, x_idx]
            if np.isnan(label_value):
                label = "-"
            else:
                label = fmt.format(label_value).replace(".", ",")
            rgba = cmap_obj(norm(value if not np.isnan(value) else 0))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "white" if luminance < 0.42 else "#1b1b1b"
            ax.text(x_idx, y_idx, label, ha="center", va="center", fontsize=8.2, color=text_color)
    return im


def draw_selection_pressure(summary: pd.DataFrame) -> None:
    selection = summary[summary["grid"] == "selection"].copy()
    if selection.empty:
        raise ValueError("No selection rows found in summary.csv")

    parents = sorted(selection["parent_count"].dropna().astype(int).unique().tolist())
    elites = sorted(selection["elite_count"].dropna().astype(int).unique().tolist())

    finish_rate = (
        selection.pivot(index="elite_count", columns="parent_count", values="last50_finish_rate")
        * 100.0
    )
    first_finish = selection.pivot(index="elite_count", columns="parent_count", values="first_finish_generation_plot")
    first_finish_labels = selection.pivot(index="elite_count", columns="parent_count", values="first_finish_generation")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.7,
            "ytick.labelsize": 8.7,
            "figure.dpi": 150,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), constrained_layout=True)
    im0 = _draw_heatmap(
        axes[0],
        finish_rate,
        parents,
        elites,
        title="Dokončenia v posledných generáciách [%]",
        cmap="YlGnBu",
        vmin=0,
        vmax=max(45, float(finish_rate.max().max())),
        fmt="{:.1f}",
    )
    im1 = _draw_heatmap(
        axes[1],
        first_finish,
        parents,
        elites,
        title="Generácia prvého dokončenia",
        cmap="magma_r",
        vmin=80,
        vmax=205,
        fmt="{:.0f}",
        annotation_data=first_finish_labels,
    )

    for ax in axes:
        ax.set_xlabel("Počet rodičov")
        ax.set_ylabel("Počet elitných jedincov")
        _highlight_cell(ax, parents, elites, 14, 1, color="#101010", dashed=False)
        _highlight_cell(ax, parents, elites, 14, 2, color="#FFB000", dashed=True)

    cbar0 = fig.colorbar(im0, ax=axes[0], shrink=0.86)
    cbar0.set_label("Dokončenia [%]")
    cbar1 = fig.colorbar(im1, ax=axes[1], shrink=0.86)
    cbar1.set_label("Generácia")

    fig.text(
        0.5,
        -0.02,
        "Plný rámček označuje najlepší bod podľa refinementu (14 rodičov, 1 elita), prerušovaný rámček praktický baseline s dvoma elitami.",
        ha="center",
        va="top",
        fontsize=9.2,
        color="#333333",
    )

    for extension in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"ga_hyperparam_selection_pressure.{extension}", bbox_inches="tight")
    plt.close(fig)


def draw_mutation_grid(summary: pd.DataFrame) -> None:
    mutation = summary[summary["grid"] == "mutation"].copy()
    if mutation.empty:
        raise ValueError("No mutation rows found in summary.csv")

    probs = sorted(mutation["mutation_prob"].dropna().unique().tolist())
    sigmas = sorted(mutation["mutation_sigma"].dropna().unique().tolist())

    finish_rate = mutation.pivot(index="mutation_sigma", columns="mutation_prob", values="last50_finish_rate") * 100.0
    penalized_time = mutation.pivot(index="mutation_sigma", columns="mutation_prob", values="last50_penalized_mean_time")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 150,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    im0 = _draw_heatmap(
        axes[0],
        finish_rate,
        probs,
        sigmas,
        title="Dokončenia v posledných generáciách [%]",
        cmap="YlGnBu",
        vmin=0,
        vmax=max(30, float(finish_rate.max().max())),
        fmt="{:.1f}",
    )
    im1 = _draw_heatmap(
        axes[1],
        penalized_time,
        probs,
        sigmas,
        title="Penalizovaný priemerný čas [s]",
        cmap="YlOrRd",
        vmin=25,
        vmax=30,
        fmt="{:.1f}",
    )

    for ax in axes:
        ax.set_xlabel("Pravdepodobnosť mutácie")
        ax.set_ylabel("Smerodajná odchýlka mutácie")
        _highlight_cell(ax, probs, sigmas, 0.10, 0.25, color="#101010", dashed=False)
        _highlight_cell(ax, probs, sigmas, 0.05, 0.325, color="#FFB000", dashed=True)

    cbar0 = fig.colorbar(im0, ax=axes[0], shrink=0.86)
    cbar0.set_label("Dokončenia [%]")
    cbar1 = fig.colorbar(im1, ax=axes[1], shrink=0.86)
    cbar1.set_label("Čas [s]")

    fig.text(
        0.5,
        -0.02,
        "Plný rámček označuje bezpečný vnútorný bod p=0,10, σ=0,25; prerušovaný rámček zaujímavý okrajový kandidát p=0,05, σ=0,325.",
        ha="center",
        va="top",
        fontsize=9.2,
        color="#333333",
    )

    for extension in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"ga_hyperparam_mutation_grid.{extension}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(SUMMARY_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    draw_selection_pressure(summary)
    draw_mutation_grid(summary)


if __name__ == "__main__":
    main()
