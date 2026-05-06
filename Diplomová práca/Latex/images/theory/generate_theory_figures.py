from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch


OUT_DIR = Path(__file__).resolve().parent

BLUE = "#4C78A8"
LIGHT_BLUE = "#EAF2FB"
ORANGE = "#F58518"
LIGHT_ORANGE = "#FFF3E6"
GREEN = "#54A24B"
LIGHT_GREEN = "#EDF7EA"
GRAY = "#6B7280"
LIGHT_GRAY = "#F3F4F6"
PURPLE = "#8E6BBE"
RED = "#C44E52"
INK = "#1F2937"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / name, bbox_inches="tight")
    plt.close(fig)


def clear_axes(ax: plt.Axes, xlim=(0, 10), ylim=(0, 6)) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    fc: str = LIGHT_BLUE,
    ec: str = BLUE,
    fontsize: int = 11,
    lw: float = 1.6,
) -> patches.FancyBboxPatch:
    patch = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
    )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
    color: str = INK,
    rad: float = 0.0,
    lw: float = 1.6,
) -> None:
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if label:
        if label_xy is None:
            label_xy = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.18)
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            ha="center",
            va="center",
            fontsize=10,
            color=color,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.88),
        )


def node(ax: plt.Axes, x: float, y: float, text: str = "", *, fc: str = "white", ec: str = BLUE, r: float = 0.18) -> None:
    circ = patches.Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=1.4)
    ax.add_patch(circ)
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=9, color=INK)


def draw_agent_environment_loop() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    clear_axes(ax, (0, 10), (0, 5.2))

    box(ax, (4.15, 4.0), 1.7, 0.8, "Agent", fc="white", ec=INK, fontsize=13, lw=1.8)
    box(ax, (3.55, 1.0), 2.9, 0.8, "Prostredie", fc="white", ec=INK, fontsize=13, lw=1.8)

    # Current state and reward enter the agent from the left.
    ax.plot([0.85, 0.85], [1.55, 4.55], color=INK, linewidth=1.7)
    arrow(ax, (0.85, 4.55), (4.15, 4.55), color=INK, lw=1.7)
    arrow(ax, (1.2, 3.85), (4.15, 4.25), color=INK, lw=1.2)
    ax.text(0.55, 3.75, "stav\n$s_t$", ha="right", va="center", fontsize=11, color=INK)
    ax.text(1.35, 3.25, "odmena\n$r_t$", ha="left", va="center", fontsize=11, color=INK)

    # The chosen action is applied to the environment.
    ax.plot([5.85, 8.8], [4.4, 4.4], color=INK, linewidth=1.7)
    ax.plot([8.8, 8.8], [4.4, 1.4], color=INK, linewidth=1.7)
    arrow(ax, (8.8, 1.4), (6.45, 1.4), color=INK, lw=1.7)
    ax.text(9.05, 2.85, "akcia\n$a_t$", ha="left", va="center", fontsize=11, color=INK)

    # Dashed boundary marks the next time step produced by the environment.
    ax.plot([2.25, 2.25], [1.0, 2.75], color=INK, linewidth=1.0, linestyle=(0, (2, 3)))
    arrow(ax, (3.55, 1.55), (2.25, 1.55), color=INK, lw=1.2)
    arrow(ax, (3.55, 1.95), (2.25, 1.95), color=INK, lw=1.2)
    ax.text(2.65, 2.15, "$r_{t+1}$", ha="center", va="bottom", fontsize=10, color=INK)
    ax.text(2.65, 1.45, "$s_{t+1}$", ha="center", va="top", fontsize=10, color=INK)

    ax.text(
        5,
        0.35,
        "Čiarkovaná hranica naznačuje posun času: prostredie po akcii vytvorí nový stav a novú odmenu.",
        ha="center",
        fontsize=9.5,
        color=GRAY,
    )
    save(fig, "theory_agent_environment_loop.pdf")


def draw_scalar_vs_lexicographic() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax in axes:
        clear_axes(ax, (0, 10), (-0.6, 5))

    ax = axes[0]
    ax.set_title("Skalárne hodnotenie")
    metric_y = [4.0, 3.0, 2.0]
    labels = ["metrika $m_1$", "metrika $m_2$", "metrika $m_3$"]
    weights = ["$w_1$", "$w_2$", "$w_3$"]
    for y, lab, w in zip(metric_y, labels, weights):
        box(ax, (0.7, y - 0.35), 2.4, 0.7, lab, fc=LIGHT_BLUE, ec=BLUE, fontsize=10)
        box(ax, (3.7, y - 0.25), 1.0, 0.5, w, fc=LIGHT_ORANGE, ec=ORANGE, fontsize=10)
        arrow(ax, (3.1, y), (3.7, y), color=GRAY, lw=1.2)
        arrow(ax, (4.7, y), (6.4, 2.5), color=GRAY, lw=1.2)
    box(ax, (6.4, 1.95), 2.6, 1.1, "$S = \\sum_i w_i m_i$", fc=LIGHT_GREEN, ec=GREEN, fontsize=12)
    ax.text(5.0, 0.15, "všetky ciele sa zmiešajú\ndo jedného čísla", ha="center", fontsize=10, color=GRAY)

    ax = axes[1]
    ax.set_title("Lexikografické poradie")
    tuple_boxes = [(0.7, 2.45, "$m_1$"), (2.3, 2.45, "$m_2$"), (3.9, 2.45, "$m_3$")]
    for i, (x, y, lab) in enumerate(tuple_boxes, start=1):
        box(ax, (x, y), 1.15, 0.9, lab, fc=LIGHT_BLUE if i == 1 else "white", ec=BLUE, fontsize=12)
        ax.text(x + 0.58, y - 0.35, f"{i}. priorita", ha="center", fontsize=9, color=GRAY)
    ax.text(2.9, 3.9, "$\\mathbf{m}(\\tau)=(m_1,m_2,m_3)$", ha="center", fontsize=12, color=INK)
    arrow(ax, (5.3, 2.9), (6.7, 2.9), label="porovnaj\npostupne", label_xy=(6.0, 3.45), color=ORANGE)
    box(ax, (6.7, 2.25), 2.4, 1.3, "najprv $m_1$\npotom $m_2$\npotom $m_3$", fc=LIGHT_ORANGE, ec=ORANGE, fontsize=10)
    ax.text(5.0, 0.15, "ciele ostávajú oddelené,\nporadie vyjadruje priority", ha="center", fontsize=10, color=GRAY)

    save(fig, "theory_scalar_vs_lexicographic.pdf")


def draw_neuron_layer_mlp() -> None:
    fig = plt.figure(figsize=(9.5, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.55, wspace=0.35)
    axes = np.array(
        [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :]),
        ],
        dtype=object,
    )
    for ax in axes[:2]:
        clear_axes(ax, (0, 8.4), (0, 5.4))
    clear_axes(axes[2], (0.3, 8.3), (0.2, 5.4))

    ax = axes[0]
    ax.set_title("Jeden neurón")
    for i, y in enumerate([4.4, 3.2, 2.0], start=1):
        node(ax, 1.4, y, f"$x_{i}$", fc=LIGHT_BLUE)
        arrow(ax, (1.65, y), (4.15, 3.2), color=GRAY, lw=1.1)
    node(ax, 4.6, 3.2, "$\\sigma$", fc=LIGHT_ORANGE, ec=ORANGE, r=0.48)
    arrow(ax, (5.1, 3.2), (7.6, 3.2), label="$h$", label_xy=(6.35, 3.55), color=BLUE)
    ax.text(4.6, 1.0, "$h = \\sigma(\\mathbf{w}^{\\top}\\mathbf{x}+b)$", ha="center", fontsize=11, color=INK)

    ax = axes[1]
    ax.set_title("Jedna vrstva")
    in_y = np.linspace(4.6, 1.4, 4)
    out_y = np.linspace(4.3, 1.7, 3)
    for y in in_y:
        node(ax, 1.2, y, fc=LIGHT_BLUE)
    for y in out_y:
        node(ax, 5.1, y, fc=LIGHT_ORANGE, ec=ORANGE)
    for yi in in_y:
        for yo in out_y:
            ax.plot([1.38, 4.92], [yi, yo], color="#CBD5E1", linewidth=0.8)
    arrow(ax, (5.35, 3.0), (8.2, 3.0), label="$\\mathbf{h}$", label_xy=(6.8, 3.35), color=BLUE)
    ax.text(4.7, 0.8, "$\\mathbf{h}=\\sigma(W\\mathbf{x}+\\mathbf{b})$", ha="center", fontsize=11, color=INK)

    ax = axes[2]
    ax.set_title("Viacvrstvová sieť")
    layer_x = [1.0, 3.2, 5.4, 7.6]
    layers = [4, 4, 3, 2]
    colors = [LIGHT_BLUE, LIGHT_ORANGE, LIGHT_ORANGE, LIGHT_GREEN]
    edges = [BLUE, ORANGE, ORANGE, GREEN]
    for x, n, fc, ec in zip(layer_x, layers, colors, edges):
        ys = np.linspace(4.6, 1.4, n)
        for y in ys:
            node(ax, x, y, fc=fc, ec=ec, r=0.16)
    for x1, x2, n1, n2 in zip(layer_x[:-1], layer_x[1:], layers[:-1], layers[1:]):
        ys1 = np.linspace(4.6, 1.4, n1)
        ys2 = np.linspace(4.6, 1.4, n2)
        for y1 in ys1:
            for y2 in ys2:
                ax.plot([x1 + 0.16, x2 - 0.16], [y1, y2], color="#CBD5E1", linewidth=0.7)
    ax.text(1.0, 0.75, "vstup", ha="center", fontsize=10, color=GRAY)
    ax.text(4.3, 0.75, "skryté vrstvy", ha="center", fontsize=10, color=GRAY)
    ax.text(7.6, 0.75, "výstup", ha="center", fontsize=10, color=GRAY)

    save(fig, "theory_neuron_layer_mlp.pdf")


def draw_activation_functions() -> None:
    x = np.linspace(-3, 3, 500)
    functions = [
        ("Sigmoid", 1.0 / (1.0 + np.exp(-x)), PURPLE, (0.0, 1.0), r"$\sigma(z)=\frac{1}{1+e^{-z}}$"),
        ("Hyperbolický tangens", np.tanh(x), BLUE, (-1.05, 1.05), r"$\tanh(z)$"),
        ("ReLU", np.maximum(0, x), ORANGE, (-0.1, 3.1), r"$\operatorname{ReLU}(z)=\max(0,z)$"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    for ax, (title, y, color, ylim, formula) in zip(axes, functions):
        ax.plot(x, y, color=color, linewidth=2.4)
        ax.axhline(0, color="#9CA3AF", linewidth=0.85)
        ax.axvline(0, color="#9CA3AF", linewidth=0.85)
        ax.set_xlim(-3, 3)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.set_xlabel("vstup $z$")
        ax.grid(True, color="#E5E7EB", linewidth=0.8)
        ax.text(0.5, -0.26, formula, transform=ax.transAxes, ha="center", va="top", fontsize=9, color=INK)
    axes[0].set_ylabel("výstup")
    fig.tight_layout(w_pad=2.0)
    save(fig, "theory_activation_functions.pdf")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_agent_environment_loop()
    draw_scalar_vs_lexicographic()
    draw_neuron_layer_mlp()
    draw_activation_functions()


if __name__ == "__main__":
    main()
