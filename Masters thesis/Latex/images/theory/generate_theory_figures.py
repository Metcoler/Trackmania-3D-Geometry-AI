from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib import patches
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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
RAY_BLUE = "#2563EB"
GA_LOW = "#D95F59"
GA_MID = "#F2C94C"
GA_HIGH = "#4FA86B"
GA_FITNESS_BAR = BLUE


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


def shaded_mesh_colors(polys: list[np.ndarray], base_color: str = LIGHT_BLUE) -> list[tuple[float, float, float, float]]:
    base = np.array(to_rgb(base_color))
    light_dir = np.array([-0.35, -0.55, 0.76])
    light_dir = light_dir / np.linalg.norm(light_dir)
    colors: list[tuple[float, float, float, float]] = []
    for tri in polys:
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            normal = normal / norm
            intensity = 0.62 + 0.34 * abs(float(np.dot(normal, light_dir)))
        else:
            intensity = 0.75
        rgb = np.clip(base * intensity, 0, 1)
        colors.append((float(rgb[0]), float(rgb[1]), float(rgb[2]), 0.92))
    return colors


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
    linestyle: str | tuple = "solid",
    alpha: float = 1.0,
) -> None:
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        alpha=alpha,
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
            alpha=alpha,
        )


def node(ax: plt.Axes, x: float, y: float, text: str = "", *, fc: str = "white", ec: str = BLUE, r: float = 0.18) -> None:
    circ = patches.Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=1.4)
    ax.add_patch(circ)
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=9, color=INK)


GA_GENES = np.array([0.12, 0.78, 0.48, 0.24, 0.92, 0.58, 0.36, 0.70])
GA_FITNESS = np.array([73, 28, 91, 44, 15, 66, 52, 84])


def ga_selection_order() -> np.ndarray:
    return np.argsort(-GA_FITNESS)


def ga_selected_parents() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parent_idx = ga_selection_order()[:4]
    return parent_idx, GA_GENES[parent_idx], GA_FITNESS[parent_idx]


def ga_crossover_specs() -> list[tuple[int, int, float]]:
    return [
        (0, 1, 0.50),
        (0, 2, 0.50),
        (1, 2, 0.50),
        (0, 3, 0.50),
        (1, 3, 0.50),
        (2, 3, 0.50),
    ]


def ga_crossover_children() -> np.ndarray:
    _, parent_genes, _ = ga_selected_parents()
    crossed = [
        weight * parent_genes[p_a] + (1.0 - weight) * parent_genes[p_b]
        for p_a, p_b, weight in ga_crossover_specs()
    ]
    return np.array(
        [
            parent_genes[0],
            parent_genes[1],
            *crossed,
        ],
        dtype=float,
    )


def ga_color(value: float) -> tuple[float, float, float]:
    low = np.array(to_rgb(GA_LOW))
    mid = np.array(to_rgb(GA_MID))
    high = np.array(to_rgb(GA_HIGH))
    value = float(np.clip(value, 0.0, 1.0))
    if value <= 0.5:
        t = value / 0.5
        rgb = (1.0 - t) * low + t * mid
    else:
        t = (value - 0.5) / 0.5
        rgb = (1.0 - t) * mid + t * high
    return tuple(float(v) for v in rgb)


def draw_ga_individual(
    ax: plt.Axes,
    x: float,
    y: float,
    gene: float,
    *,
    fitness: int | None = None,
    size: float = 0.72,
    label: str | None = None,
    crossed: bool = False,
    highlight: bool = False,
    show_bar: bool = True,
) -> None:
    edge = BLUE if highlight else INK
    lw = 2.3 if highlight else 1.3
    square = patches.FancyBboxPatch(
        (x - size / 2, y - size / 2),
        size,
        size,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=ga_color(gene),
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(square)
    if fitness is not None and show_bar:
        bar_w = size * 0.46
        bar_h = 1.28
        bar_x = x - bar_w / 2
        bar_y = y + size / 2 + 0.25
        ax.add_patch(
            patches.FancyBboxPatch(
                (bar_x, bar_y),
                bar_w,
                bar_h,
                boxstyle="round,pad=0.01,rounding_size=0.04",
                facecolor="white",
                edgecolor="#CBD5E1",
                linewidth=0.9,
            )
        )
        fill_h = bar_h * fitness / 100.0
        ax.add_patch(
            patches.FancyBboxPatch(
                (bar_x, bar_y),
                bar_w,
                fill_h,
                boxstyle="round,pad=0.01,rounding_size=0.04",
                facecolor=GA_FITNESS_BAR,
                edgecolor="none",
                alpha=0.92,
            )
        )
        ax.text(x, bar_y + bar_h + 0.16, f"{fitness} %", ha="center", va="bottom", fontsize=8.4, color=INK)
    if label:
        ax.text(x, y - size / 2 - 0.22, label, ha="center", va="top", fontsize=9, color=INK)
    if crossed:
        pad = 0.13
        ax.plot([x - size / 2 - pad, x + size / 2 + pad], [y - size / 2 - pad, y + size / 2 + pad], color=RED, lw=2.4)
        ax.plot([x - size / 2 - pad, x + size / 2 + pad], [y + size / 2 + pad, y - size / 2 - pad], color=RED, lw=2.4)


def draw_agent_environment_loop() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.9))
    clear_axes(ax, (0.90, 12.10), (0, 6.2))

    agent_x = 1.20
    agent_y = 1.00
    agent_w = 5.50
    agent_h = 4.35
    agent_top = agent_y + agent_h
    agent_bottom = agent_y
    agent_center_y = agent_y + agent_h / 2.0
    gate_margin_y = 0.82

    agent_patch = patches.FancyBboxPatch(
        (agent_x, agent_y),
        agent_w,
        agent_h,
        boxstyle="round,pad=0.06,rounding_size=0.22",
        linewidth=1.7,
        edgecolor=BLUE,
        facecolor=LIGHT_BLUE,
    )
    ax.add_patch(agent_patch)
    ax.text(agent_x + 0.30, agent_top - 0.33, "Agent", ha="left", va="center", fontsize=12, color=BLUE, weight="bold")

    policy_x = 3.05
    policy_w = 2.15
    policy_h = 0.92
    policy_cx = policy_x
    gate_x = 6.05
    gate_w = 1.35
    gate_h = 0.68
    sensor_y = agent_top - gate_margin_y
    actuator_y = agent_bottom + gate_margin_y
    policy_y = agent_center_y - policy_h / 2.0
    box(ax, (policy_cx - policy_w / 2, policy_y), policy_w, policy_h, "policy\n$\\pi_\\theta$", fc="white", ec=BLUE, fontsize=12, lw=1.6)
    box(ax, (gate_x, sensor_y - 0.34), gate_w, 0.68, "sensors", fc="white", ec=BLUE, fontsize=10, lw=1.35)
    box(ax, (gate_x, actuator_y - 0.34), gate_w, 0.68, "actuators", fc="white", ec=BLUE, fontsize=10, lw=1.35)
    env_x = 9.05
    env_w = 2.75
    env_patch = patches.FancyBboxPatch(
        (env_x, agent_y),
        env_w,
        agent_h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.7,
        edgecolor=GREEN,
        facecolor=LIGHT_GREEN,
    )
    ax.add_patch(env_patch)
    ax.text(env_x + 0.18, agent_top - 0.34, "Environment", ha="left", va="center", fontsize=11.5, color=GREEN, weight="bold")

    # Percepts enter the agent through sensors.
    arrow(
        ax,
        (env_x, sensor_y),
        (gate_x + gate_w, sensor_y),
        color=GREEN,
        lw=1.55,
    )
    ax.plot([gate_x, policy_cx], [sensor_y, sensor_y], color=BLUE, linewidth=1.35)
    arrow(ax, (policy_cx, sensor_y), (policy_cx, policy_y + policy_h), color=BLUE, lw=1.35)
    label_x = 4.70
    ax.text(
        label_x,
        sensor_y,
        "observation $o_t$\nreward $r_t$",
        ha="center",
        va="center",
        fontsize=10,
        color=BLUE,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.92),
    )

    # The policy transforms observations into actions.
    ax.plot([policy_cx, policy_cx], [policy_y, actuator_y], color=BLUE, linewidth=1.35)
    arrow(ax, (policy_cx, actuator_y), (gate_x, actuator_y), color=BLUE, lw=1.35)
    ax.text(
        label_x,
        actuator_y,
        "action\n$a_t$",
        ha="center",
        va="center",
        fontsize=10,
        color=BLUE,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.92),
    )

    # Actuators apply the action back to the environment.
    arrow(
        ax,
        (gate_x + gate_w, actuator_y),
        (env_x, actuator_y),
        color=GREEN,
        lw=1.55,
    )

    transition_x = env_x + env_w - 0.78
    ax.plot(
        [env_x, transition_x],
        [actuator_y, actuator_y],
        color=GREEN,
        linewidth=1.1,
        linestyle=(0, (2, 3)),
        alpha=0.9,
    )
    ax.plot(
        [transition_x, transition_x],
        [actuator_y, sensor_y],
        color=GREEN,
        linewidth=1.1,
        linestyle=(0, (2, 3)),
        alpha=0.9,
    )
    ax.plot(
        [transition_x, env_x],
        [sensor_y, sensor_y],
        color=GREEN,
        linewidth=1.1,
        linestyle=(0, (2, 3)),
        alpha=0.9,
    )
    ax.text(
        transition_x + 0.22,
        (actuator_y + sensor_y) / 2.0,
        "$a_t$",
        ha="center",
        va="center",
        fontsize=11,
        color=GREEN,
    )
    state_label_bbox = dict(
        boxstyle="round,pad=0.14",
        facecolor="white",
        edgecolor="none",
        alpha=0.86,
    )
    ax.text(
        transition_x,
        actuator_y,
        "$s_t$",
        ha="center",
        va="center",
        fontsize=11,
        color=INK,
        bbox=state_label_bbox,
    )
    ax.text(
        transition_x,
        sensor_y,
        "$s_{t+1}$",
        ha="center",
        va="center",
        fontsize=11,
        color=INK,
        bbox=state_label_bbox,
    )

    save(fig, "theory_agent_environment_loop.pdf")


def draw_scalar_vs_lexicographic() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax in axes:
        clear_axes(ax, (0, 10), (-0.6, 5))

    ax = axes[0]
    ax.set_title("Scalar evaluation")
    metric_y = [4.0, 3.0, 2.0]
    labels = ["metric $m_1$", "metric $m_2$", "metric $m_3$"]
    weights = ["$w_1$", "$w_2$", "$w_3$"]
    for y, lab, w in zip(metric_y, labels, weights):
        box(ax, (0.7, y - 0.35), 2.4, 0.7, lab, fc=LIGHT_BLUE, ec=BLUE, fontsize=10)
        box(ax, (3.7, y - 0.25), 1.0, 0.5, w, fc=LIGHT_ORANGE, ec=ORANGE, fontsize=10)
        arrow(ax, (3.1, y), (3.7, y), color=GRAY, lw=1.2)
        arrow(ax, (4.7, y), (6.4, 2.5), color=GRAY, lw=1.2)
    box(ax, (6.4, 1.95), 2.6, 1.1, "$S = \\sum_i w_i m_i$", fc=LIGHT_GREEN, ec=GREEN, fontsize=12)
    ax.text(5.0, 0.15, "all objectives are mixed\ninto one number", ha="center", fontsize=10, color=GRAY)

    ax = axes[1]
    ax.set_title("Lexicographic ordering")
    tuple_boxes = [(0.7, 2.45, "$m_1$"), (2.3, 2.45, "$m_2$"), (3.9, 2.45, "$m_3$")]
    for i, (x, y, lab) in enumerate(tuple_boxes, start=1):
        box(ax, (x, y), 1.15, 0.9, lab, fc=LIGHT_BLUE if i == 1 else "white", ec=BLUE, fontsize=12)
        ax.text(x + 0.58, y - 0.35, f"{i}. priorita", ha="center", fontsize=9, color=GRAY)
    ax.text(2.9, 3.9, "$\\mathbf{m}(\\tau)=(m_1,m_2,m_3)$", ha="center", fontsize=12, color=INK)
    arrow(ax, (5.3, 2.9), (6.7, 2.9), label="porovnaj\npostupne", label_xy=(6.0, 3.45), color=ORANGE)
    box(ax, (6.7, 2.25), 2.4, 1.3, "first $m_1$\nthen $m_2$\nthen $m_3$", fc=LIGHT_ORANGE, ec=ORANGE, fontsize=10)
    ax.text(5.0, 0.15, "objectives stay separate;\nthe order expresses priorities", ha="center", fontsize=10, color=GRAY)

    save(fig, "theory_scalar_vs_lexicographic.pdf")


def draw_blackbox_function() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    clear_axes(ax, (0, 10), (0, 5.6))

    input_y = [4.25, 3.75, 3.25, 2.75, 2.25, 1.75, 1.25]
    output_y = [4.05, 3.4, 2.75, 2.1, 1.45]
    input_labels = ["$x_1$", "$x_2$", "$x_3$", "$\\cdot$", "$\\cdot$", "$\\cdot$", "$x_m$"]
    output_labels = ["$y_1$", "$y_2$", "$\\cdot$", "$\\cdot$", "$y_n$"]

    for y, label in zip(input_y, input_labels):
        ax.text(1.0, y, label, ha="right", va="center", fontsize=12, color=INK)
        arrow(ax, (1.25, y), (3.35, y), color=GRAY, lw=1.15)

    blackbox = patches.FancyBboxPatch(
        (3.45, 1.05),
        3.4,
        3.4,
        boxstyle="round,pad=0.04,rounding_size=0.04",
        linewidth=1.8,
        edgecolor="#111827",
        facecolor="#111827",
    )
    ax.add_patch(blackbox)
    ax.text(5.15, 3.05, "model", ha="center", va="center", fontsize=13, color="white")
    ax.text(5.15, 2.4, "$f_\\theta$", ha="center", va="center", fontsize=19, color="white")

    for y, label in zip(output_y, output_labels):
        arrow(ax, (6.85, y), (8.5, y), color=GRAY, lw=1.15)
        ax.text(8.75, y, label, ha="left", va="center", fontsize=12, color=INK)

    ax.text(2.25, 0.38, "input vector $\\mathbf{x}$", ha="center", va="center", fontsize=10, color=GRAY)
    ax.text(7.75, 0.38, "output vector $\\mathbf{y}$", ha="center", va="center", fontsize=10, color=GRAY)
    ax.text(5.15, 5.35, "Parametric model as a mapping from inputs to outputs", ha="center", va="center", fontsize=13, color=INK)

    save(fig, "theory_blackbox_function.pdf")


def draw_neuron_layer_mlp() -> None:
    fig = plt.figure(figsize=(9.5, 6.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.18, wspace=0.35)
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
    single_input_y = [4.35, 3.2, 2.05]
    single_target_y = [3.48, 3.2, 2.92]
    for i, (y, target_y) in enumerate(zip(single_input_y, single_target_y), start=1):
        node(ax, 1.35, y, f"$x_{i}$", fc=LIGHT_BLUE, r=0.48)
        arrow(ax, (1.86, y), (4.08, target_y), color=GRAY, lw=1.1)
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
    ax.set_title("Multilayer network")
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
    ax.text(4.3, 0.75, "hidden layers", ha="center", fontsize=10, color=GRAY)
    ax.text(7.6, 0.75, "output", ha="center", fontsize=10, color=GRAY)

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
    axes[0].set_ylabel("output")
    fig.tight_layout(w_pad=2.0)
    save(fig, "theory_activation_functions.pdf")


def draw_ga_initial_population() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    clear_axes(ax, (0, 12.2), (0, 4.9))

    ax.text(6.15, 4.55, "Randomly created population", ha="center", va="center", fontsize=13, color=INK)

    xs = np.linspace(1.85, 11.25, len(GA_GENES))
    for i, (x, gene, fitness) in enumerate(zip(xs, GA_GENES, GA_FITNESS), start=1):
        draw_ga_individual(ax, x, 1.25, float(gene), fitness=int(fitness), label=f"$x_{i}$")

    ax.text(0.62, 2.50, "fitness", ha="center", va="center", fontsize=9.2, color=GA_FITNESS_BAR)
    ax.text(0.62, 1.25, "jedinec", ha="center", va="center", fontsize=9.2, color=GRAY)

    save(fig, "theory_ga_initial_population.pdf")


def draw_ga_selection() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    clear_axes(ax, (0, 11.7), (0, 4.9))

    order = ga_selection_order()
    xs = np.linspace(1.05, 10.65, len(GA_GENES))
    ax.text(5.85, 4.55, "Selection by evaluation", ha="center", va="center", fontsize=13, color=INK)
    ax.text(3.05, 3.95, "selected parents", ha="center", va="center", fontsize=10.5, color=GREEN, weight="bold")
    ax.text(8.65, 3.95, "discarded individuals", ha="center", va="center", fontsize=10.5, color=RED, weight="bold")
    ax.plot([5.85, 5.85], [0.45, 4.15], color="#CBD5E1", linewidth=1.0, linestyle="--")

    for rank, idx in enumerate(order):
        selected = rank < 4
        draw_ga_individual(
            ax,
            xs[rank],
            1.25,
            float(GA_GENES[idx]),
            fitness=int(GA_FITNESS[idx]),
            label=f"$x_{idx + 1}$",
            crossed=not selected,
            highlight=selected,
        )

    save(fig, "theory_ga_selection.pdf")


def draw_ga_crossover() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 6.28))
    clear_axes(ax, (0, 12.2), (0, 6.6))

    parent_idx, parent_genes, parent_fitness = ga_selected_parents()
    parent_xs = np.array([2.1, 4.75, 7.4, 10.05])
    child_xs = np.linspace(0.95, 11.25, 8)
    parent_y = 3.85
    child_y = 1.18

    ax.text(6.1, 6.46, "Elitism and crossover", ha="center", va="center", fontsize=13, color=INK)
    ax.text(6.1, 4.85, "selected parents", ha="center", va="center", fontsize=10.5, color=GRAY)
    ax.text(6.1, 0.28, "new generation", ha="center", va="center", fontsize=10.5, color=GRAY)

    for rank, (x, idx, gene, fitness) in enumerate(zip(parent_xs, parent_idx, parent_genes, parent_fitness), start=1):
        draw_ga_individual(ax, x, parent_y, float(gene), fitness=int(fitness), label=f"$p_{rank}$", highlight=rank <= 2)

    children = ga_crossover_children()
    pairs = [None, None, *[(p_a, p_b) for p_a, p_b, _ in ga_crossover_specs()]]

    for i, (x, gene) in enumerate(zip(child_xs, children), start=1):
        draw_ga_individual(ax, x, child_y, float(gene), label=f"$x'_{i}$", show_bar=False, size=0.66)

    for src_i, dst_i in [(0, 0), (1, 1)]:
        arrow(
            ax,
            (parent_xs[src_i], parent_y - 0.78),
            (child_xs[dst_i], child_y + 0.47),
            color=RED,
            lw=1.75,
            label="elitism" if src_i == 0 else None,
            label_xy=(1.35, 2.55),
        )

    for child_i, pair in enumerate(pairs[2:], start=2):
        dst = (child_xs[child_i], child_y + 0.43)
        p_a, p_b = pair
        arrow(ax, (parent_xs[p_a], parent_y - 0.66), dst, color=BLUE, lw=1.1, alpha=0.48, rad=-0.12)
        arrow(ax, (parent_xs[p_b], parent_y - 0.66), dst, color=BLUE, lw=1.1, alpha=0.48, rad=0.12)

    ax.text(
        7.05,
        2.54,
        "parent crossover",
        ha="center",
        va="center",
        fontsize=10,
        color=BLUE,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.9),
    )
    save(fig, "theory_ga_crossover.pdf")


def draw_ga_mutation() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.35))
    clear_axes(ax, (0, 12.2), (0, 5.45))

    before = ga_crossover_children()
    delta = np.array([0.16, -0.13, 0.10, 0.18, -0.20, 0.08, -0.14, 0.12])
    after = np.clip(before + delta, 0.0, 1.0)
    xs = np.linspace(1.15, 11.05, len(before))

    ax.text(6.1, 5.12, "Offspring mutation", ha="center", va="center", fontsize=13, color=INK)
    ax.text(0.24, 3.65, "pred", ha="left", va="center", fontsize=10.5, color=GRAY)
    ax.text(0.24, 1.55, "po", ha="left", va="center", fontsize=10.5, color=GRAY)

    for i, (x, gene_before, gene_after, eps) in enumerate(zip(xs, before, after, delta), start=1):
        ax.text(x, 4.22, f"$x'_{i}$", ha="center", va="center", fontsize=9, color=INK)
        draw_ga_individual(ax, x, 3.55, float(gene_before), show_bar=False)
        draw_ga_individual(ax, x, 1.45, float(gene_after), label=f"$x''_{i}$", show_bar=False)
        arrow(
            ax,
            (x, 3.08),
            (x, 1.92),
            color=GRAY,
            lw=1.2,
            label=f"$\\epsilon={eps:+.2f}$",
            label_xy=(x, 2.50),
        )

    save(fig, "theory_ga_mutation.pdf")


def draw_neuroevolution_genome() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 4.9))
    clear_axes(ax, (0, 12), (0, 4.9))

    ax.text(6.0, 4.55, "Parametre siete ako genóm", ha="center", va="center", fontsize=13, color=INK)

    matrix_x = 0.55
    matrix_y = 1.05
    cell_w = 0.62
    cell_h = 0.46
    rows = [
        [r"$w_{11}$", r"$w_{12}$", r"$w_{13}$", r"$\cdots$", r"$w_{1n}$"],
        [r"$w_{21}$", r"$w_{22}$", r"$w_{23}$", r"$\cdots$", r"$w_{2n}$"],
        [r"$\vdots$", r"$\vdots$", r"$\vdots$", r"$\ddots$", r"$\vdots$"],
        [r"$w_{m1}$", r"$w_{m2}$", r"$w_{m3}$", r"$\cdots$", r"$w_{mn}$"],
    ]

    ax.text(matrix_x + 1.55, 3.55, "weight matrix $W$", ha="center", va="center", fontsize=10.5, color=INK)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            x = matrix_x + c * cell_w
            y = matrix_y + (len(rows) - 1 - r) * cell_h
            face = "white" if value in {r"$\vdots$", r"$\cdots$", r"$\ddots$"} else LIGHT_BLUE
            rect = patches.FancyBboxPatch(
                (x, y),
                cell_w,
                cell_h,
                boxstyle="round,pad=0.01,rounding_size=0.04",
                facecolor=face,
                edgecolor=BLUE,
                linewidth=1.0,
            )
            ax.add_patch(rect)
            ax.text(x + cell_w / 2, y + cell_h / 2, value, ha="center", va="center", fontsize=9.4, color=INK)

    arrow(ax, (4.05, 2.0), (5.2, 2.0), color=BLUE, lw=1.5, label="flattening", label_xy=(4.62, 2.36))

    vector_x = 5.55
    vector_y = 1.72
    vector_w = 3.95
    vector_h = 0.72
    vector = patches.FancyBboxPatch(
        (vector_x, vector_y),
        vector_w,
        vector_h,
        boxstyle="round,pad=0.05,rounding_size=0.09",
        facecolor=LIGHT_GREEN,
        edgecolor=GREEN,
        linewidth=1.5,
    )
    ax.add_patch(vector)
    ax.text(
        vector_x + vector_w / 2,
        vector_y + vector_h / 2,
        r"$[\,w_{11},\,w_{12},\,\ldots,\,w_{1n},\,w_{21},\,\ldots,\,w_{mn}\,]$",
        ha="center",
        va="center",
        fontsize=10.2,
        color=INK,
    )
    ax.text(vector_x + vector_w / 2, 2.78, "vektor parametrov", ha="center", va="center", fontsize=10.5, color=INK)

    arrow(ax, (9.75, 2.0), (10.45, 2.0), color=BLUE, lw=1.5)

    genome = patches.FancyBboxPatch(
        (10.58, 1.38),
        0.95,
        1.24,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
        linewidth=1.7,
    )
    ax.add_patch(genome)
    ax.text(11.055, 2.14, r"$x$", ha="center", va="center", fontsize=17, color=INK)
    ax.text(11.055, 1.78, "genóm", ha="center", va="center", fontsize=9.6, color=ORANGE)
    ax.text(11.055, 0.94, r"$x \equiv \operatorname{vec}(W)$", ha="center", va="center", fontsize=10, color=GRAY)

    save(fig, "theory_neuroevolution_genome.pdf")


def load_stanford_bunny() -> tuple[np.ndarray, np.ndarray]:
    mesh_path = OUT_DIR / "assets" / "stanford_bunny" / "bun_zipper_res3.ply"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Stanford Bunny mesh not found: {mesh_path}")

    mesh = trimesh.load(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)

    # Reorient and normalize the model so the figure is deterministic and compact.
    vertices = vertices[:, [0, 2, 1]]
    vertices = vertices - vertices.mean(axis=0)
    vertices = vertices / np.max(np.linalg.norm(vertices, axis=1))
    vertices[:, 2] *= 1.12
    return vertices, faces


def draw_bunny_mesh_overview() -> None:
    vertices, faces = load_stanford_bunny()

    centers = vertices[faces].mean(axis=1)
    target = np.array([-0.20, 0.25, 0.00])
    target_face_index = int(np.argmin(np.linalg.norm(centers - target, axis=1)))
    target_center = centers[target_face_index]
    nearest_faces = np.argsort(np.linalg.norm(centers - target_center, axis=1))[:80]

    fig = plt.figure(figsize=(9.8, 4.35))
    grid = fig.add_gridspec(1, 2, left=0.02, right=0.99, top=0.93, bottom=0.01, wspace=-0.30)
    ax_mesh = fig.add_subplot(grid[0, 0], projection="3d")
    ax_zoom = fig.add_subplot(grid[0, 1], projection="3d")

    for ax in (ax_mesh, ax_zoom):
        ax.set_axis_off()
        ax.view_init(elev=14, azim=135)
        ax.set_box_aspect((1.0, 1.0, 0.88))

    ax_mesh.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        color=LIGHT_BLUE,
        edgecolor="#94A3B8",
        linewidth=0.08,
        alpha=0.94,
        shade=True,
    )
    highlight_edges: set[tuple[int, int]] = set()
    for face_index in nearest_faces[:80]:
        tri_indices = faces[face_index]
        for a, b in ((tri_indices[0], tri_indices[1]), (tri_indices[1], tri_indices[2]), (tri_indices[2], tri_indices[0])):
            highlight_edges.add(tuple(sorted((int(a), int(b)))))
    for a, b in highlight_edges:
        segment = vertices[[a, b]]
        ax_mesh.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=RAY_BLUE,
            linewidth=1.05,
            alpha=0.92,
            zorder=10,
        )
    ax_mesh.set_xlim(-0.68, 0.68)
    ax_mesh.set_ylim(-0.68, 0.68)
    ax_mesh.set_zlim(-0.54, 0.70)
    ax_mesh.text2D(0.5, 0.95, "Model shown as a mesh", transform=ax_mesh.transAxes, ha="center", fontsize=12, color=INK)

    zoom_polys = [vertices[faces[i]] for i in nearest_faces]
    mesh_collection = Poly3DCollection(
        zoom_polys,
        facecolors=shaded_mesh_colors(zoom_polys, LIGHT_BLUE),
        edgecolors="#64748B",
        linewidths=0.55,
        alpha=0.96,
    )
    ax_zoom.add_collection3d(mesh_collection)
    radius = 0.15
    ax_zoom.set_xlim(target_center[0] - radius, target_center[0] + radius)
    ax_zoom.set_ylim(target_center[1] - radius, target_center[1] + radius)
    ax_zoom.set_zlim(target_center[2] - radius * 0.75, target_center[2] + radius * 0.75)
    ax_zoom.text2D(0.5, 0.95, "Triangle detail", transform=ax_zoom.transAxes, ha="center", fontsize=12, color=INK)

    fig.subplots_adjust(left=0.02, right=0.99, top=0.93, bottom=0.01, wspace=-0.30)
    save(fig, "theory_bunny_mesh_overview.pdf")


def draw_mesh_raycasting_detail() -> None:
    vertices, faces = load_stanford_bunny()
    centers = vertices[faces].mean(axis=1)
    target = np.array([0.10, -0.32, 0.02])
    target_face_index = int(np.argmin(np.linalg.norm(centers - target, axis=1)))
    target_center = centers[target_face_index]
    nearest_faces = np.argsort(np.linalg.norm(centers - target_center, axis=1))[:42]

    selected_triangle_3d = vertices[faces[target_face_index]]
    all_patch_points = vertices[faces[nearest_faces]].reshape(-1, 3)[:, [0, 2]]
    mins = all_patch_points.min(axis=0)
    spans = np.maximum(all_patch_points.max(axis=0) - mins, 1e-6)

    def project(points_3d: np.ndarray) -> np.ndarray:
        points_2d = points_3d[:, [0, 2]]
        normalized = (points_2d - mins) / spans
        return np.column_stack((4.05 + normalized[:, 0] * 3.25, 1.10 + normalized[:, 1] * 3.15))

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    clear_axes(ax, (0, 10), (0, 5.8))

    for face_index in nearest_faces:
        tri = project(vertices[faces[face_index]])
        poly = patches.Polygon(tri, closed=True, facecolor=LIGHT_BLUE, edgecolor="#64748B", linewidth=0.75, alpha=0.9)
        ax.add_patch(poly)

    selected_triangle = project(selected_triangle_3d)
    selected_poly = patches.Polygon(
        selected_triangle,
        closed=True,
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
        linewidth=1.9,
        alpha=0.98,
        zorder=3,
    )
    ax.add_patch(selected_poly)

    hit = selected_triangle.mean(axis=0)
    ray_start = np.array([1.25, hit[1] - 0.72])
    ray_dir = hit - ray_start
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    ray_before = hit - ray_dir * 3.7
    ray_after = hit + ray_dir * 1.2

    arrow(ax, tuple(ray_before), tuple(hit), color=RAY_BLUE, lw=2.4)
    ax.plot([hit[0], ray_after[0]], [hit[1], ray_after[1]], color=RAY_BLUE, linewidth=1.45, linestyle="--", alpha=0.8)
    ax.scatter([ray_before[0]], [ray_before[1]], color=INK, s=28, zorder=5)
    ax.scatter([hit[0]], [hit[1]], color=RED, s=56, zorder=5)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INK, markeredgecolor=INK, markersize=6, label="ray origin"),
        Line2D([0], [0], color=RAY_BLUE, linewidth=2.4, label="cast ray"),
        Line2D([0], [0], color=RAY_BLUE, linewidth=1.6, linestyle="--", label="ray continuation"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RED, markeredgecolor=RED, markersize=6, label="bod prieniku"),
        patches.Patch(facecolor=LIGHT_ORANGE, edgecolor=ORANGE, label="tested triangle"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=True,
        framealpha=0.96,
        fontsize=8.6,
        handlelength=2.0,
        columnspacing=1.1,
    )
    save(fig, "theory_mesh_raycasting_detail.pdf")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_agent_environment_loop()
    draw_scalar_vs_lexicographic()
    draw_blackbox_function()
    draw_neuron_layer_mlp()
    draw_activation_functions()
    draw_ga_initial_population()
    draw_ga_selection()
    draw_ga_crossover()
    draw_ga_mutation()
    draw_neuroevolution_genome()
    draw_bunny_mesh_overview()
    draw_mesh_raycasting_detail()


if __name__ == "__main__":
    main()
