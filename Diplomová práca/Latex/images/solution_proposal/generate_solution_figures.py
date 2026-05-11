from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


OUT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiments.tm_map_plotting import (  # noqa: E402
    _all_road_heights,
    _projection_for_map,
    add_map_legend,
    render_map_background,
)
from Map import Map  # noqa: E402


BLUE = "#2563eb"
GREEN = "#16a34a"
ORANGE = "#f59e0b"
RED = "#dc2626"
GRAY = "#737373"
LIGHT_BG = "#f7f9fb"
ROAD = "#b9b9b9"
EDGE = "#262626"
INK = "#1F2937"


def stable_screenshot_assets() -> None:
    copies = {
        "editor_view.png": "solution_blocks_editor_view.png",
        "game_map.png": "solution_game_map_overview.png",
        "surfaces.png": "solution_surface_blocks.png",
        "height_blocks.png": "solution_height_blocks.png",
        "kayboard_controller.png": "solution_keyboard_vs_controller.png",
    }
    for source_name, target_name in copies.items():
        shutil.copyfile(OUT_DIR / source_name, OUT_DIR / target_name)


def set_axes_equal(ax) -> None:
    limits = np.asarray(
        [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()],
        dtype=np.float64,
    )
    centers = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def mesh_face_colors(
    triangles: np.ndarray,
    base_hex: str = ROAD,
    *,
    alpha: float = 0.95,
) -> list[tuple[float, float, float, float]]:
    base = np.asarray(
        [int(base_hex[i : i + 2], 16) for i in (1, 3, 5)],
        dtype=np.float64,
    ) / 255.0
    light_dir = np.asarray([-0.45, -0.35, 0.82], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)
    colors: list[tuple[float, float, float, float]] = []
    for triangle in triangles:
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            intensity = 0.72
        else:
            normal = normal / norm
            intensity = 0.58 + 0.36 * abs(float(np.dot(normal, light_dir)))
        rgb = np.clip(base * intensity, 0.0, 1.0)
        colors.append((float(rgb[0]), float(rgb[1]), float(rgb[2]), alpha))
    return colors


def add_mesh(
    ax,
    mesh: trimesh.Trimesh,
    *,
    edge_lw: float = 0.45,
    equal_axes: bool = True,
    face_alpha: float = 0.95,
) -> None:
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    collection = Poly3DCollection(
        triangles,
        facecolors=mesh_face_colors(triangles, alpha=face_alpha),
        edgecolors=(0.12, 0.12, 0.12, 0.46),
        linewidths=edge_lw,
    )
    ax.add_collection3d(collection)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    ax.set_xlim(bounds[0, 0], bounds[1, 0])
    ax.set_ylim(bounds[0, 1], bounds[1, 1])
    ax.set_zlim(bounds[0, 2], bounds[1, 2])
    if equal_axes:
        set_axes_equal(ax)
    else:
        span = np.maximum(bounds[1] - bounds[0], 1.0)
        ax.set_box_aspect((float(span[0]), max(10.0, float(span[1]) * 2.8), float(span[2])))
    ax.set_axis_off()
    ax.set_proj_type("ortho")
    ax.view_init(elev=27, azim=-58)
    ax.set_facecolor(LIGHT_BG)


def draw_block_meshes() -> None:
    specs = [
        ("RoadTechStraight.obj", "rovný blok"),
        ("RoadTechCurve1.obj", "krátka zákruta"),
        ("RoadTechCurve2.obj", "dlhšia zákruta"),
    ]
    fig = plt.figure(figsize=(12.5, 4.3), facecolor="white")
    for idx, (mesh_name, title) in enumerate(specs, start=1):
        mesh = trimesh.load(ROOT / "Meshes" / mesh_name, force="mesh")
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        add_mesh(ax, mesh)
        ax.set_title(title, fontsize=12, pad=4)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.04, wspace=0.02)
    fig.savefig(OUT_DIR / "solution_block_meshes.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_DIR / "solution_block_meshes.png", dpi=180, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _surface_profile_points(mesh: trimesh.Trimesh, *, num_points: int = 9) -> np.ndarray:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    span = bounds[1] - bounds[0]
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)

    # The height-test map contains a clear ascending road strip.  We sample only
    # upward-facing road triangles in that strip, not nearby walls.  This keeps
    # the drawn ray on the road profile instead of accidentally snapping to an
    # edge vertex.
    ramp_mask = (
        (normals[:, 1] > 0.55)
        & (centers[:, 0] > bounds[0, 0] + 0.28 * span[0])
        & (centers[:, 0] < bounds[0, 0] + 0.50 * span[0])
        & (centers[:, 2] > bounds[0, 2] + 0.40 * span[2])
        & (centers[:, 2] < bounds[0, 2] + 0.72 * span[2])
    )
    ramp_centers = centers[ramp_mask]
    if len(ramp_centers) < 4:
        raise RuntimeError("Could not find enough road-surface triangles for the height profile figure.")

    ramp_centers = ramp_centers[np.argsort(ramp_centers[:, 2])]
    groups = np.array_split(ramp_centers, min(num_points, max(4, len(ramp_centers) // 2)))
    points = np.asarray([group.mean(axis=0) for group in groups], dtype=np.float64)
    points[:, 1] += 0.7
    return points


def _height_profile_crop(mesh: trimesh.Trimesh, profile: np.ndarray) -> trimesh.Trimesh:
    x_min, x_max = float(np.min(profile[:, 0]) - 26.0), float(np.max(profile[:, 0]) + 26.0)
    z_min, z_max = float(np.min(profile[:, 2]) - 22.0), float(np.max(profile[:, 2]) + 22.0)
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    face_indices = np.flatnonzero(
        (normals[:, 1] > 0.20)
        & (triangles[:, :, 0].min(axis=1) >= x_min)
        & (triangles[:, :, 0].max(axis=1) <= x_max)
        & (triangles[:, :, 2].min(axis=1) >= z_min)
        & (triangles[:, :, 2].max(axis=1) <= z_max)
    )
    if len(face_indices) == 0:
        return mesh
    return mesh.submesh([face_indices], append=True, repair=False)


def _height_profile_visual_transform(points: np.ndarray, *, vertical_scale: float = 3.2) -> np.ndarray:
    visual = np.asarray(points, dtype=np.float64).copy()
    return np.column_stack(
        [
            visual[:, 0],
            visual[:, 2],
            visual[:, 1] * vertical_scale,
        ]
    )


def draw_height_raycast_profile() -> None:
    # Syntetická didaktická scéna: rovina -> kopec -> rovina. Reálny mesh v
    # predchádzajúcej verzii prekrýval lúč, preto tu používame jednoduchú
    # trojuholníkovú sieť, na ktorej je zlom lúča jasne viditeľný.
    x = np.asarray([-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0], dtype=np.float64)
    h = np.asarray([0.0, 0.0, 1.15, 2.35, 1.15, 0.0, 0.0], dtype=np.float64)
    half_width = 1.65

    vertices: list[list[float]] = []
    for xi, hi in zip(x, h):
        vertices.append([xi, -half_width, hi])
        vertices.append([xi, half_width, hi])
    vertices_np = np.asarray(vertices, dtype=np.float64)

    faces: list[list[int]] = []
    for idx in range(len(x) - 1):
        left0, right0 = 2 * idx, 2 * idx + 1
        left1, right1 = 2 * (idx + 1), 2 * (idx + 1) + 1
        faces.append([left0, left1, right1])
        faces.append([left0, right1, right0])
    surface_mesh = trimesh.Trimesh(vertices=vertices_np, faces=np.asarray(faces), process=False)

    surface_profile = np.column_stack([x, np.zeros_like(x), h])
    ray_profile = surface_profile.copy()
    ray_profile[:, 2] += 0.46

    fig = plt.figure(figsize=(9.2, 5.05), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    add_mesh(ax, surface_mesh, edge_lw=0.42, equal_axes=False, face_alpha=0.70)

    highlighted_quads = []
    for idx in (2, 3):
        quad = np.asarray(
            [
                vertices_np[2 * idx],
                vertices_np[2 * (idx + 1)],
                vertices_np[2 * (idx + 1) + 1],
                vertices_np[2 * idx + 1],
            ],
            dtype=np.float64,
        )
        quad[:, 2] += 0.025
        highlighted_quads.append(quad)
    highlight = Poly3DCollection(
        highlighted_quads,
        facecolors=(1.0, 0.61, 0.12, 0.38),
        edgecolors=(0.92, 0.36, 0.03, 0.95),
        linewidths=1.35,
    )
    ax.add_collection3d(highlight)

    ax.plot(
        surface_profile[:, 0],
        surface_profile[:, 1],
        surface_profile[:, 2] + 0.035,
        color="#475569",
        linewidth=1.2,
        linestyle="--",
        alpha=0.65,
    )
    ax.plot(
        ray_profile[:, 0],
        ray_profile[:, 1],
        ray_profile[:, 2],
        color=BLUE,
        linewidth=5.0,
        solid_capstyle="round",
        zorder=20,
    )
    ax.scatter(
        ray_profile[0, 0],
        ray_profile[0, 1],
        ray_profile[0, 2],
        s=42,
        color=INK,
        depthshade=False,
        zorder=5,
    )
    ax.scatter(
        ray_profile[-1, 0],
        ray_profile[-1, 1],
        ray_profile[-1, 2],
        s=58,
        color=RED,
        depthshade=False,
        zorder=6,
    )
    ax.quiver(
        ray_profile[-2, 0],
        ray_profile[-2, 1],
        ray_profile[-2, 2],
        ray_profile[-1, 0] - ray_profile[-2, 0],
        ray_profile[-1, 1] - ray_profile[-2, 1],
        ray_profile[-1, 2] - ray_profile[-2, 2],
        color=BLUE,
        linewidth=2.0,
        arrow_length_ratio=0.18,
        normalize=False,
    )

    for point, surface_point in zip(ray_profile[1:-1], surface_profile[1:-1]):
        ax.plot(
            [surface_point[0], point[0]],
            [surface_point[1], point[1]],
            [surface_point[2], point[2]],
            color="#64748b",
            linewidth=0.8,
            linestyle=":",
            alpha=0.55,
        )

    ax.set_xlim(-6.8, 6.8)
    ax.set_ylim(-2.75, 2.75)
    ax.set_zlim(-0.20, 3.45)
    ax.set_box_aspect((2.55, 1.0, 0.78))
    ax.view_init(elev=24, azim=-58)
    ax.set_facecolor("white")
    ax.set_axis_off()
    proxy_handles = [
        Line2D([0], [0], color=BLUE, lw=3.0, label="lúč po profile trate"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=INK, markersize=7, label="počiatok lúča"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markersize=8, label="koncový bod / stena"),
        patches.Patch(facecolor="#f59e0b", edgecolor="#ea580c", alpha=0.45, label="lokálna plocha"),
    ]
    ax.legend(
        handles=proxy_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=1.7,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.12)
    fig.savefig(OUT_DIR / "solution_height_raycast_profile.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_DIR / "solution_height_raycast_profile.png", dpi=190, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def load_map_without_mesh_export(map_name: str) -> Map:
    original_generate_map_mesh = Map.generate_map_mesh

    def generate_map_mesh_without_export(self) -> None:
        scene = trimesh.Scene()
        for block in self.blocks.values():
            scene.add_geometry(block.get_mesh())
        self.mesh = scene.dump(concatenate=True)

    Map.generate_map_mesh = generate_map_mesh_without_export
    try:
        return Map(map_name)
    finally:
        Map.generate_map_mesh = original_generate_map_mesh


def best_teacher_attempt() -> Path:
    summary_path = (
        ROOT
        / "Diplomová práca"
        / "Experiments"
        / "supervised_map_specialists_20260505"
        / "analysis"
        / "single_surface_flat"
        / "teacher_paths_summary.csv"
    )
    data = np.genfromtxt(summary_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    rows = np.atleast_1d(data)
    valid = [
        row
        for row in rows
        if int(row["finished"]) == 1 and int(row["crash_index"]) < 0 and float(row["finish_time"]) > 0
    ]
    if not valid:
        valid = [row for row in rows if int(row["finished"]) == 1 and float(row["finish_time"]) > 0]
    best = min(valid, key=lambda row: float(row["finish_time"]))
    return ROOT / str(best["attempt_file"])


def draw_track_drive_preview() -> None:
    attempt_path = best_teacher_attempt()
    with np.load(attempt_path) as attempt:
        positions = np.asarray(attempt["positions"], dtype=np.float64)
        speeds = np.asarray(attempt["speeds"], dtype=np.float64)
        finish_time = float(np.asarray(attempt["finish_time"]).item())

    game_map = load_map_without_mesh_export("single_surface_flat")
    projection = _projection_for_map(game_map)
    fig, ax = plt.subplots(figsize=(10.8, 7.2), facecolor="white")
    render_map_background(ax, game_map, projection=projection, show_legend=False, alpha=0.92)

    points = projection.points(positions[:, [0, 2]])
    speed_floor_mps = 100.0 / 3.6
    speed_ceiling_mps = max(speed_floor_mps + 1e-6, float(np.nanmax(speeds)))
    display_speeds = np.clip(speeds, speed_floor_mps, speed_ceiling_mps)
    norm = np.clip((display_speeds - speed_floor_mps) / (speed_ceiling_mps - speed_floor_mps), 0.0, 1.0)
    speed_cmap = plt.get_cmap("turbo_r")
    for i in range(len(points) - 1):
        ax.plot(
            points[i : i + 2, 0],
            points[i : i + 2, 1],
            color=speed_cmap(float(norm[i])),
            linewidth=2.2,
            alpha=0.95,
            solid_capstyle="round",
            zorder=120,
        )

    ax.text(
        0.02,
        0.96,
        f"zaznamenaný prejazd: {finish_time:.2f} s",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#d1d5db", alpha=0.92),
    )
    heights = _all_road_heights(game_map)
    ax.invert_xaxis()
    fig.subplots_adjust(left=0.03, right=0.86, top=0.98, bottom=0.12)
    add_map_legend(fig, ax, game_map, float(np.min(heights)), float(np.max(heights)))
    cax = fig.add_axes([0.88, 0.34, 0.018, 0.36])
    gradient = np.linspace(0.0, 1.0, 256).reshape(-1, 1)
    cax.imshow(gradient, aspect="auto", cmap=speed_cmap, origin="lower")
    cax.set_xticks([])
    cax.set_yticks([0, 255])
    speed_floor_kmh = speed_floor_mps * 3.6
    speed_ceiling_kmh = speed_ceiling_mps * 3.6
    cax.set_yticklabels([f"≤ {speed_floor_kmh:.0f} km/h", f"{speed_ceiling_kmh:.0f} km/h"], fontsize=10)
    cax.set_title("rýchlosť", fontsize=10, pad=6)
    for spine in cax.spines.values():
        spine.set_visible(False)
    fig.savefig(OUT_DIR / "solution_track_drive_preview.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_DIR / "solution_track_drive_preview.png", dpi=180, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def draw_game_vs_agent_view() -> None:
    game = mpimg.imread(OUT_DIR / "game_view.png")
    agent = mpimg.imread(OUT_DIR / "agent_view.png")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), facecolor="white")
    for ax, image, title in zip(axes, (game, agent), ("pohľad hry", "virtuálna reprezentácia")):
        ax.imshow(image)
        ax.set_title(title, fontsize=12, pad=6)
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.025)
    fig.savefig(OUT_DIR / "solution_game_vs_agent_view.png", dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def arrow(ax, start: tuple[float, float], end: tuple[float, float], *, color: str = BLUE) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, shrinkA=0, shrinkB=0),
    )


def box(ax, xy: tuple[float, float], text: str, *, fc: str, ec: str, width: float = 1.62) -> None:
    x, y = xy
    rect = plt.Rectangle(
        (x - width / 2, y - 0.33),
        width,
        0.66,
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.6,
        joinstyle="round",
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=10.5, color="#111827")


def draw_dataflow_loop() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 3.8), facecolor="white")
    ax.set_xlim(-0.12, 13.05)
    ax.set_ylim(0.0, 3.0)
    ax.axis("off")

    env_fc = "#e8f5ef"
    env_ec = "#2f8f5b"
    py_fc = "#eef2ff"
    py_ec = "#4f46e5"
    policy_fc = "#fff7ed"
    policy_ec = "#f59e0b"

    positions = {
        "tm": (0.92, 1.55),
        "script": (2.88, 1.55),
        "runtime": (4.82, 1.55),
        "world": (6.76, 1.55),
        "obs": (8.70, 1.55),
        "policy": (10.46, 1.55),
        "action": (12.02, 1.55),
    }
    widths = {
        "tm": 1.45,
        "script": 1.25,
        "runtime": 1.35,
        "world": 1.35,
        "obs": 1.35,
        "policy": 1.18,
        "action": 1.08,
    }

    for key, text, fc, ec in (
        ("tm", "Trackmania", env_fc, env_ec),
        ("script", "herný\nskript", env_fc, env_ec),
        ("runtime", "Python\nruntime", py_fc, py_ec),
        ("world", "virtuálna\nmapa", py_fc, py_ec),
        ("obs", "pozorovanie", py_fc, py_ec),
        ("policy", "politika", policy_fc, policy_ec),
        ("action", "akcia", env_fc, env_ec),
    ):
        box(ax, positions[key], text, fc=fc, ec=ec, width=widths[key])

    order = ["tm", "script", "runtime", "world", "obs", "policy", "action"]
    for left, right in zip(order, order[1:]):
        start = (positions[left][0] + widths[left] / 2 + 0.08, positions[left][1])
        end = (positions[right][0] - widths[right] / 2 - 0.08, positions[right][1])
        arrow(ax, start, end, color=BLUE if right != "action" else ORANGE)

    action_right = positions["action"][0] + widths["action"] / 2 + 0.05
    tm_left = positions["tm"][0] - widths["tm"] / 2 - 0.05
    tm_entry = (positions["tm"][0] - widths["tm"] / 2 - 0.02, positions["tm"][1])
    ax.plot(
        [action_right, action_right, tm_left, tm_left],
        [positions["action"][1], 0.45, 0.45, positions["tm"][1]],
        color=env_ec,
        lw=1.6,
    )
    ax.annotate(
        "",
        xy=tm_entry,
        xytext=(tm_left, positions["tm"][1]),
        arrowprops=dict(arrowstyle="-|>", color=env_ec, lw=1.8),
    )
    ax.text(6.22, 0.27, "akcia sa prejaví v hre a vznikne nový stav", ha="center", va="center", fontsize=9.5, color=env_ec)

    ax.text(5.92, 2.48, "spracovanie v našom systéme", ha="center", va="center", fontsize=10, color=py_ec)
    ax.plot([3.45, 9.38], [2.28, 2.28], color=py_ec, lw=1.1, alpha=0.55)
    ax.text(0.85, 2.48, "cieľové prostredie", ha="center", va="center", fontsize=10, color=env_ec)
    ax.text(10.66, 2.48, "rozhodnutie", ha="center", va="center", fontsize=10, color=policy_ec)

    fig.savefig(OUT_DIR / "solution_dataflow_loop.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_DIR / "solution_dataflow_loop.png", dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def clear_loop_axes(ax: plt.Axes, xlim=(0.90, 12.10), ylim=(0, 6.2)) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def loop_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    fc: str,
    ec: str,
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


def loop_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    lw: float = 1.6,
    linestyle: str | tuple = "solid",
    alpha: float = 1.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=lw,
            color=color,
            linestyle=linestyle,
            alpha=alpha,
            connectionstyle="arc3,rad=0",
        )
    )


def draw_game_agent_loop() -> None:
    blue = "#4C78A8"
    light_blue = "#EAF2FB"
    green = "#54A24B"
    light_green = "#EDF7EA"

    fig, ax = plt.subplots(figsize=(10.2, 5.9), facecolor="white")
    clear_loop_axes(ax, (0.90, 12.10), (0, 6.2))

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
        edgecolor=blue,
        facecolor=light_blue,
    )
    ax.add_patch(agent_patch)
    ax.text(agent_x + 0.30, agent_top - 0.33, "Agent", ha="left", va="center", fontsize=12, color=blue, weight="bold")

    policy_x = 3.05
    policy_w = 2.15
    policy_h = 0.92
    policy_cx = policy_x
    gate_x = 5.92
    gate_w = 1.62
    sensor_y = agent_top - gate_margin_y
    actuator_y = agent_bottom + gate_margin_y
    policy_y = agent_center_y - policy_h / 2.0

    loop_box(ax, (policy_cx - policy_w / 2, policy_y), policy_w, policy_h, "politika\n$\\pi_\\theta$", fc="white", ec=blue, fontsize=12)
    loop_box(ax, (gate_x, sensor_y - 0.39), gate_w, 0.78, "herný skript\n+ 3D projekcia", fc="white", ec=blue, fontsize=9.4, lw=1.35)
    loop_box(ax, (gate_x, actuator_y - 0.34), gate_w, 0.68, "Virtual\nGamepad", fc="white", ec=blue, fontsize=9.8, lw=1.35)

    env_x = 9.05
    env_w = 2.75
    env_patch = patches.FancyBboxPatch(
        (env_x, agent_y),
        env_w,
        agent_h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.7,
        edgecolor=green,
        facecolor=light_green,
    )
    ax.add_patch(env_patch)
    ax.text(env_x + 0.18, agent_top - 0.34, "Trackmania", ha="left", va="center", fontsize=11.5, color=green, weight="bold")

    loop_arrow(ax, (env_x, sensor_y), (gate_x + gate_w, sensor_y), color=green, lw=1.55)
    ax.plot([gate_x, policy_cx], [sensor_y, sensor_y], color=blue, linewidth=1.35)
    loop_arrow(ax, (policy_cx, sensor_y), (policy_cx, policy_y + policy_h), color=blue, lw=1.35)
    ax.text(
        4.60,
        sensor_y,
        "pozorovanie $o_t$\nodmena $r_t$",
        ha="center",
        va="center",
        fontsize=10,
        color=blue,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.92),
    )

    ax.plot([policy_cx, policy_cx], [policy_y, actuator_y], color=blue, linewidth=1.35)
    loop_arrow(ax, (policy_cx, actuator_y), (gate_x, actuator_y), color=blue, lw=1.35)
    ax.text(
        4.60,
        actuator_y,
        "akcia\n$a_t$",
        ha="center",
        va="center",
        fontsize=10,
        color=blue,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.92),
    )

    loop_arrow(ax, (gate_x + gate_w, actuator_y), (env_x, actuator_y), color=green, lw=1.55)

    transition_x = env_x + env_w - 0.78
    for xs, ys in (
        ([env_x, transition_x], [actuator_y, actuator_y]),
        ([transition_x, transition_x], [actuator_y, sensor_y]),
        ([transition_x, env_x], [sensor_y, sensor_y]),
    ):
        ax.plot(xs, ys, color=green, linewidth=1.1, linestyle=(0, (2, 3)), alpha=0.9)
    ax.text(transition_x + 0.22, (actuator_y + sensor_y) / 2.0, "$a_t$", ha="center", va="center", fontsize=11, color=green)
    state_label_bbox = dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.86)
    ax.text(transition_x, actuator_y, "$s_t$", ha="center", va="center", fontsize=11, color=INK, bbox=state_label_bbox)
    ax.text(transition_x, sensor_y, "$s_{t+1}$", ha="center", va="center", fontsize=11, color=INK, bbox=state_label_bbox)

    fig.savefig(OUT_DIR / "solution_game_agent_loop.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "solution_game_agent_loop.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stable_screenshot_assets()
    draw_block_meshes()
    draw_height_raycast_profile()
    draw_track_drive_preview()
    draw_game_vs_agent_view()
    draw_dataflow_loop()
    draw_game_agent_loop()
    print(f"Generated solution proposal figures in: {OUT_DIR}")


if __name__ == "__main__":
    main()
