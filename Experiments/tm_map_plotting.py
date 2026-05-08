from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend import Legend
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Map import MAP_BLOCK_SIZE, Map, MapBlock


SURFACE_COLORS = {
    "RoadTech": "#b9b9b9",
    "PlatformTech": "#b9b9b9",
    "DirtTech": "#a2642f",
    "RoadDirt": "#a2642f",
    "PlatformDirt": "#a2642f",
    "PlatformGrass": "#b7d984",
    "PlatformPlastic": "#e3cc30",
    "PlatformIce": "#8fd5e8",
    "PlatformSnow": "#dbe7ed",
    "Unknown": "#9f9f9f",
}

SPECIAL_COLORS = {
    "Start": "#16a34a",
    "Finish": "#dc2626",
    "Checkpoint": "#2563eb",
}

WALL_COLOR = "#262626"

SURFACE_LABELS = {
    "RoadTech": "Road",
    "PlatformTech": "Road",
    "DirtTech": "Dirt",
    "RoadDirt": "Dirt",
    "PlatformDirt": "Dirt",
    "PlatformGrass": "Grass",
    "PlatformPlastic": "Plastic",
    "PlatformIce": "Ice",
    "PlatformSnow": "Snow",
    "Unknown": "Unknown",
}


@dataclass(frozen=True)
class MapProjection:
    rotate: bool
    bounds: tuple[float, float, float, float]
    figsize: tuple[float, float]

    def points(self, points_xz: np.ndarray) -> np.ndarray:
        points_xz = np.asarray(points_xz, dtype=np.float64)
        if self.rotate:
            return points_xz[..., [1, 0]]
        return points_xz


def safe_output_name(map_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(map_name)).strip("_")


def _hex_to_rgb(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.asarray([int(color[idx:idx + 2], 16) for idx in (0, 2, 4)], dtype=np.float64) / 255.0


def _rgb_to_hex(rgb: np.ndarray) -> str:
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    return "#" + "".join(f"{int(round(value * 255)):02x}" for value in rgb)


def _shade_color(color: str, normalized_height: float) -> str:
    base = _hex_to_rgb(color)
    # Keep hue/surface identity dominant; sigmoid only makes non-flat maps read
    # as layered without pretending the color is an exact metric scale.
    t = float(np.clip(normalized_height, 0.0, 1.0))
    steepness = 5.0
    low = 1.0 / (1.0 + np.exp(steepness * 0.5))
    high = 1.0 / (1.0 + np.exp(-steepness * 0.5))
    shaped = (1.0 / (1.0 + np.exp(-steepness * (t - 0.5))) - low) / (high - low)
    factor = 0.56 + 0.58 * float(np.clip(shaped, 0.0, 1.0))
    return _rgb_to_hex(base * factor)


def _surface_color(block: MapBlock, *, checkpoint_as_straight: bool = False) -> str:
    if block.name in SPECIAL_COLORS and not (checkpoint_as_straight and block.name == "Checkpoint"):
        return SPECIAL_COLORS[block.name]
    return SURFACE_COLORS.get(block.surface_name, SURFACE_COLORS["Unknown"])


def _legend_items_for_map(game_map: Map, *, checkpoint_as_straight: bool = False) -> list[Patch]:
    existing_special = {
        str(block.name)
        for block in game_map.blocks.values()
        if block.name in SPECIAL_COLORS and not (checkpoint_as_straight and block.name == "Checkpoint")
    }
    existing_surfaces = {
        SURFACE_LABELS.get(str(block.surface_name), "Unknown")
        for block in game_map.blocks.values()
        if block.name not in SPECIAL_COLORS or (checkpoint_as_straight and block.name == "Checkpoint")
    }

    items: list[Patch] = []
    for name in ("Start", "Finish", "Checkpoint"):
        if name in existing_special:
            items.append(Patch(facecolor=SPECIAL_COLORS[name], label=name))

    surface_order = [
        ("Road", SURFACE_COLORS["RoadTech"]),
        ("Dirt", SURFACE_COLORS["PlatformDirt"]),
        ("Grass", SURFACE_COLORS["PlatformGrass"]),
        ("Plastic", SURFACE_COLORS["PlatformPlastic"]),
        ("Ice", SURFACE_COLORS["PlatformIce"]),
        ("Snow", SURFACE_COLORS["PlatformSnow"]),
        ("Unknown", SURFACE_COLORS["Unknown"]),
    ]
    for label, color in surface_order:
        if label in existing_surfaces:
            items.append(Patch(facecolor=color, label=label))

    if any(len(_wall_segments(block)[0]) for block in game_map.blocks.values()):
        items.append(Patch(facecolor=WALL_COLOR, label="Edges"))
    return items


def _road_triangles(block: MapBlock) -> tuple[np.ndarray, np.ndarray]:
    mesh = block.get_road_mesh()
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if triangles.size == 0:
        return np.empty((0, 3, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
    points_xz = triangles[..., [0, 2]]
    heights = triangles[..., 1].mean(axis=1)
    valid = []
    valid_heights = []
    for triangle, height in zip(points_xz, heights):
        area = abs(float(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])))
        if area > 1e-4:
            valid.append(triangle)
            valid_heights.append(float(height))
    if not valid:
        return np.empty((0, 3, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
    return np.asarray(valid, dtype=np.float64), np.asarray(valid_heights, dtype=np.float64)


def _wall_segments(block: MapBlock) -> tuple[np.ndarray, np.ndarray]:
    mesh = block.get_walls_mesh()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        return np.empty((0, 2, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
    unique: dict[tuple[float, float, float, float], tuple[np.ndarray, float]] = {}
    for face in faces:
        for a_idx, b_idx in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            a = vertices[int(a_idx)][[0, 2]]
            b = vertices[int(b_idx)][[0, 2]]
            if float(np.linalg.norm(b - a)) <= 1e-4:
                continue
            mean_height = float((vertices[int(a_idx)][1] + vertices[int(b_idx)][1]) * 0.5)
            rounded_a = tuple(np.round(a, 3))
            rounded_b = tuple(np.round(b, 3))
            key = (*rounded_a, *rounded_b)
            reverse_key = (*rounded_b, *rounded_a)
            if reverse_key in unique:
                continue
            unique[key] = (np.asarray([a, b], dtype=np.float64), mean_height)
    if not unique:
        return np.empty((0, 2, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
    segments = [value[0] for value in unique.values()]
    segment_heights = [value[1] for value in unique.values()]
    return (
        np.stack(segments).astype(np.float64, copy=False),
        np.asarray(segment_heights, dtype=np.float64),
    )


def _point_in_triangle(point: np.ndarray, triangle: np.ndarray, eps: float = 1e-6) -> bool:
    point = np.asarray(point, dtype=np.float64)
    triangle = np.asarray(triangle, dtype=np.float64)
    a, b, c = triangle
    v0 = c - a
    v1 = b - a
    v2 = point - a
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) <= eps:
        return False
    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    return u >= -eps and v >= -eps and (u + v) <= 1.0 + eps


def _segment_is_covered_by_higher_road(
    segment: np.ndarray,
    segment_height: float,
    road_triangles: np.ndarray,
    road_heights: np.ndarray,
    *,
    height_epsilon: float = 0.5,
) -> bool:
    if road_triangles.size == 0:
        return False
    candidate_indices = np.flatnonzero(road_heights > float(segment_height) + float(height_epsilon))
    if candidate_indices.size == 0:
        return False

    sample_points = [
        np.asarray(segment[0], dtype=np.float64) * (1.0 - t)
        + np.asarray(segment[1], dtype=np.float64) * t
        for t in (0.25, 0.5, 0.75)
    ]
    for point in sample_points:
        for index in candidate_indices:
            if _point_in_triangle(point, road_triangles[int(index)], eps=1e-5):
                return True
    return False


def _all_road_heights(game_map: Map) -> np.ndarray:
    values: list[np.ndarray] = []
    for block in game_map.blocks.values():
        _, heights = _road_triangles(block)
        if heights.size:
            values.append(heights)
    if not values:
        return np.asarray([0.0, 1.0], dtype=np.float64)
    return np.concatenate(values)


def _projection_for_map(game_map: Map) -> MapProjection:
    bounds = np.asarray(game_map.get_mesh().bounds, dtype=np.float64)
    min_x, max_x = float(bounds[0, 0]), float(bounds[1, 0])
    min_z, max_z = float(bounds[0, 2]), float(bounds[1, 2])
    span_x = max(1.0, max_x - min_x)
    span_z = max(1.0, max_z - min_z)
    rotate = span_z > span_x
    plot_span_x = span_z if rotate else span_x
    plot_span_y = span_x if rotate else span_z
    aspect = plot_span_x / max(1.0, plot_span_y)
    width = 13.0
    height = float(np.clip(width / max(1.25, aspect), 6.2, 8.8))
    return MapProjection(
        rotate=bool(rotate),
        bounds=(min_x, min_z, max_x, max_z),
        figsize=(width, height),
    )


def render_map_background(
    ax: Axes,
    game_map: Map,
    *,
    projection: MapProjection | None = None,
    show_legend: bool = False,
    alpha: float = 1.0,
    checkpoint_as_straight: bool = False,
) -> MapProjection:
    projection = projection or _projection_for_map(game_map)
    heights = _all_road_heights(game_map)
    height_min = float(np.min(heights))
    height_max = float(np.max(heights))
    height_range = max(1e-6, height_max - height_min)

    # Draw lower blocks first, so overpasses naturally cover lower roads in top-down projection.
    blocks = sorted(
        game_map.blocks.values(),
        key=lambda block: float(np.mean(np.asarray(block.get_bounds(), dtype=np.float64)[:, 1])),
    )
    road_triangle_parts: list[np.ndarray] = []
    road_height_parts: list[np.ndarray] = []
    for block in blocks:
        triangles, triangle_heights = _road_triangles(block)
        if triangles.size:
            road_triangle_parts.append(triangles)
            road_height_parts.append(triangle_heights)
    all_road_triangles = (
        np.concatenate(road_triangle_parts, axis=0)
        if road_triangle_parts
        else np.empty((0, 3, 2), dtype=np.float64)
    )
    all_road_heights = (
        np.concatenate(road_height_parts, axis=0)
        if road_height_parts
        else np.empty((0,), dtype=np.float64)
    )

    visible_wall_segments: list[np.ndarray] = []
    covered_wall_segments: list[np.ndarray] = []
    for block in blocks:
        triangles, triangle_heights = _road_triangles(block)
        if triangles.size:
            base_color = _surface_color(block, checkpoint_as_straight=checkpoint_as_straight)
            polygons = [projection.points(triangle) for triangle in triangles]
            colors = [
                _shade_color(
                    base_color,
                    0.5 if height_range < 1.0 else (float(height) - height_min) / height_range,
                )
                for height in triangle_heights
            ]
            collection = PolyCollection(
                polygons,
                facecolors=colors,
                edgecolors="none",
                linewidths=0.0,
                alpha=float(alpha),
                zorder=1 + float(np.mean(triangle_heights)) * 0.001,
            )
            ax.add_collection(collection)
        segments, segment_heights = _wall_segments(block)
        if len(segments):
            for segment, segment_height in zip(segments, segment_heights):
                projected_segment = projection.points(segment)
                if _segment_is_covered_by_higher_road(
                    segment,
                    float(segment_height),
                    all_road_triangles,
                    all_road_heights,
                ):
                    covered_wall_segments.append(projected_segment)
                else:
                    visible_wall_segments.append(projected_segment)

    if visible_wall_segments:
        ax.add_collection(
            LineCollection(
                visible_wall_segments,
                colors=WALL_COLOR,
                linewidths=0.95,
                alpha=0.82,
                zorder=50,
            )
        )
    if covered_wall_segments:
        ax.add_collection(
            LineCollection(
                covered_wall_segments,
                colors=WALL_COLOR,
                linewidths=0.9,
                linestyles=(0, (3.0, 3.0)),
                alpha=0.55,
                zorder=55,
            )
        )

    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.margins(0.04)
    if show_legend:
        legend_items = _legend_items_for_map(game_map, checkpoint_as_straight=checkpoint_as_straight)
        ax.legend(
            handles=legend_items,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.035),
            ncol=min(7, len(legend_items)),
            frameon=False,
            fontsize=8,
        )
    return projection


def add_map_legend(
    fig,
    ax: Axes,
    game_map: Map,
    height_min: float,
    height_max: float,
    *,
    flat_height_threshold: float = 0.5,
    checkpoint_as_straight: bool = False,
) -> Legend:
    legend_items = _legend_items_for_map(game_map, checkpoint_as_straight=checkpoint_as_straight)
    legend = fig.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.43, 0.018),
        ncol=min(7, len(legend_items)),
        frameon=False,
        fontsize=10,
        handlelength=1.2,
        handleheight=1.0,
        columnspacing=1.1,
    )

    if abs(float(height_max) - float(height_min)) < float(flat_height_threshold):
        return legend

    # Anchor the height legend to the actual rendered data extent, not to a
    # fixed figure/axes coordinate. This keeps the bar close on tall maps where
    # equal-aspect axes otherwise leave a lot of empty right-side space.
    fig.canvas.draw()
    data_bbox = ax.dataLim
    data_points = np.asarray(
        [
            [data_bbox.x0, data_bbox.y0],
            [data_bbox.x1, data_bbox.y0],
            [data_bbox.x1, data_bbox.y1],
            [data_bbox.x0, data_bbox.y1],
        ],
        dtype=np.float64,
    )
    figure_points = fig.transFigure.inverted().transform(ax.transData.transform(data_points))
    data_x1 = float(np.max(figure_points[:, 0]))
    data_y0 = float(np.min(figure_points[:, 1]))
    data_y1 = float(np.max(figure_points[:, 1]))
    data_height = max(0.1, data_y1 - data_y0)
    cax = fig.add_axes(
        [
            min(0.965, data_x1 + 0.018),
            data_y0 + 0.12 * data_height,
            0.016,
            0.64 * data_height,
        ]
    )
    gradient = np.linspace(0.0, 1.0, 256).reshape(-1, 1)
    cmap = LinearSegmentedColormap.from_list("tm_height_gradient", ["#3e3e3e", "#d8d8d8"])
    cax.imshow(gradient, aspect="auto", cmap=cmap, origin="lower")
    cax.set_xticks([])
    cax.set_yticks([0, 255])
    cax.set_yticklabels(["Low", "High"], fontsize=10)
    cax.set_title("Height", fontsize=11, pad=6)
    for spine in cax.spines.values():
        spine.set_visible(False)
    return legend


def plot_map_overview(
    map_name: str,
    output_path: Path,
    *,
    dpi: int = 180,
    show_legend: bool = False,
    title: bool = True,
    checkpoint_as_straight: bool = False,
) -> Path:
    game_map = Map(map_name)
    projection = _projection_for_map(game_map)
    fig, ax = plt.subplots(figsize=(projection.figsize[0] + 0.35, projection.figsize[1] + 0.45))
    render_map_background(
        ax,
        game_map,
        projection=projection,
        show_legend=False,
        checkpoint_as_straight=checkpoint_as_straight,
    )
    if title:
        ax.set_title(str(map_name), fontsize=13, pad=6)
    heights = _all_road_heights(game_map)
    fig.subplots_adjust(left=0.055, right=0.835, top=0.94, bottom=0.13)
    add_map_legend(
        fig,
        ax,
        game_map,
        float(np.min(heights)),
        float(np.max(heights)),
        checkpoint_as_straight=checkpoint_as_straight,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return output_path
