from __future__ import annotations

import math
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, transforms
from matplotlib.collections import LineCollection


OUT_DIR = Path(__file__).resolve().parent
CAR_ICON = OUT_DIR / "car.png"
BLOCK_SAMPLES = 80


COLORS = {
    "road_fill": "#f4f2ec",
    "road_edge": "#2d2a25",
    "block_boundary": "#4b4a45",
    "center_future": "#355c7d",
    "center_passed": "#d94a3a",
    "transition": "#2f8f5b",
    "projection": "#f59e0b",
    "ray": "#2563eb",
    "left_distance": "#0ea5e9",
    "right_distance": "#7c3aed",
    "heading": "#111827",
    "velocity": "#16a34a",
    "side_velocity": "#dc2626",
    "text": "#262626",
    "muted": "#6b7280",
    "panel_bg": "#fffaf0",
}


def bezier(p0, p1, p2, p3, n=140):
    t = np.linspace(0.0, 1.0, n)
    a = ((1 - t) ** 3)[:, None] * np.asarray(p0)
    b = (3 * ((1 - t) ** 2) * t)[:, None] * np.asarray(p1)
    c = (3 * (1 - t) * (t**2))[:, None] * np.asarray(p2)
    d = (t**3)[:, None] * np.asarray(p3)
    return a + b + c + d


def interpolate_polyline(points, samples_per_segment=BLOCK_SAMPLES):
    parts = []
    points = np.asarray(points, dtype=float)
    for idx in range(len(points) - 1):
        t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=True)
        segment = points[idx] * (1.0 - t[:, None]) + points[idx + 1] * t[:, None]
        if idx:
            segment = segment[1:]
        parts.append(segment)
    return np.vstack(parts)


def transform_track_points(*arrays, rotate_deg=-35.0):
    angle = math.radians(rotate_deg)
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    transformed = [arr @ rot.T for arr in arrays]
    all_points = np.vstack(transformed)
    shift = all_points.min(axis=0)
    return [arr - shift for arr in transformed]


def build_curve2_straight_curve2(samples_per_block=BLOCK_SAMPLES, rotate_deg=-35.0):
    # Road geometry stays smooth because real Curve2 blocks have curved edges.
    # The progress line is separate: in the runtime it is approximated by a
    # polyline connecting block transition points.
    transitions = np.array(
        [
            (0.0, 0.0),
            (2.65, -1.35),
            (5.55, -1.35),
            (8.20, 0.0),
        ],
        dtype=float,
    )

    progress = interpolate_polyline(transitions, samples_per_block)

    p0, p1, p2, p3 = transitions
    start_tangent = np.array([0.55, -1.0])
    start_tangent = start_tangent / np.linalg.norm(start_tangent)
    straight_tangent = np.array([1.0, 0.0])
    end_tangent = np.array([0.55, 1.0])
    end_tangent = end_tangent / np.linalg.norm(end_tangent)

    first_curve = bezier(
        p0,
        p0 + start_tangent * 1.55,
        p1 - straight_tangent * 1.35,
        p1,
        n=samples_per_block,
    )
    straight = interpolate_polyline([p1, p2], samples_per_block)
    second_curve = bezier(
        p2,
        p2 + straight_tangent * 1.35,
        p3 - end_tangent * 1.55,
        p3,
        n=samples_per_block,
    )
    road_center = np.vstack([first_curve[:-1], straight[:-1], second_curve])
    progress, road_center = transform_track_points(progress, road_center, rotate_deg=rotate_deg)
    return progress, road_center


def extend_polyline_end(points, length=2.4, samples=80):
    direction = points[-1] - points[-2]
    direction = direction / max(np.linalg.norm(direction), 1e-9)
    t = np.linspace(0.0, length, samples)
    extension = points[-1] + t[:, None] * direction
    return np.vstack([points, extension[1:]])


def build_centerline():
    progress, _ = build_curve2_straight_curve2(samples_per_block=BLOCK_SAMPLES, rotate_deg=-35.0)
    return progress


def build_road_centerline():
    _, road_center = build_curve2_straight_curve2(samples_per_block=BLOCK_SAMPLES, rotate_deg=-35.0)
    return road_center


def build_detail_centerline():
    progress, _ = build_curve2_straight_curve2(samples_per_block=BLOCK_SAMPLES, rotate_deg=-35.0)
    return progress


def build_detail_road_centerline():
    _, road_center = build_curve2_straight_curve2(samples_per_block=BLOCK_SAMPLES, rotate_deg=-35.0)
    return road_center


def polyline_lengths(points):
    deltas = np.diff(points, axis=0)
    steps = np.linalg.norm(deltas, axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def tangent_normals(points):
    grad = np.gradient(points, axis=0)
    tangent = grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-9)
    normals = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return tangent, normals


def project_to_polyline(point, line):
    best = None
    cumulative = 0.0
    for i in range(len(line) - 1):
        a = line[i]
        b = line[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            continue
        u = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
        foot = a + u * ab
        dist = float(np.linalg.norm(point - foot))
        seg_len = float(np.linalg.norm(ab))
        along = cumulative + u * seg_len
        if best is None or dist < best["dist"]:
            best = {"dist": dist, "foot": foot, "idx": i, "u": u, "along": along}
        cumulative += seg_len
    return best


def split_polyline_at_projection(line, projection):
    idx = projection["idx"]
    foot = projection["foot"]
    passed = np.vstack([line[: idx + 1], foot])
    future = np.vstack([foot, line[idx + 1 :]])
    return passed, future


def ray_segment_intersection(origin, direction, a, b):
    def cross2(u, v):
        return float(u[0] * v[1] - u[1] * v[0])

    v1 = origin - a
    v2 = b - a
    v3 = np.array([-direction[1], direction[0]])
    denom = float(np.dot(v2, v3))
    if abs(denom) < 1e-9:
        return None
    t1 = cross2(v2, v1) / denom
    t2 = np.dot(v1, v3) / denom
    if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
        return origin + t1 * direction, float(t1)
    return None


def first_ray_hit(origin, direction, boundaries, max_distance=3.2):
    best = None
    for boundary in boundaries:
        for i in range(len(boundary) - 1):
            hit = ray_segment_intersection(origin, direction, boundary[i], boundary[i + 1])
            if hit is None:
                continue
            point, distance = hit
            if distance > max_distance:
                continue
            if best is None or distance < best[1]:
                best = (point, distance)
    if best is not None:
        return best[0], best[1], True
    return origin + max_distance * direction, max_distance, False


def draw_centerline_progress(ax, line, projection, lw=3.0, alpha=1.0):
    passed, future = split_polyline_at_projection(line, projection)
    ax.plot(
        passed[:, 0],
        passed[:, 1],
        color=COLORS["center_passed"],
        lw=lw,
        solid_capstyle="round",
        zorder=5,
        alpha=alpha,
    )
    ax.plot(
        future[:, 0],
        future[:, 1],
        color=COLORS["center_future"],
        lw=lw,
        solid_capstyle="round",
        zorder=4,
        alpha=alpha,
    )


def draw_road(ax, road_center, left, right, progress_line, projection, lw_center=3.0):
    polygon = np.vstack([left, right[::-1]])
    ax.fill(polygon[:, 0], polygon[:, 1], color=COLORS["road_fill"], zorder=0)
    ax.plot(
        left[:, 0],
        left[:, 1],
        color=COLORS["road_edge"],
        lw=1.5,
        zorder=2,
        solid_joinstyle="round",
        solid_capstyle="round",
    )
    ax.plot(
        right[:, 0],
        right[:, 1],
        color=COLORS["road_edge"],
        lw=1.5,
        zorder=2,
        solid_joinstyle="round",
        solid_capstyle="round",
    )
    draw_centerline_progress(ax, progress_line, projection, lw=lw_center)


def draw_transition_points(
    ax,
    center,
    normals,
    road_width,
    indices,
    annotate=True,
    boundary_left=None,
    boundary_right=None,
):
    for item in indices:
        idx = item[0]
        label = item[1] if len(item) > 1 else ""
        dy = item[2] if len(item) > 2 else 0.0
        p = center[idx]
        n = normals[idx]
        if boundary_left is not None and boundary_right is not None:
            boundary_a = boundary_left[idx]
            boundary_b = boundary_right[idx]
        else:
            boundary_a = p - n * road_width * 0.50
            boundary_b = p + n * road_width * 0.50
        ax.plot(
            [boundary_a[0], boundary_b[0]],
            [boundary_a[1], boundary_b[1]],
            color=COLORS["block_boundary"],
            lw=0.9,
            alpha=0.55,
            zorder=3,
            solid_capstyle="round",
        )
        ax.scatter(
            [p[0]],
            [p[1]],
            s=54,
            color=COLORS["transition"],
            edgecolor="white",
            linewidth=1.0,
            zorder=8,
        )
        if annotate and label:
            label_pos = p + n * road_width * 0.34 + np.array([0.0, dy])
            ax.text(
                label_pos[0],
                label_pos[1],
                label,
                color=COLORS["transition"],
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82),
                zorder=12,
            )


def draw_car(ax, position, heading, size=0.78):
    # heading points along the vehicle forward direction in data coordinates.
    if CAR_ICON.exists():
        img = mpimg.imread(CAR_ICON)
        img_h, img_w = img.shape[:2]
        # Preserve the original icon ratio. The icon itself already contains the
        # car silhouette, so squeezing it would visually distort the vehicle.
        height = size
        width = size * (img_w / img_h)
        extent = [
            position[0] - width / 2,
            position[0] + width / 2,
            position[1] - height / 2,
            position[1] + height / 2,
        ]
        im = ax.imshow(img, extent=extent, zorder=20)
        # The source icon points in the opposite vertical direction than our
        # local heading convention, therefore we rotate it by an additional pi.
        tr = transforms.Affine2D().rotate_around(position[0], position[1], heading + math.pi / 2)
        im.set_transform(tr + ax.transData)
        return

    # Fallback: simple vector car, useful while tuning the diagram without icon asset.
    width = size * 0.62
    height = size
    base = patches.FancyBboxPatch(
        (position[0] - width / 2, position[1] - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor="#9f1d14",
        edgecolor="#4a0f0b",
        linewidth=1.0,
        zorder=20,
    )
    tr = transforms.Affine2D().rotate_around(position[0], position[1], heading - math.pi / 2) + ax.transData
    base.set_transform(tr)
    ax.add_patch(base)
    for wx, wy in [(-0.21, -0.32), (0.21, -0.32), (-0.21, 0.32), (0.21, 0.32)]:
        wheel = patches.Rectangle(
            (position[0] + wx * size - 0.07, position[1] + wy * size - 0.08),
            0.14,
            0.16,
            facecolor="#1f2937",
            edgecolor="none",
            zorder=21,
        )
        wheel.set_transform(tr)
        ax.add_patch(wheel)


def draw_vector(ax, start, vec, color, label, label_offset=(0.0, 0.0), lw=2.0):
    end = start + vec
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=0, shrinkB=0),
        zorder=24,
    )
    if label:
        ax.text(
            end[0] + label_offset[0],
            end[1] + label_offset[1],
            label,
            color=color,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75),
            zorder=25,
        )


def set_clean_axis(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(COLORS["panel_bg"])
    ax.axis("off")


def draw_observation_from_geometry():
    center_base = build_centerline()
    road_center_base = build_road_centerline()
    center = extend_polyline_end(center_base, length=2.7, samples=90)
    road_center = extend_polyline_end(road_center_base, length=2.7, samples=90)
    tangents, normals = tangent_normals(center)
    road_tangents, road_normals = tangent_normals(road_center)
    road_width = 1.55
    left = road_center + road_normals * road_width / 2
    right = road_center - road_normals * road_width / 2
    base_road_tangents, base_road_normals = tangent_normals(road_center_base)
    left_base = road_center_base + base_road_normals * road_width / 2
    right_base = road_center_base - base_road_normals * road_width / 2

    # Pick a point inside the first curve and place the car off the centerline.
    car_idx = 108
    base = center[car_idx]
    tangent = tangents[car_idx]
    normal = normals[car_idx]
    car_pos = base - normal * 0.34 + tangent * 0.08
    heading = math.atan2(tangent[1], tangent[0]) - 0.23
    car_forward = np.array([math.cos(heading), math.sin(heading)])
    car_left = np.array([-car_forward[1], car_forward[0]])
    projection = project_to_polyline(car_pos, center)

    fig = plt.figure(figsize=(7.0, 3.5), dpi=180)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.03)
    ax_top = fig.add_subplot(gs[0])
    ax_detail = fig.add_subplot(gs[1])

    # Upper context panel.
    draw_road(ax_top, road_center, left, right, center, projection, lw_center=3.2)
    transition_indices = [
        (0, "vstup bloku", 0.12),
        (BLOCK_SAMPLES - 1, "block transition", -0.10),
        (2 * BLOCK_SAMPLES - 2, "block transition", -0.10),
        (len(center_base) - 1, "output bloku", 0.12),
    ]
    draw_transition_points(
        ax_top,
        center,
        normals,
        road_width,
        transition_indices,
        annotate=False,
        boundary_left=left,
        boundary_right=right,
    )
    draw_car(ax_top, car_pos, heading, size=0.55)
    ax_top.scatter(
        [projection["foot"][0]],
        [projection["foot"][1]],
        s=58,
        color=COLORS["projection"],
        edgecolor="white",
        linewidth=1.0,
        zorder=30,
    )
    top_points = np.vstack([left_base, right_base, center_base, car_pos[None, :]])
    top_min = top_points.min(axis=0)
    top_max = top_points.max(axis=0)
    ax_top.set_xlim(top_min[0] + 0.22, top_max[0] - 0.08)
    ax_top.set_ylim(top_min[1] - 0.03, top_max[1] + 0.04)
    set_clean_axis(ax_top)

    # Lower detail panel.
    center_d = build_detail_centerline()
    road_center_d = build_detail_road_centerline()
    tang_d, norm_d = tangent_normals(center_d)
    road_tang_d, road_norm_d = tangent_normals(road_center_d)
    left_d = road_center_d + road_norm_d * road_width / 2
    right_d = road_center_d - road_norm_d * road_width / 2
    detail_idx = 96
    detail_base = center_d[detail_idx]
    detail_tangent = tang_d[detail_idx]
    detail_normal = norm_d[detail_idx]
    detail_car_pos = detail_base - detail_normal * 0.34 + detail_tangent * 0.08
    detail_heading = math.atan2(detail_tangent[1], detail_tangent[0]) - 0.23
    detail_forward = np.array([math.cos(detail_heading), math.sin(detail_heading)])
    detail_left = np.array([-detail_forward[1], detail_forward[0]])
    projection_d = project_to_polyline(detail_car_pos, center_d)
    draw_road(ax_detail, road_center_d, left_d, right_d, center_d, projection_d, lw_center=3.1)
    # Keep nearby block transitions visible in the zoom so the reader sees that
    # the progress line is anchored in block connection points.
    draw_transition_points(
        ax_detail,
        center_d,
        norm_d,
        road_width,
        [(BLOCK_SAMPLES - 1,), (2 * BLOCK_SAMPLES - 2,)],
        annotate=False,
        boundary_left=left_d,
        boundary_right=right_d,
    )

    foot = projection_d["foot"]
    ax_detail.plot(
        [detail_car_pos[0], foot[0]],
        [detail_car_pos[1], foot[1]],
        color=COLORS["projection"],
        lw=1.5,
        zorder=18,
    )
    segment = center_d[projection_d["idx"] + 1] - center_d[projection_d["idx"]]
    tangent_unit = segment / max(np.linalg.norm(segment), 1e-9)
    perpendicular_unit = detail_car_pos - foot
    perpendicular_unit = perpendicular_unit / max(np.linalg.norm(perpendicular_unit), 1e-9)
    marker_radius = 0.105
    passed_unit = -tangent_unit
    arc_t = np.linspace(0.0, math.pi / 2.0, 18)
    right_angle = foot + marker_radius * (
        np.cos(arc_t)[:, None] * passed_unit + np.sin(arc_t)[:, None] * perpendicular_unit
    )
    fill_points = np.vstack([foot, right_angle, foot])
    ax_detail.fill(
        fill_points[:, 0],
        fill_points[:, 1],
        color=COLORS["projection"],
        alpha=0.14,
        zorder=30,
    )
    ax_detail.plot(
        right_angle[:, 0],
        right_angle[:, 1],
        color=COLORS["projection"],
        lw=1.35,
        solid_capstyle="round",
        zorder=31,
    )
    right_angle_dot = foot + marker_radius * 0.58 * (passed_unit + perpendicular_unit) / math.sqrt(2.0)
    ax_detail.scatter(
        [right_angle_dot[0]],
        [right_angle_dot[1]],
        s=9,
        color=COLORS["projection"],
        edgecolor="none",
        zorder=32,
    )
    ax_detail.scatter(
        [foot[0]],
        [foot[1]],
        s=58,
        color=COLORS["projection"],
        edgecolor="white",
        linewidth=1.0,
        zorder=30,
    )
    boundaries = [left_d, right_d]
    ray_angles = np.deg2rad(np.linspace(-90.0, 90.0, 11))
    ray_segments = []
    ray_hits = []
    for angle in ray_angles:
        ca, sa = math.cos(angle), math.sin(angle)
        direction = np.array(
            [
                detail_forward[0] * ca - detail_forward[1] * sa,
                detail_forward[0] * sa + detail_forward[1] * ca,
            ]
        )
        direction = direction / np.linalg.norm(direction)
        hit, distance, collided = first_ray_hit(detail_car_pos, direction, boundaries, max_distance=9.0)
        ray_segments.append([detail_car_pos, hit])
        ray_hits.append((hit, distance, angle, collided))
    ax_detail.add_collection(
        LineCollection(
            ray_segments,
            colors=COLORS["ray"],
            linewidths=1.1,
            linestyles="dashed",
            alpha=0.82,
            zorder=12,
        )
    )
    for hit, _, _, collided in ray_hits:
        if collided:
            ax_detail.scatter([hit[0]], [hit[1]], s=12, color=COLORS["ray"], alpha=0.7, zorder=16)

    draw_car(ax_detail, detail_car_pos, detail_heading, size=0.54)
    vector_origin = detail_car_pos + detail_forward * 0.18
    forward_component = detail_forward * 0.56
    side_component = detail_left * 0.24
    velocity_vec = forward_component + side_component
    draw_vector(
        ax_detail,
        vector_origin,
        velocity_vec,
        COLORS["heading"],
        "",
        (0.0, 0.0),
        lw=2.0,
    )
    draw_vector(
        ax_detail,
        vector_origin,
        forward_component,
        COLORS["velocity"],
        "",
        (0.0, 0.0),
        lw=2.2,
    )
    draw_vector(
        ax_detail,
        vector_origin,
        side_component,
        COLORS["side_velocity"],
        "",
        (0.0, 0.0),
        lw=1.9,
    )
    zoom_half_width = 1.66
    zoom_half_height = 1.22
    ax_detail.set_xlim(detail_car_pos[0] - zoom_half_width, detail_car_pos[0] + zoom_half_width)
    ax_detail.set_ylim(detail_car_pos[1] - zoom_half_height, detail_car_pos[1] + zoom_half_height)
    set_clean_axis(ax_detail)

    # Make the figure height follow the data aspect ratios. With equal-aspect
    # axes this avoids empty side bands and keeps both views next to each other.
    top_xlim = ax_top.get_xlim()
    top_ylim = ax_top.get_ylim()
    detail_xlim = ax_detail.get_xlim()
    detail_ylim = ax_detail.get_ylim()
    top_aspect = (top_ylim[1] - top_ylim[0]) / (top_xlim[1] - top_xlim[0])
    detail_aspect = (detail_ylim[1] - detail_ylim[0]) / (detail_xlim[1] - detail_xlim[0])
    if detail_aspect < top_aspect:
        # Fit the detail view to the same physical height as the overview. To
        # preserve the right-side ray context, crop the extra width from the
        # left side of the detail view.
        detail_y_span = detail_ylim[1] - detail_ylim[0]
        detail_x_right = detail_xlim[1]
        detail_x_span = detail_y_span / top_aspect
        ax_detail.set_xlim(detail_x_right - detail_x_span, detail_x_right)
        detail_xlim = ax_detail.get_xlim()
        detail_aspect = (detail_ylim[1] - detail_ylim[0]) / (detail_xlim[1] - detail_xlim[0])

    fig_width = 7.0
    side_margin = 0.10
    panel_gap = 0.10
    legend_height = 0.80
    top_margin = 0.06
    panel_width = (fig_width - 2 * side_margin - panel_gap) / 2.0
    top_panel_height = panel_width * top_aspect
    detail_panel_height = panel_width * detail_aspect
    panel_height = max(top_panel_height, detail_panel_height)
    fig_height = top_margin + panel_height + legend_height
    fig.set_size_inches(fig_width, fig_height, forward=True)

    left_x = side_margin
    right_x = side_margin + panel_width + panel_gap
    panel_bottom = legend_height / fig_height
    left_norm = left_x / fig_width
    right_norm = right_x / fig_width
    width_norm = panel_width / fig_width
    top_bottom = panel_bottom + (panel_height - top_panel_height) / (2.0 * fig_height)
    detail_bottom = panel_bottom + (panel_height - detail_panel_height) / (2.0 * fig_height)
    ax_top.set_position([left_norm, top_bottom, width_norm, top_panel_height / fig_height])
    ax_detail.set_position([right_norm, detail_bottom, width_norm, detail_panel_height / fig_height])

    def legend_item(x, y, label, color, style, fontsize=8.8):
        if style == "o":
            fig.text(x, y, "●", color=color, fontsize=8.9, ha="right", va="center", zorder=90)
        elif style == "->":
            fig.text(x, y, "━━▶", color=color, fontsize=8.0, ha="right", va="center", zorder=90)
        else:
            fig.text(
                x,
                y,
                "━━━━" if style == "-" else "┄┄┄┄",
                color=color,
                fontsize=8.0,
                ha="right",
                va="center",
                zorder=90,
            )
        fig.text(x + 0.010, y, label, color=COLORS["text"], fontsize=fontsize, ha="left", va="center", zorder=90)

    legend_box_bottom = 0.10 / fig_height
    legend_box_height = 0.70 / fig_height
    legend_left = left_norm
    legend_width = right_norm + width_norm - left_norm
    fig.add_artist(
        patches.FancyBboxPatch(
            (legend_left, legend_box_bottom),
            legend_width,
            legend_box_height,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            transform=fig.transFigure,
            facecolor="white",
            edgecolor="#d7cbb7",
            linewidth=0.8,
            alpha=0.96,
            zorder=80,
        )
    )

    col = [legend_left + legend_width * k for k in (0.11, 0.42, 0.72)]
    row = [
        legend_box_bottom + legend_box_height * 0.73,
        legend_box_bottom + legend_box_height * 0.47,
        legend_box_bottom + legend_box_height * 0.21,
    ]
    unified_legend = [
        (col[0], row[0], "completed progress", COLORS["center_passed"], "-"),
        (col[0], row[1], "future progress", COLORS["center_future"], "-"),
        (col[0], row[2], "block transitions", COLORS["transition"], "o"),
        (col[1], row[0], "progress point", COLORS["projection"], "o"),
        (col[1], row[1], "perpendicular to axis", COLORS["projection"], "-"),
        (col[1], row[2], "wall distances", COLORS["ray"], "--"),
        (col[2], row[0], "velocity vector", COLORS["heading"], "->"),
        (col[2], row[1], "forward velocity", COLORS["velocity"], "->"),
        (col[2], row[2], "lateral velocity", COLORS["side_velocity"], "->"),
    ]
    for x, y, label, color, style in unified_legend:
        legend_item(x, y, label, color, style, fontsize=6.9)

    fig.add_artist(
        plt.Line2D(
            [0.5, 0.5],
            [legend_box_bottom + legend_box_height + 0.04 / fig_height, 0.985],
            transform=fig.transFigure,
            color="#d7cbb7",
            lw=1.35,
            alpha=0.95,
            zorder=70,
        )
    )

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"solution_observation_from_geometry.{ext}", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


if __name__ == "__main__":
    draw_observation_from_geometry()
