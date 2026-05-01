from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiments.tm2d_geometry import RaycastResult, TM2DGeometry


@dataclass(frozen=True)
class ArcPrimitive:
    center: np.ndarray
    radius: float
    angle_start: float
    angle_end: float
    ccw: bool
    source_block: str
    segment_count: int
    rms_error: float


def _project_xz(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return points[..., [0, 2]]


def _angle_normalize(angle: np.ndarray | float) -> np.ndarray | float:
    return np.mod(angle, 2.0 * np.pi)


def _angle_span_ccw(start: float, end: float) -> float:
    return float((end - start) % (2.0 * np.pi))


def _angle_on_arc(angle: np.ndarray, start: float, end: float, ccw: bool, eps: float = 1e-5) -> np.ndarray:
    angle = _angle_normalize(angle)
    start = float(_angle_normalize(start))
    end = float(_angle_normalize(end))
    if ccw:
        span = _angle_span_ccw(start, end)
        rel = np.mod(angle - start, 2.0 * np.pi)
    else:
        span = _angle_span_ccw(end, start)
        rel = np.mod(start - angle, 2.0 * np.pi)
    return rel <= span + eps


def _minimal_ccw_angle_interval(angles: np.ndarray) -> tuple[float, float, float]:
    """Smallest CCW interval covering circular angles.

    The interval starts after the largest angular gap and ends before it. This
    handles arcs that cross the 0/2pi boundary without special cases.
    """

    values = np.sort(np.mod(np.asarray(angles, dtype=np.float64), 2.0 * np.pi))
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    if len(values) == 1:
        value = float(values[0])
        return value, value, 0.0
    gaps = np.diff(np.concatenate([values, values[:1] + 2.0 * np.pi]))
    gap_idx = int(np.argmax(gaps))
    start = float(values[(gap_idx + 1) % len(values)])
    end = float(values[gap_idx])
    span = float((end - start) % (2.0 * np.pi))
    return start, end, span


def _dedup_segments(segments: list[np.ndarray], precision: float = 0.01) -> np.ndarray:
    unique: dict[tuple[int, int, int, int], np.ndarray] = {}
    for segment in segments:
        a = np.asarray(segment[0], dtype=np.float32)
        b = np.asarray(segment[1], dtype=np.float32)
        if float(np.linalg.norm(b - a)) <= 1e-4:
            continue
        qa = tuple(np.round(a / precision).astype(np.int64))
        qb = tuple(np.round(b / precision).astype(np.int64))
        key_points = sorted((qa, qb))
        key = (*key_points[0], *key_points[1])
        unique[key] = np.asarray([a, b], dtype=np.float32)
    if not unique:
        return np.empty((0, 2, 2), dtype=np.float32)
    return np.stack(list(unique.values())).astype(np.float32, copy=False)


def _segments_from_block_sensor_walls(block) -> np.ndarray:
    mesh = block.get_sensor_walls_mesh()
    vertices = _project_xz(np.asarray(mesh.vertices, dtype=np.float32))
    faces = np.asarray(mesh.faces, dtype=np.int64)
    segments: list[np.ndarray] = []
    for face in faces:
        for a_idx, b_idx in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            a = vertices[int(a_idx)]
            b = vertices[int(b_idx)]
            segments.append(np.asarray([a, b], dtype=np.float32))
    return _dedup_segments(segments)


def _segment_components(segments: np.ndarray, precision: float = 0.01) -> list[list[int]]:
    if len(segments) == 0:
        return []
    endpoint_to_segments: dict[tuple[int, int], list[int]] = {}
    for idx, segment in enumerate(segments):
        for point in segment:
            key = tuple(np.round(point / precision).astype(np.int64))
            endpoint_to_segments.setdefault(key, []).append(int(idx))

    neighbors: list[set[int]] = [set() for _ in range(len(segments))]
    for connected in endpoint_to_segments.values():
        for idx in connected:
            neighbors[idx].update(other for other in connected if other != idx)

    components: list[list[int]] = []
    visited: set[int] = set()
    for start_idx in range(len(segments)):
        if start_idx in visited:
            continue
        stack = [start_idx]
        visited.add(start_idx)
        component: list[int] = []
        while stack:
            idx = stack.pop()
            component.append(idx)
            for next_idx in neighbors[idx]:
                if next_idx not in visited:
                    visited.add(next_idx)
                    stack.append(next_idx)
        components.append(component)
    return components


def _fit_circle(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    points = np.asarray(points, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]
    design = np.column_stack([x, y, np.ones_like(x)])
    rhs = -(x * x + y * y)
    a, b, c = np.linalg.lstsq(design, rhs, rcond=None)[0]
    center = np.asarray([-a * 0.5, -b * 0.5], dtype=np.float64)
    radius_sq = float(center[0] * center[0] + center[1] * center[1] - c)
    radius = math.sqrt(max(radius_sq, 0.0))
    radial_error = np.linalg.norm(points - center.reshape(1, 2), axis=1) - radius
    rms = float(math.sqrt(float(np.mean(radial_error * radial_error))))
    return center.astype(np.float32), float(radius), rms


def _ordered_chain_points(component_segments: np.ndarray, precision: float = 0.01) -> np.ndarray:
    endpoint_to_points: dict[tuple[int, int], np.ndarray] = {}
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for segment in component_segments:
        keys = []
        for point in segment:
            key = tuple(np.round(point / precision).astype(np.int64))
            endpoint_to_points[key] = np.asarray(point, dtype=np.float32)
            adjacency.setdefault(key, [])
            keys.append(key)
        adjacency[keys[0]].append(keys[1])
        adjacency[keys[1]].append(keys[0])

    endpoints = [key for key, connected in adjacency.items() if len(connected) == 1]
    start = endpoints[0] if endpoints else next(iter(adjacency))
    ordered_keys = [start]
    previous = None
    current = start
    while True:
        candidates = [key for key in adjacency[current] if key != previous]
        if not candidates:
            break
        previous, current = current, candidates[0]
        if current == start:
            break
        ordered_keys.append(current)
        if len(ordered_keys) > len(adjacency) + 2:
            break
    return np.asarray([endpoint_to_points[key] for key in ordered_keys], dtype=np.float32)


def fit_curve_block_arcs(
    geometry: TM2DGeometry,
    min_segments: int = 4,
    max_rms_error: float = 0.35,
) -> tuple[list[ArcPrimitive], np.ndarray]:
    arcs: list[ArcPrimitive] = []
    remaining_segments: list[np.ndarray] = []

    for block in geometry.game_map.blocks.values():
        block_segments = _segments_from_block_sensor_walls(block)
        if "Curve" not in getattr(block, "mesh_name", ""):
            remaining_segments.extend(block_segments)
            continue

        consumed_segments: set[int] = set()
        for component in _segment_components(block_segments):
            if len(component) < int(min_segments):
                continue
            component_segments = block_segments[component]
            chain_points = _ordered_chain_points(component_segments)
            unique_points = np.unique(np.round(chain_points, 3), axis=0)
            if len(unique_points) < 4:
                continue
            center, radius, rms = _fit_circle(unique_points)
            if radius <= 1e-6 or rms > float(max_rms_error):
                continue

            angles = np.unwrap(np.arctan2(chain_points[:, 1] - center[1], chain_points[:, 0] - center[0]))
            if len(angles) < 2:
                continue
            ccw = bool(angles[-1] >= angles[0])
            span = abs(float(angles[-1] - angles[0]))
            if span < math.radians(10.0) or span > math.radians(130.0):
                continue
            arcs.append(
                ArcPrimitive(
                    center=center.astype(np.float32),
                    radius=float(radius),
                    angle_start=float(_angle_normalize(angles[0])),
                    angle_end=float(_angle_normalize(angles[-1])),
                    ccw=ccw,
                    source_block=str(getattr(block, "raw_name", getattr(block, "mesh_name", "unknown"))),
                    segment_count=len(component),
                    rms_error=float(rms),
                )
            )
            consumed_segments.update(int(index) for index in component)

        for idx, segment in enumerate(block_segments):
            if idx not in consumed_segments:
                remaining_segments.append(segment)

    if remaining_segments:
        remaining = _dedup_segments(remaining_segments)
    else:
        remaining = np.empty((0, 2, 2), dtype=np.float32)
    return arcs, remaining


def analytic_curve_block_arcs(
    geometry: TM2DGeometry,
    radius_tolerance: float = 0.25,
) -> tuple[list[ArcPrimitive], np.ndarray]:
    """Build exact quarter-circle primitives from Trackmania curve block geometry.

    Curve blocks are authored as quarter-ring roads. For a curve of size N,
    the inner and outer wall radii are:

    - inner = (N - 1) * 32 + 2.7
    - outer = N * 32 - 2.7

    The block may be rotated in the map, so the curve center is inferred from
    the transformed wall-vertex bounding-box corner whose vertex radii best
    match those two expected radii.
    """

    arcs: list[ArcPrimitive] = []
    remaining_segments: list[np.ndarray] = []

    for block in geometry.game_map.blocks.values():
        block_segments = _segments_from_block_sensor_walls(block)
        if "Curve" not in getattr(block, "mesh_name", ""):
            remaining_segments.extend(block_segments)
            continue

        mesh = block.get_sensor_walls_mesh()
        vertices = _project_xz(np.asarray(mesh.vertices, dtype=np.float32))
        if len(vertices) == 0:
            remaining_segments.extend(block_segments)
            continue

        block_size = int(getattr(block, "block_size", 1))
        inner_radius = float((block_size - 1) * 32.0 + 2.7)
        outer_radius = float(block_size * 32.0 - 2.7)
        min_xy = np.min(vertices, axis=0)
        max_xy = np.max(vertices, axis=0)
        candidate_centers = [
            np.asarray([min_xy[0], min_xy[1]], dtype=np.float32),
            np.asarray([min_xy[0], max_xy[1]], dtype=np.float32),
            np.asarray([max_xy[0], min_xy[1]], dtype=np.float32),
            np.asarray([max_xy[0], max_xy[1]], dtype=np.float32),
        ]

        best_center = candidate_centers[0]
        best_error = float("inf")
        for center in candidate_centers:
            radii = np.linalg.norm(vertices - center.reshape(1, 2), axis=1)
            radius_error = np.minimum(np.abs(radii - inner_radius), np.abs(radii - outer_radius))
            # Median is robust against straight seam/helper fragments.
            error = float(np.median(radius_error))
            if error < best_error:
                best_error = error
                best_center = center

        consumed_segments: set[int] = set()
        for radius in (inner_radius, outer_radius):
            radii = np.linalg.norm(vertices - best_center.reshape(1, 2), axis=1)
            on_arc_points = vertices[np.abs(radii - radius) <= radius_tolerance]
            if len(on_arc_points) < 2:
                continue

            raw_angles = np.arctan2(on_arc_points[:, 1] - best_center[1], on_arc_points[:, 0] - best_center[0])
            start_angle, end_angle, span = _minimal_ccw_angle_interval(raw_angles)
            ccw = True

            if span < math.radians(20.0) or span > math.radians(120.0):
                continue

            point_radii_error = np.linalg.norm(on_arc_points - best_center.reshape(1, 2), axis=1) - radius
            rms = float(math.sqrt(float(np.mean(point_radii_error * point_radii_error))))
            radius_segment_count = int(
                sum(
                    1
                    for segment in block_segments
                    if np.all(
                        np.abs(np.linalg.norm(segment - best_center.reshape(1, 2), axis=1) - radius)
                        <= radius_tolerance
                    )
                )
            )
            arcs.append(
                ArcPrimitive(
                    center=best_center.astype(np.float32),
                    radius=radius,
                    angle_start=start_angle,
                    angle_end=end_angle,
                    ccw=ccw,
                    source_block=str(getattr(block, "raw_name", getattr(block, "mesh_name", "unknown"))),
                    segment_count=radius_segment_count,
                    rms_error=rms,
                )
            )

        for idx, segment in enumerate(block_segments):
            endpoint_radii = np.linalg.norm(segment - best_center.reshape(1, 2), axis=1)
            on_inner = np.all(np.abs(endpoint_radii - inner_radius) <= radius_tolerance)
            on_outer = np.all(np.abs(endpoint_radii - outer_radius) <= radius_tolerance)
            if on_inner or on_outer:
                consumed_segments.add(idx)

        for idx, segment in enumerate(block_segments):
            if idx not in consumed_segments:
                remaining_segments.append(segment)

    remaining = _dedup_segments(remaining_segments) if remaining_segments else np.empty((0, 2, 2), dtype=np.float32)
    return arcs, remaining


def cast_lasers_segments_and_arcs(
    geometry: TM2DGeometry,
    center: np.ndarray,
    heading: float,
    segments: np.ndarray,
    arcs: list[ArcPrimitive],
) -> RaycastResult:
    center = np.asarray(center, dtype=np.float32)
    angles = float(heading) + geometry._laser_offsets_radians
    directions = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
    distances = np.full(geometry.num_lasers, geometry.laser_max_distance, dtype=np.float32)

    if len(segments) > 0:
        seg_start = segments[:, 0, :].astype(np.float32, copy=False)
        seg_vec = (segments[:, 1, :] - segments[:, 0, :]).astype(np.float32, copy=False)
        p = seg_start.reshape(1, -1, 2)
        s = seg_vec.reshape(1, -1, 2)
        d = directions.reshape(-1, 1, 2)
        rel = p - center.reshape(1, 1, 2)
        denom = d[..., 0] * s[..., 1] - d[..., 1] * s[..., 0]
        valid = np.abs(denom) > 1e-7
        safe_denom = np.where(valid, denom, 1.0)
        t = (rel[..., 0] * s[..., 1] - rel[..., 1] * s[..., 0]) / safe_denom
        u = (rel[..., 0] * d[..., 1] - rel[..., 1] * d[..., 0]) / safe_denom
        hit = valid & (t >= 0.0) & (t <= geometry.laser_max_distance) & (u >= 0.0) & (u <= 1.0)
        hit_t = np.where(hit, t, np.inf)
        closest = np.min(hit_t, axis=1)
        distances = np.where(np.isfinite(closest), closest, distances).astype(np.float32)

    if arcs:
        for arc in arcs:
            oc = center.reshape(1, 2) - arc.center.reshape(1, 2)
            b = 2.0 * np.sum(directions * oc, axis=1)
            c = float(np.sum(oc * oc)) - arc.radius * arc.radius
            disc = b * b - 4.0 * c
            valid = disc >= 0.0
            if not np.any(valid):
                continue
            sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
            roots = np.stack([(-b - sqrt_disc) * 0.5, (-b + sqrt_disc) * 0.5], axis=1)
            for root_idx in range(2):
                t = roots[:, root_idx]
                points = center.reshape(1, 2) + directions * t.reshape(-1, 1)
                hit_angles = np.arctan2(points[:, 1] - arc.center[1], points[:, 0] - arc.center[0])
                hit = (
                    valid
                    & (t >= 0.0)
                    & (t <= geometry.laser_max_distance)
                    & (t < distances)
                    & _angle_on_arc(hit_angles, arc.angle_start, arc.angle_end, arc.ccw)
                )
                distances = np.where(hit, t.astype(np.float32), distances)

    endpoints = center.reshape(1, 2) + directions * distances.reshape(-1, 1)
    return RaycastResult(distances=distances, endpoints=endpoints, directions=directions)


def _angle_on_arcs_vectorized(
    angles: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    ccw: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    angles = np.mod(angles, 2.0 * np.pi)
    starts = np.mod(starts.reshape(1, -1), 2.0 * np.pi)
    ends = np.mod(ends.reshape(1, -1), 2.0 * np.pi)
    ccw = ccw.reshape(1, -1)

    ccw_span = np.mod(ends - starts, 2.0 * np.pi)
    ccw_rel = np.mod(angles - starts, 2.0 * np.pi)
    cw_span = np.mod(starts - ends, 2.0 * np.pi)
    cw_rel = np.mod(starts - angles, 2.0 * np.pi)
    return np.where(ccw, ccw_rel <= ccw_span + eps, cw_rel <= cw_span + eps)


def cast_lasers_segments_and_arcs_vectorized(
    geometry: TM2DGeometry,
    center: np.ndarray,
    heading: float,
    segments: np.ndarray,
    arcs: list[ArcPrimitive],
) -> RaycastResult:
    center = np.asarray(center, dtype=np.float32)
    angles = float(heading) + geometry._laser_offsets_radians
    directions = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
    distances = np.full(geometry.num_lasers, geometry.laser_max_distance, dtype=np.float32)

    if len(segments) > 0:
        seg_start = segments[:, 0, :].astype(np.float32, copy=False)
        seg_vec = (segments[:, 1, :] - segments[:, 0, :]).astype(np.float32, copy=False)
        p = seg_start.reshape(1, -1, 2)
        s = seg_vec.reshape(1, -1, 2)
        d = directions.reshape(-1, 1, 2)
        rel = p - center.reshape(1, 1, 2)
        denom = d[..., 0] * s[..., 1] - d[..., 1] * s[..., 0]
        valid = np.abs(denom) > 1e-7
        safe_denom = np.where(valid, denom, 1.0)
        t = (rel[..., 0] * s[..., 1] - rel[..., 1] * s[..., 0]) / safe_denom
        u = (rel[..., 0] * d[..., 1] - rel[..., 1] * d[..., 0]) / safe_denom
        hit = valid & (t >= 0.0) & (t <= geometry.laser_max_distance) & (u >= 0.0) & (u <= 1.0)
        hit_t = np.where(hit, t, np.inf)
        closest = np.min(hit_t, axis=1)
        distances = np.where(np.isfinite(closest), closest, distances).astype(np.float32)

    if arcs:
        arc_centers = np.asarray([arc.center for arc in arcs], dtype=np.float32)
        arc_radii = np.asarray([arc.radius for arc in arcs], dtype=np.float32)
        starts = np.asarray([arc.angle_start for arc in arcs], dtype=np.float32)
        ends = np.asarray([arc.angle_end for arc in arcs], dtype=np.float32)
        ccw = np.asarray([arc.ccw for arc in arcs], dtype=bool)

        oc = center.reshape(1, 2) - arc_centers
        b = 2.0 * np.sum(directions.reshape(-1, 1, 2) * oc.reshape(1, -1, 2), axis=2)
        c = np.sum(oc * oc, axis=1).reshape(1, -1) - arc_radii.reshape(1, -1) ** 2
        disc = b * b - 4.0 * c
        valid_disc = disc >= 0.0
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        roots = np.stack([(-b - sqrt_disc) * 0.5, (-b + sqrt_disc) * 0.5], axis=2)

        best_arc_t = np.full((geometry.num_lasers,), np.inf, dtype=np.float32)
        for root_idx in range(2):
            t = roots[:, :, root_idx]
            points = (
                center.reshape(1, 1, 2)
                + directions.reshape(-1, 1, 2) * t.reshape(geometry.num_lasers, -1, 1)
            )
            hit_angles = np.arctan2(points[:, :, 1] - arc_centers[:, 1].reshape(1, -1), points[:, :, 0] - arc_centers[:, 0].reshape(1, -1))
            angle_hit = _angle_on_arcs_vectorized(hit_angles, starts, ends, ccw)
            hit = valid_disc & (t >= 0.0) & (t <= geometry.laser_max_distance) & angle_hit
            hit_t = np.where(hit, t, np.inf)
            best_arc_t = np.minimum(best_arc_t, np.min(hit_t, axis=1).astype(np.float32))
        distances = np.minimum(distances, best_arc_t)

    endpoints = center.reshape(1, 2) + directions * distances.reshape(-1, 1)
    return RaycastResult(distances=distances, endpoints=endpoints, directions=directions)


def benchmark(map_name: str, samples: int, arc_source: str) -> None:
    geometry = TM2DGeometry(map_name)
    if str(arc_source).strip().lower() == "analytic":
        arcs, remaining_segments = analytic_curve_block_arcs(geometry)
    elif str(arc_source).strip().lower() == "fit":
        arcs, remaining_segments = fit_curve_block_arcs(geometry)
    else:
        raise ValueError("arc_source must be analytic or fit.")

    centers: list[np.ndarray] = []
    headings: list[float] = []
    for i in range(int(samples)):
        idx = i % len(geometry.path_points)
        centers.append(
            geometry.path_points[idx]
            + np.asarray([0.2 * (i % 7), -0.15 * (i % 5)], dtype=np.float32)
        )
        headings.append((i % 360) * np.pi / 180.0)

    max_diff = 0.0
    p95_diffs: list[float] = []
    for center, heading in zip(centers[: min(500, len(centers))], headings[: min(500, len(headings))]):
        segment_distances = geometry.cast_lasers(center, heading).distances
        arc_distances = cast_lasers_segments_and_arcs_vectorized(
            geometry,
            center,
            heading,
            remaining_segments,
            arcs,
        ).distances
        diff = np.abs(segment_distances - arc_distances)
        max_diff = max(max_diff, float(np.max(diff)))
        p95_diffs.extend(float(value) for value in diff)

    print("\n[Arc experiment]")
    print(f"map_name={map_name}")
    print(f"arc_source={arc_source}")
    print(f"original_segments={len(geometry.wall_segments)}")
    print(f"arc_count={len(arcs)}")
    print(f"arc_replaced_segments={sum(arc.segment_count for arc in arcs)}")
    print(f"remaining_segments={len(remaining_segments)}")
    if arcs:
        print(
            "arc_rms_error_avg="
            f"{np.mean([arc.rms_error for arc in arcs]):.4f} "
            f"max={np.max([arc.rms_error for arc in arcs]):.4f}"
        )
    print(f"distance_diff_max={max_diff:.4f}")
    print(f"distance_diff_p95={np.percentile(p95_diffs, 95):.4f}")

    for name, fn in [
        ("segment_baseline", lambda c, h: geometry.cast_lasers(c, h)),
        ("segments_plus_arcs_loop", lambda c, h: cast_lasers_segments_and_arcs(geometry, c, h, remaining_segments, arcs)),
        ("segments_plus_arcs_vec", lambda c, h: cast_lasers_segments_and_arcs_vectorized(geometry, c, h, remaining_segments, arcs)),
    ]:
        for center, heading in zip(centers[:50], headings[:50]):
            fn(center, heading)
        timings: list[float] = []
        for _ in range(3):
            start = time.perf_counter()
            for center, heading in zip(centers, headings):
                fn(center, heading)
            timings.append(time.perf_counter() - start)
        best = min(timings)
        print(f"{name}: calls={len(centers)} best_elapsed={best:.4f}s per_call_ms={best / len(centers) * 1000:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental ray/arc lidar benchmark for TM2D maps.")
    parser.add_argument("--map-name", default="AI Training #5")
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--arc-source", choices=["analytic", "fit"], default="analytic")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark(args.map_name, args.samples, args.arc_source)
