from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Car import Car
from Map import MAP_BLOCK_SIZE, Map


def _cross_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _normalize_2d(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        if fallback is None:
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.asarray(fallback, dtype=np.float32)
    return (vector / norm).astype(np.float32, copy=False)


@dataclass(frozen=True)
class RaycastResult:
    distances: np.ndarray
    endpoints: np.ndarray
    directions: np.ndarray


class TM2DGeometry:
    """2D projection of our Trackmania map data.

    The source of truth remains `Map.py` and the exported Trackmania block files.
    We project road/wall meshes onto XZ and keep only a fast 2D representation for
    local experiments.
    """

    def __init__(
        self,
        map_name: str,
        laser_max_distance: float = Car.LASER_MAX_DISTANCE,
        num_lasers: int = Car.NUM_LASERS,
        laser_angle_degrees: float = Car.ANGLE,
    ) -> None:
        self.map_name = str(map_name)
        self.game_map = Map(self.map_name)
        self.laser_max_distance = float(laser_max_distance)
        self.num_lasers = int(num_lasers)
        self.laser_angle_degrees = float(laser_angle_degrees)
        self._laser_offsets_radians = (
            np.linspace(
                -self.laser_angle_degrees * 0.5,
                self.laser_angle_degrees * 0.5,
                self.num_lasers,
                dtype=np.float32,
            )
            * np.pi
            / 180.0
        ).astype(np.float32, copy=False)

        self.wall_segments = self._segments_from_wall_mesh()
        if len(self.wall_segments) == 0:
            self.wall_segments = self._segments_from_road_boundary()
        self.wall_segments = self._merge_collinear_segments(self.wall_segments)
        self._seg_start = self.wall_segments[:, 0, :].astype(np.float32, copy=False)
        self._seg_vec = (
            self.wall_segments[:, 1, :] - self.wall_segments[:, 0, :]
        ).astype(np.float32, copy=False)
        self._segment_grid = self._build_segment_grid()

        self.road_triangles = self._road_triangles_xz()
        self._triangle_grid = self._build_triangle_grid()
        self.path_tiles_xz = [
            (int(tile[0]), int(tile[2]))
            for tile in self.game_map.path_tiles
        ]
        self.path_points = np.asarray(
            [
                [float(tile[0]) * MAP_BLOCK_SIZE + MAP_BLOCK_SIZE * 0.5,
                 float(tile[2]) * MAP_BLOCK_SIZE + MAP_BLOCK_SIZE * 0.5]
                for tile in self.game_map.path_tiles
            ],
            dtype=np.float32,
        )
        self.path_instructions = list(self.game_map.path_instructions)
        self.path_surface_instructions = list(self.game_map.path_surface_instructions)
        self.path_height_instructions = list(self.game_map.path_height_instructions)
        self.progress_bucket = 100.0 / max(1, len(self.path_tiles_xz) - 1)

        self.start_position = self.path_points[0].astype(np.float32, copy=True)
        start_dir_3d = np.asarray(self.game_map.get_start_direction(), dtype=np.float32)
        self.start_forward = _normalize_2d(
            np.array([start_dir_3d[0], start_dir_3d[2]], dtype=np.float32)
        )

    @property
    def estimated_path_length(self) -> float:
        return float(max(1, len(self.path_tiles_xz) - 1) * MAP_BLOCK_SIZE)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        points = []
        if len(self.road_triangles) > 0:
            points.append(self.road_triangles.reshape(-1, 2))
        if len(self.wall_segments) > 0:
            points.append(self.wall_segments.reshape(-1, 2))
        if not points:
            return 0.0, 0.0, 1.0, 1.0
        all_points = np.vstack(points)
        min_xy = np.min(all_points, axis=0)
        max_xy = np.max(all_points, axis=0)
        return float(min_xy[0]), float(min_xy[1]), float(max_xy[0]), float(max_xy[1])

    @staticmethod
    def _project_xz(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        return points[..., [0, 2]]

    def _segments_from_wall_mesh(self) -> np.ndarray:
        mesh = self.game_map.get_sensor_walls_mesh()
        vertices = self._project_xz(np.asarray(mesh.vertices, dtype=np.float32))
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(vertices) == 0 or len(faces) == 0:
            return np.empty((0, 2, 2), dtype=np.float32)

        unique: dict[tuple[int, int, int, int], np.ndarray] = {}
        for face in faces:
            for a_idx, b_idx in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                a = vertices[int(a_idx)]
                b = vertices[int(b_idx)]
                if float(np.linalg.norm(b - a)) <= 1e-4:
                    continue
                rounded_a = tuple(np.round(a, 3).astype(np.int64))
                rounded_b = tuple(np.round(b, 3).astype(np.int64))
                key = (*rounded_a, *rounded_b)
                reverse_key = (*rounded_b, *rounded_a)
                if reverse_key in unique:
                    continue
                unique[key] = np.asarray([a, b], dtype=np.float32)
        if not unique:
            return np.empty((0, 2, 2), dtype=np.float32)
        return np.stack(list(unique.values())).astype(np.float32, copy=False)

    def _segments_from_road_boundary(self) -> np.ndarray:
        mesh = self.game_map.get_road_mesh()
        vertices = self._project_xz(np.asarray(mesh.vertices, dtype=np.float32))
        faces = np.asarray(mesh.faces, dtype=np.int64)
        edge_counts: dict[tuple[tuple[float, float], tuple[float, float]], int] = {}
        edge_points: dict[tuple[tuple[float, float], tuple[float, float]], np.ndarray] = {}
        for face in faces:
            for a_idx, b_idx in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                a = vertices[int(a_idx)]
                b = vertices[int(b_idx)]
                if float(np.linalg.norm(b - a)) <= 1e-4:
                    continue
                rounded_a = tuple(np.round(a, 3))
                rounded_b = tuple(np.round(b, 3))
                key = tuple(sorted((rounded_a, rounded_b)))
                edge_counts[key] = edge_counts.get(key, 0) + 1
                edge_points[key] = np.asarray([a, b], dtype=np.float32)
        boundary = [edge_points[key] for key, count in edge_counts.items() if count == 1]
        if not boundary:
            return np.empty((0, 2, 2), dtype=np.float32)
        return np.stack(boundary).astype(np.float32, copy=False)

    def _road_triangles_xz(self) -> np.ndarray:
        mesh = self.game_map.get_road_mesh()
        triangles = self._project_xz(np.asarray(mesh.triangles, dtype=np.float32))
        valid = []
        for triangle in triangles:
            area = abs(float(_cross_2d(triangle[1] - triangle[0], triangle[2] - triangle[0])))
            if area > 1e-5:
                valid.append(triangle)
        if not valid:
            return np.empty((0, 3, 2), dtype=np.float32)
        return np.stack(valid).astype(np.float32, copy=False)

    def _build_triangle_grid(self) -> dict[tuple[int, int], list[int]]:
        grid: dict[tuple[int, int], list[int]] = {}
        for idx, triangle in enumerate(self.road_triangles):
            mins = np.floor(np.min(triangle, axis=0) / MAP_BLOCK_SIZE).astype(int)
            maxs = np.floor(np.max(triangle, axis=0) / MAP_BLOCK_SIZE).astype(int)
            for cell_x in range(int(mins[0]), int(maxs[0]) + 1):
                for cell_z in range(int(mins[1]), int(maxs[1]) + 1):
                    grid.setdefault((cell_x, cell_z), []).append(int(idx))
        return grid

    def _build_segment_grid(self) -> dict[tuple[int, int], list[int]]:
        grid: dict[tuple[int, int], list[int]] = {}
        for idx, segment in enumerate(self.wall_segments):
            mins = np.floor(np.min(segment, axis=0) / MAP_BLOCK_SIZE).astype(int) - 1
            maxs = np.floor(np.max(segment, axis=0) / MAP_BLOCK_SIZE).astype(int) + 1
            for cell_x in range(int(mins[0]), int(maxs[0]) + 1):
                for cell_z in range(int(mins[1]), int(maxs[1]) + 1):
                    grid.setdefault((cell_x, cell_z), []).append(int(idx))
        return grid

    @staticmethod
    def _merge_collinear_segments(
        segments: np.ndarray,
        angle_precision_degrees: float = 0.25,
        offset_precision: float = 0.05,
        gap_tolerance: float = 0.05,
    ) -> np.ndarray:
        """Merge overlapping/touching collinear wall fragments.

        Straight/checkpoint blocks are already simple, but start/finish meshes
        and triangulated wall surfaces can still contain tiny collinear pieces.
        Merging only same-line intervals keeps raycast geometry effectively
        unchanged while reducing the number of segment intersections.
        """

        segments = np.asarray(segments, dtype=np.float32)
        if len(segments) <= 1:
            return segments

        groups: dict[tuple[float, float], list[tuple[float, float, np.ndarray, np.ndarray, float]]] = {}
        for segment in segments:
            a = np.asarray(segment[0], dtype=np.float64)
            b = np.asarray(segment[1], dtype=np.float64)
            vec = b - a
            length = float(np.linalg.norm(vec))
            if length <= 1e-4:
                continue
            direction = vec / length
            if direction[0] < -1e-9 or (abs(direction[0]) <= 1e-9 and direction[1] < 0.0):
                direction = -direction
                a, b = b, a

            angle = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)
            angle_key = round(angle / angle_precision_degrees) * angle_precision_degrees
            normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
            offset = float(np.dot(normal, a))
            offset_key = round(offset / offset_precision) * offset_precision
            t0 = float(np.dot(direction, a))
            t1 = float(np.dot(direction, b))
            if t1 < t0:
                t0, t1 = t1, t0
            groups.setdefault((angle_key, offset_key), []).append((t0, t1, direction, normal, offset))

        merged: list[np.ndarray] = []
        for intervals in groups.values():
            intervals.sort(key=lambda item: item[0])
            current_start, current_end, direction, normal, offset = intervals[0]
            for start, end, _, _, _ in intervals[1:]:
                if start <= current_end + gap_tolerance:
                    current_end = max(current_end, end)
                else:
                    merged.append(
                        np.asarray(
                            [
                                direction * current_start + normal * offset,
                                direction * current_end + normal * offset,
                            ],
                            dtype=np.float32,
                        )
                    )
                    current_start, current_end = start, end
            merged.append(
                np.asarray(
                    [
                        direction * current_start + normal * offset,
                        direction * current_end + normal * offset,
                    ],
                    dtype=np.float32,
                )
            )

        if not merged:
            return np.empty((0, 2, 2), dtype=np.float32)
        return np.stack(merged).astype(np.float32, copy=False)

    def point_to_cell(self, point: np.ndarray) -> tuple[int, int]:
        point = np.asarray(point, dtype=np.float32)
        return int(math.floor(float(point[0]) / MAP_BLOCK_SIZE)), int(
            math.floor(float(point[1]) / MAP_BLOCK_SIZE)
        )

    def point_on_road(self, point: np.ndarray, margin: float = 1e-4) -> bool:
        point = np.asarray(point, dtype=np.float32)
        candidates = self._triangle_grid.get(self.point_to_cell(point), [])
        if not candidates:
            return False
        triangles = self.road_triangles[candidates]
        a = triangles[:, 0, :]
        b = triangles[:, 1, :]
        c = triangles[:, 2, :]
        v0 = c - a
        v1 = b - a
        v2 = point.reshape(1, 2) - a
        dot00 = np.einsum("ij,ij->i", v0, v0)
        dot01 = np.einsum("ij,ij->i", v0, v1)
        dot02 = np.einsum("ij,ij->i", v0, v2)
        dot11 = np.einsum("ij,ij->i", v1, v1)
        dot12 = np.einsum("ij,ij->i", v1, v2)
        denom = dot00 * dot11 - dot01 * dot01
        valid = np.abs(denom) > 1e-8
        inv = np.zeros_like(denom)
        inv[valid] = 1.0 / denom[valid]
        u = (dot11 * dot02 - dot01 * dot12) * inv
        v = (dot00 * dot12 - dot01 * dot02) * inv
        inside = (u >= -margin) & (v >= -margin) & ((u + v) <= 1.0 + margin)
        return bool(np.any(inside & valid))

    def car_on_road(self, center: np.ndarray, heading: float, length: float, width: float) -> bool:
        forward = np.array([math.cos(heading), math.sin(heading)], dtype=np.float32)
        right = np.array([-forward[1], forward[0]], dtype=np.float32)
        half_l = float(length) * 0.5
        half_w = float(width) * 0.5
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                corner = center + forward * half_l * sx + right * half_w * sy
                if not self.point_on_road(corner):
                    return False
        return True

    def update_path_index(self, point: np.ndarray, path_index: int) -> int:
        current_cell = self.point_to_cell(point)
        if 0 <= path_index < len(self.path_tiles_xz) and current_cell == self.path_tiles_xz[path_index]:
            return path_index
        if (
            path_index < len(self.path_tiles_xz) - 1
            and current_cell == self.path_tiles_xz[path_index + 1]
        ):
            return path_index + 1
        if path_index > 0 and current_cell == self.path_tiles_xz[path_index - 1]:
            return path_index - 1
        return path_index

    def progress_for_index(self, path_index: int) -> float:
        return float(path_index / max(1, len(self.path_tiles_xz) - 1) * 100.0)

    def dense_progress_for_point(self, point: np.ndarray, path_index: int) -> float:
        """Continuous progress across the current path tile.

        This mirrors `Car.dense_progress()`: the discrete path index advances at
        tile boundaries, so dense progress must fill the whole current tile
        span instead of only the center-to-center half segment.
        """
        point = np.asarray(point, dtype=np.float32)
        if len(self.path_points) <= 1:
            return 0.0
        idx0 = int(np.clip(path_index, 0, len(self.path_points) - 1))
        if idx0 >= len(self.path_points) - 1:
            return 100.0

        current = self.path_points[idx0].astype(np.float32, copy=False)
        next_point = self.path_points[idx0 + 1].astype(np.float32, copy=False)
        previous = (
            self.path_points[idx0 - 1].astype(np.float32, copy=False)
            if idx0 > 0
            else None
        )

        exit_direction = _normalize_2d(next_point - current)
        if previous is None:
            entry_direction = exit_direction
        else:
            entry_direction = _normalize_2d(current - previous, fallback=exit_direction)

        half_tile = MAP_BLOCK_SIZE * 0.5
        entry = current - entry_direction * half_tile
        exit_point = current + exit_direction * half_tile

        def project_segment(p: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, float, float]:
            segment = end - start
            segment_len_sq = float(np.dot(segment, segment))
            if segment_len_sq <= 1e-6:
                return 0.0, 0.0, float(np.linalg.norm(p - start))
            segment_len = float(math.sqrt(segment_len_sq))
            t = float(np.dot(p - start, segment) / segment_len_sq)
            t = float(np.clip(t, 0.0, 1.0))
            closest = start + segment * t
            distance = float(np.linalg.norm(p - closest))
            return t, segment_len, distance

        entry_t, entry_len, entry_dist = project_segment(point, entry, current)
        exit_t, exit_len, exit_dist = project_segment(point, current, exit_point)
        total_len = entry_len + exit_len
        if total_len <= 1e-6:
            fraction = 0.0
        elif entry_dist <= exit_dist:
            fraction = float(np.clip((entry_t * entry_len) / total_len, 0.0, 1.0))
        else:
            fraction = float(np.clip((entry_len + exit_t * exit_len) / total_len, 0.0, 1.0))
        dense_index = float(idx0) + fraction
        return float(dense_index / max(1, len(self.path_points) - 1) * 100.0)

    def next_path_vectors(self, path_index: int) -> tuple[np.ndarray, np.ndarray]:
        idx0 = min(max(int(path_index), 0), len(self.path_points) - 1)
        idx1 = min(idx0 + 1, len(self.path_points) - 1)
        idx2 = min(idx1 + 1, len(self.path_points) - 1)
        current = _normalize_2d(self.path_points[idx1] - self.path_points[idx0])
        next_vec = _normalize_2d(self.path_points[idx2] - self.path_points[idx1], fallback=current)
        return current, next_vec

    def signed_heading_error(self, forward: np.ndarray, target: np.ndarray) -> float:
        forward = _normalize_2d(forward)
        target = _normalize_2d(target)
        dot = float(np.clip(np.dot(forward, target), -1.0, 1.0))
        cross = float(forward[0] * target[1] - forward[1] * target[0])
        return float(math.atan2(cross, dot) / math.pi)

    def path_instructions_at(self, path_index: int, count: int = Car.SIGHT_TILES) -> list[float]:
        return self._window(self.path_instructions, path_index, count, 0.0)

    def surface_instructions_at(self, path_index: int, count: int = Car.SIGHT_TILES) -> list[float]:
        return self._window(self.path_surface_instructions, path_index, count, 1.0)

    def height_instructions_at(self, path_index: int, count: int = Car.SIGHT_TILES) -> list[float]:
        return self._window(self.path_height_instructions, path_index, count, 0.0)

    @staticmethod
    def _window(values: list[float], start: int, count: int, pad: float) -> list[float]:
        result = list(values[int(start): int(start) + int(count)])
        if not result:
            result = [pad]
        while len(result) < count:
            result.append(result[-1])
        return result[:count]

    def cast_lasers(self, center: np.ndarray, heading: float) -> RaycastResult:
        center = np.asarray(center, dtype=np.float32)
        angles = float(heading) + self._laser_offsets_radians
        directions = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
        distances = np.full(self.num_lasers, self.laser_max_distance, dtype=np.float32)

        # For the current TM block maps, broadcasting all laser/segment pairs in
        # NumPy is faster than per-ray grid traversal because it avoids Python
        # loops. Keep the grid path for very large maps where candidate pruning
        # can outweigh that overhead.
        if 0 < len(self.wall_segments) <= 2_000:
            p = self._seg_start.reshape(1, -1, 2)
            s = self._seg_vec.reshape(1, -1, 2)
            d = directions.reshape(-1, 1, 2)
            rel = p - center.reshape(1, 1, 2)
            denom = d[..., 0] * s[..., 1] - d[..., 1] * s[..., 0]
            valid = np.abs(denom) > 1e-7
            safe_denom = np.where(valid, denom, 1.0)
            t = (rel[..., 0] * s[..., 1] - rel[..., 1] * s[..., 0]) / safe_denom
            u = (rel[..., 0] * d[..., 1] - rel[..., 1] * d[..., 0]) / safe_denom
            hit = (
                valid
                & (t >= 0.0)
                & (t <= self.laser_max_distance)
                & (u >= 0.0)
                & (u <= 1.0)
            )
            hit_t = np.where(hit, t, np.inf)
            closest = np.min(hit_t, axis=1)
            distances = np.where(np.isfinite(closest), closest, distances).astype(np.float32)
        elif len(self.wall_segments) > 0:
            for laser_idx, direction in enumerate(directions):
                candidates = self._candidate_segments_for_ray(center, direction)
                if candidates.size == 0:
                    continue
                p = self._seg_start[candidates]
                s = self._seg_vec[candidates]
                rel = p - center.reshape(1, 2)
                denom = direction[0] * s[:, 1] - direction[1] * s[:, 0]
                valid = np.abs(denom) > 1e-7
                if not np.any(valid):
                    continue
                safe_denom = np.where(valid, denom, 1.0)
                t = (rel[:, 0] * s[:, 1] - rel[:, 1] * s[:, 0]) / safe_denom
                u = (rel[:, 0] * direction[1] - rel[:, 1] * direction[0]) / safe_denom
                hit = (
                    valid
                    & (t >= 0.0)
                    & (t <= self.laser_max_distance)
                    & (u >= 0.0)
                    & (u <= 1.0)
                )
                if np.any(hit):
                    distances[laser_idx] = float(np.min(t[hit]))

        endpoints = center.reshape(1, 2) + directions * distances.reshape(-1, 1)
        return RaycastResult(distances=distances, endpoints=endpoints, directions=directions)

    def _candidate_segments_for_ray(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        origin = np.asarray(origin, dtype=np.float32)
        direction = _normalize_2d(direction)
        cell_x, cell_z = self.point_to_cell(origin)
        step_x = 1 if direction[0] >= 0.0 else -1
        step_z = 1 if direction[1] >= 0.0 else -1

        if abs(float(direction[0])) <= 1e-7:
            t_delta_x = float("inf")
            t_max_x = float("inf")
        else:
            next_x = (cell_x + (1 if step_x > 0 else 0)) * MAP_BLOCK_SIZE
            t_max_x = (next_x - float(origin[0])) / float(direction[0])
            t_delta_x = MAP_BLOCK_SIZE / abs(float(direction[0]))

        if abs(float(direction[1])) <= 1e-7:
            t_delta_z = float("inf")
            t_max_z = float("inf")
        else:
            next_z = (cell_z + (1 if step_z > 0 else 0)) * MAP_BLOCK_SIZE
            t_max_z = (next_z - float(origin[1])) / float(direction[1])
            t_delta_z = MAP_BLOCK_SIZE / abs(float(direction[1]))

        candidates: set[int] = set()
        max_steps = int(math.ceil(self.laser_max_distance / MAP_BLOCK_SIZE)) + 4
        traveled = 0.0
        for _ in range(max_steps):
            candidates.update(self._segment_grid.get((cell_x, cell_z), ()))
            if t_max_x < t_max_z:
                traveled = float(t_max_x)
                t_max_x += t_delta_x
                cell_x += step_x
            else:
                traveled = float(t_max_z)
                t_max_z += t_delta_z
                cell_z += step_z
            if traveled > self.laser_max_distance:
                break
        if not candidates:
            return np.empty(0, dtype=np.int64)
        return np.fromiter(candidates, dtype=np.int64)
