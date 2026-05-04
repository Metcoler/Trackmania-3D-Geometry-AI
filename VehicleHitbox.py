from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class VehicleHitbox:
    """Empirical AABB footprint used to turn center lidar into clearance lidar.

    The raycasts still start in the car center for performance.  For each laser
    direction we subtract the distance from the center to this AABB boundary,
    so a normalized lidar value of zero means contact.
    """

    half_width: float = 1.10
    front_half_length: float = 2.15
    rear_half_length: float = 1.95
    half_height: float = 0.60
    safety_margin: float = 0.7
    source: str = "empirical_mesh_and_supervised_aabb_20260503"

    @classmethod
    def stadium_car_sport(cls) -> "VehicleHitbox":
        return cls()

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def laser_angles_radians(num_lasers: int, angle_degrees: float) -> np.ndarray:
        num_lasers = int(num_lasers)
        if num_lasers <= 1:
            return np.zeros(max(1, num_lasers), dtype=np.float32)
        angle_range = np.radians(float(angle_degrees))
        return (
            np.arange(num_lasers, dtype=np.float32) * (angle_range / float(num_lasers - 1))
            - angle_range / 2.0
        ).astype(np.float32, copy=False)

    def support_distances_for_angles(self, angles_radians) -> np.ndarray:
        """Return distance from center to the AABB footprint along each angle.

        Angles are relative to the car forward direction. Positive/negative
        lateral angles are symmetric.  The implementation also supports rear
        rays even though the current 180 degree lidar fan only covers the front.
        """

        angles = np.asarray(angles_radians, dtype=np.float32).reshape(-1)
        forward_component = np.cos(angles)
        lateral_component = np.sin(angles)

        forward_half = np.where(
            forward_component >= 0.0,
            float(self.front_half_length),
            float(self.rear_half_length),
        )
        eps = np.float32(1e-6)
        longitudinal_limit = np.full_like(angles, np.inf, dtype=np.float32)
        lateral_limit = np.full_like(angles, np.inf, dtype=np.float32)

        longitudinal_mask = np.abs(forward_component) > eps
        longitudinal_limit[longitudinal_mask] = (
            forward_half[longitudinal_mask] / np.abs(forward_component[longitudinal_mask])
        )

        lateral_mask = np.abs(lateral_component) > eps
        lateral_limit[lateral_mask] = float(self.half_width) / np.abs(lateral_component[lateral_mask])

        distances = np.minimum(longitudinal_limit, lateral_limit)
        distances[~np.isfinite(distances)] = 0.0
        distances = distances + float(self.safety_margin)
        return distances.astype(np.float32, copy=False)

    def laser_offsets_2d(
        self,
        num_lasers: int,
        angle_degrees: float,
    ) -> np.ndarray:
        return self.support_distances_for_angles(
            self.laser_angles_radians(num_lasers=num_lasers, angle_degrees=angle_degrees)
        )

    def clearances(
        self,
        raw_distances,
        *,
        num_lasers: int,
        angle_degrees: float,
    ) -> np.ndarray:
        raw = np.asarray(raw_distances, dtype=np.float32).reshape(-1)
        offsets = self.laser_offsets_2d(num_lasers=num_lasers, angle_degrees=angle_degrees)
        if raw.size < offsets.size:
            raw = np.pad(raw, (0, offsets.size - raw.size), constant_values=np.inf)
        elif raw.size > offsets.size:
            raw = raw[: offsets.size]
        return (raw - offsets).astype(np.float32, copy=False)

    def normalized_clearances(
        self,
        raw_distances,
        *,
        num_lasers: int,
        angle_degrees: float,
        laser_max_distance: float,
    ) -> np.ndarray:
        raw = np.asarray(raw_distances, dtype=np.float32).reshape(-1)
        offsets = self.laser_offsets_2d(num_lasers=num_lasers, angle_degrees=angle_degrees)
        if raw.size < offsets.size:
            raw = np.pad(raw, (0, offsets.size - raw.size), constant_values=float(laser_max_distance))
        elif raw.size > offsets.size:
            raw = raw[: offsets.size]
        denominators = np.maximum(1e-6, float(laser_max_distance) - offsets)
        normalized = (raw - offsets) / denominators
        return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)
