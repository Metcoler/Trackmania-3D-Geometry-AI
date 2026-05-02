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
from Individual import Individual
from ObservationEncoder import ObservationEncoder

from Experiments.tm2d_geometry import TM2DGeometry, _normalize_2d


@dataclass
class TM2DPhysicsConfig:
    min_dt: float = 1.0 / 100.0
    max_dt: float = 1.0 / 30.0
    max_speed: float = 120.0
    reverse_speed: float = 10.0
    gas_accel: float = 19.563
    brake_accel: float = 17.463
    drag: float = 0.1439
    rolling_drag: float = 5.549
    lateral_grip: float = 5.5
    max_yaw_rate: float = 1.011
    car_length: float = 4.8
    car_width: float = 2.6

    def with_fixed_fps(self, fps: float | None) -> "TM2DPhysicsConfig":
        if fps is None or float(fps) <= 0.0:
            return self
        fixed_dt = 1.0 / float(fps)
        return TM2DPhysicsConfig(
            min_dt=fixed_dt,
            max_dt=fixed_dt,
            max_speed=self.max_speed,
            reverse_speed=self.reverse_speed,
            gas_accel=self.gas_accel,
            brake_accel=self.brake_accel,
            drag=self.drag,
            rolling_drag=self.rolling_drag,
            lateral_grip=self.lateral_grip,
            max_yaw_rate=self.max_yaw_rate,
            car_length=self.car_length,
            car_width=self.car_width,
        )


@dataclass
class TM2DRewardConfig:
    mode: str = "progress_primary_delta"
    terminal_fitness_scale: float = 1_000_000.0
    finish_bonus: float = Individual.FINISHED_WEIGHT / 1_000_000.0
    pace_target_time: float | None = None


class TM2DSimEnv:
    """Fast 2D local Trackmania-like environment.

    It intentionally keeps the same high-level observation contract as the live
    Trackmania environment, while replacing realtime I/O with deterministic local
    simulation.
    """

    def __init__(
        self,
        map_name: str = "AI Training #5",
        max_time: float = 45.0,
        reward_config: TM2DRewardConfig | None = None,
        physics_config: TM2DPhysicsConfig | None = None,
        seed: int | None = None,
        random_start_jitter: float = 0.0,
        collision_mode: str = "laser",
        collision_distance_threshold: float = 2.0,
        start_idle_max_time: float = 5.0,
        start_idle_progress_epsilon: float = 0.5,
        start_idle_speed_threshold: float = 3.0,
        stuck_timeout_speed_threshold: float = 2.5,
        stuck_timeout_duration: float = 2.5,
        vertical_mode: bool = False,
        multi_surface_mode: bool = False,
        binary_gas_brake: bool = True,
    ) -> None:
        self.geometry = TM2DGeometry(map_name=map_name)
        self.max_time = float(max_time)
        self.reward_config = reward_config or TM2DRewardConfig()
        self.physics = physics_config or TM2DPhysicsConfig()
        self.random_start_jitter = float(random_start_jitter)
        self.collision_mode = str(collision_mode).strip().lower()
        if self.collision_mode not in {"center", "corners", "laser", "lidar"}:
            raise ValueError("collision_mode must be 'center', 'corners', or 'laser'.")
        self.collision_distance_threshold = float(collision_distance_threshold)
        self.start_idle_max_time = float(start_idle_max_time)
        self.start_idle_progress_epsilon = float(start_idle_progress_epsilon)
        self.start_idle_speed_threshold = float(start_idle_speed_threshold)
        self.stuck_timeout_speed_threshold = float(stuck_timeout_speed_threshold)
        self.stuck_timeout_duration = float(stuck_timeout_duration)
        self.binary_gas_brake = bool(binary_gas_brake)
        self.rng = np.random.default_rng(seed)
        self.obs_encoder = ObservationEncoder(
            vertical_mode=bool(vertical_mode),
            multi_surface_mode=bool(multi_surface_mode),
        )
        self._clearance_windows = ObservationEncoder.clearance_window_bounds()
        self._obs_buffer = np.zeros(self.obs_encoder.obs_dim, dtype=np.float32)
        self._path_instr_buffer = np.zeros(Car.SIGHT_TILES, dtype=np.float32)
        self._surface_instr_buffer = np.ones(Car.SIGHT_TILES, dtype=np.float32)
        self._height_instr_buffer = np.zeros(Car.SIGHT_TILES, dtype=np.float32)
        self.observation_space_shape = (self.obs_encoder.obs_dim,)
        self.action_space_shape = (3,)
        self.reset()

    @property
    def obs_dim(self) -> int:
        return self.obs_encoder.obs_dim

    @property
    def act_dim(self) -> int:
        return 3

    @property
    def progress_bucket(self) -> float:
        return self.geometry.progress_bucket

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.obs_encoder.reset()
        self.position = self.geometry.start_position.astype(np.float32, copy=True)
        self.forward = self.geometry.start_forward.astype(np.float32, copy=True)
        if self.random_start_jitter > 0.0:
            right = np.array([-self.forward[1], self.forward[0]], dtype=np.float32)
            self.position += right * float(self.rng.uniform(-self.random_start_jitter, self.random_start_jitter))
        self.heading = float(math.atan2(self.forward[1], self.forward[0]))
        self.velocity = self.forward * 8.0
        self.speed = float(np.linalg.norm(self.velocity))
        self.side_speed = 0.0
        self.distance = 0.0
        self.time = 0.0
        self.path_index = 0
        self.finished = 0
        self.crashes = 0
        self.done = False
        self._stuck_since_time = None
        self.last_score = self._score_state(
            finished=0,
            crashes=0,
            progress=0.0,
            time_value=0.0,
            distance=0.0,
        )
        self.last_yaw_rate = 0.0
        self.current_dt = self.obs_encoder.dt_ref
        self.previous_speed = None
        self.previous_side_speed = None
        self.previous_clearance_windows = None
        self._obs_buffer.fill(0.0)
        obs, info = self._build_observation()
        return obs, info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.done:
            obs, info = self._build_observation()
            return obs, 0.0, True, False, info

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 3:
            action = np.pad(action, (0, 3 - action.size), constant_values=0.0)
        gas = float(np.clip(action[0], 0.0, 1.0))
        brake = float(np.clip(action[1], 0.0, 1.0))
        steer = float(np.clip(action[2], -1.0, 1.0))
        if self.binary_gas_brake:
            gas = 1.0 if gas > 0.5 else 0.0
            brake = 1.0 if brake > 0.5 else 0.0

        dt = float(self.rng.uniform(self.physics.min_dt, self.physics.max_dt))
        self.current_dt = dt
        old_heading = float(self.heading)
        old_position = self.position.copy()

        forward = np.array([math.cos(self.heading), math.sin(self.heading)], dtype=np.float32)
        right = np.array([-forward[1], forward[0]], dtype=np.float32)
        forward_speed = float(np.dot(self.velocity, forward))
        side_speed = float(np.dot(self.velocity, right))

        accel = gas * self.physics.gas_accel - brake * self.physics.brake_accel
        accel -= self.physics.drag * forward_speed
        if abs(forward_speed) > 1e-3:
            accel -= math.copysign(self.physics.rolling_drag, forward_speed)
        forward_speed += accel * dt
        forward_speed = float(np.clip(forward_speed, -self.physics.reverse_speed, self.physics.max_speed))

        speed_factor = float(np.clip(abs(forward_speed) / 70.0, 0.0, 1.0))
        yaw_rate = steer * self.physics.max_yaw_rate * (0.25 + 0.75 * speed_factor)
        self.heading += yaw_rate * dt
        self.forward = np.array([math.cos(self.heading), math.sin(self.heading)], dtype=np.float32)
        right = np.array([-self.forward[1], self.forward[0]], dtype=np.float32)

        side_speed *= math.exp(-self.physics.lateral_grip * dt)
        self.velocity = self.forward * forward_speed + right * side_speed
        self.position = self.position + self.velocity * dt
        self.time += dt
        self.distance += float(np.linalg.norm(self.position - old_position))
        self.speed = float(np.linalg.norm(self.velocity))
        self.side_speed = float(side_speed)
        self.last_yaw_rate = float((self.heading - old_heading) / max(dt, 1e-6))

        self.path_index = self.geometry.update_path_index(self.position, self.path_index)
        progress = self.geometry.progress_for_index(self.path_index)
        raycast = None
        if self.collision_mode in {"laser", "lidar"}:
            raycast = self.geometry.cast_lasers(self.position, self.heading)

        terminated = False
        truncated = False
        finished = 0
        crashes = 0
        if self._crash_detected(raycast=raycast):
            terminated = True
            crashes = 1
        elif self.path_index >= len(self.geometry.path_tiles_xz) - 1:
            terminated = True
            finished = 1
        elif self.time >= self.max_time:
            truncated = True
        elif (
            self.time >= self.start_idle_max_time
            and progress <= self.start_idle_progress_epsilon
            and self.speed <= self.start_idle_speed_threshold
        ):
            terminated = True
            crashes = 1
        elif (
            progress > self.start_idle_progress_epsilon
            and self.speed <= self.stuck_timeout_speed_threshold
        ):
            if self._stuck_since_time is None:
                self._stuck_since_time = self.time
            elif (self.time - float(self._stuck_since_time)) >= self.stuck_timeout_duration:
                terminated = True
                crashes = 1
        else:
            self._stuck_since_time = None

        self.done = bool(terminated or truncated)
        self.finished = int(finished)
        self.crashes = int(crashes)
        obs, info = self._build_observation(raycast=raycast)
        score_progress = float(info.get("dense_progress", progress))
        score = self._score_state(
            finished=finished if self.done else 0,
            crashes=crashes if self.done else 0,
            progress=score_progress,
            time_value=self.time,
            distance=self.distance,
        )
        reward = float(score - self.last_score)
        self.last_score = score
        if self.done and finished > 0 and self._uses_external_finish_bonus():
            reward += float(self.reward_config.finish_bonus)
        info.update(
            {
                "finished": int(finished),
                "crashes": int(crashes),
                "timeout": int(self.done and int(finished) == 0 and int(crashes) == 0),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "reward_score": float(score),
                "reward": float(reward),
                "dt": float(dt),
            }
        )
        return obs, reward, bool(terminated), bool(truncated), info

    def _uses_external_finish_bonus(self) -> bool:
        mode = str(self.reward_config.mode).strip().lower()
        return mode in {"progress_delta", "progress_primary_delta", "pace_delta", "progress_rate"}

    def _crash_detected(self, raycast=None) -> bool:
        if self.collision_mode in {"laser", "lidar"}:
            if raycast is None:
                raycast = self.geometry.cast_lasers(self.position, self.heading)
            distances = np.asarray(raycast.distances, dtype=np.float32)
            if distances.size <= 0:
                return False
            return bool(float(np.min(distances)) < self.collision_distance_threshold)

        if self.collision_mode == "center":
            return not self.geometry.point_on_road(self.position)

        return not self.geometry.car_on_road(
            self.position,
            self.heading,
            length=self.physics.car_length,
            width=self.physics.car_width,
        )

    def _score_state(self, finished: int, crashes: int, progress: float, time_value: float, distance: float) -> float:
        mode = str(self.reward_config.mode).strip().lower()
        finished = 1 if int(finished) > 0 else 0
        crashes = max(0, int(crashes))
        progress = float(progress)
        time_value = max(0.0, float(time_value))
        distance = float(distance)
        if mode == "progress_delta":
            return progress
        if mode == "progress_primary_delta":
            time_penalty = self.geometry.progress_bucket * (time_value / max(1e-6, self.max_time))
            return progress - time_penalty
        if mode == "pace_delta":
            pace_target = self.reward_config.pace_target_time
            if pace_target is None:
                pace_target = self.max_time * 0.5
            expected_progress = 100.0 * time_value / max(1e-6, float(pace_target))
            return 2.0 * progress - expected_progress
        if mode == "terminal_fitness":
            if not self._is_terminal_score_state(finished, crashes, time_value):
                return 0.0
            return (
                Individual.compute_scalar_fitness_for(finished, crashes, progress, time_value, distance)
                / max(1e-6, self.reward_config.terminal_fitness_scale)
            )
        if mode == "individual_dense":
            return (
                Individual.compute_scalar_fitness_for(finished, crashes, progress, time_value, distance)
                / max(1e-6, self.reward_config.terminal_fitness_scale)
            )
        if mode == "terminal_lexicographic":
            if not self._is_terminal_score_state(finished, crashes, time_value):
                return 0.0
            return self._lexicographic_score(
                finished=finished,
                crashes=crashes,
                progress=progress,
                time_value=time_value,
                distance=distance,
                include_failure_term=True,
                ignore_time_below_progress=0.0,
                distance_mode="all",
            )
        if mode == "terminal_lexicographic_no_distance":
            if not self._is_terminal_score_state(finished, crashes, time_value):
                return 0.0
            return self._lexicographic_score(
                finished=finished,
                crashes=crashes,
                progress=progress,
                time_value=time_value,
                distance=distance,
                include_failure_term=True,
                ignore_time_below_progress=0.0,
                distance_mode="none",
            )
        if mode == "terminal_lexicographic_progress20":
            if not self._is_terminal_score_state(finished, crashes, time_value):
                return 0.0
            return self._lexicographic_score(
                finished=finished,
                crashes=crashes,
                progress=progress,
                time_value=time_value,
                distance=distance,
                include_failure_term=True,
                ignore_time_below_progress=20.0,
                distance_mode="finish",
            )
        if mode == "delta_lexicographic":
            return self._lexicographic_score(
                finished=finished,
                crashes=crashes,
                progress=progress,
                time_value=time_value,
                distance=distance,
                include_failure_term=False,
                ignore_time_below_progress=0.0,
                distance_mode="all",
            )
        if mode == "delta_lexicographic_terminal":
            return self._lexicographic_score(
                finished=finished,
                crashes=crashes,
                progress=progress,
                time_value=time_value,
                distance=distance,
                include_failure_term=True,
                ignore_time_below_progress=0.0,
                distance_mode="all",
            )
        if mode == "terminal_progress_time_efficiency":
            return Individual.compute_terminal_progress_time_efficiency_score_for(
                finished=finished,
                crashes=crashes,
                progress=progress,
                time_value=time_value,
                distance=distance,
                max_time=self.max_time,
                path_tile_count=len(self.geometry.path_tiles_xz),
                progress_bucket=self.geometry.progress_bucket,
                estimated_path_length=self.geometry.estimated_path_length,
                max_episode_distance=float(self.physics.max_speed) * float(self.max_time),
            )
        if mode == "delta_progress_time_efficiency":
            return Individual.compute_delta_progress_time_efficiency_score_for(
                finished=finished,
                crashes=crashes,
                progress=progress,
                time_value=time_value,
                distance=distance,
                max_time=self.max_time,
                path_tile_count=len(self.geometry.path_tiles_xz),
                progress_bucket=self.geometry.progress_bucket,
                estimated_path_length=self.geometry.estimated_path_length,
                max_episode_distance=float(self.physics.max_speed) * float(self.max_time),
            )
        if mode == "terminal_finished_progress_time":
            return Individual.compute_terminal_finished_progress_time_score_for(
                finished=finished,
                crashes=crashes,
                progress=progress,
                time_value=time_value,
                distance=distance,
                max_time=self.max_time,
                path_tile_count=len(self.geometry.path_tiles_xz),
                progress_bucket=self.geometry.progress_bucket,
            )
        if mode == "delta_finished_progress_time":
            return Individual.compute_delta_finished_progress_time_score_for(
                finished=finished,
                crashes=crashes,
                progress=progress,
                time_value=time_value,
                distance=distance,
                max_time=self.max_time,
                path_tile_count=len(self.geometry.path_tiles_xz),
                progress_bucket=self.geometry.progress_bucket,
            )
        if mode == "terminal_progress_time_safety":
            return Individual.compute_terminal_progress_time_safety_score_for(
                finished=finished,
                crashes=crashes,
                progress=progress,
                time_value=time_value,
                distance=distance,
                max_time=self.max_time,
                path_tile_count=len(self.geometry.path_tiles_xz),
                progress_bucket=self.geometry.progress_bucket,
            )
        if mode == "delta_progress_time_safety":
            return Individual.compute_delta_progress_time_safety_score_for(
                finished=finished,
                crashes=crashes,
                progress=progress,
                time_value=time_value,
                distance=distance,
                max_time=self.max_time,
                path_tile_count=len(self.geometry.path_tiles_xz),
                progress_bucket=self.geometry.progress_bucket,
            )
        if mode == "terminal_progress_time_block_penalty":
            return Individual.compute_terminal_progress_time_block_penalty_score_for(
                finished=finished,
                crashes=crashes,
                progress=progress,
                time_value=time_value,
                distance=distance,
                max_time=self.max_time,
                path_tile_count=len(self.geometry.path_tiles_xz),
                progress_bucket=self.geometry.progress_bucket,
            )
        if mode == "delta_progress_time_block_penalty":
            return Individual.compute_delta_progress_time_block_penalty_score_for(
                finished=finished,
                crashes=crashes,
                progress=progress,
                time_value=time_value,
                distance=distance,
                max_time=self.max_time,
                path_tile_count=len(self.geometry.path_tiles_xz),
                progress_bucket=self.geometry.progress_bucket,
            )
        if mode == "progress_rate":
            if time_value <= 1e-6:
                return 0.0
            progress_norm = np.clip(progress / 100.0, 0.0, 1.0)
            time_norm = max(time_value / max(1e-6, self.max_time), self._progress_unit_norm())
            return float(progress_norm / time_norm)
        raise ValueError(
            "Unknown reward mode. Use progress_delta, progress_primary_delta, "
            "pace_delta, terminal_fitness, individual_dense, terminal_lexicographic, "
            "terminal_lexicographic_no_distance, terminal_lexicographic_progress20, "
            "delta_lexicographic, delta_lexicographic_terminal, "
            "terminal_progress_time_efficiency, delta_progress_time_efficiency, "
            "terminal_finished_progress_time, delta_finished_progress_time, "
            "terminal_progress_time_safety, delta_progress_time_safety, "
            "terminal_progress_time_block_penalty, delta_progress_time_block_penalty, "
            "or progress_rate."
        )

    def _is_terminal_score_state(self, finished: int, crashes: int, time_value: float) -> bool:
        return int(finished) > 0 or int(crashes) > 0 or float(time_value) >= self.max_time

    def _progress_unit_norm(self) -> float:
        return 1.0 / max(1, len(self.geometry.path_tiles_xz) - 1)

    def _lexicographic_score(
        self,
        finished: int,
        crashes: int,
        progress: float,
        time_value: float,
        distance: float,
        include_failure_term: bool,
        ignore_time_below_progress: float,
        distance_mode: str,
    ) -> float:
        """Bounded scalar encoding of the intended ranking.

        The units are chosen from map/time geometry, not fitted constants:
        one whole path tile of progress is always larger than the entire time
        tie-break range, and the time tie-break is larger than the distance
        tie-break range. This keeps "go farther" dominant while still ranking
        equally far agents by speed and then by path efficiency.
        """

        terminal = self._is_terminal_score_state(finished, crashes, time_value)
        progress_norm = float(np.clip(float(progress) / 100.0, 0.0, 1.0))
        tile_unit = self._progress_unit_norm()

        score = progress_norm
        if terminal and int(finished) > 0:
            score += 1.0
        elif terminal and include_failure_term:
            score -= 1.0

        if float(progress) >= float(ignore_time_below_progress):
            time_norm = float(np.clip(float(time_value) / max(1e-6, self.max_time), 0.0, 1.0))
            score += tile_unit * (1.0 - time_norm)

        use_distance = distance_mode == "all" or (distance_mode == "finish" and terminal and int(finished) > 0)
        if use_distance and float(progress) >= float(ignore_time_below_progress):
            max_episode_distance = max(
                1e-6,
                float(self.geometry.estimated_path_length),
                float(self.physics.max_speed) * float(self.max_time),
            )
            distance_norm = float(np.clip(float(distance) / max_episode_distance, 0.0, 1.0))
            score += (tile_unit * tile_unit) * (1.0 - distance_norm)

        return float(score)

    def _build_observation(self, raycast=None) -> tuple[np.ndarray, dict]:
        if raycast is None:
            raycast = self.geometry.cast_lasers(self.position, self.heading)
        path_vec, next_path_vec = self.geometry.next_path_vectors(self.path_index)
        forward = _normalize_2d(self.forward)
        progress = self.geometry.progress_for_index(self.path_index)
        dense_progress = self.geometry.dense_progress_for_point(self.position, self.path_index)
        dense_progress = max(float(progress), float(dense_progress))
        slip_mean = float(
            np.clip(abs(self.side_speed) / max(8.0, abs(float(np.dot(self.velocity, forward)))), 0.0, 1.0)
        )
        info = {
            "speed": float(self.speed),
            "side_speed": float(self.side_speed),
            "distance": float(self.distance),
            "x": float(self.position[0]),
            "y": 0.0,
            "z": float(self.position[1]),
            "dx": float(forward[0]),
            "dy": 0.0,
            "dz": float(forward[1]),
            "time": float(self.time),
            "discrete_progress": float(progress),
            "dense_progress": float(dense_progress),
            "done": 1.0 if int(self.finished) > 0 else 0.0,
            "finished": int(self.finished),
            "crashes": int(self.crashes),
            "timeout": int(self.done and int(self.finished) == 0 and int(self.crashes) == 0),
            "collision_mode": self.collision_mode,
            "min_laser_distance": float(np.min(raycast.distances)) if len(raycast.distances) else float("inf"),
            "collision_distance_threshold": float(self.collision_distance_threshold),
            "slip_mean": slip_mean,
            "slip_fl": slip_mean,
            "slip_fr": slip_mean,
            "slip_rl": slip_mean,
            "slip_rr": slip_mean,
            "next_surface_instructions": self.geometry.surface_instructions_at(self.path_index),
            "next_height_instructions": self.geometry.height_instructions_at(self.path_index),
            "segment_heading_error": self.geometry.signed_heading_error(forward, path_vec),
            "next_segment_heading_error": self.geometry.signed_heading_error(forward, next_path_vec),
            "laser_endpoints_2d": raycast.endpoints,
            "laser_directions_2d": raycast.directions,
            "path_index": int(self.path_index),
            "progress_bucket": float(self.geometry.progress_bucket),
        }
        obs = self._build_fast_observation(raycast.distances, info)
        return obs, info

    def _build_fast_observation(self, distances: np.ndarray, info: dict) -> np.ndarray:
        distances = np.asarray(distances, dtype=np.float32)
        self._path_instr_buffer[:] = self.geometry.path_instructions_at(self.path_index)
        if self.obs_encoder.multi_surface_mode:
            self._surface_instr_buffer[:] = info["next_surface_instructions"]
        if self.obs_encoder.vertical_mode:
            self._height_instr_buffer[:] = info["next_height_instructions"]
        speed = float(info["speed"])
        side_speed = float(info["side_speed"])
        dt = max(1e-6, float(self.current_dt))
        dt_ratio = float(np.clip(dt / self.obs_encoder.dt_ref, 0.0, self.obs_encoder.dt_ratio_clip))
        clearance_windows = np.asarray(
            [
                float(np.mean(distances[start:end]))
                for start, end in self._clearance_windows
            ],
            dtype=np.float32,
        )
        longitudinal_accel = 0.0
        lateral_accel = 0.0
        clearance_rates = np.zeros(len(self._clearance_windows), dtype=np.float32)
        if self.previous_speed is not None:
            longitudinal_accel = (speed - float(self.previous_speed)) / dt
        if self.previous_side_speed is not None:
            lateral_accel = (side_speed - float(self.previous_side_speed)) / dt
        if self.previous_clearance_windows is not None:
            clearance_rates = (
                clearance_windows - np.asarray(self.previous_clearance_windows, dtype=np.float32)
            ) / dt

        self.previous_speed = speed
        self.previous_side_speed = side_speed
        self.previous_clearance_windows = clearance_windows

        info["dt_ratio"] = dt_ratio
        info["longitudinal_accel"] = float(longitudinal_accel)
        info["lateral_accel"] = float(lateral_accel)
        info["yaw_rate"] = float(self.last_yaw_rate)
        for idx, value in enumerate(clearance_rates):
            info[f"clearance_rate_sector_{idx}"] = float(value)

        obs = self._obs_buffer
        offset = 0
        obs[offset:offset + Car.NUM_LASERS] = distances / self.obs_encoder.laser_max_distance
        np.clip(obs[offset:offset + Car.NUM_LASERS], 0.0, 1.0, out=obs[offset:offset + Car.NUM_LASERS])
        offset += Car.NUM_LASERS

        obs[offset:offset + Car.SIGHT_TILES] = (
            self._path_instr_buffer / self.obs_encoder.path_instruction_abs_max
        )
        np.clip(obs[offset:offset + Car.SIGHT_TILES], -1.0, 1.0, out=obs[offset:offset + Car.SIGHT_TILES])
        offset += Car.SIGHT_TILES

        obs[offset] = min(1.0, max(-1.0, speed / self.obs_encoder.speed_abs_max))
        obs[offset + 1] = min(1.0, max(-1.0, side_speed / self.obs_encoder.side_speed_abs_max))
        obs[offset + 2] = min(1.0, max(-1.0, float(info["segment_heading_error"])))
        obs[offset + 3] = min(1.0, max(-1.0, float(info["next_segment_heading_error"])))
        obs[offset + 4] = dt_ratio
        offset += 5

        obs[offset] = min(1.0, max(0.0, float(info["slip_mean"])))
        offset += 1

        if self.obs_encoder.multi_surface_mode:
            obs[offset:offset + Car.SIGHT_TILES] = self._surface_instr_buffer
            np.clip(obs[offset:offset + Car.SIGHT_TILES], 0.0, 1.0, out=obs[offset:offset + Car.SIGHT_TILES])
            offset += Car.SIGHT_TILES

        if self.obs_encoder.vertical_mode:
            obs[offset:offset + Car.SIGHT_TILES] = self._height_instr_buffer
            np.clip(obs[offset:offset + Car.SIGHT_TILES], -1.0, 1.0, out=obs[offset:offset + Car.SIGHT_TILES])
            offset += Car.SIGHT_TILES

        obs[offset] = min(1.0, max(-1.0, longitudinal_accel / self.obs_encoder.accel_abs_max))
        obs[offset + 1] = min(1.0, max(-1.0, lateral_accel / self.obs_encoder.accel_abs_max))
        obs[offset + 2] = min(1.0, max(-1.0, self.last_yaw_rate / self.obs_encoder.yaw_rate_abs_max))
        obs[offset + 3:offset + 3 + len(clearance_rates)] = (
            clearance_rates / self.obs_encoder.clearance_rate_abs_max
        )
        np.clip(
            obs[offset + 3:offset + 3 + len(clearance_rates)],
            -1.0,
            1.0,
            out=obs[offset + 3:offset + 3 + len(clearance_rates)],
        )
        offset += 3 + len(clearance_rates)

        if self.obs_encoder.vertical_mode:
            vertical_speed = 0.0
            forward_y = 0.0
            support_normal_y = 1.0
            cross_slope = 0.0
            elevation_windows = np.zeros(
                len(ObservationEncoder.VERTICAL_FEATURE_NAMES) - 4,
                dtype=np.float32,
            )
            info["vertical_speed"] = vertical_speed
            info["forward_y"] = forward_y
            info["support_normal_y"] = support_normal_y
            info["cross_slope"] = cross_slope
            for idx, value in enumerate(elevation_windows):
                info[f"surface_elevation_sector_{idx}"] = float(value)

            obs[offset] = vertical_speed
            obs[offset + 1] = forward_y
            obs[offset + 2] = support_normal_y
            obs[offset + 3] = cross_slope
            obs[offset + 4:offset + 4 + len(elevation_windows)] = elevation_windows
        return obs.copy()

    def rollout_policy(
        self,
        policy,
        max_steps: int = 5000,
        collect_trajectory: bool = False,
    ) -> dict:
        obs, info = self.reset()
        total_reward = 0.0
        trajectory = []
        terminated = False
        truncated = False
        for step in range(int(max_steps)):
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = self.step(action)
            total_reward += float(reward)
            if collect_trajectory:
                trajectory.append(
                    {
                        "position": self.position.copy(),
                        "heading": float(self.heading),
                        "lasers": np.asarray(info.get("laser_endpoints_2d"), dtype=np.float32).copy(),
                        "progress": float(info.get("discrete_progress", 0.0)),
                        "dense_progress": float(info.get("dense_progress", 0.0)),
                    }
                )
            if terminated or truncated:
                break
        finished = int(info.get("finished", int(getattr(self, "finished", 0))))
        crashes = int(info.get("crashes", int(getattr(self, "crashes", 0))))
        if terminated and finished <= 0 and crashes <= 0:
            crashes = 1
        progress = float(info.get("discrete_progress", 0.0))
        dense_progress = float(info.get("dense_progress", progress))
        time_value = float(info.get("time", self.time))
        distance = float(info.get("distance", self.distance))
        fitness_progress = (
            dense_progress
            if Individual.RANKING_PROGRESS_SOURCE == "dense_progress"
            else progress
        )
        fitness = Individual.compute_scalar_fitness_for(
            finished,
            crashes,
            fitness_progress,
            time_value,
            distance,
        )
        result = {
            "finished": finished,
            "crashes": crashes,
            "timeout": int(bool(truncated) and finished <= 0 and crashes <= 0),
            "progress": progress,
            "dense_progress": dense_progress,
            "time": time_value,
            "distance": distance,
            "reward": total_reward,
            "fitness": float(fitness),
            "steps": step + 1,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        if collect_trajectory:
            result["trajectory"] = trajectory
        return result
