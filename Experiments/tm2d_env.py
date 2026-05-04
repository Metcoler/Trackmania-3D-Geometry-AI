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
    dt_ref: float = 1.0 / 100.0
    physics_tick_profile: str = "supervised_v2d"
    physics_tick_values: tuple[int, ...] = (1, 2, 3, 4)
    physics_tick_probabilities: tuple[float, ...] = (0.938285, 0.060381, 0.000562, 0.000772)
    max_speed: float = 130.0
    reverse_speed: float = 10.0
    gas_accel: float = 22.0
    brake_accel: float = 17.463
    drag: float = 0.1439
    rolling_drag: float = 5.549
    lateral_grip: float = 5.5
    max_yaw_rate: float = 1.5
    car_length: float = 4.8
    car_width: float = 2.6

    @staticmethod
    def normalized_tick_distribution(
        tick_values: tuple[int, ...],
        tick_probabilities: tuple[float, ...],
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        values = tuple(max(1, int(value)) for value in tick_values)
        probs = np.asarray(tick_probabilities, dtype=np.float64)
        if len(values) != int(probs.size) or not values:
            raise ValueError("physics tick values and probabilities must have the same non-zero length.")
        if not np.all(np.isfinite(probs)) or float(np.sum(probs)) <= 0.0:
            raise ValueError("physics tick probabilities must be positive finite values.")
        probs = np.maximum(probs, 0.0)
        probs = probs / float(np.sum(probs))
        return values, tuple(float(value) for value in probs)

    @staticmethod
    def profile_distribution(profile: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
        profile = str(profile).strip().lower()
        if profile == "fixed100":
            return (1,), (1.0,)
        if profile == "supervised_v2d":
            return (1, 2, 3, 4), (0.938285, 0.060381, 0.000562, 0.000772)
        if profile == "custom":
            return (1,), (1.0,)
        raise ValueError("physics_tick_profile must be fixed100, supervised_v2d, or custom.")

    def with_tick_profile(
        self,
        profile: str,
        tick_values: tuple[int, ...] | None = None,
        tick_probabilities: tuple[float, ...] | None = None,
    ) -> "TM2DPhysicsConfig":
        profile = str(profile).strip().lower()
        if profile == "custom":
            if tick_values is None or tick_probabilities is None:
                raise ValueError("custom physics tick profile requires explicit probabilities.")
            values, probs = self.normalized_tick_distribution(tick_values, tick_probabilities)
        else:
            values, probs = self.profile_distribution(profile)
        return TM2DPhysicsConfig(
            dt_ref=self.dt_ref,
            physics_tick_profile=profile,
            physics_tick_values=values,
            physics_tick_probabilities=probs,
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

    def sample_tick_count(self, rng: np.random.Generator) -> int:
        values, probs = self.normalized_tick_distribution(
            self.physics_tick_values,
            self.physics_tick_probabilities,
        )
        return int(rng.choice(np.asarray(values, dtype=np.int32), p=np.asarray(probs, dtype=np.float64)))


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
        max_touches: int = 1,
        collision_bounce_speed_retention: float = 0.40,
        collision_bounce_backoff: float = 0.05,
        touch_release_clearance_threshold: float = 0.50,
        mask_physics_delay_observation: bool = False,
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
        self.lidar_mode = "aabb_clearance"
        self.start_idle_max_time = float(start_idle_max_time)
        self.start_idle_progress_epsilon = float(start_idle_progress_epsilon)
        self.start_idle_speed_threshold = float(start_idle_speed_threshold)
        self.stuck_timeout_speed_threshold = float(stuck_timeout_speed_threshold)
        self.stuck_timeout_duration = float(stuck_timeout_duration)
        self.binary_gas_brake = bool(binary_gas_brake)
        self.max_touches = max(1, int(max_touches))
        self.collision_bounce_speed_retention = float(np.clip(collision_bounce_speed_retention, 0.0, 1.0))
        self.collision_bounce_backoff = max(0.0, float(collision_bounce_backoff))
        self.touch_release_clearance_threshold = max(0.0, float(touch_release_clearance_threshold))
        self.mask_physics_delay_observation = bool(mask_physics_delay_observation)
        self.rng = np.random.default_rng(seed)
        self.obs_encoder = ObservationEncoder(
            vertical_mode=bool(vertical_mode),
            multi_surface_mode=bool(multi_surface_mode),
        )
        self.vehicle_hitbox = self.obs_encoder.vehicle_hitbox
        self.laser_hitbox_offsets = self.obs_encoder.laser_hitbox_offsets
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
        self.touch_count = 0
        self._laser_touch_latched = False
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
        self.current_physics_tick_count = 1
        self.current_physics_hz = 1.0 / self.obs_encoder.dt_ref
        self.current_physics_hz_norm = 1.0
        self.current_physics_delay_norm = 0.0
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

        physics_ticks = int(self.physics.sample_tick_count(self.rng))
        dt = float(physics_ticks * self.physics.dt_ref)
        self.current_dt = dt
        self.current_physics_tick_count = physics_ticks
        self.current_physics_hz_norm = 1.0 / float(physics_ticks)
        self.current_physics_hz = self.current_physics_hz_norm / float(self.physics.dt_ref)
        self.current_physics_delay_norm = float(1.0 - self.current_physics_hz_norm)
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
        touch_reason = ""
        contact = self._laser_contact_state(raycast=raycast)
        if contact["min_clearance"] > self.touch_release_clearance_threshold:
            self._laser_touch_latched = False

        if self._crash_detected_from_contact(contact):
            if self.max_touches <= 1:
                self.touch_count = max(1, self.touch_count + 1)
                terminated = True
                touch_reason = "laser"
            else:
                if not self._laser_touch_latched:
                    self.touch_count += 1
                    self._laser_touch_latched = True
                    touch_reason = "laser"
                    if self.touch_count >= self.max_touches:
                        terminated = True
                    else:
                        self._apply_collision_bounce(contact)
                        raycast = None
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
            self.touch_count = max(1, self.touch_count)
            touch_reason = "start_idle"
        elif (
            progress > self.start_idle_progress_epsilon
            and self.speed <= self.stuck_timeout_speed_threshold
        ):
            if self._stuck_since_time is None:
                self._stuck_since_time = self.time
            elif (self.time - float(self._stuck_since_time)) >= self.stuck_timeout_duration:
                terminated = True
                self.touch_count = max(1, self.touch_count)
                touch_reason = "stuck_after_progress"
        else:
            self._stuck_since_time = None

        self.done = bool(terminated or truncated)
        self.finished = int(finished)
        crashes = int(self.touch_count)
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
                "touch_count": int(self.touch_count),
                "max_touches": int(self.max_touches),
                "touch_reason": touch_reason,
            }
        )
        return obs, reward, bool(terminated), bool(truncated), info

    def _uses_external_finish_bonus(self) -> bool:
        mode = str(self.reward_config.mode).strip().lower()
        return mode in {"progress_delta", "progress_primary_delta", "pace_delta", "progress_rate"}

    def _laser_contact_state(self, raycast=None) -> dict:
        if self.collision_mode in {"laser", "lidar"}:
            if raycast is None:
                raycast = self.geometry.cast_lasers(self.position, self.heading)
            distances = np.asarray(raycast.distances, dtype=np.float32)
            if distances.size <= 0:
                return {"min_clearance": float("inf"), "index": -1, "raycast": raycast}
            clearances = distances - self.laser_hitbox_offsets
            index = int(np.argmin(clearances))
            return {
                "min_clearance": float(clearances[index]),
                "index": index,
                "raycast": raycast,
            }
        return {"min_clearance": float("inf"), "index": -1, "raycast": raycast}

    def _crash_detected_from_contact(self, contact: dict) -> bool:
        if self.collision_mode in {"laser", "lidar"}:
            return bool(float(contact.get("min_clearance", float("inf"))) <= 0.0)
        return self._crash_detected()

    def _crash_detected(self, raycast=None) -> bool:
        if self.collision_mode in {"laser", "lidar"}:
            return self._crash_detected_from_contact(self._laser_contact_state(raycast=raycast))

        if self.collision_mode == "center":
            return not self.geometry.point_on_road(self.position)

        return not self.geometry.car_on_road(
            self.position,
            self.heading,
            length=self.physics.car_length,
            width=self.physics.car_width,
        )

    def _apply_collision_bounce(self, contact: dict) -> None:
        raycast = contact.get("raycast")
        index = int(contact.get("index", -1))
        if raycast is None or index < 0:
            self.velocity *= self.collision_bounce_speed_retention
            self.speed = float(np.linalg.norm(self.velocity))
            return

        directions = np.asarray(raycast.directions, dtype=np.float32)
        if index >= len(directions):
            self.velocity *= self.collision_bounce_speed_retention
            self.speed = float(np.linalg.norm(self.velocity))
            return

        ray_direction = _normalize_2d(directions[index])
        normal = -ray_direction
        velocity = np.asarray(self.velocity, dtype=np.float32)
        if float(np.dot(velocity, normal)) < 0.0:
            velocity = velocity - (2.0 * float(np.dot(velocity, normal)) * normal)
        velocity = velocity * self.collision_bounce_speed_retention

        penetration = max(0.0, -float(contact.get("min_clearance", 0.0)))
        self.position = self.position + normal * float(penetration + self.collision_bounce_backoff)
        self.velocity = velocity.astype(np.float32, copy=False)
        self.speed = float(np.linalg.norm(self.velocity))
        forward = np.array([math.cos(self.heading), math.sin(self.heading)], dtype=np.float32)
        right = np.array([-forward[1], forward[0]], dtype=np.float32)
        self.side_speed = float(np.dot(self.velocity, right))

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
            "min_raw_laser_distance": float(np.min(raycast.distances)) if len(raycast.distances) else float("inf"),
            "collision_distance_threshold": float(self.collision_distance_threshold),
            "lidar_mode": self.lidar_mode,
            "vehicle_hitbox": self.vehicle_hitbox.as_dict(),
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
        laser_clearances = distances - self.laser_hitbox_offsets
        min_clearance_index = int(np.argmin(laser_clearances)) if laser_clearances.size else -1
        info["raw_laser_distances"] = distances.astype(np.float32, copy=True)
        info["laser_hitbox_offsets"] = self.laser_hitbox_offsets.astype(np.float32, copy=True)
        info["laser_clearances"] = laser_clearances.astype(np.float32, copy=True)
        info["min_laser_clearance"] = (
            float(laser_clearances[min_clearance_index]) if min_clearance_index >= 0 else float("inf")
        )
        info["min_laser_distance"] = float(info["min_laser_clearance"])
        info["min_laser_clearance_index"] = int(min_clearance_index)
        info["hitbox_offset_at_min_clearance"] = (
            float(self.laser_hitbox_offsets[min_clearance_index])
            if min_clearance_index >= 0
            else float("nan")
        )
        self._path_instr_buffer[:] = self.geometry.path_instructions_at(self.path_index)
        if self.obs_encoder.multi_surface_mode:
            self._surface_instr_buffer[:] = info["next_surface_instructions"]
        if self.obs_encoder.vertical_mode:
            self._height_instr_buffer[:] = info["next_height_instructions"]
        speed = float(info["speed"])
        side_speed = float(info["side_speed"])
        dt = max(1e-6, float(self.current_dt))
        action_dt_ratio = float(
            np.clip(dt / self.obs_encoder.dt_ref, 0.0, self.obs_encoder.action_dt_ratio_clip)
        )
        physics_tick_count = max(1, int(round(dt / self.obs_encoder.dt_ref)))
        physics_hz_norm = 1.0 / float(physics_tick_count)
        physics_hz = physics_hz_norm / float(self.obs_encoder.dt_ref)
        physics_delay_norm = float(np.clip(1.0 - physics_hz_norm, 0.0, 1.0))
        if self.mask_physics_delay_observation:
            observation_physics_delay_norm = 0.0
        else:
            observation_physics_delay_norm = physics_delay_norm
        clearance_windows = np.asarray(
            [
                float(np.mean(laser_clearances[start:end]))
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

        info["game_dt"] = float(dt)
        info["action_dt_ratio"] = float(action_dt_ratio)
        info["physics_tick_count"] = int(physics_tick_count)
        info["physics_hz"] = float(physics_hz)
        info["physics_hz_norm"] = float(physics_hz_norm)
        info["physics_delay_norm"] = float(physics_delay_norm)
        info["observation_physics_delay_norm"] = float(observation_physics_delay_norm)
        info["longitudinal_accel"] = float(longitudinal_accel)
        info["lateral_accel"] = float(lateral_accel)
        info["yaw_rate"] = float(self.last_yaw_rate)
        for idx, value in enumerate(clearance_rates):
            info[f"clearance_rate_sector_{idx}"] = float(value)

        obs = self._obs_buffer
        offset = 0
        laser_denominators = np.maximum(
            1e-6,
            self.obs_encoder.laser_max_distance - self.laser_hitbox_offsets,
        )
        obs[offset:offset + Car.NUM_LASERS] = laser_clearances / laser_denominators
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
        obs[offset + 4] = observation_physics_delay_norm
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
        trajectory_log_actions: bool = False,
        reset_seed: int | None = None,
        mirrored: bool = False,
    ) -> dict:
        obs, info = self.reset(seed=reset_seed)
        total_reward = 0.0
        trajectory = []
        terminated = False
        truncated = False
        physics_tick_sum = 0.0
        physics_hz_sum = 0.0
        physics_delay_sum = 0.0
        timing_samples = 0
        for step in range(int(max_steps)):
            policy_obs = (
                ObservationEncoder.mirror_observation(
                    obs,
                    vertical_mode=self.obs_encoder.vertical_mode,
                    multi_surface_mode=self.obs_encoder.multi_surface_mode,
                )
                if mirrored
                else obs
            )
            action = policy.act(policy_obs)
            if mirrored:
                action = ObservationEncoder.mirror_action(action)
            obs, reward, terminated, truncated, info = self.step(action)
            total_reward += float(reward)
            physics_tick_sum += float(info.get("physics_tick_count", 1.0))
            physics_hz_sum += float(info.get("physics_hz", 1.0 / self.obs_encoder.dt_ref))
            physics_delay_sum += float(info.get("physics_delay_norm", 0.0))
            timing_samples += 1
            if collect_trajectory:
                record = {
                    "step": float(step + 1),
                    "time": float(info.get("time", self.time)),
                    "x": float(info.get("x", self.position[0])),
                    "y": float(info.get("y", 0.0)),
                    "z": float(info.get("z", self.position[1])),
                    "speed": float(info.get("speed", self.speed)),
                    "dense_progress": float(info.get("dense_progress", 0.0)),
                    "physics_hz": float(info.get("physics_hz", 1.0 / self.obs_encoder.dt_ref)),
                    "physics_delay_norm": float(info.get("physics_delay_norm", 0.0)),
                }
                if trajectory_log_actions:
                    record["action"] = np.asarray(action, dtype=np.float32).copy()
                trajectory.append(record)
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
            "physics_tick_mean": float(physics_tick_sum / max(1, timing_samples)),
            "physics_hz_mean": float(physics_hz_sum / max(1, timing_samples)),
            "physics_delay_norm_mean": float(physics_delay_sum / max(1, timing_samples)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "mirrored": int(bool(mirrored)),
        }
        if collect_trajectory:
            result["trajectory"] = trajectory
        return result
