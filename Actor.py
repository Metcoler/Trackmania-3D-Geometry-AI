import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np
try:
    import vgamepad
except ImportError:  # pragma: no cover - only needed for live perturbation collection.
    vgamepad = None

from Car import Car
from Map import Map
from ObservationEncoder import ObservationEncoder
from XboxController import XboxControllerReader, XboxControllerState


@dataclass
class AttemptSample:
    observation: np.ndarray
    action: np.ndarray
    executed_action: np.ndarray
    game_time: float
    discrete_progress: float
    dense_progress: float
    distance: float
    speed: float
    side_speed: float
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    slip_mean: float
    dt_ratio: float
    raw_laser_distances: np.ndarray
    laser_hitbox_offsets: np.ndarray
    laser_clearances: np.ndarray
    finished: int
    crashes: int
    timeout: int


@dataclass
class ActionPerturbationConfig:
    enabled: bool = False
    random_seed: int = 20260502
    steer_sigma: float = 0.18
    steer_bias_decay: float = 0.96
    gas_flip_rate_per_second: float = 0.20
    brake_flip_rate_per_second: float = 0.10
    min_flip_hold_seconds: float = 0.08
    max_flip_hold_seconds: float = 0.20

    def as_dict(self) -> Dict[str, float | int | bool]:
        return {
            "enabled": bool(self.enabled),
            "random_seed": int(self.random_seed),
            "steer_sigma": float(self.steer_sigma),
            "steer_bias_decay": float(self.steer_bias_decay),
            "gas_flip_rate_per_second": float(self.gas_flip_rate_per_second),
            "brake_flip_rate_per_second": float(self.brake_flip_rate_per_second),
            "min_flip_hold_seconds": float(self.min_flip_hold_seconds),
            "max_flip_hold_seconds": float(self.max_flip_hold_seconds),
        }


class ActionPerturber:
    """Inject persistent recoverable mistakes into human driving actions."""

    def __init__(self, config: ActionPerturbationConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(int(config.random_seed))
        self.steer_bias = 0.0
        self.gas_flip_until = 0.0
        self.brake_flip_until = 0.0
        self.last_time = time.perf_counter()

    @staticmethod
    def _event_probability(rate_per_second: float, dt: float) -> float:
        rate = max(0.0, float(rate_per_second))
        return float(1.0 - np.exp(-rate * max(0.0, float(dt))))

    def _maybe_start_flip(self, now: float, current_until: float, rate: float, dt: float) -> float:
        if now < current_until:
            return current_until
        if self.rng.random() >= self._event_probability(rate, dt):
            return current_until
        hold = float(
            self.rng.uniform(
                max(0.0, self.config.min_flip_hold_seconds),
                max(self.config.min_flip_hold_seconds, self.config.max_flip_hold_seconds),
            )
        )
        return now + hold

    def apply(self, action: np.ndarray) -> np.ndarray:
        clean = np.asarray(action, dtype=np.float32).reshape(3)
        if not self.config.enabled:
            return clean.copy()

        now = time.perf_counter()
        dt = max(1e-3, now - self.last_time)
        self.last_time = now

        decay = float(np.clip(self.config.steer_bias_decay, 0.0, 0.9999))
        innovation_scale = np.sqrt(max(0.0, 1.0 - decay * decay))
        self.steer_bias = (
            decay * self.steer_bias
            + innovation_scale * float(self.rng.normal(0.0, self.config.steer_sigma))
        )

        self.gas_flip_until = self._maybe_start_flip(
            now=now,
            current_until=self.gas_flip_until,
            rate=self.config.gas_flip_rate_per_second,
            dt=dt,
        )
        self.brake_flip_until = self._maybe_start_flip(
            now=now,
            current_until=self.brake_flip_until,
            rate=self.config.brake_flip_rate_per_second,
            dt=dt,
        )

        perturbed = clean.copy()
        perturbed[2] = float(np.clip(perturbed[2] + self.steer_bias, -1.0, 1.0))
        if now < self.gas_flip_until:
            perturbed[0] = 1.0 - float(perturbed[0])
        if now < self.brake_flip_until:
            perturbed[1] = 1.0 - float(perturbed[1])
        return perturbed.astype(np.float32, copy=False)


class VirtualGamepadActionOutput:
    def __init__(self) -> None:
        if vgamepad is None:
            raise RuntimeError("vgamepad is not installed; cannot apply perturbed actions.")
        self.gamepad = vgamepad.VX360Gamepad()
        self.gamepad.reset()
        self.gamepad.update()

    def apply(self, action: np.ndarray, state: XboxControllerState) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(3)
        self.gamepad.reset()
        self.gamepad.right_trigger_float(float(np.clip(action[0], 0.0, 1.0)))
        self.gamepad.left_trigger_float(float(np.clip(action[1], 0.0, 1.0)))
        self.gamepad.left_joystick_float(float(np.clip(action[2], -1.0, 1.0)), 0.0)
        if int(state.button_a):
            self.gamepad.press_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_A)
        if int(state.button_b):
            self.gamepad.press_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self.gamepad.update()

    def close(self) -> None:
        self.gamepad.reset()
        self.gamepad.update()


class AttemptWriter:
    SUMMARY_HEADERS = [
        "attempt_index",
        "saved",
        "num_frames",
        "finish_time",
        "discrete_progress",
        "dense_progress",
        "finished",
        "crashes",
        "timeout",
        "distance",
        "path",
    ]

    def __init__(
        self,
        base_dir: str,
        map_name: str,
        encoder: ObservationEncoder,
        perturbation_config: ActionPerturbationConfig,
        apply_to_virtual_gamepad: bool,
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lidar_tag = (
            f"{'v3d' if encoder.vertical_mode else 'v2d'}_"
            f"{'surface' if encoder.multi_surface_mode else 'asphalt'}"
        )
        run_name = f"{timestamp}_map_{map_name}_{lidar_tag}_target_dataset"
        self.run_dir = os.path.join(base_dir, run_name)
        self.attempts_dir = os.path.join(self.run_dir, "attempts")
        os.makedirs(self.attempts_dir, exist_ok=False)
        self.summary_path = os.path.join(self.run_dir, "attempts.csv")
        with open(self.summary_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.SUMMARY_HEADERS)
            writer.writeheader()

        config = {
            "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "map_name": map_name,
            "observation_dim": encoder.obs_dim,
            "observation_layout": ObservationEncoder.feature_names(
                vertical_mode=encoder.vertical_mode,
                multi_surface_mode=encoder.multi_surface_mode,
            ),
            "vertical_mode": bool(encoder.vertical_mode),
            "multi_surface_mode": bool(encoder.multi_surface_mode),
            "dt_ref": encoder.dt_ref,
            "dt_ratio_clip": encoder.dt_ratio_clip,
            "lidar_mode": "aabb_clearance",
            "vehicle_hitbox": encoder.vehicle_hitbox.as_dict(),
            "action_mode": "target",
            "action_layout": ["gas", "brake", "steer"],
            "recorded_action": "human_correction_action",
            "executed_action": "executed_action_after_perturbation",
            "action_perturbation": perturbation_config.as_dict(),
            "action_perturbation_applied_to_virtual_gamepad": bool(apply_to_virtual_gamepad),
        }
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=True)

    def save_attempt(
        self,
        attempt_index: int,
        samples: List[AttemptSample],
        finish_info: Dict[str, float],
    ) -> str:
        if not samples:
            raise ValueError("Cannot save an empty attempt.")

        observations = np.stack([sample.observation for sample in samples]).astype(np.float32)
        actions = np.stack([sample.action for sample in samples]).astype(np.float32)
        executed_actions = np.stack([sample.executed_action for sample in samples]).astype(np.float32)
        game_times = np.array([sample.game_time for sample in samples], dtype=np.float32)
        discrete_progress = np.array([sample.discrete_progress for sample in samples], dtype=np.float32)
        dense_progress = np.array([sample.dense_progress for sample in samples], dtype=np.float32)
        distances = np.array([sample.distance for sample in samples], dtype=np.float32)
        speeds = np.array([sample.speed for sample in samples], dtype=np.float32)
        side_speeds = np.array([sample.side_speed for sample in samples], dtype=np.float32)
        positions = np.array([[sample.x, sample.y, sample.z] for sample in samples], dtype=np.float32)
        directions = np.array([[sample.dx, sample.dy, sample.dz] for sample in samples], dtype=np.float32)
        slip_mean = np.array([sample.slip_mean for sample in samples], dtype=np.float32)
        dt_ratio = np.array([sample.dt_ratio for sample in samples], dtype=np.float32)
        raw_laser_distances = np.stack(
            [sample.raw_laser_distances for sample in samples]
        ).astype(np.float32)
        laser_hitbox_offsets = np.stack(
            [sample.laser_hitbox_offsets for sample in samples]
        ).astype(np.float32)
        laser_clearances = np.stack(
            [sample.laser_clearances for sample in samples]
        ).astype(np.float32)
        finished = np.array([sample.finished for sample in samples], dtype=np.int32)
        crashes = np.array([sample.crashes for sample in samples], dtype=np.int32)
        timeout = np.array([sample.timeout for sample in samples], dtype=np.int32)

        output_path = os.path.join(self.attempts_dir, f"attempt_{attempt_index:04d}.npz")
        np.savez(
            output_path,
            observations=observations,
            # `actions` is intentionally the human correction target for behavior cloning.
            # `executed_actions` is the actual perturbed action applied to Trackmania.
            actions=actions,
            human_actions=actions,
            executed_actions=executed_actions,
            game_times=game_times,
            discrete_progress=discrete_progress,
            dense_progress=dense_progress,
            distances=distances,
            speeds=speeds,
            side_speeds=side_speeds,
            positions=positions,
            directions=directions,
            slip_mean=slip_mean,
            dt_ratio=dt_ratio,
            raw_laser_distances=raw_laser_distances,
            laser_hitbox_offsets=laser_hitbox_offsets,
            laser_clearances=laser_clearances,
            finished=finished,
            crashes=crashes,
            timeout=timeout,
            finish_time=np.array([float(finish_info.get("time", 0.0))], dtype=np.float32),
            finish_progress=np.array([float(finish_info.get("discrete_progress", 0.0))], dtype=np.float32),
            finish_dense_progress=np.array([float(finish_info.get("dense_progress", finish_info.get("discrete_progress", 0.0)))], dtype=np.float32),
            finish_distance=np.array([float(finish_info.get("distance", 0.0))], dtype=np.float32),
            finish_finished=np.array([int(finish_info.get("finished", 0))], dtype=np.int32),
            finish_crashes=np.array([int(finish_info.get("crashes", 0))], dtype=np.int32),
            finish_timeout=np.array([int(finish_info.get("timeout", 0))], dtype=np.int32),
        )
        self._append_summary(
            dict(
                attempt_index=attempt_index,
                saved=1,
                num_frames=len(samples),
                finish_time=float(finish_info.get("time", 0.0)),
                discrete_progress=float(finish_info.get("discrete_progress", 0.0)),
                dense_progress=float(finish_info.get("dense_progress", finish_info.get("discrete_progress", 0.0))),
                finished=int(finish_info.get("finished", 0)),
                crashes=int(finish_info.get("crashes", 0)),
                timeout=int(finish_info.get("timeout", 0)),
                distance=float(finish_info.get("distance", 0.0)),
                path=output_path,
            )
        )
        return output_path

    def log_discard(self, attempt_index: int, samples: List[AttemptSample], finish_info: Dict[str, float]) -> None:
        self._append_summary(
            dict(
                attempt_index=attempt_index,
                saved=0,
                num_frames=len(samples),
                finish_time=float(finish_info.get("time", 0.0)),
                discrete_progress=float(finish_info.get("discrete_progress", 0.0)),
                dense_progress=float(finish_info.get("dense_progress", finish_info.get("discrete_progress", 0.0))),
                finished=int(finish_info.get("finished", 0)),
                crashes=int(finish_info.get("crashes", 0)),
                timeout=int(finish_info.get("timeout", 0)),
                distance=float(finish_info.get("distance", 0.0)),
                path="",
            )
        )

    def _append_summary(self, row: Dict) -> None:
        with open(self.summary_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.SUMMARY_HEADERS)
            writer.writerow(row)


def controller_to_action(state: XboxControllerState) -> np.ndarray:
    gas = 1.0 if float(state.gas) > 0.5 else 0.0
    brake = 1.0 if float(state.brake) > 0.5 else 0.0
    return np.array([gas, brake, state.steer], dtype=np.float32)


def rising_edge(current: int, previous: int) -> bool:
    return bool(current) and not bool(previous)


if __name__ == "__main__":
    map_name = "AI Training #5"
    #map_name = "small_map"
    #map_name = "AI Training #4"
    vertical_mode = False
    multi_surface_mode = False
    base_dir = "logs/supervised_data"
    perturbation_config = ActionPerturbationConfig(
        enabled=False,
        random_seed=20260502,
        steer_sigma=0.18,
        steer_bias_decay=0.96,
        gas_flip_rate_per_second=0.20,
        brake_flip_rate_per_second=0.10,
        min_flip_hold_seconds=0.08,
        max_flip_hold_seconds=0.20,
    )
    apply_perturbed_action_to_virtual_gamepad = True
    encoder = ObservationEncoder(
        dt_ref=1.0 / 100.0,
        dt_ratio_clip=3.0,
        vertical_mode=vertical_mode,
        multi_surface_mode=multi_surface_mode,
    )
    writer = AttemptWriter(
        base_dir=base_dir,
        map_name=map_name,
        encoder=encoder,
        perturbation_config=perturbation_config,
        apply_to_virtual_gamepad=apply_perturbed_action_to_virtual_gamepad,
    )

    print(f"Saving supervised attempts into: {writer.run_dir}")
    print("Workflow:")
    print("- recording starts when game time becomes > 0")
    print("- press B during an attempt to discard it")
    print("- after finish, press A to save or B to discard")
    print("- stop script with Ctrl+C")
    if perturbation_config.enabled:
        print("Action perturbation is ENABLED.")
        print(
            "Important: for physically consistent data, Trackmania should listen "
            "to the virtual gamepad output, not directly to the physical controller."
        )
        if not apply_perturbed_action_to_virtual_gamepad:
            raise RuntimeError(
                "Refusing to record perturbed labels without applying them to the game. "
                "Set apply_perturbed_action_to_virtual_gamepad=True or disable perturbation."
            )

    game_map = Map(map_name)
    car = Car(game_map, vertical_mode=vertical_mode)
    controller = XboxControllerReader()
    perturber = ActionPerturber(perturbation_config)
    action_output = (
        VirtualGamepadActionOutput()
        if perturbation_config.enabled and apply_perturbed_action_to_virtual_gamepad
        else None
    )

    attempt_index = 1
    state = "waiting_for_start"
    attempt_samples: List[AttemptSample] = []
    last_buttons = {"a": 0, "b": 0}
    finish_info: Dict[str, float] = {}

    try:
        while True:
            distances, instructions, info = car.get_data()
            snapshot = controller.snapshot()
            human_action = controller_to_action(snapshot)
            action = perturber.apply(human_action)
            if action_output is not None:
                action_output.apply(action, snapshot)
            a_pressed = rising_edge(snapshot.button_a, last_buttons["a"])
            b_pressed = rising_edge(snapshot.button_b, last_buttons["b"])
            last_buttons["a"] = snapshot.button_a
            last_buttons["b"] = snapshot.button_b

            game_time = float(info.get("time", 0.0))
            discrete_progress = float(info.get("discrete_progress", 0.0))
            dense_progress = float(info.get("dense_progress", discrete_progress))
            total_distance = float(info.get("distance", 0.0))
            finished = bool(info.get("done", 0.0) == 1.0)

            if state == "waiting_for_start":
                if game_time > 0.0:
                    encoder.reset()
                    attempt_samples = []
                    state = "recording"
                    print(f"Attempt {attempt_index:04d} started.")

            if state == "recording":
                observation = encoder.build_observation(
                    distances=distances,
                    instructions=instructions,
                    info=info,
                )
                raw_laser_distances = np.asarray(
                    info.get("raw_laser_distances", distances),
                    dtype=np.float32,
                ).reshape(-1)
                laser_hitbox_offsets = np.asarray(
                    info.get("laser_hitbox_offsets", encoder.laser_hitbox_offsets),
                    dtype=np.float32,
                ).reshape(-1)
                laser_clearances = np.asarray(
                    info.get("laser_clearances", raw_laser_distances - laser_hitbox_offsets),
                    dtype=np.float32,
                ).reshape(-1)
                finished_int = int(finished)
                crashes = int(info.get("crashes", 0))
                timeout = int(info.get("timeout", 0))
                attempt_samples.append(
                    AttemptSample(
                        observation=observation,
                        action=human_action.copy(),
                        executed_action=action.copy(),
                        game_time=game_time,
                        discrete_progress=discrete_progress,
                        dense_progress=float(info.get("dense_progress", dense_progress)),
                        distance=total_distance,
                        speed=float(info.get("speed", 0.0)),
                        side_speed=float(info.get("side_speed", 0.0)),
                        x=float(info.get("x", 0.0)),
                        y=float(info.get("y", 0.0)),
                        z=float(info.get("z", 0.0)),
                        dx=float(info.get("dx", 0.0)),
                        dy=float(info.get("dy", 0.0)),
                        dz=float(info.get("dz", 0.0)),
                        slip_mean=float(info.get("slip_mean", 0.0)),
                        dt_ratio=float(info.get("dt_ratio", 1.0)),
                        raw_laser_distances=raw_laser_distances.copy(),
                        laser_hitbox_offsets=laser_hitbox_offsets.copy(),
                        laser_clearances=laser_clearances.copy(),
                        finished=finished_int,
                        crashes=crashes,
                        timeout=timeout,
                    )
                )

                if b_pressed:
                    print(f"Attempt {attempt_index:04d} discarded by B restart.")
                    writer.log_discard(attempt_index, attempt_samples, dict(
                        time=game_time,
                        discrete_progress=discrete_progress,
                        dense_progress=float(info.get("dense_progress", dense_progress)),
                        distance=total_distance,
                        finished=0,
                        crashes=crashes,
                        timeout=timeout,
                    ))
                    attempt_index += 1
                    attempt_samples = []
                    state = "waiting_for_reset"
                elif finished:
                    finish_info = dict(
                        time=game_time,
                        discrete_progress=discrete_progress,
                        dense_progress=float(info.get("dense_progress", dense_progress)),
                        distance=total_distance,
                        finished=1,
                        crashes=0,
                        timeout=0,
                    )
                    print(
                        f"Attempt {attempt_index:04d} finished in {game_time:.2f}s. "
                        "Press A to save or B to discard."
                    )
                    state = "await_finish_confirmation"
                elif game_time <= 0.0:
                    print(f"Attempt {attempt_index:04d} reset before finish. Discarded.")
                    writer.log_discard(attempt_index, attempt_samples, dict(
                        time=game_time,
                        discrete_progress=discrete_progress,
                        dense_progress=float(info.get("dense_progress", dense_progress)),
                        distance=total_distance,
                        finished=0,
                        crashes=0,
                        timeout=0,
                    ))
                    attempt_index += 1
                    attempt_samples = []
                    state = "waiting_for_start"

            elif state == "await_finish_confirmation":
                if a_pressed:
                    output_path = writer.save_attempt(attempt_index, attempt_samples, finish_info)
                    print(f"Attempt {attempt_index:04d} saved: {output_path}")
                    attempt_index += 1
                    attempt_samples = []
                    state = "waiting_for_reset"
                elif b_pressed:
                    print(f"Attempt {attempt_index:04d} discarded after finish.")
                    writer.log_discard(attempt_index, attempt_samples, finish_info)
                    attempt_index += 1
                    attempt_samples = []
                    state = "waiting_for_reset"

            elif state == "waiting_for_reset":
                if game_time <= 0.0:
                    state = "waiting_for_start"

    except KeyboardInterrupt:
        print("\nStopped data collection.")
    finally:
        if action_output is not None:
            action_output.close()
        controller.close()
