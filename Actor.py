import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np

from Car import Car
from Map import Map
from ObservationEncoder import ObservationEncoder
from XboxController import XboxControllerReader, XboxControllerState


@dataclass
class AttemptSample:
    observation: np.ndarray
    action: np.ndarray
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
    finished: int
    crashes: int
    timeout: int


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

    def __init__(self, base_dir: str, map_name: str, encoder: ObservationEncoder) -> None:
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
            "action_mode": "target",
            "action_layout": ["gas", "brake", "steer"],
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
        finished = np.array([sample.finished for sample in samples], dtype=np.int32)
        crashes = np.array([sample.crashes for sample in samples], dtype=np.int32)
        timeout = np.array([sample.timeout for sample in samples], dtype=np.int32)

        output_path = os.path.join(self.attempts_dir, f"attempt_{attempt_index:04d}.npz")
        np.savez(
            output_path,
            observations=observations,
            actions=actions,
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
    vertical_mode = True
    multi_surface_mode = True
    base_dir = "logs/supervised_data"
    encoder = ObservationEncoder(
        dt_ref=1.0 / 100.0,
        dt_ratio_clip=3.0,
        vertical_mode=vertical_mode,
        multi_surface_mode=multi_surface_mode,
    )
    writer = AttemptWriter(base_dir=base_dir, map_name=map_name, encoder=encoder)

    print(f"Saving supervised attempts into: {writer.run_dir}")
    print("Workflow:")
    print("- recording starts when game time becomes > 0")
    print("- press B during an attempt to discard it")
    print("- after finish, press A to save or B to discard")
    print("- stop script with Ctrl+C")

    game_map = Map(map_name)
    car = Car(game_map, vertical_mode=vertical_mode)
    controller = XboxControllerReader()

    attempt_index = 1
    state = "waiting_for_start"
    attempt_samples: List[AttemptSample] = []
    last_buttons = {"a": 0, "b": 0}
    finish_info: Dict[str, float] = {}

    try:
        while True:
            distances, instructions, info = car.get_data()
            snapshot = controller.snapshot()
            action = controller_to_action(snapshot)
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
                finished_int = int(finished)
                crashes = int(info.get("crashes", 0))
                timeout = int(info.get("timeout", 0))
                attempt_samples.append(
                    AttemptSample(
                        observation=observation,
                        action=action.copy(),
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
        controller.close()
