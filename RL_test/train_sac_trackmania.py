import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch.nn as nn
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from Car import Car
from Enviroment import RacingGameEnviroment
from Individual import Individual
from Map import MAP_BLOCK_SIZE


MAP_NAME = "AI Training #5"
EPISODES_TO_RUN = 10_000
TOTAL_TIMESTEPS = 50_000_000
MAX_RUNTIME_HOURS = 8.0
ENV_MAX_TIME = 45.0
CHECKPOINT_EVERY_EPISODES = 50

# SAC debug default after the 2026-04-30 local sandbox sweep:
#   - dense progress signal so SAC is not trained only from sparse terminal reward
#   - timeout/crash penalty equal to one map-derived progress bucket
#   - gas_steer first, because gas_brake_steer made early exploration much worse
REWARD_MODE = "delta_progress_time_block_penalty"
TERMINAL_FITNESS_SCALE = 1_000_000.0
PACE_TARGET_TIME = ENV_MAX_TIME / 2.0
INITIAL_MODEL_PATH = None

ACTION_LAYOUT = "gas_steer"  # gas_steer / gas_brake_steer / throttle_steer / target_3d
VERTICAL_MODE = True
MAX_TOUCHES = 1
START_IDLE_MAX_TIME = 2.0
LOG_LIVE_RESETS = True

SAC_LEARNING_RATE = 3e-4
SAC_BUFFER_SIZE = 300_000
SAC_LEARNING_STARTS = 1_000
SAC_BATCH_SIZE = 256
SAC_GAMMA = 0.9995
SAC_TAU = 0.005
SAC_GRADIENT_STEPS = 8
SAC_TRAIN_FREQ = (1, "episode")
SAC_ENT_COEF = "0.01"
SAC_POLICY_NET_ARCH = [32, 32]
SAC_ACTIVATION_FN = "relu"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_name(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return "".join(ch if ch in allowed else "_" for ch in value)


def resolve_pace_target_time(env_max_time: float, pace_target_time: Optional[float] = None) -> float:
    if pace_target_time is not None:
        return float(pace_target_time)
    return float(env_max_time) / 2.0


class ContinuousTargetRacingEnv(RacingGameEnviroment):
    """Racing env variant that keeps analog gas/brake for SB3 continuous actions."""

    def perform_action(self, action_input):
        if self.race_terminated != 0:
            self.controller.reset()
            self.controller.update()
            return

        action_input = np.asarray(action_input, dtype=np.float32)
        if action_input.shape != (3,) or not np.all(np.isfinite(action_input)):
            return

        if self.action_mode != "target":
            return super().perform_action(action_input)

        action = np.array(
            [
                float(np.clip(action_input[0], 0.0, 1.0)),
                float(np.clip(action_input[1], 0.0, 1.0)),
                float(np.clip(action_input[2], -1.0, 1.0)),
            ],
            dtype=np.float32,
        )
        self.previous_action = action
        self.controller.reset()
        self.controller.right_trigger_float(action[0])
        self.controller.left_trigger_float(action[1])
        self.controller.left_joystick_float(action[2], 0.0)
        self.controller.update()


class TrackmaniaSB3Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        map_name: str,
        reward_mode: str,
        action_layout: str,
        env_max_time: float,
        vertical_mode: bool = True,
        max_touches: int = 1,
        start_idle_max_time: float = 2.0,
        terminal_fitness_scale: float = TERMINAL_FITNESS_SCALE,
        pace_target_time: float = PACE_TARGET_TIME,
    ) -> None:
        super().__init__()
        self.reward_mode = str(reward_mode).strip().lower()
        if self.reward_mode not in {
            "terminal_progress",
            "terminal_fitness",
            "fitness_delta",
            "delta_lexicographic",
            "terminal_lexicographic",
            "delta_progress_time_safety",
            "terminal_progress_time_safety",
            "delta_progress_time_block_penalty",
            "terminal_progress_time_block_penalty",
        }:
            raise ValueError(
                "reward_mode must be terminal_progress, terminal_fitness, fitness_delta, "
                "delta_lexicographic, terminal_lexicographic, "
                "delta_progress_time_safety, terminal_progress_time_safety, "
                "delta_progress_time_block_penalty, or terminal_progress_time_block_penalty."
            )
        self.action_layout = str(action_layout).strip().lower()
        if self.action_layout not in {"gas_steer", "gas_brake_steer", "throttle_steer", "target_3d"}:
            raise ValueError(
                "action_layout must be gas_steer, gas_brake_steer, throttle_steer, or target_3d."
            )

        self.env = ContinuousTargetRacingEnv(
            map_name=map_name,
            never_quit=False,
            action_mode="target",
            dt_ref=1.0 / 100.0,
            dt_ratio_clip=3.0,
            vertical_mode=vertical_mode,
            surface_step_size=Car.SURFACE_STEP_SIZE,
            surface_probe_height=Car.SURFACE_PROBE_HEIGHT,
            surface_ray_lift=Car.SURFACE_RAY_LIFT,
            max_time=env_max_time,
            max_touches=max_touches,
            start_idle_max_time=start_idle_max_time,
        )
        self.observation_space = self.env.observation_space
        if self.action_layout in {"gas_steer", "throttle_steer"}:
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        self.terminal_fitness_scale = float(terminal_fitness_scale)
        self.pace_target_time = float(pace_target_time)
        if not np.isfinite(self.pace_target_time) or self.pace_target_time <= 0.0:
            raise ValueError("pace_target_time must be a positive finite number.")
        self.path_tile_count = len(getattr(self.env.map, "path_tiles", []) or [])
        self.progress_bucket = 100.0 / max(1, self.path_tile_count - 1)
        self.estimated_path_length = max(1.0, max(1, self.path_tile_count - 1) * MAP_BLOCK_SIZE)
        self.last_delta_score = 0.0
        self.last_info: Dict[str, Any] = {}
        self.episode_index = 0

    def _to_env_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.action_layout == "gas_steer":
            gas = 0.5 * (float(np.clip(action[0], -1.0, 1.0)) + 1.0)
            steer = float(np.clip(action[1], -1.0, 1.0))
            return np.array([gas, 0.0, steer], dtype=np.float32)
        if self.action_layout == "gas_brake_steer":
            return np.array(
                [
                    float(np.clip(action[0], 0.0, 1.0)),
                    float(np.clip(action[1], 0.0, 1.0)),
                    float(np.clip(action[2], -1.0, 1.0)),
                ],
                dtype=np.float32,
            )
        if self.action_layout == "throttle_steer":
            throttle = float(np.clip(action[0], -1.0, 1.0))
            steer = float(np.clip(action[1], -1.0, 1.0))
            gas = max(throttle, 0.0)
            brake = max(-throttle, 0.0)
            return np.array([gas, brake, steer], dtype=np.float32)
        return np.array(
            [
                float(np.clip(action[0], 0.0, 1.0)),
                float(np.clip(action[1], 0.0, 1.0)),
                float(np.clip(action[2], -1.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def _terminal_metrics(self, info: Dict[str, Any]) -> Dict[str, float]:
        progress = float(info.get("total_progress", 0.0))
        time_value = float(info.get("time", 0.0))
        if time_value <= 0.0 or not np.isfinite(time_value):
            time_value = 0.0
        distance = float(info.get("distance", 0.0))
        term = int(getattr(self.env, "race_terminated", 0))
        if info.get("done", 0.0) == 1.0 and term == 0:
            term = 1
        fitness = Individual.compute_scalar_fitness_for(
            term=term,
            progress=progress,
            time_value=time_value,
            distance=distance,
        )
        return dict(
            term=float(term),
            progress=progress,
            time=time_value,
            distance=distance,
            fitness=float(fitness),
        )

    def _lexicographic_score(self, metrics: Dict[str, float]) -> float:
        return Individual.compute_delta_lexicographic_score_for(
            term=int(metrics["term"]),
            progress=float(metrics["progress"]),
            time_value=float(metrics["time"]),
            distance=float(metrics["distance"]),
            max_time=float(self.env.max_time),
            path_tile_count=int(self.path_tile_count),
            progress_bucket=float(self.progress_bucket),
            estimated_path_length=float(self.estimated_path_length),
        )

    def _terminal_lexicographic_score(self, metrics: Dict[str, float]) -> float:
        return Individual.compute_terminal_lexicographic_score_for(
            term=int(metrics["term"]),
            progress=float(metrics["progress"]),
            time_value=float(metrics["time"]),
            distance=float(metrics["distance"]),
            max_time=float(self.env.max_time),
            path_tile_count=int(self.path_tile_count),
            progress_bucket=float(self.progress_bucket),
            estimated_path_length=float(self.estimated_path_length),
        )

    def _progress_time_safety_score(
        self,
        metrics: Dict[str, float],
        terminal_only: bool = False,
    ) -> float:
        return Individual.compute_progress_time_safety_score_for(
            term=int(metrics["term"]),
            progress=float(metrics["progress"]),
            time_value=float(metrics["time"]),
            distance=float(metrics["distance"]),
            max_time=float(self.env.max_time),
            path_tile_count=int(self.path_tile_count),
            progress_bucket=float(self.progress_bucket),
            terminal_only=terminal_only,
        )

    def _progress_time_block_penalty_score(
        self,
        metrics: Dict[str, float],
        terminal_only: bool = False,
    ) -> float:
        return Individual.compute_progress_time_block_penalty_score_for(
            term=int(metrics["term"]),
            progress=float(metrics["progress"]),
            time_value=float(metrics["time"]),
            distance=float(metrics["distance"]),
            max_time=float(self.env.max_time),
            path_tile_count=int(self.path_tile_count),
            progress_bucket=float(self.progress_bucket),
            terminal_only=terminal_only,
        )

    def _delta_reward_score(
        self,
        metrics: Dict[str, float],
        terminated: bool,
        truncated: bool,
    ) -> float:
        progress = max(0.0, float(metrics["progress"]))
        time_value = max(0.0, float(metrics["time"]))
        expected_progress_by_time = 100.0 * time_value / self.pace_target_time
        pace_score = progress - expected_progress_by_time
        score = progress + pace_score

        # Keep term out of the dense shaping. A crash should end the episode and
        # remove future progress opportunities, not erase credit for good driving
        # earlier in the same run. Finishing still gets the lexicographic finish
        # bonus from Individual's scoring scale.
        if terminated or truncated:
            term = int(metrics["term"])
            if term > 0:
                score += Individual.TERM_WEIGHT / self.terminal_fitness_scale
                score -= (
                    max(0.0, float(metrics["distance"]))
                    * Individual.DISTANCE_WEIGHT
                    / self.terminal_fitness_scale
                )

        metrics["pace_target_time"] = float(self.pace_target_time)
        metrics["pace_expected_progress"] = float(expected_progress_by_time)
        metrics["pace_score"] = float(pace_score)
        return float(score)

    def _reward(
        self,
        info: Dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> tuple[float, Dict[str, float]]:
        metrics = self._terminal_metrics(info)

        reward = 0.0
        if self.reward_mode == "delta_lexicographic":
            delta_score = self._lexicographic_score(metrics)
            reward = delta_score - self.last_delta_score
            self.last_delta_score = delta_score
            metrics["lexicographic_score"] = float(delta_score)
        elif self.reward_mode == "delta_progress_time_safety":
            delta_score = self._progress_time_safety_score(metrics, terminal_only=False)
            reward = delta_score - self.last_delta_score
            self.last_delta_score = delta_score
            metrics["progress_time_safety_score"] = float(delta_score)
        elif self.reward_mode == "delta_progress_time_block_penalty":
            delta_score = self._progress_time_block_penalty_score(metrics, terminal_only=False)
            reward = delta_score - self.last_delta_score
            self.last_delta_score = delta_score
            metrics["progress_time_block_penalty_score"] = float(delta_score)
        elif self.reward_mode == "fitness_delta":
            delta_score = self._delta_reward_score(
                metrics,
                terminated=terminated,
                truncated=truncated,
            )
            reward = delta_score - self.last_delta_score
            self.last_delta_score = delta_score
        elif terminated or truncated:
            if self.reward_mode == "terminal_progress":
                reward = float(metrics["progress"])
            elif self.reward_mode == "terminal_fitness":
                reward = float(metrics["fitness"]) / self.terminal_fitness_scale
            elif self.reward_mode == "terminal_lexicographic":
                reward = self._terminal_lexicographic_score(metrics)
                metrics["lexicographic_score"] = float(reward)
            elif self.reward_mode == "terminal_progress_time_safety":
                reward = self._progress_time_safety_score(metrics, terminal_only=True)
                metrics["progress_time_safety_score"] = float(reward)
            elif self.reward_mode == "terminal_progress_time_block_penalty":
                reward = self._progress_time_block_penalty_score(metrics, terminal_only=True)
                metrics["progress_time_block_penalty_score"] = float(reward)

        metrics["delta_score"] = float(self.last_delta_score)
        metrics.setdefault("lexicographic_score", 0.0)
        metrics.setdefault("progress_time_safety_score", 0.0)
        metrics.setdefault("progress_time_block_penalty_score", 0.0)
        metrics["terminal_fitness_score"] = float(
            float(metrics["fitness"]) / self.terminal_fitness_scale
        )
        metrics["reward"] = float(reward)
        return float(reward), metrics

    def _perform_live_reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(seed=seed)
        while info.get("done", 0.0) != 0:
            obs, info = self.env.reset(seed=seed)
        reset_metrics = self._terminal_metrics(info)
        if self.reward_mode == "delta_lexicographic":
            self.last_delta_score = self._lexicographic_score(reset_metrics)
        elif self.reward_mode == "delta_progress_time_safety":
            self.last_delta_score = self._progress_time_safety_score(
                reset_metrics,
                terminal_only=False,
            )
        elif self.reward_mode == "delta_progress_time_block_penalty":
            self.last_delta_score = self._progress_time_block_penalty_score(
                reset_metrics,
                terminal_only=False,
            )
        else:
            self.last_delta_score = self._delta_reward_score(
                reset_metrics,
                terminated=False,
                truncated=False,
            )
        self.last_info = dict(info)
        self.episode_index += 1
        if LOG_LIVE_RESETS:
            print(
                "[SB3 SAC] live reset confirmed "
                f"episode={self.episode_index} "
                f"attempts={getattr(self.env, 'last_reset_attempts', '?')} "
                f"reset_s={float(getattr(self.env, 'last_reset_seconds', 0.0)):.2f} "
                f"time={float(info.get('time', 0.0)):.2f} "
                f"progress={float(info.get('total_progress', 0.0)):.2f}% "
                f"distance={float(info.get('distance', 0.0)):.2f}"
            )
        return np.asarray(obs, dtype=np.float32), dict(info)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        return self._perform_live_reset(seed=seed)

    def step(self, action):
        env_action = self._to_env_action(action)
        obs, _, done, truncated, info = self.env.step(env_action)
        race_term = int(getattr(self.env, "race_terminated", 0))
        info_done = info.get("done", 0.0) == 1.0
        terminated = bool(done or info_done or race_term != 0)
        truncated = bool(truncated)
        reward, metrics = self._reward(info, terminated=terminated, truncated=truncated)
        info = dict(info)
        info["sb3_action"] = np.asarray(action, dtype=np.float32).tolist()
        info["env_action"] = env_action.tolist()
        info["episode_metrics"] = metrics
        self.last_info = info
        return np.asarray(obs, dtype=np.float32), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()
        super().close()


class EpisodeMetricsCallback:
    def __init__(self, run_dir: Path, checkpoint_every_episodes: int = 50, verbose: int = 1) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer: "EpisodeMetricsCallback") -> None:
                super().__init__(verbose=outer.verbose)
                self.outer = outer

            def _on_training_start(self) -> None:
                self.outer.model = self.model

            def _on_step(self) -> bool:
                infos = self.locals.get("infos", [])
                dones = self.locals.get("dones", [])
                for done, info in zip(dones, infos):
                    if bool(done):
                        self.outer.on_episode_end(info)
                return True

        self.run_dir = run_dir
        self.checkpoints_dir = run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = run_dir / "episode_metrics.csv"
        self.checkpoint_every_episodes = max(1, int(checkpoint_every_episodes))
        self.verbose = int(verbose)
        self.episode = 0
        self.best_progress = -float("inf")
        self.best_reward = -float("inf")
        self.model = None
        self.callback = _Callback(self)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers())
            writer.writeheader()

    @staticmethod
    def headers() -> list[str]:
        return [
            "timestamp_utc",
            "episode",
            "term",
            "progress",
            "time",
            "distance",
            "fitness",
            "pace_target_time",
            "pace_expected_progress",
            "pace_score",
            "delta_score",
            "lexicographic_score",
            "progress_time_safety_score",
            "progress_time_block_penalty_score",
            "terminal_fitness_score",
            "episode_reward",
            "timesteps",
        ]

    def on_episode_end(self, info: Dict[str, Any]) -> None:
        self.episode += 1
        metrics = dict(info.get("episode_metrics", {}))
        monitor_episode = info.get("episode", {})
        episode_reward = float(monitor_episode.get("r", metrics.get("reward", 0.0)))
        progress = float(metrics.get("progress", 0.0))
        row = dict(
            timestamp_utc=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            episode=self.episode,
            term=int(metrics.get("term", 0)),
            progress=progress,
            time=float(metrics.get("time", 0.0)),
            distance=float(metrics.get("distance", 0.0)),
            fitness=float(metrics.get("fitness", 0.0)),
            pace_target_time=float(metrics.get("pace_target_time", 0.0)),
            pace_expected_progress=float(metrics.get("pace_expected_progress", 0.0)),
            pace_score=float(metrics.get("pace_score", 0.0)),
            delta_score=float(metrics.get("delta_score", 0.0)),
            lexicographic_score=float(metrics.get("lexicographic_score", 0.0)),
            progress_time_safety_score=float(metrics.get("progress_time_safety_score", 0.0)),
            progress_time_block_penalty_score=float(
                metrics.get("progress_time_block_penalty_score", 0.0)
            ),
            terminal_fitness_score=float(metrics.get("terminal_fitness_score", 0.0)),
            episode_reward=episode_reward,
            timesteps=int(getattr(self.model, "num_timesteps", 0)) if self.model is not None else 0,
        )
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers())
            writer.writerow(row)

        if self.model is not None:
            self.model.save(self.run_dir / "latest_model")
            if progress > self.best_progress or (
                np.isclose(progress, self.best_progress) and episode_reward > self.best_reward
            ):
                self.best_progress = progress
                self.best_reward = episode_reward
                self.model.save(self.run_dir / "best_model")
            if self.episode % self.checkpoint_every_episodes == 0:
                self.model.save(self.checkpoints_dir / f"sac_episode_{self.episode:05d}")

        if self.verbose:
            print(
                f"[SB3 SAC] ep={self.episode} progress={progress:.2f}% "
                f"reward={episode_reward:.3f} term={int(row['term'])} "
                f"time={row['time']:.2f}s"
            )


class StopTrainingOnWallClock:
    def __init__(self, max_runtime_hours: Optional[float], verbose: int = 1) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer: "StopTrainingOnWallClock") -> None:
                super().__init__(verbose=outer.verbose)
                self.outer = outer

            def _on_training_start(self) -> None:
                self.outer.started_at = time.monotonic()

            def _on_step(self) -> bool:
                if self.outer.max_runtime_seconds is None:
                    return True
                elapsed = time.monotonic() - self.outer.started_at
                if elapsed < self.outer.max_runtime_seconds:
                    return True
                if self.verbose:
                    print(
                        "[SB3 SAC] Max runtime reached: "
                        f"{elapsed / 3600.0:.2f}h >= {self.outer.max_runtime_hours:.2f}h"
                    )
                return False

        self.max_runtime_hours = None if max_runtime_hours is None else float(max_runtime_hours)
        self.max_runtime_seconds = (
            None
            if self.max_runtime_hours is None or self.max_runtime_hours <= 0.0
            else self.max_runtime_hours * 3600.0
        )
        self.verbose = int(verbose)
        self.started_at = time.monotonic()
        self.callback = _Callback(self)


def activation_fn_from_name(name: str):
    value = str(name).strip().lower()
    if value == "relu":
        return nn.ReLU
    if value == "tanh":
        return nn.Tanh
    if value == "elu":
        return nn.ELU
    if value == "leaky_relu":
        return nn.LeakyReLU
    raise ValueError("activation name must be relu, tanh, elu, or leaky_relu.")


def build_run_dir(
    base_dir: str,
    map_name: str,
    reward_mode: str,
    action_layout: str,
    vertical_mode: bool,
) -> Path:
    lidar_mode = "3d_lidar" if vertical_mode else "2d_lidar"
    run_name = (
        f"{timestamp()}_sac_map_{map_name}_{lidar_mode}_{reward_mode}_{action_layout}"
    )
    run_dir = Path(base_dir) / sanitize_name(run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable-Baselines3 SAC Trackmania experiment.")
    parser.add_argument("--episodes", type=int, default=EPISODES_TO_RUN)
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--max-runtime-hours", type=float, default=MAX_RUNTIME_HOURS)
    parser.add_argument(
        "--reward-mode",
        default=REWARD_MODE,
        choices=[
            "terminal_progress",
            "terminal_fitness",
            "fitness_delta",
            "delta_lexicographic",
            "terminal_lexicographic",
            "delta_progress_time_safety",
            "terminal_progress_time_safety",
            "delta_progress_time_block_penalty",
            "terminal_progress_time_block_penalty",
        ],
    )
    parser.add_argument(
        "--action-layout",
        default=ACTION_LAYOUT,
        choices=["gas_steer", "gas_brake_steer", "throttle_steer", "target_3d"],
    )
    parser.add_argument("--net-arch", default=",".join(str(dim) for dim in SAC_POLICY_NET_ARCH))
    parser.add_argument("--activation-fn", default=SAC_ACTIVATION_FN, choices=["relu", "tanh", "elu", "leaky_relu"])
    parser.add_argument("--learning-rate", type=float, default=SAC_LEARNING_RATE)
    parser.add_argument("--env-max-time", type=float, default=ENV_MAX_TIME)
    parser.add_argument("--checkpoint-every-episodes", type=int, default=CHECKPOINT_EVERY_EPISODES)
    parser.add_argument("--terminal-fitness-scale", type=float, default=TERMINAL_FITNESS_SCALE)
    parser.add_argument(
        "--pace-target-time",
        type=float,
        default=None,
        help="Target finish time for fitness_delta pace shaping. Defaults to env_max_time / 2.",
    )
    parser.add_argument("--initial-model-path", default=INITIAL_MODEL_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--import-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnMaxEpisodes
    from stable_baselines3.common.monitor import Monitor

    if args.import_check:
        print("Stable-Baselines3 import check OK.")
        return
    pace_target_time = resolve_pace_target_time(args.env_max_time, args.pace_target_time)

    run_dir = Path(args.run_dir) if args.run_dir else build_run_dir(
        "logs/sb3_runs",
        map_name=MAP_NAME,
        reward_mode=args.reward_mode,
        action_layout=args.action_layout,
        vertical_mode=VERTICAL_MODE,
    )
    monitor_path = str(run_dir / "monitor.csv")
    raw_env = TrackmaniaSB3Env(
        map_name=MAP_NAME,
        reward_mode=args.reward_mode,
        action_layout=args.action_layout,
        env_max_time=args.env_max_time,
        vertical_mode=VERTICAL_MODE,
        max_touches=MAX_TOUCHES,
        start_idle_max_time=START_IDLE_MAX_TIME,
        terminal_fitness_scale=args.terminal_fitness_scale,
        pace_target_time=pace_target_time,
    )
    env = Monitor(raw_env, filename=monitor_path, info_keywords=("episode_metrics",))

    config = dict(
        algorithm="SAC",
        map_name=MAP_NAME,
        initial_model_path=args.initial_model_path,
        episodes_to_run=int(args.episodes),
        total_timesteps=int(args.total_timesteps),
        max_runtime_hours=float(args.max_runtime_hours),
        env_max_time=float(args.env_max_time),
        reward_mode=args.reward_mode,
        action_layout=args.action_layout,
        vertical_mode=VERTICAL_MODE,
        terminal_fitness_scale=float(args.terminal_fitness_scale),
        pace_target_time=float(pace_target_time),
        path_tile_count=int(raw_env.path_tile_count),
        progress_bucket=float(raw_env.progress_bucket),
        estimated_path_length=float(raw_env.estimated_path_length),
        train_freq=list(SAC_TRAIN_FREQ),
        gradient_steps=SAC_GRADIENT_STEPS,
        learning_rate=float(args.learning_rate),
        buffer_size=SAC_BUFFER_SIZE,
        learning_starts=SAC_LEARNING_STARTS,
        batch_size=SAC_BATCH_SIZE,
        gamma=SAC_GAMMA,
        tau=SAC_TAU,
        ent_coef=SAC_ENT_COEF,
        policy_net_arch=[int(dim) for dim in str(args.net_arch).split(",") if str(dim).strip()],
        activation_fn=args.activation_fn,
        action_space=str(env.action_space),
        observation_space=str(env.observation_space),
        device=args.device,
    )
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=True)

    print("\n[SB3 SAC] Prepared Trackmania SAC experiment")
    print(f"[SB3 SAC] run_dir={run_dir}")
    print(f"[SB3 SAC] map_name={MAP_NAME} lidar_mode={'3D' if VERTICAL_MODE else '2D'}")
    print(f"[SB3 SAC] reward_mode={args.reward_mode} action_layout={args.action_layout}")
    print(f"[SB3 SAC] pace_target_time={float(pace_target_time):.2f}s")
    print(
        f"[SB3 SAC] path_tiles={int(raw_env.path_tile_count)} "
        f"progress_bucket={float(raw_env.progress_bucket):.4f}% "
        f"estimated_path_length={float(raw_env.estimated_path_length):.1f}"
    )
    print(f"[SB3 SAC] initial_model_path={args.initial_model_path or 'None'}")
    print(
        f"[SB3 SAC] max_runtime_hours={float(args.max_runtime_hours):.2f} "
        f"episodes_cap={int(args.episodes)} total_timesteps_cap={int(args.total_timesteps)}"
    )
    print(f"[SB3 SAC] learning_rate={float(args.learning_rate):.6g}")
    print(f"[SB3 SAC] train_freq={SAC_TRAIN_FREQ}, gradient_steps={SAC_GRADIENT_STEPS}")
    print(f"[SB3 SAC] action_space={env.action_space}")
    print(f"[SB3 SAC] observation_space={env.observation_space}")

    net_arch = [int(dim) for dim in str(args.net_arch).split(",") if str(dim).strip()]
    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn=activation_fn_from_name(args.activation_fn),
    )
    initial_model_path = None
    if args.initial_model_path:
        initial_model_path = Path(args.initial_model_path)
        if not initial_model_path.is_absolute():
            initial_model_path = PROJECT_ROOT / initial_model_path
        if not initial_model_path.exists():
            raise FileNotFoundError(initial_model_path)

    if initial_model_path is not None:
        print(f"[SB3 SAC] Loading initial model: {initial_model_path}")
        model = SAC.load(
            initial_model_path,
            env=env,
            learning_rate=float(args.learning_rate),
            buffer_size=SAC_BUFFER_SIZE,
            learning_starts=SAC_LEARNING_STARTS,
            batch_size=SAC_BATCH_SIZE,
            tau=SAC_TAU,
            gamma=SAC_GAMMA,
            train_freq=SAC_TRAIN_FREQ,
            gradient_steps=SAC_GRADIENT_STEPS,
            ent_coef=SAC_ENT_COEF,
            verbose=1,
            tensorboard_log=str(run_dir / "tensorboard"),
            device=args.device,
        )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=float(args.learning_rate),
            buffer_size=SAC_BUFFER_SIZE,
            learning_starts=SAC_LEARNING_STARTS,
            batch_size=SAC_BATCH_SIZE,
            tau=SAC_TAU,
            gamma=SAC_GAMMA,
            train_freq=SAC_TRAIN_FREQ,
            gradient_steps=SAC_GRADIENT_STEPS,
            ent_coef=SAC_ENT_COEF,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(run_dir / "tensorboard"),
            device=args.device,
        )
    episode_callback = EpisodeMetricsCallback(
        run_dir=run_dir,
        checkpoint_every_episodes=args.checkpoint_every_episodes,
        verbose=1,
    )
    wall_clock_callback = StopTrainingOnWallClock(
        max_runtime_hours=args.max_runtime_hours,
        verbose=1,
    )
    callbacks = CallbackList(
        [
            episode_callback.callback,
            StopTrainingOnMaxEpisodes(max_episodes=int(args.episodes), verbose=1),
            wall_clock_callback.callback,
        ]
    )

    try:
        model.learn(
            total_timesteps=int(args.total_timesteps),
            callback=callbacks,
            log_interval=1,
            progress_bar=False,
        )
        model.save(run_dir / "final_model")
        model.save(run_dir / "latest_model")
        print(f"[SB3 SAC] Finished. Artifacts saved in: {run_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
