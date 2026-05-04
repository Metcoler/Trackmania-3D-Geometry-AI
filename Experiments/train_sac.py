from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Individual import Individual

from Experiments.tm2d_env import TM2DPhysicsConfig, TM2DRewardConfig, TM2DSimEnv


DEFAULT_MAP_NAME = "AI Training #5"
DEFAULT_REWARD_MODE = "delta_finished_progress_time"
DEFAULT_ACTION_LAYOUT = "gas_steer"
DEFAULT_NET_ARCH = [32, 32]
DEFAULT_LOG_DIR = "Experiments/runs_rl"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_name(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return "".join(ch if ch in allowed else "_" for ch in value)


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def activation_fn_from_name(name: str):
    import torch.nn as nn

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


class TM2DSB3Env(gym.Env):
    """Gymnasium wrapper around the local 2D Trackmania simulator."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        map_name: str,
        reward_mode: str,
        env_max_time: float,
        action_layout: str = DEFAULT_ACTION_LAYOUT,
        collision_mode: str = "laser",
        collision_distance_threshold: float = 2.0,
        seed: int | None = None,
        terminal_fitness_scale: float = 1_000_000.0,
        pace_target_time: float | None = None,
        fixed_fps: float | None = None,
        physics_tick_profile: str = "supervised_v2d",
        vertical_mode: bool = False,
        multi_surface_mode: bool = False,
        binary_gas_brake: bool = True,
    ) -> None:
        super().__init__()
        self.reward_config = TM2DRewardConfig(
            mode=str(reward_mode),
            terminal_fitness_scale=float(terminal_fitness_scale),
            pace_target_time=pace_target_time,
        )
        self.env = TM2DSimEnv(
            map_name=map_name,
            max_time=float(env_max_time),
            reward_config=self.reward_config,
            physics_config=TM2DPhysicsConfig().with_tick_profile(
                "fixed100" if fixed_fps is not None else physics_tick_profile
            ),
            seed=seed,
            collision_mode=collision_mode,
            collision_distance_threshold=collision_distance_threshold,
            vertical_mode=vertical_mode,
            multi_surface_mode=multi_surface_mode,
            binary_gas_brake=binary_gas_brake,
        )
        low, high = self.env.obs_encoder.get_observation_bounds(action_mode="target")
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_layout = str(action_layout).strip().lower()
        if self.action_layout == "gas_brake_steer":
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        elif self.action_layout in {"gas_steer", "throttle_steer"}:
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            raise ValueError("action_layout must be gas_brake_steer, gas_steer, or throttle_steer.")
        self.last_info: dict[str, Any] = {}

    def _to_env_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.action_layout == "gas_steer":
            return np.asarray(
                [
                    0.5 * (float(np.clip(action[0], -1.0, 1.0)) + 1.0),
                    0.0,
                    float(np.clip(action[1], -1.0, 1.0)),
                ],
                dtype=np.float32,
            )
        if self.action_layout == "throttle_steer":
            throttle = float(np.clip(action[0], -1.0, 1.0))
            return np.asarray(
                [max(throttle, 0.0), max(-throttle, 0.0), float(np.clip(action[1], -1.0, 1.0))],
                dtype=np.float32,
            )
        if action.size < 3:
            action = np.pad(action, (0, 3 - action.size), constant_values=0.0)
        return np.asarray(
            [
                float(np.clip(action[0], 0.0, 1.0)),
                float(np.clip(action[1], 0.0, 1.0)),
                float(np.clip(action[2], -1.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def _episode_metrics(self, info: dict[str, Any]) -> dict[str, float]:
        finished = int(info.get("finished", 0))
        crashes = int(info.get("crashes", 0))
        progress = float(info.get("discrete_progress", 0.0))
        dense_progress = float(info.get("dense_progress", progress))
        time_value = float(info.get("time", 0.0))
        distance = float(info.get("distance", 0.0))
        fitness_progress = (
            dense_progress
            if Individual.RANKING_PROGRESS_SOURCE == "dense_progress"
            else progress
        )
        fitness = Individual.compute_scalar_fitness_for(finished, crashes, fitness_progress, time_value, distance)
        return {
            "finished": float(finished),
            "crashes": float(crashes),
            "timeout": float(int(finished <= 0 and crashes <= 0)),
            "progress": progress,
            "dense_progress": dense_progress,
            "time": time_value,
            "distance": distance,
            "reward": float(info.get("reward", 0.0)),
            "reward_score": float(info.get("reward_score", 0.0)),
            "fitness": float(fitness),
        }

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)
        self.last_info = dict(info)
        return np.asarray(obs, dtype=np.float32), dict(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(self._to_env_action(action))
        info = dict(info)
        info["episode_metrics"] = self._episode_metrics(info)
        self.last_info = info
        return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        return None


class EpisodeMetricsCallback:
    def __init__(
        self,
        run_dir: Path,
        checkpoint_every_episodes: int,
        verbose: int = 1,
        algorithm: str = "SAC",
    ) -> None:
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
        self.algorithm = str(algorithm).upper()
        self.episode = 0
        self.best_progress = -float("inf")
        self.best_dense_progress = -float("inf")
        self.best_reward = -float("inf")
        self.model = None
        self.callback = _Callback(self)
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.headers())
            writer.writeheader()

    @staticmethod
    def headers() -> list[str]:
        return [
            "timestamp_utc",
            "episode",
            "finished",
            "crashes",
            "timeout",
            "progress",
            "dense_progress",
            "time",
            "distance",
            "fitness",
            "reward_score",
            "episode_reward",
            "timesteps",
        ]

    def on_episode_end(self, info: dict[str, Any]) -> None:
        self.episode += 1
        metrics = dict(info.get("episode_metrics", {}))
        monitor_episode = info.get("episode", {})
        episode_reward = float(monitor_episode.get("r", metrics.get("reward", 0.0)))
        progress = float(metrics.get("progress", 0.0))
        dense_progress = float(metrics.get("dense_progress", progress))
        row = {
            "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "episode": self.episode,
            "finished": int(metrics.get("finished", 0)),
            "crashes": int(metrics.get("crashes", 0)),
            "timeout": int(metrics.get("timeout", 0)),
            "progress": progress,
            "dense_progress": dense_progress,
            "time": float(metrics.get("time", 0.0)),
            "distance": float(metrics.get("distance", 0.0)),
            "fitness": float(metrics.get("fitness", 0.0)),
            "reward_score": float(metrics.get("reward_score", 0.0)),
            "episode_reward": episode_reward,
            "timesteps": int(getattr(self.model, "num_timesteps", 0)) if self.model is not None else 0,
        }
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.headers())
            writer.writerow(row)

        if self.model is not None:
            self.model.save(self.run_dir / "latest_model")
            improved = dense_progress > self.best_dense_progress or (
                np.isclose(dense_progress, self.best_dense_progress)
                and episode_reward > self.best_reward
            )
            if improved:
                self.best_progress = progress
                self.best_dense_progress = dense_progress
                self.best_reward = episode_reward
                self.model.save(self.run_dir / "best_model")
            if self.episode % self.checkpoint_every_episodes == 0:
                self.model.save(self.checkpoints_dir / f"{self.algorithm.lower()}_episode_{self.episode:05d}")

        if self.verbose:
            print(
                f"[TM2D {self.algorithm}] ep={self.episode} dense={dense_progress:.2f}% "
                f"progress={progress:.2f}% reward={episode_reward:.3f} "
                f"fin={int(row['finished'])} crashes={int(row['crashes'])} "
                f"time={row['time']:.2f}s"
            )


class StopTrainingOnWallClock:
    def __init__(self, max_runtime_minutes: float | None, verbose: int = 1) -> None:
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
                        "[TM2D RL] Max runtime reached: "
                        f"{elapsed / 60.0:.2f}min >= {self.outer.max_runtime_minutes:.2f}min"
                    )
                return False

        self.max_runtime_minutes = None if max_runtime_minutes is None else float(max_runtime_minutes)
        self.max_runtime_seconds = (
            None
            if self.max_runtime_minutes is None or self.max_runtime_minutes <= 0.0
            else self.max_runtime_minutes * 60.0
        )
        self.verbose = int(verbose)
        self.started_at = time.monotonic()
        self.callback = _Callback(self)


def build_run_dir(base_dir: str, algorithm: str, map_name: str, reward_mode: str, action_layout: str) -> Path:
    run_name = f"{timestamp()}_tm2d_{algorithm.lower()}_{map_name}_{reward_mode}_{action_layout}"
    run_dir = Path(base_dir) / sanitize_name(run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable-Baselines3 RL over the fast TM2D simulator.")
    parser.add_argument("--algorithm", default="SAC", choices=["SAC", "PPO", "TD3"])
    parser.add_argument("--map-name", default=DEFAULT_MAP_NAME)
    parser.add_argument(
        "--reward-mode",
        default=DEFAULT_REWARD_MODE,
        choices=[
            "progress_delta",
            "progress_primary_delta",
            "pace_delta",
            "terminal_fitness",
            "individual_dense",
            "terminal_lexicographic",
            "terminal_lexicographic_no_distance",
            "terminal_lexicographic_progress20",
            "delta_lexicographic",
            "delta_lexicographic_terminal",
            "terminal_progress_time_efficiency",
            "delta_progress_time_efficiency",
            "terminal_finished_progress_time",
            "delta_finished_progress_time",
            "terminal_progress_time_safety",
            "delta_progress_time_safety",
            "terminal_progress_time_block_penalty",
            "delta_progress_time_block_penalty",
            "progress_rate",
        ],
    )
    parser.add_argument(
        "--action-layout",
        default=DEFAULT_ACTION_LAYOUT,
        choices=["gas_brake_steer", "gas_steer", "throttle_steer"],
    )
    parser.add_argument("--env-max-time", type=float, default=30.0)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--max-runtime-minutes", type=float, default=30.0)
    parser.add_argument("--checkpoint-every-episodes", type=int, default=50)
    parser.add_argument("--collision-mode", choices=["center", "corners", "laser", "lidar"], default="laser")
    parser.add_argument(
        "--collision-distance-threshold",
        type=float,
        default=2.0,
        help="Laser/lidar collision threshold used when --collision-mode laser is selected.",
    )
    parser.add_argument(
        "--vertical-mode",
        action="store_true",
        help=(
            "Use the 3D-compatible observation layout with neutral "
            "vertical features while keeping TM2D physics flat."
        ),
    )
    parser.add_argument(
        "--multi-surface-mode",
        action="store_true",
        help="Append surface traction instructions to the observation.",
    )
    parser.add_argument(
        "--continuous-gas-brake",
        action="store_true",
        help="Disable live-TM-style gas/brake binarization in TM2D diagnostics.",
    )
    parser.add_argument(
        "--fixed-fps",
        type=float,
        default=None,
        help="Legacy alias for deterministic 100 Hz TM2D physics when set to 100.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.9995)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gradient-steps", type=int, default=8)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--train-freq-unit", choices=["step", "episode"], default="episode")
    parser.add_argument("--ent-coef", default="0.01")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--action-noise-sigma", type=float, default=0.10)
    parser.add_argument("--net-arch", default=",".join(str(dim) for dim in DEFAULT_NET_ARCH))
    parser.add_argument("--activation-fn", choices=["relu", "tanh", "elu", "leaky_relu"], default="relu")
    parser.add_argument("--terminal-fitness-scale", type=float, default=1_000_000.0)
    parser.add_argument("--pace-target-time", type=float, default=None)
    parser.add_argument("--initial-model-path", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--import-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnMaxEpisodes
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise

    if args.import_check:
        print("Stable-Baselines3 import check OK.")
        return

    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else build_run_dir(args.log_dir, args.algorithm, args.map_name, args.reward_mode, args.action_layout)
    )
    raw_env = TM2DSB3Env(
        map_name=args.map_name,
        reward_mode=args.reward_mode,
        env_max_time=args.env_max_time,
        action_layout=args.action_layout,
        collision_mode=args.collision_mode,
        collision_distance_threshold=args.collision_distance_threshold,
        seed=args.seed,
        terminal_fitness_scale=args.terminal_fitness_scale,
        pace_target_time=args.pace_target_time,
        fixed_fps=args.fixed_fps,
        vertical_mode=args.vertical_mode,
        multi_surface_mode=args.multi_surface_mode,
        binary_gas_brake=not args.continuous_gas_brake,
    )
    env = Monitor(raw_env, filename=str(run_dir / "monitor.csv"), info_keywords=("episode_metrics",))
    net_arch = parse_int_list(args.net_arch)
    train_freq = (int(args.train_freq), str(args.train_freq_unit))
    config = {
        "algorithm": args.algorithm,
        "map_name": args.map_name,
        "reward_mode": args.reward_mode,
        "action_layout": args.action_layout,
        "env_max_time": float(args.env_max_time),
        "episodes": int(args.episodes),
        "total_timesteps": int(args.total_timesteps),
        "max_runtime_minutes": float(args.max_runtime_minutes),
        "collision_mode": args.collision_mode,
        "collision_distance_threshold": float(args.collision_distance_threshold),
        "lidar_mode": "aabb_clearance",
        "vehicle_hitbox": raw_env.env.vehicle_hitbox.as_dict(),
        "vertical_mode": bool(args.vertical_mode),
        "multi_surface_mode": bool(args.multi_surface_mode),
        "binary_gas_brake": bool(not args.continuous_gas_brake),
        "fixed_fps": args.fixed_fps,
        "seed": int(args.seed),
        "learning_rate": float(args.learning_rate),
        "buffer_size": int(args.buffer_size),
        "learning_starts": int(args.learning_starts),
        "batch_size": int(args.batch_size),
        "gamma": float(args.gamma),
        "tau": float(args.tau),
        "gradient_steps": int(args.gradient_steps),
        "train_freq": list(train_freq),
        "ent_coef": args.ent_coef,
        "n_steps": int(args.n_steps),
        "n_epochs": int(args.n_epochs),
        "gae_lambda": float(args.gae_lambda),
        "clip_range": float(args.clip_range),
        "action_noise_sigma": float(args.action_noise_sigma),
        "net_arch": list(net_arch),
        "activation_fn": args.activation_fn,
        "terminal_fitness_scale": float(args.terminal_fitness_scale),
        "pace_target_time": args.pace_target_time,
        "initial_model_path": args.initial_model_path,
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
        "physics": asdict(raw_env.env.physics),
        "progress_bucket": raw_env.env.progress_bucket,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    algorithm = str(args.algorithm).upper()
    print(f"\n[TM2D {algorithm}] Prepared local RL experiment")
    print(f"[TM2D {algorithm}] run_dir={run_dir}")
    print(f"[TM2D {algorithm}] map_name={args.map_name}")
    print(f"[TM2D {algorithm}] reward_mode={args.reward_mode} action_layout={args.action_layout}")
    print(f"[TM2D {algorithm}] train_freq={train_freq} gradient_steps={args.gradient_steps}")
    print(f"[TM2D {algorithm}] net_arch={net_arch} activation={args.activation_fn}")
    print(f"[TM2D {algorithm}] observation_space={env.observation_space}")
    print(f"[TM2D {algorithm}] action_space={env.action_space}")

    policy_kwargs = {
        "net_arch": net_arch,
        "activation_fn": activation_fn_from_name(args.activation_fn),
    }
    algo_classes = {
        "SAC": SAC,
        "PPO": PPO,
        "TD3": TD3,
    }
    algo_class = algo_classes[algorithm]
    if args.initial_model_path:
        initial_model_path = Path(args.initial_model_path)
        if not initial_model_path.is_absolute():
            initial_model_path = ROOT / initial_model_path
        if not initial_model_path.exists():
            raise FileNotFoundError(initial_model_path)
        print(f"[TM2D {algorithm}] Loading initial model: {initial_model_path}")
        model = algo_class.load(
            initial_model_path,
            env=env,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            verbose=1,
            tensorboard_log=str(run_dir / "tensorboard"),
            device=args.device,
        )
    elif algorithm == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=train_freq,
            gradient_steps=args.gradient_steps,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(run_dir / "tensorboard"),
            device=args.device,
        )
    elif algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=float(args.ent_coef),
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(run_dir / "tensorboard"),
            device=args.device,
        )
    elif algorithm == "TD3":
        action_noise = NormalActionNoise(
            mean=np.zeros(env.action_space.shape[-1], dtype=np.float32),
            sigma=float(args.action_noise_sigma) * np.ones(env.action_space.shape[-1], dtype=np.float32),
        )
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=train_freq,
            gradient_steps=args.gradient_steps,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(run_dir / "tensorboard"),
            device=args.device,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    episode_callback = EpisodeMetricsCallback(
        run_dir=run_dir,
        checkpoint_every_episodes=args.checkpoint_every_episodes,
        verbose=1,
        algorithm=algorithm,
    )
    callbacks = CallbackList(
        [
            episode_callback.callback,
            StopTrainingOnMaxEpisodes(max_episodes=int(args.episodes), verbose=1),
            StopTrainingOnWallClock(args.max_runtime_minutes, verbose=1).callback,
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
        print(f"[TM2D {algorithm}] Finished. Artifacts saved in: {run_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
