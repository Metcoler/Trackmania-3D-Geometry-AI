from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiments.train_sac import DEFAULT_ACTION_LAYOUT, DEFAULT_MAP_NAME, DEFAULT_REWARD_MODE, TM2DSB3Env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved SB3 policy in TM2D.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--algorithm", default="PPO", choices=["SAC", "PPO", "TD3"])
    parser.add_argument("--map-name", default=DEFAULT_MAP_NAME)
    parser.add_argument("--reward-mode", default=DEFAULT_REWARD_MODE)
    parser.add_argument("--action-layout", default=DEFAULT_ACTION_LAYOUT)
    parser.add_argument("--collision-mode", default="laser", choices=["center", "corners", "laser", "lidar"])
    parser.add_argument("--collision-distance-threshold", type=float, default=2.0)
    parser.add_argument("--env-max-time", type=float, default=45.0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def load_model(algorithm: str, model_path: Path):
    from stable_baselines3 import PPO, SAC, TD3

    classes = {"SAC": SAC, "PPO": PPO, "TD3": TD3}
    return classes[str(algorithm).upper()].load(model_path, device="cpu")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = load_model(args.algorithm, model_path)
    rows: list[dict] = []
    for episode in range(1, int(args.episodes) + 1):
        env = TM2DSB3Env(
            map_name=args.map_name,
            reward_mode=args.reward_mode,
            env_max_time=args.env_max_time,
            action_layout=args.action_layout,
            collision_mode=args.collision_mode,
            collision_distance_threshold=args.collision_distance_threshold,
            seed=int(args.seed) + episode,
        )
        obs, info = env.reset(seed=int(args.seed) + episode)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=not bool(args.stochastic))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
        metrics = dict(info.get("episode_metrics", {}))
        row = {
            "episode": episode,
            "finished": int(metrics.get("finished", 0)),
            "crashes": int(metrics.get("crashes", 0)),
            "timeout": int(metrics.get("timeout", 0)),
            "progress": float(metrics.get("progress", 0.0)),
            "dense_progress": float(metrics.get("dense_progress", 0.0)),
            "time": float(metrics.get("time", 0.0)),
            "distance": float(metrics.get("distance", 0.0)),
            "episode_reward": float(total_reward),
            "steps": int(steps),
        }
        rows.append(row)
        print(
            f"ep={episode:03d} dense={row['dense_progress']:.2f}% "
            f"progress={row['progress']:.2f}% fin={row['finished']} "
            f"crashes={row['crashes']} time={row['time']:.3f}s reward={row['episode_reward']:.3f}"
        )
        env.close()

    dense = np.asarray([row["dense_progress"] for row in rows], dtype=np.float64)
    finished = np.asarray([row["finished"] for row in rows], dtype=np.float64)
    finish_times = np.asarray([row["time"] for row in rows if row["finished"] > 0], dtype=np.float64)
    print("\nEvaluation summary")
    print(f"episodes={len(rows)}")
    print(f"finish_count={int(np.sum(finished))}")
    print(f"mean_dense_progress={float(np.mean(dense)):.3f}%")
    print(f"max_dense_progress={float(np.max(dense)):.3f}%")
    if finish_times.size:
        print(f"best_finish_time={float(np.min(finish_times)):.3f}s")
        print(f"mean_finish_time={float(np.mean(finish_times)):.3f}s")

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved evaluation CSV to {output_path}")


if __name__ == "__main__":
    main()
