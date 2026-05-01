from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiments.tm2d_env import TM2DRewardConfig, TM2DSimEnv


REWARD_MODES = [
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
    "progress_rate",
]


class ConstantPolicy:
    def __init__(self, action) -> None:
        self.action = np.asarray(action, dtype=np.float32)

    def act(self, obs) -> np.ndarray:
        del obs
        return self.action


class HeuristicPolicy:
    def __init__(self, gas: float = 1.0, gain: float = 2.5, oscillation: float = 0.0) -> None:
        self.gas = float(gas)
        self.gain = float(gain)
        self.oscillation = float(oscillation)
        self.step = 0

    def act(self, obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        segment_heading_error = float(obs[15 + 5 + 2])
        steer = float(np.clip(segment_heading_error * self.gain, -1.0, 1.0))
        if self.oscillation:
            steer = float(np.clip(steer + math.sin(self.step * 0.2) * self.oscillation, -1.0, 1.0))
        self.step += 1
        return np.asarray([self.gas, 0.0, steer], dtype=np.float32)


def static_scenarios(env: TM2DSimEnv) -> list[dict]:
    bucket = float(env.progress_bucket)
    max_time = float(env.max_time)
    path = float(env.geometry.estimated_path_length)
    return [
        {"name": "idle_timeout", "term": 0, "progress": 0.0, "time": max_time, "distance": 0.0},
        {"name": "early_kamikaze", "term": -1, "progress": 0.5 * bucket, "time": 2.0, "distance": 20.0},
        {"name": "first_block_fast_crash", "term": -1, "progress": bucket, "time": 3.0, "distance": 42.0},
        {"name": "first_block_slow_crash", "term": -1, "progress": bucket, "time": 10.0, "distance": 42.0},
        {"name": "mid_fast_crash", "term": -1, "progress": 20.0, "time": 15.0, "distance": 320.0},
        {"name": "mid_slow_crash", "term": -1, "progress": 20.0, "time": 30.0, "distance": 320.0},
        {"name": "finish_fast_clean", "term": 1, "progress": 100.0, "time": 0.5 * max_time, "distance": path},
        {"name": "finish_fast_messy", "term": 1, "progress": 100.0, "time": 0.5 * max_time, "distance": 1.35 * path},
        {"name": "finish_slow_clean", "term": 1, "progress": 100.0, "time": max_time, "distance": path},
    ]


def print_static_rankings(env: TM2DSimEnv, modes: list[str]) -> None:
    scenarios = static_scenarios(env)
    print("\n=== Static terminal scenario ranking ===")
    print("Higher is better. Scenarios are artificial, not simulated policies.")
    for mode in modes:
        env.reward_config = TM2DRewardConfig(mode=mode)
        ranked = []
        for scenario in scenarios:
            score = env._score_state(
                term=int(scenario["term"]),
                progress=float(scenario["progress"]),
                time_value=float(scenario["time"]),
                distance=float(scenario["distance"]),
            )
            ranked.append((float(score), scenario["name"]))
        ranked.sort(reverse=True)
        print(f"\nmode={mode}")
        for score, name in ranked:
            print(f"  {score:10.4f}  {name}")


def print_policy_rollouts(map_name: str, max_time: float, modes: list[str], seed: int) -> None:
    policies = {
        "idle": ConstantPolicy([0.0, 0.0, 0.0]),
        "full_gas_straight": ConstantPolicy([1.0, 0.0, 0.0]),
        "heuristic_full": HeuristicPolicy(gas=1.0, gain=2.5),
        "heuristic_soft": HeuristicPolicy(gas=0.65, gain=2.0),
        "heuristic_wavy": HeuristicPolicy(gas=1.0, gain=2.5, oscillation=0.45),
    }
    print("\n=== Simulated policy rollout ranking ===")
    for mode in modes:
        env = TM2DSimEnv(
            map_name=map_name,
            max_time=max_time,
            reward_config=TM2DRewardConfig(mode=mode),
            seed=seed,
        )
        results = []
        for name, policy in policies.items():
            if isinstance(policy, HeuristicPolicy):
                policy.step = 0
            metrics = env.rollout_policy(policy)
            results.append(
                (
                    float(metrics["reward"]),
                    name,
                    float(metrics["dense_progress"]),
                    float(metrics["progress"]),
                    float(metrics["time"]),
                    int(metrics["term"]),
                    int(metrics["steps"]),
                )
            )
        results.sort(reverse=True)
        print(f"\nmode={mode}")
        for reward, name, dense, progress, time_value, term, steps in results:
            print(
                f"  reward={reward:10.4f} dense={dense:6.2f}% progress={progress:6.2f}% "
                f"time={time_value:6.2f}s term={term:2d} steps={steps:4d}  {name}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reward sanity sweep for TM2D experiments.")
    parser.add_argument("--map-name", default="AI Training #5")
    parser.add_argument("--max-time", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--modes", default=",".join(REWARD_MODES))
    args = parser.parse_args()

    modes = [mode.strip() for mode in str(args.modes).split(",") if mode.strip()]
    env = TM2DSimEnv(map_name=args.map_name, max_time=args.max_time, seed=args.seed)
    print(
        f"map={args.map_name} max_time={args.max_time:.2f}s "
        f"progress_bucket={env.progress_bucket:.4f}% path_tiles={len(env.geometry.path_tiles_xz)}"
    )
    print_static_rankings(env, modes)
    print_policy_rollouts(args.map_name, args.max_time, modes, args.seed)


if __name__ == "__main__":
    main()
