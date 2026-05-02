from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NeuralPolicy import NeuralPolicy

from Experiments.tm2d_env import TM2DPhysicsConfig, TM2DRewardConfig, TM2DSimEnv
from Experiments.tm2d_viewer import TM2DViewer


class HeuristicPolicy:
    def act(self, obs):
        # A tiny baseline: gas forward, steer against segment heading error.
        obs = np.asarray(obs, dtype=np.float32)
        segment_heading_error = float(obs[15 + 5 + 2])
        steer = float(np.clip(segment_heading_error * 2.5, -1.0, 1.0))
        return np.array([1.0, 0.0, steer], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the fast 2D TM simulator.")
    parser.add_argument("--map-name", default="AI Training #5")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-time", type=float, default=45.0)
    parser.add_argument(
        "--fixed-fps",
        type=float,
        default=None,
        help="Use deterministic fixed simulation FPS by setting min_dt=max_dt=1/fps.",
    )
    parser.add_argument("--vertical-mode", action="store_true")
    parser.add_argument("--multi-surface-mode", action="store_true")
    parser.add_argument(
        "--continuous-gas-brake",
        action="store_true",
        help="Disable live-TM-style gas/brake binarization in TM2D diagnostics.",
    )
    args = parser.parse_args()

    env = TM2DSimEnv(
        map_name=args.map_name,
        max_time=args.max_time,
        reward_config=TM2DRewardConfig(mode="progress_primary_delta"),
        physics_config=TM2DPhysicsConfig().with_fixed_fps(args.fixed_fps),
        vertical_mode=args.vertical_mode,
        multi_surface_mode=args.multi_surface_mode,
        binary_gas_brake=not args.continuous_gas_brake,
    )
    if args.model_path:
        policy, _ = NeuralPolicy.load(args.model_path)
    else:
        policy = HeuristicPolicy()
    viewer = TM2DViewer(env)
    obs, info = env.reset()
    while True:
        action = policy.act(obs)
        obs, _, terminated, truncated, info = env.step(action)
        viewer.update(info)
        time.sleep(1.0 / 60.0)
        if terminated or truncated:
            time.sleep(0.75)
            obs, info = env.reset()


if __name__ == "__main__":
    main()
