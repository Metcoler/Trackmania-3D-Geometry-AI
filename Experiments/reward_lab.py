from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiments.tm2d_env import TM2DRewardConfig, TM2DSimEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Print reward tables for one TM2D map.")
    parser.add_argument("--map-name", default="AI Training #5")
    parser.add_argument("--max-time", type=float, default=45.0)
    args = parser.parse_args()

    env = TM2DSimEnv(map_name=args.map_name, max_time=args.max_time)
    progresses = [0.0, env.progress_bucket, 2 * env.progress_bucket, 5 * env.progress_bucket, 20.0, 50.0, 100.0]
    times = [2.0, 5.0, 10.0, args.max_time * 0.5, args.max_time]
    modes = [
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
    print(f"map={args.map_name} progress_bucket={env.progress_bucket:.4f}% max_time={args.max_time:.2f}s")
    for mode in modes:
        env.reward_config = TM2DRewardConfig(mode=mode)
        print(f"\nmode={mode}")
        header = "progress/time".ljust(14) + "".join(f"{t:>12.1f}s" for t in times)
        print(header)
        for progress in progresses:
            values = [
                env._score_state(finished=0, crashes=1, progress=progress, time_value=t, distance=0.0)
                for t in times
            ]
            print(f"{progress:>10.3f}%  " + "".join(f"{value:12.3f}" for value in values))


if __name__ == "__main__":
    main()
