from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_ROOTS = [
    "logs/supervised_data/20260502_153421_map_AI Training #5_v2d_asphalt_target_dataset",
    "logs/supervised_data/20260502_154227_map_AI Training #5_v2d_asphalt_target_dataset",
]


def collect_attempt_files(roots: list[str]) -> list[Path]:
    files: list[Path] = []
    for root_text in roots:
        root = Path(root_text)
        if root.is_file() and root.suffix.lower() == ".npz":
            files.append(root)
            continue
        files.extend(sorted(root.glob("attempts/attempt_*.npz")))
        files.extend(sorted(root.glob("**/attempts/attempt_*.npz")))
    return sorted(set(files))


def tick_stats_for_file(path: Path, dt_ref: float, max_game_dt: float) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if "game_times" not in data:
            return {
                "path": str(path),
                "frames": 0,
                "valid_deltas": 0,
                "status": "missing_game_times",
            }
        game_times = np.asarray(data["game_times"], dtype=np.float64)

    if game_times.size < 2:
        return {
            "path": str(path),
            "frames": int(game_times.size),
            "valid_deltas": 0,
            "status": "too_short",
        }

    deltas = np.diff(game_times)
    valid = deltas[np.isfinite(deltas) & (deltas > 1e-6) & (deltas <= max_game_dt)]
    if valid.size <= 0:
        return {
            "path": str(path),
            "frames": int(game_times.size),
            "valid_deltas": 0,
            "status": "no_valid_deltas",
        }

    ticks = np.maximum(1, np.rint(valid / dt_ref).astype(np.int32))
    return {
        "path": str(path),
        "frames": int(game_times.size),
        "valid_deltas": int(valid.size),
        "status": "ok",
        "dt_mean": float(np.mean(valid)),
        "dt_std": float(np.std(valid)),
        "fps_mean": float(np.mean(1.0 / valid)),
        "fps_std": float(np.std(1.0 / valid)),
        "tick_mean": float(np.mean(ticks)),
        "tick_std": float(np.std(ticks)),
        "ticks": ticks,
    }


def format_profile(values: np.ndarray, probs: np.ndarray) -> str:
    return ",".join(f"{int(value)}:{float(prob):.6f}" for value, prob in zip(values, probs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Trackmania supervised physics tick skips.")
    parser.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    parser.add_argument("--output-dir", default="Experiments/analysis/supervised_physics_ticks")
    parser.add_argument("--dt-ref", type=float, default=1.0 / 100.0)
    parser.add_argument("--max-game-dt", type=float, default=0.25)
    parser.add_argument("--max-profile-tick", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_attempt_files([str(root) for root in args.roots])
    rows = [tick_stats_for_file(path, dt_ref=float(args.dt_ref), max_game_dt=float(args.max_game_dt)) for path in files]
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        raise RuntimeError("No attempt files with valid game_times were found.")

    all_ticks = np.concatenate([np.asarray(row["ticks"], dtype=np.int32) for row in ok_rows])
    clipped_ticks = np.minimum(all_ticks, int(args.max_profile_tick))
    values, counts = np.unique(clipped_ticks, return_counts=True)
    probs = counts.astype(np.float64) / float(np.sum(counts))
    profile = format_profile(values, probs)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "path",
            "frames",
            "valid_deltas",
            "status",
            "dt_mean",
            "dt_std",
            "fps_mean",
            "fps_std",
            "tick_mean",
            "tick_std",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    distribution_path = output_dir / "tick_distribution.csv"
    with distribution_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["physics_tick_count", "physics_hz", "probability", "count"])
        writer.writeheader()
        for value, count, prob in zip(values, counts, probs):
            writer.writerow(
                {
                    "physics_tick_count": int(value),
                    "physics_hz": float(1.0 / (float(value) * float(args.dt_ref))),
                    "probability": float(prob),
                    "count": int(count),
                }
            )

    recommended = {
        "dt_ref": float(args.dt_ref),
        "max_game_dt": float(args.max_game_dt),
        "max_profile_tick": int(args.max_profile_tick),
        "attempt_files": len(files),
        "valid_attempt_files": len(ok_rows),
        "valid_deltas": int(all_ticks.size),
        "physics_tick_probs": profile,
        "physics_tick_values": [int(value) for value in values],
        "physics_tick_probabilities": [float(prob) for prob in probs],
    }
    (output_dir / "recommended_profile.json").write_text(
        json.dumps(recommended, indent=2),
        encoding="utf-8",
    )

    report = [
        "# Supervised Physics Tick Analysis",
        "",
        f"- Attempt files: {len(files)}",
        f"- Valid attempt files: {len(ok_rows)}",
        f"- Valid frame deltas: {int(all_ticks.size)}",
        f"- Recommended `--physics-tick-probs`: `{profile}`",
        "",
        "## Distribution",
        "",
        "| ticks | physics Hz | probability | count |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for value, count, prob in zip(values, counts, probs):
        report.append(
            f"| {int(value)} | {1.0 / (float(value) * float(args.dt_ref)):.2f} | {float(prob):.4f} | {int(count)} |"
        )
    report.extend(
        [
            "",
            "Interpretation: the observation should encode physics-tick delay, not render FPS. "
            "`physics_delay_norm = 1 - 1 / tick_count` is zero at 100 Hz and grows when game physics updates are skipped.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote {summary_path}")
    print(f"Recommended --physics-tick-probs \"{profile}\"")


if __name__ == "__main__":
    main()
