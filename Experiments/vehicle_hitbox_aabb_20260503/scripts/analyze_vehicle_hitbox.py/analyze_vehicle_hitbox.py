from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Car import Car
from VehicleHitbox import VehicleHitbox


MESH_ESTIMATES = {
    "visual_main_body": {
        "size": [2.1432, 3.7842, 0.8822],
        "center": [0.0004, -0.2737, 0.4292],
        "note": "MainBody lod1 visual mesh OBJ export.",
    },
    "car_primitives": {
        "size": [2.1326, 4.0792, 1.1792],
        "center": [0.0, 0.1051, -0.5795],
        "note": "ManiaPark primitive-style CPlugSolid estimate.",
    },
}


def find_attempt_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".npz":
            files.append(root)
        elif root.exists():
            files.extend(root.rglob("attempt_*.npz"))
    return sorted(set(files))


def raw_lasers_from_attempt(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if "raw_laser_distances" in data:
            raw = np.asarray(data["raw_laser_distances"], dtype=np.float32)
        elif "observations" in data:
            observations = np.asarray(data["observations"], dtype=np.float32)
            if observations.ndim != 2 or observations.shape[1] < Car.NUM_LASERS:
                raise ValueError(f"{path} has invalid observations shape {observations.shape}.")
            raw = observations[:, : Car.NUM_LASERS] * float(Car.LASER_MAX_DISTANCE)
        else:
            raise ValueError(f"{path} has neither raw_laser_distances nor observations.")
    if raw.ndim != 2 or raw.shape[1] < Car.NUM_LASERS:
        raise ValueError(f"{path} has invalid laser array shape {raw.shape}.")
    return raw[:, : Car.NUM_LASERS].astype(np.float32, copy=False)


def summarize_lasers(raw: np.ndarray) -> dict[str, np.ndarray]:
    raw = np.asarray(raw, dtype=np.float32)
    stats = {
        "min": np.min(raw, axis=0),
        "q001": np.quantile(raw, 0.001, axis=0),
        "q01": np.quantile(raw, 0.01, axis=0),
        "q05": np.quantile(raw, 0.05, axis=0),
        "median": np.median(raw, axis=0),
    }
    for key, values in list(stats.items()):
        mirrored = values.copy()
        for idx in range(Car.NUM_LASERS):
            other = Car.NUM_LASERS - 1 - idx
            mirrored[idx] = min(float(values[idx]), float(values[other]))
        stats[f"sym_{key}"] = mirrored
    return stats


def write_summary_csv(path: Path, angles: np.ndarray, stats: dict[str, np.ndarray], hitbox: VehicleHitbox) -> None:
    offsets = hitbox.laser_offsets_2d(Car.NUM_LASERS, Car.ANGLE)
    fieldnames = [
        "laser_index",
        "angle_deg",
        "aabb_offset",
        "raw_min",
        "raw_q001",
        "raw_q01",
        "raw_q05",
        "raw_median",
        "sym_min",
        "sym_q001",
        "sym_q01",
        "sym_q05",
        "sym_median",
        "sym_min_minus_aabb",
        "sym_q01_minus_aabb",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(Car.NUM_LASERS):
            writer.writerow(
                {
                    "laser_index": idx,
                    "angle_deg": float(angles[idx]),
                    "aabb_offset": float(offsets[idx]),
                    "raw_min": float(stats["min"][idx]),
                    "raw_q001": float(stats["q001"][idx]),
                    "raw_q01": float(stats["q01"][idx]),
                    "raw_q05": float(stats["q05"][idx]),
                    "raw_median": float(stats["median"][idx]),
                    "sym_min": float(stats["sym_min"][idx]),
                    "sym_q001": float(stats["sym_q001"][idx]),
                    "sym_q01": float(stats["sym_q01"][idx]),
                    "sym_q05": float(stats["sym_q05"][idx]),
                    "sym_median": float(stats["sym_median"][idx]),
                    "sym_min_minus_aabb": float(stats["sym_min"][idx] - offsets[idx]),
                    "sym_q01_minus_aabb": float(stats["sym_q01"][idx] - offsets[idx]),
                }
            )


def maybe_write_plot(path: Path, angles: np.ndarray, stats: dict[str, np.ndarray], hitbox: VehicleHitbox) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    offsets = hitbox.laser_offsets_2d(Car.NUM_LASERS, Car.ANGLE)
    plt.figure(figsize=(10, 5))
    plt.plot(angles, offsets, marker="o", label="AABB offset")
    plt.plot(angles, stats["sym_min"], marker="o", label="sym min supervised raw")
    plt.plot(angles, stats["sym_q01"], marker="o", label="sym q01 supervised raw")
    plt.plot(angles, stats["sym_q05"], marker="o", label="sym q05 supervised raw")
    plt.xlabel("Laser angle relative to car forward [deg]")
    plt.ylabel("Distance from car center [world units]")
    plt.title("Vehicle AABB support vs supervised near-contact lidar distances")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def write_report(
    path: Path,
    *,
    attempt_files: list[Path],
    frame_count: int,
    stats: dict[str, np.ndarray],
    hitbox: VehicleHitbox,
    plot_written: bool,
) -> None:
    offsets = hitbox.laser_offsets_2d(Car.NUM_LASERS, Car.ANGLE)
    margins = stats["sym_min"] - offsets
    q01_margins = stats["sym_q01"] - offsets
    report = f"""# Vehicle AABB Hitbox Analysis

## Inputs
- Attempt files: `{len(attempt_files)}`
- Frames: `{frame_count}`
- Raw laser source: `raw_laser_distances` when available, otherwise reconstructed as `observations[:, :15] * {Car.LASER_MAX_DISTANCE}`
- Mesh estimates:
```json
{json.dumps(MESH_ESTIMATES, indent=2)}
```

## Selected Empirical AABB
```json
{json.dumps(hitbox.as_dict(), indent=2)}
```

## Key Checks
- Minimum supervised-minus-AABB margin: `{float(np.min(margins)):.3f}`
- Median supervised-minus-AABB margin: `{float(np.median(margins)):.3f}`
- Minimum q01 supervised-minus-AABB margin: `{float(np.min(q01_margins)):.3f}`
- Median q01 supervised-minus-AABB margin: `{float(np.median(q01_margins)):.3f}`
- Plot written: `{bool(plot_written)}`

## Interpretation
The current AABB is intentionally empirical. It is based on the primitive mesh
footprint and checked against supervised near-contact lidar distances. The
supervised minima should not be treated as exact collision truth because the
latest near-contact run finished successfully; they are mainly a sanity check
that the selected AABB does not exceed observed close-pass distances.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Trackmania vehicle AABB lidar offsets.")
    parser.add_argument(
        "--data-roots",
        nargs="+",
        default=["logs/supervised_data"],
        help="Supervised data roots or direct attempt_*.npz files.",
    )
    parser.add_argument("--output-dir", default="Experiments/analysis/vehicle_hitbox")
    args = parser.parse_args()

    roots = [Path(value) for value in args.data_roots]
    attempt_files = find_attempt_files(roots)
    if not attempt_files:
        raise FileNotFoundError(f"No attempt_*.npz files found under {args.data_roots}.")

    raw_arrays = [raw_lasers_from_attempt(path) for path in attempt_files]
    raw = np.concatenate(raw_arrays, axis=0)
    stats = summarize_lasers(raw)
    hitbox = VehicleHitbox.stadium_car_sport()
    angles = np.degrees(VehicleHitbox.laser_angles_radians(Car.NUM_LASERS, Car.ANGLE))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(output_dir / "vehicle_hitbox_laser_summary.csv", angles, stats, hitbox)
    plot_written = maybe_write_plot(output_dir / "vehicle_hitbox_laser_summary.png", angles, stats, hitbox)
    write_report(
        output_dir / "REPORT.md",
        attempt_files=attempt_files,
        frame_count=int(raw.shape[0]),
        stats=stats,
        hitbox=hitbox,
        plot_written=plot_written,
    )
    print(f"Analyzed {len(attempt_files)} attempts / {raw.shape[0]} frames.")
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
