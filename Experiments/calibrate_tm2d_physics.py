from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ObservationEncoder import ObservationEncoder


def _feature_index(name: str, observation_dim: int) -> int | None:
    """Return the current canonical observation index, if it fits the sample."""

    for vertical_mode in (True, False):
        names = ObservationEncoder.feature_names(vertical_mode=vertical_mode)
        if observation_dim == len(names) and name in names:
            return names.index(name)
    names = ObservationEncoder.feature_names(vertical_mode=False)
    if name in names:
        index = names.index(name)
        if index < observation_dim:
            return index
    return None


def _legacy_feature_index(name: str, observation_dim: int) -> int | None:
    """Best-effort support for pre-5-lookahead supervised datasets.

    Historical datasets used 15 lasers + 10 path instructions, so the base
    feature block started at index 25. Current datasets use 15 + 5 and start
    at index 20. Prefer explicit raw arrays when available.
    """

    legacy = {
        "speed": 25,
        "side_speed": 26,
    }
    index = legacy.get(name)
    if index is None or index >= observation_dim:
        return None
    return index


def _read_feature(data, observations: np.ndarray, raw_key: str, feature_name: str) -> np.ndarray:
    if raw_key in data.files:
        return np.asarray(data[raw_key], dtype=np.float64)

    index = _feature_index(feature_name, observations.shape[1])
    if index is None:
        index = _legacy_feature_index(feature_name, observations.shape[1])
    if index is None:
        return np.zeros(observations.shape[0], dtype=np.float64)

    scale = 1000.0 if feature_name in {"speed", "side_speed"} else 1.0
    return np.asarray(observations[:, index], dtype=np.float64) * scale


def load_supervised_samples(data_root: str) -> dict[str, np.ndarray]:
    files = sorted(glob.glob(str(Path(data_root) / "**" / "attempts" / "attempt_*.npz"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No supervised attempt_*.npz files found under {data_root!r}.")

    rows = {
        "speed": [],
        "side_speed": [],
        "dt": [],
        "accel": [],
        "gas": [],
        "brake": [],
        "steer": [],
        "distance_speed": [],
    }
    attempt_summaries = []
    for path in files:
        data = np.load(path)
        observations = np.asarray(data["observations"], dtype=np.float64)
        actions = np.asarray(data["actions"], dtype=np.float64)
        times = np.asarray(data["game_times"], dtype=np.float64)
        distances = np.asarray(data["distances"], dtype=np.float64)
        if observations.ndim != 2:
            continue

        speed = _read_feature(data, observations, "speeds", "speed")
        side_speed = _read_feature(data, observations, "side_speeds", "side_speed")
        dt = np.diff(times)
        ds = np.diff(distances)
        valid = (
            (dt > 0.004)
            & (dt < 0.08)
            & np.isfinite(dt)
            & np.isfinite(ds)
            & np.isfinite(speed[:-1])
            & np.isfinite(speed[1:])
            & (speed[:-1] >= 0.0)
            & (speed[:-1] < 220.0)
            & (speed[1:] >= 0.0)
            & (speed[1:] < 220.0)
        )
        if not np.any(valid):
            continue

        accel = np.full_like(dt, np.nan, dtype=np.float64)
        accel[valid] = (speed[1:][valid] - speed[:-1][valid]) / dt[valid]
        valid = valid & np.isfinite(accel) & (np.abs(accel) < 400.0)
        if not np.any(valid):
            continue

        rows["speed"].append(speed[:-1][valid])
        rows["side_speed"].append(side_speed[:-1][valid])
        rows["dt"].append(dt[valid])
        rows["accel"].append(accel[valid])
        rows["gas"].append(actions[:-1][valid, 0])
        rows["brake"].append(actions[:-1][valid, 1])
        rows["steer"].append(actions[:-1][valid, 2])
        distance_speed = np.zeros(np.count_nonzero(valid), dtype=np.float64)
        distance_speed[:] = ds[valid] / dt[valid]
        rows["distance_speed"].append(distance_speed)
        attempt_summaries.append(
            {
                "path": path,
                "frames": int(len(times)),
                "finish_time": float(data.get("finish_time", [times[-1]])[0]),
                "finish_progress": float(data.get("finish_progress", [np.nan])[0]),
                "finish_dense_progress": float(
                    data.get("finish_dense_progress", data.get("finish_progress", [np.nan]))[0]
                ),
                "finish_distance": float(data.get("finish_distance", [distances[-1]])[0]),
                "finish_finished": int(data.get("finish_finished", [0])[0]),
                "finish_crashes": int(data.get("finish_crashes", [0])[0]),
            }
        )

    return {
        key: np.concatenate(value) if value else np.empty(0, dtype=np.float64)
        for key, value in rows.items()
    } | {"attempt_summaries": attempt_summaries, "files": files}


def fit_longitudinal_model(samples: dict[str, np.ndarray]) -> dict[str, float]:
    speed = samples["speed"]
    accel = samples["accel"]
    gas = samples["gas"]
    brake = samples["brake"]
    matrix = np.column_stack([np.ones_like(speed), gas, brake, speed])
    beta, *_ = np.linalg.lstsq(matrix, accel, rcond=None)
    pred = matrix @ beta
    intercept, gas_coef, brake_coef, speed_coef = [float(value) for value in beta]
    return {
        "intercept": intercept,
        "gas_accel": max(1.0, gas_coef),
        "brake_accel": max(1.0, -brake_coef),
        "drag": max(0.0, -speed_coef),
        "rolling_drag": max(0.0, -intercept),
        "rmse": float(np.sqrt(np.mean((pred - accel) ** 2))),
        "gas_equilibrium_speed": float((intercept + gas_coef) / max(1e-9, -speed_coef)),
    }


def percentile_dict(values: np.ndarray) -> dict[str, float]:
    percentiles = [1, 5, 25, 50, 75, 90, 95, 99]
    result = np.percentile(values, percentiles)
    return {f"p{p}": float(v) for p, v in zip(percentiles, result)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate TM2D physics parameters from supervised attempts.")
    parser.add_argument("--data-root", default="logs/supervised_data")
    parser.add_argument("--output", default="Experiments/physics_calibration.json")
    args = parser.parse_args()

    samples = load_supervised_samples(args.data_root)
    fit = fit_longitudinal_model(samples)
    speed_stats = percentile_dict(samples["speed"])
    distance_speed_stats = percentile_dict(samples["distance_speed"])
    side_stats = percentile_dict(samples["side_speed"])
    dt_stats = percentile_dict(samples["dt"])
    steer_abs_stats = percentile_dict(np.abs(samples["steer"]))

    suggested = {
        "min_dt": 1.0 / 120.0,
        "max_dt": 1.0 / 25.0,
        "max_speed": max(
            120.0,
            min(
                170.0,
                max(
                    speed_stats["p99"] * 1.25,
                    distance_speed_stats["p99"] * 1.05,
                ),
            ),
        ),
        "reverse_speed": 10.0,
        "gas_accel": round(fit["gas_accel"], 3),
        "brake_accel": round(fit["brake_accel"], 3),
        "drag": round(fit["drag"], 4),
        "rolling_drag": round(fit["rolling_drag"], 3),
        "lateral_grip": 5.5,
        "max_yaw_rate": 2.35,
        "car_length": 4.8,
        "car_width": 2.6,
    }
    payload = {
        "attempt_files": len(samples["files"]),
        "usable_samples": int(samples["speed"].shape[0]),
        "speed_stats_from_observation": speed_stats,
        "speed_stats_from_distance_derivative": distance_speed_stats,
        "side_speed_stats": side_stats,
        "dt_stats": dt_stats,
        "abs_steer_stats": steer_abs_stats,
        "longitudinal_fit": fit,
        "suggested_tm2d_physics_config": suggested,
        "notes": [
            "Fit model: accel ~= intercept + gas*gas_coef + brake*brake_coef + speed*speed_coef.",
            "TM2DPhysicsConfig implements this as gas_accel, brake_accel, rolling_drag and drag.",
            "New supervised datasets save raw speeds/side_speeds directly. Older datasets fall back to observation layout indices.",
            "Steering/yaw is not directly present in older supervised attempts, so max_yaw_rate remains a hand-tuned curve feasibility parameter.",
        ],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved calibration to {output_path}")


if __name__ == "__main__":
    main()
