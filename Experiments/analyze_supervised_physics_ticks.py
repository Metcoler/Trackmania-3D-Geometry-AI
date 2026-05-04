from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def write_tick_distribution_plot(
    output_dir: Path,
    values: np.ndarray,
    counts: np.ndarray,
    probs: np.ndarray,
    dt_ref: float,
) -> Path:
    hz_values = np.asarray([1.0 / (float(value) * float(dt_ref)) for value in values], dtype=np.float64)
    delay_values = 1.0 - (1.0 / values.astype(np.float64))
    labels = [
        f"{int(value)} tick\n{hz:.1f} Hz\ndelay={delay:.3f}"
        for value, hz, delay in zip(values, hz_values, delay_values)
    ]
    percentages = probs * 100.0
    skip_mask = values > 1
    skip_percent = float(np.sum(probs[skip_mask]) * 100.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    fig.suptitle("Empirická distribúcia Trackmania physics tickov zo supervised dát", fontsize=13)

    color_main = "#2f6f73"
    color_skip = "#c76637"
    colors = [color_skip if int(value) > 1 else color_main for value in values]

    axes[0].bar(labels, percentages, color=colors, edgecolor="#1f2d2e", linewidth=0.8)
    axes[0].set_ylabel("Podiel frame delta [%]")
    axes[0].set_title("Celá distribúcia")
    axes[0].set_ylim(0.0, max(100.0, float(np.max(percentages)) * 1.08))
    axes[0].grid(axis="y", alpha=0.25)
    for idx, percentage in enumerate(percentages):
        axes[0].text(idx, percentage + 1.0, f"{percentage:.2f}%", ha="center", va="bottom", fontsize=9)

    if np.any(skip_mask):
        skipped_labels = [labels[idx] for idx, flag in enumerate(skip_mask) if flag]
        skipped_percentages = percentages[skip_mask]
        axes[1].bar(skipped_labels, skipped_percentages, color=color_skip, edgecolor="#1f2d2e", linewidth=0.8)
        axes[1].set_ylim(0.0, max(0.25, float(np.max(skipped_percentages)) * 1.25))
        for idx, percentage in enumerate(skipped_percentages):
            axes[1].text(idx, percentage + 0.03, f"{percentage:.3f}%", ha="center", va="bottom", fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "Bez vynechaných physics tickov", ha="center", va="center")
        axes[1].set_xticks([])
    axes[1].set_ylabel("Podiel zo všetkých frame delta [%]")
    axes[1].set_title("Zoom na vynechané physics update-y")
    axes[1].grid(axis="y", alpha=0.25)

    note = (
        f"Vynechaný physics update sa v dátach objavil v {skip_percent:.2f}% frame delta. "
        "Preto variable-TM2D testy treba interpretovať cez physics tick delay, nie render FPS."
    )
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=10)

    output_path = output_dir / "physics_tick_distribution_thesis.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


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

    plot_path = write_tick_distribution_plot(
        output_dir=output_dir,
        values=values,
        counts=counts,
        probs=probs,
        dt_ref=float(args.dt_ref),
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
        f"- Thesis plot: `{plot_path.name}`",
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
            "",
            "Elite-cache motivation: fixed-100 Hz TM2D experiments are deterministic, but live Trackmania can miss "
            "physics updates. Re-evaluating an elite in a different tick sequence can therefore change the outcome "
            "even when the genotype did not change. This does not prove that elite cache is always better, but it "
            "explains why it is a meaningful training variant to test under variable physics timing.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote {summary_path}")
    print(f"Recommended --physics-tick-probs \"{profile}\"")


if __name__ == "__main__":
    main()
