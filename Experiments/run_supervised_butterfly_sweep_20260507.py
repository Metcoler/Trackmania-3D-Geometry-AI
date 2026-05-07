from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Experiments" / "analysis" / "supervised_butterfly_sweep_20260507"
SCRIPT = ROOT / "Experiments" / "plot_supervised_butterfly_effect.py"


CASES = [
    {
        "tag": "p00_right_80",
        "trigger": 0,
        "duration": 80,
        "args": ["--perturb-steer", "-1.0"],
        "label": "Hard-right from start block",
    },
    {
        "tag": "p00_left_80",
        "trigger": 0,
        "duration": 80,
        "args": ["--perturb-steer", "1.0"],
        "label": "Hard-left from start block",
    },
    {
        "tag": "p01_right_160",
        "trigger": 1,
        "duration": 160,
        "args": ["--perturb-steer", "-1.0"],
        "label": "Hard-right in first corner",
    },
    {
        "tag": "p01_left_160",
        "trigger": 1,
        "duration": 160,
        "args": ["--perturb-steer", "1.0"],
        "label": "Hard-left in first corner",
    },
    {
        "tag": "p02_right_120",
        "trigger": 2,
        "duration": 120,
        "args": ["--perturb-steer", "-1.0"],
        "label": "Hard-right after first bend",
    },
    {
        "tag": "p02_left_120",
        "trigger": 2,
        "duration": 120,
        "args": ["--perturb-steer", "1.0"],
        "label": "Hard-left after first bend",
    },
    {
        "tag": "p04_right_120",
        "trigger": 4,
        "duration": 120,
        "args": ["--perturb-steer", "-1.0"],
        "label": "Hard-right before upper turn",
    },
    {
        "tag": "p04_left_120",
        "trigger": 4,
        "duration": 120,
        "args": ["--perturb-steer", "1.0"],
        "label": "Hard-left before upper turn",
    },
    {
        "tag": "p01_brake_80",
        "trigger": 1,
        "duration": 80,
        "args": ["--perturb-brake", "1.0", "--perturb-gas", "0.0"],
        "label": "Brake pulse in first corner",
    },
    {
        "tag": "p01_gas_cut_120",
        "trigger": 1,
        "duration": 120,
        "args": ["--perturb-gas", "0.0", "--perturb-brake", "0.0"],
        "label": "Gas cut in first corner",
    },
]


def read_metrics(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    baseline = rows[0]
    perturbed = rows[1]
    return {
        "baseline_time": baseline["time"],
        "baseline_crashes": baseline["crashes"],
        "perturbed_time": perturbed["time"],
        "perturbed_crashes": perturbed["crashes"],
        "perturbed_finished": perturbed["finished"],
        "perturbed_progress": perturbed["progress"],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str]] = []

    for case in CASES:
        command = [
            sys.executable,
            str(SCRIPT),
            "--map-name",
            "single_surface_flat",
            "--specialist-root",
            "logs/supervised_runs_map_specialists_20260505",
            "--output-dir",
            str(OUTPUT_DIR),
            "--trigger-path-index",
            str(case["trigger"]),
            "--perturb-duration-steps",
            str(case["duration"]),
            "--output-tag",
            case["tag"],
            "--perturb-label",
            case["label"],
            *case["args"],
        ]
        print("Running", case["tag"])
        subprocess.run(command, cwd=ROOT, check=True)
        metrics_path = OUTPUT_DIR / "single_surface_flat" / f"butterfly_metrics_{case['tag']}.csv"
        metrics = read_metrics(metrics_path)
        baseline_time = float(metrics["baseline_time"])
        perturbed_time = float(metrics["perturbed_time"])
        baseline_crashes = int(float(metrics["baseline_crashes"]))
        perturbed_crashes = int(float(metrics["perturbed_crashes"]))
        summary_rows.append(
            {
                "tag": case["tag"],
                "trigger_path_index": str(case["trigger"]),
                "duration_steps": str(case["duration"]),
                "label": case["label"],
                **metrics,
                "delta_time": f"{perturbed_time - baseline_time:.6f}",
                "delta_crashes": str(perturbed_crashes - baseline_crashes),
                "image": str(OUTPUT_DIR / "single_surface_flat" / f"single_surface_flat_butterfly_{case['tag']}.png"),
            }
        )

    summary_path = OUTPUT_DIR / "butterfly_sweep_summary.csv"
    fieldnames = [
        "tag",
        "trigger_path_index",
        "duration_steps",
        "label",
        "baseline_time",
        "baseline_crashes",
        "perturbed_time",
        "perturbed_crashes",
        "perturbed_finished",
        "perturbed_progress",
        "delta_time",
        "delta_crashes",
        "image",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
