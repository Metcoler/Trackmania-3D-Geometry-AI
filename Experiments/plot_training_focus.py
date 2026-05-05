from __future__ import annotations

import argparse
from pathlib import Path

from training_focus_plots import load_focus_run, plot_focus_progress, write_focus_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Create focus training progress plots for one run or an A/B test.")
    parser.add_argument("--run-dir", action="append", required=True, help="Run directory containing generation_metrics.csv.")
    parser.add_argument("--label", action="append", default=None, help="Optional label. Repeat once per --run-dir.")
    parser.add_argument("--output-dir", default="Experiments/analysis/training_focus", help="Output directory.")
    parser.add_argument("--title", default="", help="Optional figure title.")
    parser.add_argument("--density", choices=["auto", "always", "off"], default="auto")
    args = parser.parse_args()

    labels = list(args.label or [])
    if labels and len(labels) != len(args.run_dir):
        raise ValueError("--label must be repeated exactly once per --run-dir, or omitted entirely.")

    runs = [
        load_focus_run(run_dir, label=labels[index] if labels else None)
        for index, run_dir in enumerate(args.run_dir)
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_focus_progress(
        runs,
        output_dir / "focus_progress.png",
        title=args.title or None,
        density=args.density,
    )
    summary_path = write_focus_summary(runs, output_dir / "focus_summary.csv")
    print(f"Wrote {plot_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
