from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiments.tm_map_plotting import plot_map_overview, safe_output_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Trackmania map as a 2D thesis-friendly overview.")
    parser.add_argument("--map-name", required=True)
    parser.add_argument("--output-dir", default="Experiments/analysis/map_overviews_20260505")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--legend", action="store_true", help="Add a compact surface legend below the map.")
    parser.add_argument("--no-title", action="store_true", help="Do not draw the map name above the figure.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_path = output_dir / f"{safe_output_name(args.map_name)}.png"
    saved = plot_map_overview(
        args.map_name,
        output_path,
        dpi=args.dpi,
        show_legend=bool(args.legend),
        title=not bool(args.no_title),
    )
    print(f"Saved map overview: {saved}")


if __name__ == "__main__":
    main()
