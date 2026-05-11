from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiments.tm_map_plotting import render_map_background
from Map import Map


RUNS = [
    {
        "key": "single_surface_flat",
        "label": "Rovinná trať",
        "description": "Rovinná trať s jedným povrchom",
        "run_dir": Path(
            "logs/tm_finetune_runs/"
            "20260510_092555_tm_finetune_map_single_surface_flat_v2d_asphalt_h48x24_p48_src_resume_population_gen_0170"
        ),
        "generation_run_dirs": [
            Path(
                "logs/tm_finetune_runs/"
                "20260506_004011_tm_seed_map_single_surface_flat_v2d_asphalt_h48x24_p48_src_best_model"
            ),
            Path(
                "logs/tm_finetune_runs/"
                "20260509_144133_tm_finetune_map_single_surface_flat_v2d_asphalt_h48x24_p48_src_resume_population_gen_0110"
            ),
            Path(
                "logs/tm_finetune_runs/"
                "20260510_024403_tm_finetune_map_single_surface_flat_v2d_asphalt_h48x24_p48_src_resume_population_gen_0150"
            ),
            Path(
                "logs/tm_finetune_runs/"
                "20260510_092555_tm_finetune_map_single_surface_flat_v2d_asphalt_h48x24_p48_src_resume_population_gen_0170"
            ),
        ],
    },
    {
        "key": "single_surface_height",
        "label": "Výškové rozdiely",
        "description": "Trať s výškovými rozdielmi",
        "run_dir": Path(
            "logs/tm_finetune_runs/"
            "20260506_160030_tm_seed_map_single_surface_height_v3d_asphalt_h48x24_p48_src_best_model"
        ),
    },
    {
        "key": "multi_surface_flat",
        "label": "Rôzne povrchy",
        "description": "Rovinná trať s rôznymi povrchmi",
        "run_dir": Path(
            "logs/tm_finetune_runs/"
            "20260507_090226_tm_seed_map_multi_surface_flat_v2d_surface_h48x24_p48_src_best_model"
        ),
    },
]


OUTPUT_DIR = ROOT / "Diplomová práca" / "Experiments" / "final_ga_training_20260507"
IMAGE_DIR = ROOT / "Diplomová práca" / "Latex" / "images" / "training_policy"


@dataclass
class RunData:
    key: str
    label: str
    description: str
    run_dir: Path
    config: dict[str, Any]
    generation: pd.DataFrame
    individual: pd.DataFrame
    manifest: list[dict[str, str]]
    best_trajectory_row: dict[str, str]
    best_trajectory: dict[str, np.ndarray]
    summary: dict[str, Any]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _to_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _choose_best_trajectory(rows: list[dict[str, str]]) -> dict[str, str]:
    finishers = [row for row in rows if _to_int(row.get("finished")) > 0]
    if finishers:
        return min(finishers, key=lambda row: _to_float(row.get("time"), float("inf")))
    return max(rows, key=lambda row: _to_float(row.get("dense_progress"), -1.0))


def _load_trajectory(run_dir: Path, row: dict[str, str]) -> dict[str, np.ndarray]:
    path = run_dir / str(row["path_file"])
    data = np.load(path)
    return {key: np.asarray(data[key]) for key in data.files}


def _read_generation_history(spec: dict[str, Any]) -> pd.DataFrame:
    run_dirs = spec.get("generation_run_dirs")
    if not run_dirs:
        return _read_csv((ROOT / spec["run_dir"]) / "generation_summary.csv")

    frames: list[pd.DataFrame] = []
    for order, relative in enumerate(run_dirs):
        path = ROOT / relative / "generation_summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["_source_order"] = order
        frames.append(frame)

    if not frames:
        return _read_csv((ROOT / spec["run_dir"]) / "generation_summary.csv")

    merged = pd.concat(frames, ignore_index=True)
    merged["generation"] = pd.to_numeric(merged["generation"], errors="coerce")
    merged = merged.dropna(subset=["generation"])
    merged = merged.sort_values(["generation", "_source_order"])
    merged = merged.drop_duplicates(subset=["generation"], keep="last")
    merged = merged.drop(columns=["_source_order"])
    return merged.sort_values("generation").reset_index(drop=True)


def _load_run(spec: dict[str, Any]) -> RunData:
    run_dir = ROOT / spec["run_dir"]
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    generation = _read_generation_history(spec)
    individual = _read_csv(run_dir / "individual_metrics.csv")
    manifest = _read_manifest(run_dir / "trajectories" / "trajectory_manifest.csv")
    best_row = _choose_best_trajectory(manifest)
    trajectory = _load_trajectory(run_dir, best_row)

    finishes = individual[pd.to_numeric(individual["finished"], errors="coerce").fillna(0) > 0]
    finish_count_total = int(len(finishes))
    first_finish_generation = None
    best_time = None
    generation_finish = pd.to_numeric(generation.get("finish_count"), errors="coerce").fillna(0)
    if generation_finish.gt(0).any():
        first_finish_generation = int(
            pd.to_numeric(generation.loc[generation_finish.gt(0), "generation"], errors="coerce").min()
        )
    if finish_count_total > 0:
        best_time = float(pd.to_numeric(finishes["time"], errors="coerce").min())

    best_progress = float(pd.to_numeric(individual["progress"], errors="coerce").max())
    generation_count = int(len(generation))
    final_generation = int(pd.to_numeric(generation["generation"], errors="coerce").max())
    final_row = generation.sort_values("generation").iloc[-1]

    summary = {
        "key": spec["key"],
        "description": spec["description"],
        "run_dir": str(spec["run_dir"]),
        "map_name": config.get("map_name"),
        "obs_dim": int(config.get("obs_dim")),
        "hidden_dim": "x".join(str(value) for value in config.get("hidden_dim", [])),
        "hidden_activation": ",".join(str(value) for value in config.get("hidden_activation", [])),
        "population": int(config.get("pop_size")),
        "parents": int(config.get("parent_count")),
        "elites": int(config.get("elite_count")),
        "mutation_prob": float(config.get("mutation_prob")),
        "mutation_sigma": float(config.get("mutation_sigma")),
        "ranking_key": str(config.get("ranking_key_expression") or config.get("ranking_key")),
        "generation_rows": generation_count,
        "final_generation": final_generation,
        "finish_count_total": finish_count_total,
        "first_finish_generation": first_finish_generation if first_finish_generation is not None else "",
        "best_time_s": best_time if best_time is not None else "",
        "best_progress_pct": best_progress,
        "max_finishers_in_generation": int(pd.to_numeric(generation["finish_count"], errors="coerce").max()),
        "final_finish_count": int(_to_float(final_row.get("finish_count"))),
        "final_mean_progress_pct": float(_to_float(final_row.get("mean_progress"))),
        "final_best_progress_pct": float(_to_float(final_row.get("best_progress"))),
        "best_trajectory_file": best_row["path_file"],
        "best_trajectory_generation": int(_to_float(best_row.get("generation"))),
        "best_trajectory_finished": int(_to_float(best_row.get("finished"))),
        "best_trajectory_time_s": float(_to_float(best_row.get("time"))),
        "best_trajectory_progress_pct": float(_to_float(best_row.get("dense_progress"))),
    }

    return RunData(
        key=str(spec["key"]),
        label=str(spec["label"]),
        description=str(spec["description"]),
        run_dir=run_dir,
        config=config,
        generation=generation,
        individual=individual,
        manifest=manifest,
        best_trajectory_row=best_row,
        best_trajectory=trajectory,
        summary=summary,
    )


def _normalized_x(df: pd.DataFrame) -> np.ndarray:
    generation = pd.to_numeric(df["generation"], errors="coerce").to_numpy(dtype=np.float64)
    if generation.size == 0:
        return generation
    low = float(np.nanmin(generation))
    high = float(np.nanmax(generation))
    if high <= low:
        return np.zeros_like(generation)
    return (generation - low) / (high - low) * 100.0


def _plot_progress(runs: list[RunData]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9), sharey=True)
    colors = {
        "best": "#1d4ed8",
        "mean": "#64748b",
        "finish": "#dc2626",
    }
    for axis, run in zip(axes, runs):
        df = run.generation.sort_values("generation")
        x = _normalized_x(df)
        best = pd.to_numeric(df["best_progress"], errors="coerce").to_numpy(dtype=np.float64)
        mean = pd.to_numeric(df["mean_progress"], errors="coerce").to_numpy(dtype=np.float64)
        finish_count = pd.to_numeric(df["finish_count"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)

        axis.plot(x, best, color=colors["best"], linewidth=2.25, label="najlepší progres")
        axis.plot(x, mean, color=colors["mean"], linewidth=1.8, linestyle="--", label="priemer populácie")
        first_finish = run.summary["first_finish_generation"]
        if first_finish != "":
            generations = pd.to_numeric(df["generation"], errors="coerce").to_numpy(dtype=np.float64)
            finish_x = x[np.where(generations == float(first_finish))[0][0]]
            axis.axvline(finish_x, color=colors["finish"], linewidth=1.35, linestyle=":", label="prvé dokončenie")
        if np.nanmax(finish_count) > 0:
            finish_scaled = finish_count / max(float(run.config.get("pop_size", 48)), 1.0) * 100.0
            axis.fill_between(x, 0, finish_scaled, color="#16a34a", alpha=0.12, linewidth=0, label="podiel dokončení")

        axis.set_title(run.label, fontsize=12)
        axis.set_xlabel("Relatívny priebeh tréningu [%]")
        axis.set_ylim(0, 103)
        axis.grid(True, alpha=0.22)

    axes[0].set_ylabel("Progres [%]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Finálne tréningové behy genetického algoritmu", fontsize=14, y=0.99)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.82, bottom=0.22, wspace=0.12)
    output = IMAGE_DIR / "final_ga_training_progress.pdf"
    fig.savefig(output, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    return output


def _project_points(projection: Any, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.column_stack([x, z])
    if projection is not None:
        points = projection.points(points)
    return points[:, 0], points[:, 1]


def _plot_trajectories(runs: list[RunData]) -> Path:
    all_speeds_kmh: list[np.ndarray] = []
    for run in runs:
        speed = np.asarray(run.best_trajectory["speed"], dtype=np.float64) * 3.6
        if speed.size:
            all_speeds_kmh.append(speed)
    speed_values = np.concatenate(all_speeds_kmh) if all_speeds_kmh else np.asarray([0.0])
    vmax = max(180.0, float(np.nanpercentile(speed_values, 98)))
    norm = Normalize(vmin=100.0, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.6))
    collection_for_colorbar = None
    for axis, run in zip(axes, runs):
        game_map = Map(str(run.config["map_name"]))
        projection = render_map_background(axis, game_map, alpha=0.86)
        data = run.best_trajectory
        x = np.asarray(data["x"], dtype=np.float64)
        z = np.asarray(data["z"], dtype=np.float64)
        px, pz = _project_points(projection, x, z)
        speed_kmh = np.asarray(data["speed"], dtype=np.float64) * 3.6

        points = np.column_stack([px, pz]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        collection = LineCollection(segments, cmap="turbo_r", norm=norm, linewidth=2.25, zorder=100)
        collection.set_array(speed_kmh[:-1] if speed_kmh.size > 1 else np.asarray([0.0]))
        axis.add_collection(collection)
        collection_for_colorbar = collection

        time_text = f"{run.summary['best_trajectory_time_s']:.2f} s".replace(".", ",")
        progress_text = f"{run.summary['best_trajectory_progress_pct']:.1f} %".replace(".", ",")
        title = f"{run.label}\nčas {time_text}, progres {progress_text}"
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_xticks([])
        axis.set_yticks([])

    block_legend = [
        Patch(facecolor="#b9b9b9", edgecolor="#555555", label="asfalt"),
        Patch(facecolor="#b7d984", edgecolor="#6b8f3a", label="tráva"),
        Patch(facecolor="#a2642f", edgecolor="#6b3d1e", label="hlina"),
        Patch(facecolor="#16a34a", edgecolor="#0f6f32", label="štart"),
        Patch(facecolor="#dc2626", edgecolor="#8f1d1d", label="cieľ"),
    ]
    fig.legend(
        handles=block_legend,
        loc="center right",
        bbox_to_anchor=(0.995, 0.48),
        frameon=True,
        fontsize=8,
        title="Bloky",
        title_fontsize=9,
        borderpad=0.55,
        labelspacing=0.45,
        handlelength=1.4,
    )

    if collection_for_colorbar is not None:
        colorbar = fig.colorbar(
            collection_for_colorbar,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            fraction=0.045,
            pad=0.04,
            aspect=52,
        )
        colorbar.set_label("Rýchlosť [km/h]  (modrá = rýchlejšie, červená = pomalšie)", fontsize=8)
        colorbar.ax.tick_params(labelsize=8)
    fig.suptitle("Najlepšie zaznamenané trajektórie finálnych behov", fontsize=14, y=0.99)
    fig.subplots_adjust(left=0.02, right=0.89, top=0.82, bottom=0.14, wspace=0.18)
    output = IMAGE_DIR / "final_ga_training_trajectories.pdf"
    fig.savefig(output, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    return output


def _write_summary(runs: list[RunData]) -> Path:
    path = OUTPUT_DIR / "summary.csv"
    rows = [run.summary for run in runs]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_metadata(runs: list[RunData], figures: list[Path]) -> Path:
    metadata = {
        "experiment_id": "final_ga_training_20260507",
        "generated_by": "prepare_thesis_figures.py",
        "purpose": "Curated final GA training runs for thesis chapter 7.",
        "source_runs": [run.summary["run_dir"] for run in runs],
        "figures": [str(path.relative_to(ROOT)) for path in figures],
    }
    path = OUTPUT_DIR / "metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _format_optional(value: Any, suffix: str = "") -> str:
    if value == "" or value is None:
        return "--"
    if isinstance(value, float):
        text = f"{value:.2f}".replace(".", ",")
    else:
        text = str(value)
    return f"{text}{suffix}"


def _write_report(runs: list[RunData], figures: list[Path]) -> Path:
    lines: list[str] = [
        "# Final GA Training Runs 20260507",
        "",
        "## Summary",
        "",
        "This curated package summarizes the three final GA training runs used at the end of thesis chapter 7.",
        "The runs demonstrate the same lexicographic neuroevolution pipeline on a flat track, a height track, and a multi-surface track.",
        "",
        "## Source Runs",
        "",
    ]
    for run in runs:
        s = run.summary
        lines.extend(
            [
                f"### {s['description']}",
                "",
                f"- Source: `{s['run_dir']}`",
                f"- Map: `{s['map_name']}`",
                f"- Observation dimension: `{s['obs_dim']}`",
                f"- Network: `{s['hidden_dim']} {s['hidden_activation']}`",
                f"- Ranking: `{s['ranking_key']}`",
                f"- Population/parents/elites: `{s['population']}/{s['parents']}/{s['elites']}`",
                f"- Mutation: `p={s['mutation_prob']}`, `sigma={s['mutation_sigma']}`",
                f"- Logged generations: `{s['generation_rows']}`",
                f"- First finish generation: `{_format_optional(s['first_finish_generation'])}`",
                f"- Total finishing individuals: `{s['finish_count_total']}`",
                f"- Best finish time: `{_format_optional(s['best_time_s'], ' s')}`",
                f"- Best progress: `{_format_optional(s['best_progress_pct'], ' %')}`",
                f"- Best trajectory: `{s['best_trajectory_file']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Generated Files",
            "",
            "- `summary.csv`",
            "- `metadata.json`",
        ]
    )
    for figure in figures:
        lines.append(f"- `{figure.relative_to(ROOT)}`")
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- The x-axis in the progress figure is normalized because the logged runs have different lengths.",
            "- These runs are final training evidence, not the final evaluation chapter.",
            "- Trajectory colors use speed in km/h: colder colors indicate faster sections and warmer colors slower sections.",
            "",
        ]
    )
    path = OUTPUT_DIR / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    runs = [_load_run(spec) for spec in RUNS]
    progress = _plot_progress(runs)
    trajectories = _plot_trajectories(runs)
    summary = _write_summary(runs)
    metadata = _write_metadata(runs, [progress, trajectories])
    report = _write_report(runs, [progress, trajectories])
    print(f"Wrote {summary}")
    print(f"Wrote {metadata}")
    print(f"Wrote {report}")
    print(f"Wrote {progress}")
    print(f"Wrote {trajectories}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
