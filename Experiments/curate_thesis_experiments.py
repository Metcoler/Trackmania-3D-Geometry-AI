from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
THESIS_EXPERIMENTS_ROOT = REPO_ROOT / "Diplomová práca" / "Experiments"
DISCARDED_ROOT = REPO_ROOT / "Experiments" / "_discarded"
MANIFEST_PATH = REPO_ROOT / "Experiments" / "curation_manifest.csv"

SKIP_FILE_NAMES = {
    "combined_individual_metrics.csv",
}
MAX_COPY_SIZE_BYTES = 95 * 1024 * 1024


@dataclass(frozen=True)
class ExperimentPackage:
    experiment_id: str
    title: str
    category: str
    status: str
    thesis_relevance: str
    interpretation: str
    analysis_sources: tuple[str, ...] = ()
    run_sources: tuple[str, ...] = ()
    script_sources: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscardCandidate:
    source: str
    reason: str
    status: str = "discard"


@dataclass
class CopyStats:
    files_copied: int = 0
    files_skipped: int = 0
    bytes_copied: int = 0
    skipped_files: list[str] = field(default_factory=list)


PACKAGES: tuple[ExperimentPackage, ...] = (
    ExperimentPackage(
        experiment_id="lex_reward_aabb_lidar_fixed100_20260503",
        title="Lexicographic reward sweep with AABB-clearance lidar",
        category="thesis_grade",
        status="single_seed_strong_signal",
        thesis_relevance=(
            "Main reward-function comparison after switching to AABB-clearance lidar, "
            "fixed FPS 100, binary gas/brake, no elite cache."
        ),
        interpretation=(
            "Current evidence favors (finished, progress, -time, -crashes) as the base "
            "lexicographic GA ranking tuple."
        ),
        analysis_sources=("Experiments/analysis/lex_sweep_aabb_lidar_fixed100_20260503",),
        run_sources=("Experiments/runs_ga/lex_sweep_aabb_lidar_fixed100_seed_2026050306",),
        script_sources=(
            "Experiments/run_lexicographic_sweep_aabb_lidar_fixed100.ps1",
            "Experiments/run_lexicographic_sweep_aabb_lidar_fixed100_2.ps1",
            "Experiments/analyze_lexicographic_sweep.py",
        ),
        keywords=("lexicographic_ga", "reward_sweep", "aabb_lidar", "fixed100"),
    ),
    ExperimentPackage(
        experiment_id="ga_hyperparam_refinement_20260504",
        title="GA hyperparameter refinement sweep",
        category="thesis_grade",
        status="screening_single_seed_complete",
        thesis_relevance=(
            "Refined parent/elite and mutation probability/sigma screening for the selected "
            "reward tuple, using direct parent_count x elite_count heatmaps."
        ),
        interpretation=(
            "Best practical baseline is population=48, parent_count=14, elite_count=1, "
            "mutation_prob=0.10, mutation_sigma=0.25. The older coarse selection grid is "
            "superseded by this refinement."
        ),
        analysis_sources=("Experiments/analysis/ga_hyperparam_refinement_20260504",),
        run_sources=(
            "Experiments/runs_ga_hyperparam/pc1_selection_refinement_seed_2026050401",
            "Experiments/runs_ga_hyperparam/pc2_mutation_refinement_seed_2026050401",
        ),
        script_sources=(
            "Experiments/run_ga_hyperparam_selection_refinement_pc1.ps1",
            "Experiments/run_ga_hyperparam_mutation_refinement_pc2.ps1",
            "Experiments/analyze_ga_hyperparam_sweep.py",
        ),
        keywords=("ga_hyperparameters", "selection_pressure", "mutation_sigma", "mutation_probability"),
    ),
    ExperimentPackage(
        experiment_id="ga_mutation_grid_20260504",
        title="GA mutation probability and sigma grid",
        category="thesis_grade",
        status="screening_single_seed",
        thesis_relevance=(
            "Mutation probability/sigma screening for the selected reward tuple and "
            "baseline selection settings."
        ),
        interpretation=(
            "Best region suggests mutating fewer weights with medium-to-larger steps, "
            "around mutation_prob=0.10 and mutation_sigma=0.25-0.30."
        ),
        analysis_sources=("Experiments/analysis/ga_hyperparam_mutation_grid_20260504",),
        run_sources=("Experiments/runs_ga_hyperparam/pc1_mutation_grid_seed_2026050311",),
        script_sources=(
            "Experiments/run_ga_hyperparam_mutation_grid_pc1.ps1",
            "Experiments/run_ga_hyperparam_mutation_grid_pc2_prob25_30.ps1",
            "Experiments/analyze_ga_hyperparam_sweep.py",
        ),
        keywords=("ga_hyperparameters", "mutation_probability", "mutation_sigma"),
    ),
    ExperimentPackage(
        experiment_id="vehicle_hitbox_aabb_20260503",
        title="Vehicle AABB hitbox analysis",
        category="thesis_grade",
        status="empirical_model",
        thesis_relevance=(
            "Justification for replacing a global raw lidar threshold with AABB-relative "
            "clearance lidar."
        ),
        interpretation=(
            "The selected AABB is empirical, derived from mesh estimates and sanity-checked "
            "against near-contact supervised data."
        ),
        analysis_sources=("Experiments/analysis/vehicle_hitbox_20260503_smoke",),
        script_sources=("Experiments/analyze_vehicle_hitbox.py",),
        keywords=("aabb_lidar", "vehicle_hitbox", "observation_design"),
    ),
    ExperimentPackage(
        experiment_id="ga_architecture_activation_ablation_20260504",
        title="Closed-loop architecture activation ablation",
        category="thesis_grade",
        status="closed_loop_single_seed",
        thesis_relevance=(
            "Closed-loop GA comparison of 32x16/48x24 and relu,tanh versus relu,relu "
            "under the selected reward tuple."
        ),
        interpretation=(
            "relu,tanh outperformed relu,relu in the closed-loop ablation; 48x24 is the "
            "stronger candidate, while 32x16 remains the cheaper experimental baseline."
        ),
        analysis_sources=("Experiments/analysis/architecture_ablation_debug_20260504",),
        run_sources=(
            "Experiments/runs_ga_architecture_ablation/fixed100_lidar_finished_progress_time_crashes_seed_2026050314",
        ),
        script_sources=("Experiments/train_ga.py",),
        keywords=("architecture", "activation", "relu_tanh", "closed_loop_ga"),
    ),
)


WORKING_SOURCES: tuple[tuple[str, str], ...] = (
    ("Experiments/runs_rl", "RL comparison branch; keep until thesis role is decided."),
)


DISCARD_CANDIDATES: tuple[DiscardCandidate, ...] = (
    DiscardCandidate(
        source="Experiments/runs_smoke",
        reason="Smoke tests only; not thesis evidence.",
    ),
    DiscardCandidate(
        source="Experiments/analysis/lexicographic_sweep_20260502",
        reason="Old fair sweep with obsolete raw lidar threshold; kept only as historical context.",
    ),
    DiscardCandidate(
        source="Experiments/runs_ga/lex_sweep_seed_2026050201",
        reason="Raw runs for obsolete raw-threshold lidar sweep.",
    ),
    DiscardCandidate(
        source="Experiments/runs_ga/lex_sweep_seed_2026050202",
        reason="Raw runs for obsolete raw-threshold lidar sweep.",
    ),
    DiscardCandidate(
        source="Experiments/runs_ga_ablation",
        reason="Internal diagnostic ablations, not final thesis evidence.",
    ),
    DiscardCandidate(
        source="Experiments/runs_ga_diagnostic",
        reason="Internal diagnostic runs used to find FPS/lidar issues, not final thesis evidence.",
    ),
    DiscardCandidate(
        source="Experiments/analysis/ga_hyperparam_selection_grid_20260503",
        reason="Superseded coarse selection-pressure analysis; replaced by refined parent_count x elite_count grid.",
    ),
    DiscardCandidate(
        source="Experiments/runs_ga_hyperparam/pc2_selection_grid_seed_2026050311",
        reason="Superseded coarse selection-pressure runs; replaced by refined parent_count x elite_count grid.",
    ),
)


def repo_path(relative_path: str) -> Path:
    return REPO_ROOT / relative_path.replace("/", "\\")


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def iter_files(source: Path) -> Iterable[Path]:
    if source.is_file():
        yield source
    elif source.is_dir():
        yield from (path for path in source.rglob("*") if path.is_file())


def should_skip_file(path: Path) -> bool:
    if path.name in SKIP_FILE_NAMES:
        return True
    try:
        return path.stat().st_size > MAX_COPY_SIZE_BYTES
    except OSError:
        return True


def copy_source(source: Path, destination: Path, *, dry_run: bool) -> CopyStats:
    stats = CopyStats()
    if not source.exists():
        stats.files_skipped += 1
        stats.skipped_files.append(f"missing:{relative_to_repo(source)}")
        return stats

    for file_path in iter_files(source):
        if should_skip_file(file_path):
            stats.files_skipped += 1
            stats.skipped_files.append(relative_to_repo(file_path))
            continue

        relative = file_path.name if source.is_file() else file_path.relative_to(source)
        target = destination / relative
        stats.files_copied += 1
        stats.bytes_copied += file_path.stat().st_size
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target)
    return stats


def write_text(path: Path, content: str, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def package_report(package: ExperimentPackage, copied_at: str) -> str:
    return (
        f"# {package.title}\n\n"
        f"## Status\n\n"
        f"- Experiment ID: `{package.experiment_id}`\n"
        f"- Category: `{package.category}`\n"
        f"- Status: `{package.status}`\n"
        f"- Curated at: `{copied_at}`\n\n"
        f"## Thesis Relevance\n\n{package.thesis_relevance}\n\n"
        f"## Interpretation\n\n{package.interpretation}\n\n"
        "## Package Contents\n\n"
        "- `runs/`: selected raw run logs copied from the working experiment folders.\n"
        "- `analysis/`: summary tables, reports, and generated plots.\n"
        "- `scripts/`: scripts needed to reproduce or analyze the experiment.\n"
        "- `metadata.json`: source paths, keywords, and curation metadata.\n\n"
        "## Notes\n\n"
        "Large aggregate files such as `combined_individual_metrics.csv` are intentionally "
        "not copied into the thesis package. They remain local working artifacts if needed.\n"
    )


def merge_report(package_dir: Path, package: ExperimentPackage, *, dry_run: bool) -> None:
    report_path = package_dir / "REPORT.md"
    if report_path.exists() and not dry_run:
        return
    write_text(report_path, package_report(package, datetime.now().isoformat(timespec="seconds")), dry_run=dry_run)


def curate_package(package: ExperimentPackage, *, dry_run: bool) -> dict:
    package_dir = THESIS_EXPERIMENTS_ROOT / package.experiment_id
    stats = CopyStats()
    copied_at = datetime.now().isoformat(timespec="seconds")

    for source_name in package.analysis_sources:
        source = repo_path(source_name)
        target = package_dir / "analysis" / source.name
        source_stats = copy_source(source, target, dry_run=dry_run)
        stats.files_copied += source_stats.files_copied
        stats.files_skipped += source_stats.files_skipped
        stats.bytes_copied += source_stats.bytes_copied
        stats.skipped_files.extend(source_stats.skipped_files)

    for source_name in package.run_sources:
        source = repo_path(source_name)
        target = package_dir / "runs" / source.name
        source_stats = copy_source(source, target, dry_run=dry_run)
        stats.files_copied += source_stats.files_copied
        stats.files_skipped += source_stats.files_skipped
        stats.bytes_copied += source_stats.bytes_copied
        stats.skipped_files.extend(source_stats.skipped_files)

    for source_name in package.script_sources:
        source = repo_path(source_name)
        target = package_dir / "scripts" / source.name
        source_stats = copy_source(source, target, dry_run=dry_run)
        stats.files_copied += source_stats.files_copied
        stats.files_skipped += source_stats.files_skipped
        stats.bytes_copied += source_stats.bytes_copied
        stats.skipped_files.extend(source_stats.skipped_files)

    merge_report(package_dir, package, dry_run=dry_run)
    metadata = {
        "experiment_id": package.experiment_id,
        "title": package.title,
        "category": package.category,
        "status": package.status,
        "thesis_relevance": package.thesis_relevance,
        "interpretation": package.interpretation,
        "keywords": list(package.keywords),
        "analysis_sources": list(package.analysis_sources),
        "run_sources": list(package.run_sources),
        "script_sources": list(package.script_sources),
        "curated_at": copied_at,
        "copy_policy": {
            "skipped_file_names": sorted(SKIP_FILE_NAMES),
            "max_copy_size_bytes": MAX_COPY_SIZE_BYTES,
        },
        "skipped_files": stats.skipped_files,
    }
    write_text(
        package_dir / "metadata.json",
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        dry_run=dry_run,
    )

    return {
        "action": "package",
        "status": package.status,
        "experiment_id": package.experiment_id,
        "source": ";".join(package.analysis_sources + package.run_sources + package.script_sources),
        "destination": relative_to_repo(package_dir),
        "reason": package.thesis_relevance,
        "files_copied": stats.files_copied,
        "files_skipped": stats.files_skipped,
        "mb_copied": round(stats.bytes_copied / (1024 * 1024), 3),
    }


def archive_discard(candidate: DiscardCandidate, *, dry_run: bool, hard_delete: bool, timestamp: str) -> dict:
    source = repo_path(candidate.source)
    destination = DISCARDED_ROOT / timestamp / source.name
    exists = source.exists()
    if exists and not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if hard_delete:
            if source.is_dir():
                shutil.rmtree(source)
            else:
                source.unlink()
        else:
            if destination.exists():
                shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
            shutil.move(str(source), str(destination))
    return {
        "action": "hard_delete" if hard_delete else "archive_discard",
        "status": candidate.status if exists else "missing",
        "experiment_id": "",
        "source": candidate.source,
        "destination": "" if hard_delete else relative_to_repo(destination),
        "reason": candidate.reason,
        "files_copied": "",
        "files_skipped": "",
        "mb_copied": "",
    }


def working_row(source: str, reason: str) -> dict:
    exists = repo_path(source).exists()
    return {
        "action": "keep_working",
        "status": "working" if exists else "missing",
        "experiment_id": "",
        "source": source,
        "destination": "",
        "reason": reason,
        "files_copied": "",
        "files_skipped": "",
        "mb_copied": "",
    }


def write_manifest(rows: list[dict], *, dry_run: bool, output_path: Path) -> None:
    fieldnames = [
        "action",
        "status",
        "experiment_id",
        "source",
        "destination",
        "reason",
        "files_copied",
        "files_skipped",
        "mb_copied",
    ]
    if dry_run:
        output_path = output_path.with_name(output_path.stem + "_dry_run.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate experiment artifacts into thesis-grade packages.")
    parser.add_argument("--dry-run", action="store_true", help="Only write a dry-run manifest; do not copy or move artifacts.")
    parser.add_argument("--apply", action="store_true", help="Create thesis packages.")
    parser.add_argument("--archive-discarded", action="store_true", help="Soft-move discard candidates into Experiments/_discarded/<timestamp>.")
    parser.add_argument("--hard-delete", action="store_true", help="Permanently delete discard candidates. Requires --archive-discarded.")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="Path for the curation manifest CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hard_delete and not args.archive_discarded:
        raise SystemExit("--hard-delete requires --archive-discarded")
    if not args.dry_run and not args.apply and not args.archive_discarded:
        raise SystemExit("Use --dry-run first, or pass --apply and/or --archive-discarded.")

    dry_run = bool(args.dry_run)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: list[dict] = []

    for package in PACKAGES:
        if args.apply or dry_run:
            rows.append(curate_package(package, dry_run=dry_run or not args.apply))

    for source, reason in WORKING_SOURCES:
        rows.append(working_row(source, reason))

    for candidate in DISCARD_CANDIDATES:
        if args.archive_discarded or dry_run:
            rows.append(
                archive_discard(
                    candidate,
                    dry_run=dry_run or not args.archive_discarded,
                    hard_delete=bool(args.hard_delete),
                    timestamp=timestamp,
                )
            )

    write_manifest(rows, dry_run=dry_run, output_path=Path(args.manifest))
    manifest_path = Path(args.manifest)
    if dry_run:
        manifest_path = manifest_path.with_name(manifest_path.stem + "_dry_run.csv")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
