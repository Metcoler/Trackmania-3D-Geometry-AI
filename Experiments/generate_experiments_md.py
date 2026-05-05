from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "Diplomová práca" / "Experiments" / "EXPERIMENTS.md"


@dataclass(frozen=True)
class ReportEntry:
    title: str
    status: str
    source: str
    note: str


REPORTS: tuple[ReportEntry, ...] = (
    ReportEntry(
        title="Lexicographic Reward Sweep With AABB Lidar",
        status="thesis-grade",
        source="Diplomová práca/Experiments/lex_reward_aabb_lidar_fixed100_20260503/REPORT.md",
        note="Reward tuple selection; current best lexicographic tuple.",
    ),
    ReportEntry(
        title="Lexicographic Reward Sweep Detailed Analysis",
        status="thesis-grade detail",
        source="Diplomová práca/Experiments/lex_reward_aabb_lidar_fixed100_20260503/analysis/lex_sweep_aabb_lidar_fixed100_20260503/REPORT.md",
        note="Detailed plots and metric interpretation for the reward sweep.",
    ),
    ReportEntry(
        title="GA Hyperparameter Refinement",
        status="thesis-grade",
        source="Diplomová práca/Experiments/ga_hyperparam_refinement_20260504/REPORT.md",
        note="Refined parent/elite and mutation baseline evidence.",
    ),
    ReportEntry(
        title="GA Hyperparameter Refinement Interpretation",
        status="thesis-grade detail",
        source="Diplomová práca/Experiments/ga_hyperparam_refinement_20260504/analysis/ga_hyperparam_refinement_20260504/INTERPRETATION_SK.md",
        note="Slovak interpretation of parent/elite and mutation ranges.",
    ),
    ReportEntry(
        title="GA Mutation Grid",
        status="thesis-grade",
        source="Diplomová práca/Experiments/ga_mutation_grid_20260504/REPORT.md",
        note="Original mutation probability/sigma grid evidence.",
    ),
    ReportEntry(
        title="GA Mutation Grid Detailed Analysis",
        status="thesis-grade detail",
        source="Diplomová práca/Experiments/ga_mutation_grid_20260504/analysis/ga_hyperparam_mutation_grid_20260504/REPORT.md",
        note="Detailed mutation grid report.",
    ),
    ReportEntry(
        title="Architecture And Activation Ablation",
        status="thesis-grade",
        source="Diplomová práca/Experiments/ga_architecture_activation_ablation_20260504/REPORT.md",
        note="Closed-loop evidence for relu,tanh and 32x16/48x24 tradeoff.",
    ),
    ReportEntry(
        title="AABB Vehicle Hitbox",
        status="thesis-grade",
        source="Diplomová práca/Experiments/vehicle_hitbox_aabb_20260503/REPORT.md",
        note="Justification for AABB-clearance lidar.",
    ),
    ReportEntry(
        title="AABB Vehicle Hitbox Detailed Report",
        status="thesis-grade detail",
        source="Diplomová práca/Experiments/vehicle_hitbox_aabb_20260503/analysis/vehicle_hitbox_20260503_smoke/REPORT.md",
        note="Detailed hitbox sanity checks.",
    ),
    ReportEntry(
        title="Supervised Map Specialists",
        status="thesis-grade visualization",
        source="Diplomová práca/Experiments/supervised_map_specialists_20260505/REPORT.md",
        note="Teacher/agent path plots and supervised specialist result package.",
    ),
    ReportEntry(
        title="Latest GA Training Improvements",
        status="thesis-grade and diagnostic mix",
        source="Diplomová práca/Experiments/training_improvements_20260505/analysis/latest_training_results_20260505/REPORT.md",
        note="Elite cache, decay, mirror, max-touch, MOO and live TM summary.",
    ),
    ReportEntry(
        title="RL Reward-Equivalent Sweep",
        status="comparison / useful negative",
        source="Diplomová práca/Experiments/rl_reward_equivalent_sweep_20260505/analysis/rl_reward_equivalent_sweep_20260505/DEEP_REPORT.md",
        note="PPO vs SAC vs TD3 under reward-equivalent scalarization.",
    ),
    ReportEntry(
        title="GA Supervised-Seeded Hybrid",
        status="thesis-grade positive hybrid",
        source="Diplomová práca/Experiments/ga_supervised_seeded_20260505/analysis/ga_supervised_seeded_20260505/DEEP_REPORT.md",
        note="Behavior cloning initialization plus GA fine-tuning.",
    ),
    ReportEntry(
        title="Supervised Physics Tick Distribution",
        status="supporting diagnostic",
        source="Experiments/analysis/supervised_physics_ticks_20260504/REPORT.md",
        note="Timing evidence for physics tick delay and 100Hz/variable tick discussion.",
    ),
)


def repo_path(relative: str) -> Path:
    return REPO_ROOT / relative.replace("/", "\\")


def read_report(entry: ReportEntry) -> str:
    path = repo_path(entry.source)
    if not path.exists():
        return f"> Missing source report: `{entry.source}`\n"
    return path.read_text(encoding="utf-8").strip()


def build_index() -> str:
    lines = [
        "# Experiment Reports Index",
        "",
        "This file aggregates the relevant experiment reports used while writing the thesis.",
        "It is a working source document, not the final thesis chapter.",
        "",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Index",
        "",
        "| # | Experiment | Status | Source | Note |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for index, entry in enumerate(REPORTS, start=1):
        anchor = entry.title.lower().replace(" ", "-").replace("/", "")
        lines.append(f"| {index} | [{entry.title}](#{anchor}) | {entry.status} | `{entry.source}` | {entry.note} |")
    lines.extend(
        [
            "",
            "## Reading Notes",
            "",
            "- `progress` means dense/continuous progress unless explicitly named `block_progress`.",
            "- Reports from `Experiments/_discarded`, smoke tests, and temporary bounce/butterfly probes are intentionally excluded.",
            "- Thesis packages are preferred over duplicate working reports when both exist.",
            "",
        ]
    )
    return "\n".join(lines)


def build_document() -> str:
    sections = [build_index()]
    for index, entry in enumerate(REPORTS, start=1):
        sections.extend(
            [
                "\n---\n",
                f"## {entry.title}",
                "",
                f"- Status: `{entry.status}`",
                f"- Source: `{entry.source}`",
                f"- Note: {entry.note}",
                "",
                read_report(entry),
                "",
            ]
        )
    return "\n".join(sections).rstrip() + "\n"


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_document(), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
