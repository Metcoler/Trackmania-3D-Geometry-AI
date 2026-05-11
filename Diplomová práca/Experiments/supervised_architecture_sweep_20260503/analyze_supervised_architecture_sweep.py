from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
RAW_DIR = REPO_ROOT / "logs" / "supervised_architecture_sweep" / "20260503_214049_v2d_asphalt_capacity_sweep"
LATEX_IMAGE_DIR = REPO_ROOT / "Masters thesis" / "Latex" / "images" / "training_policy"

ANALYSIS_DIR = PACKAGE_DIR / "analysis"
RAW_COPY_DIR = PACKAGE_DIR / "raw"

WIDE_STEM = "supervised_architecture_val_loss_vs_params"
FOCUS_STEM = "supervised_architecture_focus_32x16_48x24"
ALL_CURVES_STEM = "supervised_architecture_training_curves_all"
PRACTICAL_CURVES_STEM = "supervised_architecture_training_curves_practical"


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "parameter_count",
            "epochs",
            "best_epoch",
            "best_val_loss",
            "best_val_steer_mae",
            "best_val_gas_accuracy",
            "best_val_brake_accuracy",
            "train_samples",
            "val_samples",
        ):
            if key in row and row[key] != "":
                row[key] = float(row[key]) if "." in str(row[key]) else int(row[key])
    return rows


def save_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_epoch_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["epoch"] = int(row["epoch"])
        for key in (
            "train_loss",
            "val_loss",
            "val_steer_mae",
            "val_gas_accuracy",
            "val_brake_accuracy",
            "val_gas_bce",
            "val_brake_bce",
        ):
            row[key] = float(row[key])
    return rows


def activation_style(activation: str) -> tuple[str, str, str]:
    styles = {
        "relu,tanh": ("#1b9e77", "o", "ReLU, tanh"),
        "relu,relu": ("#377eb8", "s", "ReLU, ReLU"),
        "tanh,tanh": ("#e6ab02", "^", "tanh, tanh"),
        "sigmoid,sigmoid": ("#d95f02", "X", "sigmoid, sigmoid"),
        "relu": ("#7570b3", "D", "ReLU"),
        "tanh": ("#a6761d", "v", "tanh"),
        "sigmoid": ("#e7298a", "P", "sigmoid"),
        "relu,relu,relu": ("#66a61e", "h", "ReLU, ReLU, ReLU"),
        "relu,relu,tanh": ("#1f78b4", "p", "ReLU, ReLU, tanh"),
        "tanh,tanh,tanh": ("#b2df8a", "<", "tanh, tanh, tanh"),
        "sigmoid,sigmoid,sigmoid": ("#fb9a99", ">", "sigmoid, sigmoid, sigmoid"),
    }
    return styles.get(activation, ("#666666", "o", activation))


def candidate_label(row: dict[str, object]) -> str:
    activation = str(row["activation"]).replace(",", ", ")
    return f"{row['architecture']} {activation}"


def rows_by_candidate(epoch_rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in epoch_rows:
        grouped.setdefault(str(row["candidate_id"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["epoch"]))
    return grouped


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    for directory in (ANALYSIS_DIR, LATEX_IMAGE_DIR):
        fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(directory / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_all_training_curves(rows: list[dict[str, object]], epoch_rows: list[dict[str, object]]) -> None:
    by_id = rows_by_candidate(epoch_rows)
    by_summary = {str(row["candidate_id"]): row for row in rows}

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), sharey=False)
    legend_seen: set[str] = set()
    handles = []
    labels = []

    highlight_ids = {
        "c022_arch_32x16_act_relu_tanh",
        "c026_arch_48x24_act_relu_tanh",
        "c038_arch_128x64_act_relu_tanh",
    }

    for candidate_id, series in sorted(by_id.items()):
        summary = by_summary[candidate_id]
        activation = str(summary["activation"])
        color, _marker, activation_label = activation_style(activation)
        is_highlight = candidate_id in highlight_ids
        linewidth = 2.1 if is_highlight else 0.85
        alpha = 0.95 if is_highlight else 0.34
        zorder = 5 if is_highlight else 2
        linestyle = "-" if "sigmoid" not in activation else "--"
        x = [int(row["epoch"]) for row in series]
        y = [float(row["val_loss"]) for row in series]
        line = axes[0].plot(x, y, color=color, lw=linewidth, alpha=alpha, zorder=zorder, linestyle=linestyle)[0]
        axes[1].plot(x, y, color=color, lw=linewidth, alpha=alpha, zorder=zorder, linestyle=linestyle)
        if activation_label not in legend_seen:
            handles.append(line)
            labels.append(activation_label)
            legend_seen.add(activation_label)

    axes[0].set_title("Full training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation loss")
    axes[0].grid(True, alpha=0.24)
    axes[0].set_xlim(1, 120)
    axes[0].set_ylim(0.052, 0.49)

    axes[1].set_title("Zoom on final epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.24)
    axes[1].set_xlim(80, 120)
    axes[1].set_ylim(0.052, 0.112)

    for candidate_id, offset in {
        "c022_arch_32x16_act_relu_tanh": (3, 0.002),
        "c026_arch_48x24_act_relu_tanh": (3, -0.004),
        "c038_arch_128x64_act_relu_tanh": (-28, -0.002),
    }.items():
        series = by_id[candidate_id]
        summary = by_summary[candidate_id]
        y = float(series[-1]["val_loss"])
        x = int(series[-1]["epoch"])
        axes[1].annotate(
            candidate_label(summary),
            xy=(x, y),
            xytext=(x + offset[0], y + offset[1]),
            fontsize=8,
            arrowprops={"arrowstyle": "-", "lw": 0.8, "color": "#333333"},
        )

    fig.suptitle("Training curves for all architectures", y=1.02)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    for directory in (ANALYSIS_DIR, LATEX_IMAGE_DIR):
        fig.savefig(directory / f"{ALL_CURVES_STEM}.pdf", bbox_inches="tight")
        fig.savefig(directory / f"{ALL_CURVES_STEM}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_practical_training_curves(rows: list[dict[str, object]], epoch_rows: list[dict[str, object]]) -> None:
    by_id = rows_by_candidate(epoch_rows)
    by_summary = {str(row["candidate_id"]): row for row in rows}
    keep = [
        ("c021_arch_32x16_act_relu_relu", "#377eb8", "-", "32x16 ReLU, ReLU"),
        ("c022_arch_32x16_act_relu_tanh", "#1b9e77", "-", "32x16 ReLU, tanh"),
        ("c025_arch_48x24_act_relu_relu", "#377eb8", "--", "48x24 ReLU, ReLU"),
        ("c026_arch_48x24_act_relu_tanh", "#1b9e77", "--", "48x24 ReLU, tanh"),
        ("c037_arch_128x64_act_relu_relu", "#555555", ":", "128x64 ReLU, ReLU"),
        ("c038_arch_128x64_act_relu_tanh", "#111111", ":", "128x64 ReLU, tanh"),
    ]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for candidate_id, color, linestyle, label in keep:
        series = [row for row in by_id[candidate_id] if int(row["epoch"]) >= 20]
        ax.plot(
            [int(row["epoch"]) for row in series],
            [float(row["val_loss"]) for row in series],
            label=label,
            color=color,
            linestyle=linestyle,
            lw=2.0,
        )
        best_row = min(by_id[candidate_id], key=lambda row: float(row["val_loss"]))
        ax.scatter(
            [int(best_row["epoch"])],
            [float(best_row["val_loss"])],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            s=34,
            zorder=5,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title("Practical candidates after initial drop")
    ax.set_xlim(20, 120)
    ax.set_ylim(0.052, 0.115)
    ax.grid(True, alpha=0.24)
    ax.legend(loc="upper right", fontsize=8.4, frameon=True, ncol=2)
    save_figure(fig, PRACTICAL_CURVES_STEM)


def plot_wide(rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    seen: set[str] = set()
    for activation in sorted({str(row["activation"]) for row in rows}):
        subset = [row for row in rows if row["activation"] == activation]
        color, marker, label = activation_style(activation)
        ax.scatter(
            [int(row["parameter_count"]) for row in subset],
            [float(row["best_val_loss"]) for row in subset],
            s=58,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.92,
            label=label if activation not in seen else None,
        )
        seen.add(activation)

    highlights = {
        ("32x16", "relu,tanh"): ("32x16\nReLU,tanh", (16, 16)),
        ("48x24", "relu,tanh"): ("48x24\nReLU,tanh", (16, -26)),
        ("128x64", "relu,tanh"): ("128x64\nreferencia", (-72, 10)),
    }
    for row in rows:
        key = (str(row["architecture"]), str(row["activation"]))
        if key not in highlights:
            continue
        label, offset = highlights[key]
        x = int(row["parameter_count"])
        y = float(row["best_val_loss"])
        ax.scatter([x], [y], s=150, facecolors="none", edgecolors="black", linewidth=1.6, zorder=5)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": "#333333", "lw": 0.9},
        )

    ax.set_xscale("log")
    ax.set_xlabel("Count parametrov siete")
    ax.set_ylabel("Best validation loss")
    ax.grid(True, which="both", axis="both", alpha=0.22)
    ax.legend(loc="upper right", fontsize=8, frameon=True, ncol=2)
    ax.set_title("Supervised architecture sweep")
    fig.tight_layout()
    for directory in (ANALYSIS_DIR, LATEX_IMAGE_DIR):
        fig.savefig(directory / f"{WIDE_STEM}.pdf", bbox_inches="tight")
        fig.savefig(directory / f"{WIDE_STEM}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_focus(rows: list[dict[str, object]]) -> None:
    keep = [
        ("32x16", "relu,relu"),
        ("32x16", "relu,tanh"),
        ("48x24", "relu,relu"),
        ("48x24", "relu,tanh"),
    ]
    selected = []
    for architecture, activation in keep:
        selected.append(
            next(
                row
                for row in rows
                if str(row["architecture"]) == architecture and str(row["activation"]) == activation
            )
        )

    colors = ["#377eb8", "#1b9e77", "#377eb8", "#1b9e77"]
    labels = [f"{row['architecture']}\n{row['activation']}" for row in selected]
    values = [float(row["best_val_loss"]) for row in selected]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bars = ax.bar(range(len(selected)), values, color=colors, edgecolor="#222222", linewidth=0.8)
    ax.set_xticks(range(len(selected)), labels)
    ax.set_ylabel("Best validation loss")
    ax.set_title("Practical candidate detail")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0.058, max(values) + 0.004)
    for bar, row in zip(bars, selected):
        value = float(row["best_val_loss"])
        params = int(row["parameter_count"])
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.00035,
            f"{value:.4f}\n{params} params",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    fig.tight_layout()
    for directory in (ANALYSIS_DIR, LATEX_IMAGE_DIR):
        fig.savefig(directory / f"{FOCUS_STEM}.pdf", bbox_inches="tight")
        fig.savefig(directory / f"{FOCUS_STEM}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(config: dict[str, object], rows: list[dict[str, object]]) -> None:
    selected_keys = {
        ("32x16", "relu,relu"),
        ("32x16", "relu,tanh"),
        ("48x24", "relu,relu"),
        ("48x24", "relu,tanh"),
        ("128x64", "relu,relu"),
        ("128x64", "relu,tanh"),
    }
    selected = [
        row
        for row in rows
        if (str(row["architecture"]), str(row["activation"])) in selected_keys
    ]
    selected = sorted(selected, key=lambda row: (int(row["parameter_count"]), str(row["activation"])))

    fieldnames = [
        "architecture",
        "activation",
        "parameter_count",
        "best_val_loss",
        "best_val_steer_mae",
        "best_val_gas_accuracy",
        "best_val_brake_accuracy",
        "best_epoch",
    ]
    save_csv(ANALYSIS_DIR / "selected_architecture_candidates.csv", selected, fieldnames)

    metadata = {
        "experiment_id": "supervised_architecture_sweep_20260503",
        "source_run": str(RAW_DIR.relative_to(REPO_ROOT)),
        "purpose": "Supervised capacity sweep for selecting practical MLP policy architectures.",
        "train_samples": config["train_dataset_stats"]["frames_final"],
        "val_samples": config["val_dataset_stats"]["frames_final"],
        "obs_dim": config["obs_dim"],
        "act_dim": config["act_dim"],
        "epochs": config["epochs"],
        "optimizer": "Adam",
        "loss": "SmoothL1Loss with action weights [1, 1, 3]",
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "batch_size": config["batch_size"],
        "train_augment_mirror": config["train_augment_mirror"],
        "val_augment_mirror": config["val_augment_mirror"],
        "random_seed": config["random_seed"],
    }
    (PACKAGE_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    table_lines = []
    for row in selected:
        table_lines.append(
            "| {architecture} | {activation} | {parameter_count} | {best_val_loss:.6f} | {best_val_steer_mae:.4f} | {best_val_gas_accuracy:.3f} | {best_val_brake_accuracy:.3f} | {best_epoch} |".format(
                **row
            )
        )

    report = f"""# Supervised Architecture Sweep 20260503

This curated package summarizes the supervised architecture-capacity sweep used in the thesis chapter on policy training.

## Source Run

- Raw run: `{RAW_DIR.relative_to(REPO_ROOT)}`
- Dataset: v2d/asphalt human-driving observations on `AI Training #5`
- Train samples: {metadata['train_samples']}
- Validation samples: {metadata['val_samples']}
- Observation dimension: {metadata['obs_dim']}
- Action dimension: {metadata['act_dim']}
- Epochs: {metadata['epochs']}
- Optimizer: Adam
- Loss: SmoothL1Loss with action weights `[1, 1, 3]`

## Selected Candidates

| architecture | activation | params | val loss | steer MAE | gas acc. | brake acc. | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_lines)}

## Interpretation

This sweep is a representation-capacity test. It shows how well a fixed MLP architecture can approximate human actions on recorded data. It does not prove closed-loop driving quality. Larger networks reach lower validation loss, but they also create a larger parameter vector for later evolutionary search. The thesis therefore keeps `32x16 relu,tanh` as a cheap experimental baseline and `48x24 relu,tanh` as the stronger practical candidate.

## Generated Figures

- `{ALL_CURVES_STEM}.pdf`
- `{PRACTICAL_CURVES_STEM}.pdf`
- `{WIDE_STEM}.pdf`
- `{FOCUS_STEM}.pdf`
"""
    (PACKAGE_DIR / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Missing raw run directory: {RAW_DIR}")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_COPY_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_rows(RAW_DIR / "summary.csv")
    epoch_rows = read_epoch_rows(RAW_DIR / "epoch_metrics.csv")
    config = json.loads((RAW_DIR / "config.json").read_text(encoding="utf-8"))

    shutil.copy2(RAW_DIR / "summary.csv", RAW_COPY_DIR / "summary.csv")
    shutil.copy2(RAW_DIR / "config.json", RAW_COPY_DIR / "config.json")
    shutil.copy2(RAW_DIR / "epoch_metrics.csv", RAW_COPY_DIR / "epoch_metrics.csv")

    plot_all_training_curves(rows, epoch_rows)
    plot_practical_training_curves(rows, epoch_rows)
    plot_wide(rows)
    plot_focus(rows)
    write_report(config, rows)


if __name__ == "__main__":
    main()
