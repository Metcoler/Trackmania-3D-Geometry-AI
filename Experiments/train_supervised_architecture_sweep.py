from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NeuralPolicy import NeuralPolicy, normalize_hidden_activations, normalize_hidden_dims
from ObservationEncoder import ObservationEncoder
from SupervisedTraining import (
    build_dataset_from_attempts,
    choose_batch_size,
    choose_device,
    choose_num_workers,
    parse_bool,
    parse_csv_values,
    split_attempt_files,
)


DEFAULT_DATA_ROOTS = [
    "logs/supervised_data/20260502_153421_map_AI Training #5_v2d_asphalt_target_dataset",
    "logs/supervised_data/20260502_154227_map_AI Training #5_v2d_asphalt_target_dataset",
]

DEFAULT_ARCHITECTURES = "8;16;24;32;16,8;24,12;32,16;48,24;64,32;32,16,8;128,64"
DEFAULT_ACTIVATIONS = "auto"
DEFAULT_ACTIVATIONS_BY_DEPTH = {
    1: "relu;tanh;sigmoid",
    2: "relu,relu;relu,tanh;tanh,tanh;sigmoid,sigmoid",
    3: "relu,relu,relu;tanh,tanh,tanh;relu,relu,tanh;sigmoid,sigmoid,sigmoid",
}
DEFAULT_MASK_FEATURES = (
    "steer,gas,brake,input_steer,input_gas,input_brake,"
    "previous_steer,previous_gas,previous_brake"
)


def find_attempt_files_in_roots(roots: Iterable[str]) -> list[str]:
    files: list[str] = []
    for root in roots:
        pattern = os.path.join(str(root), "**", "attempts", "attempt_*.npz")
        files.extend(glob.glob(pattern, recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No attempt files found under roots: {list(roots)}")
    return files


def parse_architectures(value: str) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = []
    for item in str(value).split(";"):
        item = item.strip()
        if not item:
            continue
        dims = tuple(int(part.strip()) for part in item.split(",") if part.strip())
        result.append(normalize_hidden_dims(dims))
    if not result:
        raise ValueError("Architecture list is empty.")
    return result


def parse_activation_candidates(value: str) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = []
    for item in str(value).split(";"):
        item = item.strip()
        if not item:
            continue
        result.append(tuple(parse_csv_values(item)))
    if not result:
        raise ValueError("Activation list is empty.")
    return result


def activation_candidates_for_depth(value: str, depth: int) -> list[tuple[str, ...]]:
    value = str(value).strip()
    if value.lower() == "auto":
        if int(depth) not in DEFAULT_ACTIVATIONS_BY_DEPTH:
            raise ValueError(f"No default activation candidates configured for {depth} hidden layers.")
        raw_candidates = parse_activation_candidates(DEFAULT_ACTIVATIONS_BY_DEPTH[int(depth)])
    else:
        raw_candidates = parse_activation_candidates(value)

    normalized_candidates: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in raw_candidates:
        try:
            normalized = normalize_hidden_activations(candidate, num_hidden_layers=int(depth))
        except ValueError:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_candidates.append(normalized)
    if not normalized_candidates:
        raise ValueError(f"No compatible activation candidates for {depth} hidden layers.")
    return normalized_candidates


def dims_label(dims: tuple[int, ...]) -> str:
    return "x".join(str(value) for value in dims)


def activation_label(activations: tuple[str, ...]) -> str:
    return ",".join(str(value) for value in activations)


def safe_label(value: str) -> str:
    return (
        str(value)
        .replace(",", "_")
        .replace(" ", "")
        .replace("(", "")
        .replace(")", "")
    )


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_grad_scaler(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def make_loader(
    obs: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    num_workers = choose_num_workers(device=device) if shuffle else 0
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(torch.from_numpy(obs), torch.from_numpy(actions)),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator if shuffle else None,
    )


def compute_loss(predictions: torch.Tensor, targets: torch.Tensor, loss_weights: torch.Tensor) -> torch.Tensor:
    criterion = nn.SmoothL1Loss(reduction="none")
    per_value_loss = criterion(predictions, targets)
    return (per_value_loss * loss_weights).mean()


@torch.no_grad()
def evaluate_model(
    model: NeuralPolicy,
    loader: DataLoader,
    device: torch.device,
    loss_weights: torch.Tensor,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_steer_abs = 0.0
    total_gas_correct = 0
    total_brake_correct = 0
    total_gas_bce = 0.0
    total_brake_bce = 0.0
    total_gas_abs = 0.0
    total_brake_abs = 0.0
    eps = 1e-6

    for batch_obs, batch_actions in loader:
        batch_obs = batch_obs.to(device, non_blocking=device.type == "cuda")
        batch_actions = batch_actions.to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            predictions = model(batch_obs)
            loss = compute_loss(predictions, batch_actions, loss_weights)

        batch_size = int(batch_obs.shape[0])
        total_samples += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size

        gas_pred = predictions[:, 0].float().clamp(eps, 1.0 - eps)
        brake_pred = predictions[:, 1].float().clamp(eps, 1.0 - eps)
        steer_pred = predictions[:, 2].float()
        gas_target = batch_actions[:, 0].float()
        brake_target = batch_actions[:, 1].float()
        steer_target = batch_actions[:, 2].float()

        total_steer_abs += float(torch.abs(steer_pred - steer_target).sum().detach().cpu())
        total_gas_abs += float(torch.abs(gas_pred - gas_target).sum().detach().cpu())
        total_brake_abs += float(torch.abs(brake_pred - brake_target).sum().detach().cpu())
        total_gas_correct += int(((gas_pred >= 0.5) == (gas_target >= 0.5)).sum().detach().cpu())
        total_brake_correct += int(((brake_pred >= 0.5) == (brake_target >= 0.5)).sum().detach().cpu())
        total_gas_bce += float(F.binary_cross_entropy(gas_pred, gas_target, reduction="sum").detach().cpu())
        total_brake_bce += float(F.binary_cross_entropy(brake_pred, brake_target, reduction="sum").detach().cpu())

    denominator = max(1, total_samples)
    return {
        "loss": total_loss / denominator,
        "steer_mae": total_steer_abs / denominator,
        "gas_accuracy": total_gas_correct / denominator,
        "brake_accuracy": total_brake_correct / denominator,
        "gas_bce": total_gas_bce / denominator,
        "brake_bce": total_brake_bce / denominator,
        "gas_mae": total_gas_abs / denominator,
        "brake_mae": total_brake_abs / denominator,
        "samples": float(total_samples),
    }


def train_candidate(
    *,
    candidate_id: str,
    hidden_dims: tuple[int, ...],
    hidden_activations: tuple[str, ...],
    train_obs: np.ndarray,
    train_actions: np.ndarray,
    val_obs: np.ndarray,
    val_actions: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    random_seed: int,
    amp_enabled: bool,
) -> tuple[dict, list[dict]]:
    set_all_seeds(random_seed)
    obs_dim = int(train_obs.shape[1])
    act_dim = int(train_actions.shape[1])
    model = NeuralPolicy(
        obs_dim=obs_dim,
        hidden_dim=hidden_dims,
        act_dim=act_dim,
        action_mode="target",
        hidden_activation=hidden_activations,
        action_scale=np.ones(act_dim, dtype=np.float32),
        device=device,
    )
    optimizer = optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    loss_weights = torch.tensor([1.0, 1.0, 3.0], dtype=torch.float32, device=device)
    scaler = make_grad_scaler(device=device, enabled=amp_enabled)
    train_loader = make_loader(
        train_obs,
        train_actions,
        batch_size=batch_size,
        shuffle=True,
        device=device,
        seed=random_seed + 17,
    )
    train_eval_loader = make_loader(
        train_obs,
        train_actions,
        batch_size=batch_size,
        shuffle=False,
        device=device,
        seed=random_seed + 23,
    )
    val_loader = make_loader(
        val_obs,
        val_actions,
        batch_size=batch_size,
        shuffle=False,
        device=device,
        seed=random_seed + 29,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_val_metrics: dict[str, float] = {}
    final_train_loss = float("nan")
    epoch_rows: list[dict] = []
    wall_start = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        for batch_obs, batch_actions in train_loader:
            batch_obs = batch_obs.to(device, non_blocking=device.type == "cuda")
            batch_actions = batch_actions.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions = model(batch_obs)
                loss = compute_loss(predictions, batch_actions, loss_weights)

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            batch_size_actual = int(batch_obs.shape[0])
            running_loss += float(loss.detach().cpu()) * batch_size_actual
            running_count += batch_size_actual

        final_train_loss = running_loss / max(1, running_count)
        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            loss_weights=loss_weights,
            amp_enabled=amp_enabled,
        )
        row = {
            "candidate_id": candidate_id,
            "epoch": epoch,
            "train_loss": final_train_loss,
            "val_loss": val_metrics["loss"],
            "val_steer_mae": val_metrics["steer_mae"],
            "val_gas_accuracy": val_metrics["gas_accuracy"],
            "val_brake_accuracy": val_metrics["brake_accuracy"],
            "val_gas_bce": val_metrics["gas_bce"],
            "val_brake_bce": val_metrics["brake_bce"],
        }
        epoch_rows.append(row)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = int(epoch)
            best_val_metrics = dict(val_metrics)

    train_metrics = evaluate_model(
        model=model,
        loader=train_eval_loader,
        device=device,
        loss_weights=loss_weights,
        amp_enabled=amp_enabled,
    )
    parameter_count = int(model.genome_size)
    wall_seconds = float(time.perf_counter() - wall_start)
    summary = {
        "candidate_id": candidate_id,
        "hidden_dims": json.dumps(list(hidden_dims)),
        "hidden_activations": json.dumps(list(hidden_activations)),
        "architecture": dims_label(hidden_dims),
        "activation": activation_label(hidden_activations),
        "parameter_count": parameter_count,
        "epochs": int(epochs),
        "best_epoch": best_epoch,
        "final_train_loss": final_train_loss,
        "final_train_eval_loss": train_metrics["loss"],
        "best_val_loss": best_val_loss,
        "best_val_steer_mae": best_val_metrics.get("steer_mae", float("nan")),
        "best_val_gas_accuracy": best_val_metrics.get("gas_accuracy", float("nan")),
        "best_val_brake_accuracy": best_val_metrics.get("brake_accuracy", float("nan")),
        "best_val_gas_bce": best_val_metrics.get("gas_bce", float("nan")),
        "best_val_brake_bce": best_val_metrics.get("brake_bce", float("nan")),
        "best_val_gas_mae": best_val_metrics.get("gas_mae", float("nan")),
        "best_val_brake_mae": best_val_metrics.get("brake_mae", float("nan")),
        "train_samples": int(train_obs.shape[0]),
        "val_samples": int(val_obs.shape[0]),
        "wall_seconds": wall_seconds,
        "random_seed": int(random_seed),
    }
    return summary, epoch_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_heatmap(summary_rows: list[dict], output_dir: Path) -> None:
    if not summary_rows:
        return
    import pandas as pd

    df = pd.DataFrame(summary_rows)
    pivot = df.pivot_table(
        index="architecture",
        columns="activation",
        values="best_val_loss",
        aggfunc="min",
    )
    if pivot.empty:
        return
    architecture_order: list[str] = []
    for row in summary_rows:
        architecture = str(row["architecture"])
        if architecture not in architecture_order:
            architecture_order.append(architecture)
    pivot = pivot.reindex(architecture_order)
    pivot = pivot.loc[~pivot.index.duplicated(keep="first")]
    values = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    plt.figure(figsize=(11, 7))
    image = plt.imshow(masked, aspect="auto", cmap=cmap)
    plt.colorbar(image, label="Best validation loss")
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.xlabel("Hidden activation")
    plt.ylabel("Architecture")
    plt.title("Architecture capacity sweep: validation loss")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                plt.text(col_idx, row_idx, f"{value:.4f}", ha="center", va="center", fontsize=8, color="white")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_architecture_activation_val_loss.png", dpi=160)
    plt.close()


def plot_scatter(summary_rows: list[dict], output_dir: Path, metric: str, ylabel: str, filename: str) -> None:
    if not summary_rows:
        return
    plt.figure(figsize=(9, 6))
    activations = sorted(set(str(row["activation"]) for row in summary_rows))
    for activation in activations:
        rows = [row for row in summary_rows if str(row["activation"]) == activation]
        rows.sort(key=lambda row: int(row["parameter_count"]))
        plt.scatter(
            [int(row["parameter_count"]) for row in rows],
            [float(row[metric]) for row in rows],
            label=activation,
            alpha=0.8,
            s=55,
        )
    plt.xlabel("Parameter count")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs parameter count")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=160)
    plt.close()


def markdown_table(rows: list[dict], columns: list[str], limit: int = 12) -> str:
    if not rows:
        return "_No data._"
    selected = rows[:limit]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.5f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def tolerance_rows(summary_rows: list[dict], best_loss: float) -> list[dict]:
    rows: list[dict] = []
    if not summary_rows or not np.isfinite(best_loss):
        return rows
    for tolerance in (0.05, 0.10, 0.15, 0.20):
        limit = best_loss * (1.0 + tolerance)
        candidates = [
            row for row in sorted(summary_rows, key=lambda item: int(item["parameter_count"]))
            if float(row["best_val_loss"]) <= limit
        ]
        if not candidates:
            rows.append(
                {
                    "tolerance": f"{int(tolerance * 100)}%",
                    "candidate_id": "",
                    "architecture": "",
                    "activation": "",
                    "parameter_count": "",
                    "best_val_loss": "",
                    "loss_ratio": "",
                    "best_val_steer_mae": "",
                    "best_val_gas_accuracy": "",
                    "best_val_brake_accuracy": "",
                    "best_epoch": "",
                }
            )
            continue
        candidate = candidates[0]
        enriched = dict(candidate)
        enriched["tolerance"] = f"{int(tolerance * 100)}%"
        enriched["loss_ratio"] = float(candidate["best_val_loss"]) / best_loss
        rows.append(enriched)
    return rows


def write_report(output_dir: Path, summary_rows: list[dict], skipped_rows: list[dict], config: dict) -> None:
    ranked = sorted(summary_rows, key=lambda row: (float(row["best_val_loss"]), int(row["parameter_count"])))
    best_loss = float(ranked[0]["best_val_loss"]) if ranked else float("nan")
    columns = [
        "candidate_id",
        "architecture",
        "activation",
        "parameter_count",
        "best_val_loss",
        "best_val_steer_mae",
        "best_val_gas_accuracy",
        "best_val_brake_accuracy",
        "best_epoch",
    ]
    tolerance_columns = [
        "tolerance",
        "candidate_id",
        "architecture",
        "activation",
        "parameter_count",
        "best_val_loss",
        "loss_ratio",
        "best_val_steer_mae",
        "best_val_gas_accuracy",
        "best_val_brake_accuracy",
        "best_epoch",
    ]
    smallest_tolerances = tolerance_rows(summary_rows, best_loss)
    report = f"""# Supervised Architecture Capacity Sweep

## Purpose

This experiment estimates whether a small MLP can approximate the mapping `observation -> human action` on a fixed supervised dataset. It is not a closed-loop driving validation.

## Dataset

- Data roots: `{config["data_roots"]}`
- Train attempts: `{len(config["train_attempt_files"])}`
- Validation attempts: `{len(config["val_attempt_files"])}`
- Train frames after preprocessing: `{config["train_dataset_stats"]["frames_final"]}`
- Validation frames after preprocessing: `{config["val_dataset_stats"]["frames_final"]}`
- Observation dim: `{config["obs_dim"]}`
- Action dim: `{config["act_dim"]}`
- Train mirroring: `{config["train_augment_mirror"]}`
- Validation mirroring: `{config["val_augment_mirror"]}`

## Best Candidates By Validation Loss

{markdown_table(ranked, columns=columns, limit=12)}

## Smallest Candidates Within Loss Tolerances

{markdown_table(smallest_tolerances, columns=tolerance_columns, limit=4)}

## Skipped Or Duplicate Configurations

Skipped incompatible or duplicate architecture/activation combinations: `{len(skipped_rows)}`.

## Generated Plots

- `heatmap_architecture_activation_val_loss.png`
- `validation_loss_vs_parameter_count.png`
- `steer_mae_vs_parameter_count.png`
- `gas_accuracy_vs_parameter_count.png`
- `brake_accuracy_vs_parameter_count.png`

## Interpretation Guide

Prefer the smallest architecture close to the best validation loss, not the largest network with the absolute best score. For GA, parameter count is genome size, so bigger networks directly increase the evolutionary search space.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised MLP architecture capacity sweep.")
    parser.add_argument("--data-roots", nargs="+", default=DEFAULT_DATA_ROOTS)
    parser.add_argument("--output-root", default="logs/supervised_architecture_sweep")
    parser.add_argument("--vertical-mode", type=parse_bool, default=False)
    parser.add_argument("--multi-surface-mode", type=parse_bool, default=False)
    parser.add_argument("--architectures", default=DEFAULT_ARCHITECTURES)
    parser.add_argument(
        "--hidden-activations",
        default=DEFAULT_ACTIVATIONS,
        help=(
            "'auto' uses depth-specific clean candidates; otherwise provide a semicolon-separated "
            "activation list such as 'relu;relu,tanh;tanh,tanh'."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--boring-keep-probability", type=float, default=0.9)
    parser.add_argument("--max-frames-after-filter", type=int, default=None)
    parser.add_argument("--disable-boring-filter", action="store_true")
    parser.add_argument("--disable-train-mirror", action="store_true")
    parser.add_argument("--mirror-validation", action="store_true")
    parser.add_argument("--mask-feature-names", default=DEFAULT_MASK_FEATURES)
    parser.add_argument("--random-seed", type=int, default=51951)
    parser.add_argument("--disable-amp", action="store_true")
    args = parser.parse_args()

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_v2d_asphalt_capacity_sweep"
    output_dir = Path(args.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    architectures = parse_architectures(args.architectures)
    activation_candidates_by_depth: dict[int, list[tuple[str, ...]]] = {}
    for hidden_dims in architectures:
        depth = len(hidden_dims)
        if depth not in activation_candidates_by_depth:
            activation_candidates_by_depth[depth] = activation_candidates_for_depth(
                args.hidden_activations,
                depth=depth,
            )
    mask_feature_names = parse_csv_values(args.mask_feature_names)
    random_seed = int(args.random_seed)
    split_rng = np.random.default_rng(random_seed)
    train_rng = np.random.default_rng(random_seed + 1)
    val_rng = np.random.default_rng(random_seed + 2)

    attempt_files = find_attempt_files_in_roots(args.data_roots)
    train_files, val_files = split_attempt_files(
        attempt_files=attempt_files,
        validation_fraction=float(args.validation_fraction),
        rng=split_rng,
    )
    train_obs, train_actions, train_dataset_stats = build_dataset_from_attempts(
        attempt_files=train_files,
        rng=train_rng,
        boring_keep_probability=float(args.boring_keep_probability),
        max_frames_after_filter=args.max_frames_after_filter,
        apply_boring_filter=not bool(args.disable_boring_filter),
        apply_frame_cap=False,
        augment_mirror=not bool(args.disable_train_mirror),
        target_vertical_mode=bool(args.vertical_mode),
        target_multi_surface_mode=bool(args.multi_surface_mode),
        mask_feature_names=mask_feature_names,
    )
    val_obs, val_actions, val_dataset_stats = build_dataset_from_attempts(
        attempt_files=val_files,
        rng=val_rng,
        boring_keep_probability=float(args.boring_keep_probability),
        max_frames_after_filter=args.max_frames_after_filter,
        apply_boring_filter=not bool(args.disable_boring_filter),
        apply_frame_cap=False,
        augment_mirror=bool(args.mirror_validation),
        target_vertical_mode=bool(args.vertical_mode),
        target_multi_surface_mode=bool(args.multi_surface_mode),
        mask_feature_names=mask_feature_names,
    )
    obs_dim = int(train_obs.shape[1])
    act_dim = int(train_actions.shape[1])
    if obs_dim != int(val_obs.shape[1]) or act_dim != int(val_actions.shape[1]):
        raise ValueError("Train and validation dimensions do not match.")

    device = choose_device()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    amp_enabled = bool(device.type == "cuda" and not args.disable_amp)
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else choose_batch_size(num_train_samples=train_obs.shape[0], device=device)
    )

    config = {
        "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_roots": list(args.data_roots),
        "all_attempt_files": attempt_files,
        "train_attempt_files": train_files,
        "val_attempt_files": val_files,
        "train_dataset_stats": train_dataset_stats,
        "val_dataset_stats": val_dataset_stats,
        "vertical_mode": bool(args.vertical_mode),
        "multi_surface_mode": bool(args.multi_surface_mode),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "observation_layout": ObservationEncoder.feature_names(
            vertical_mode=bool(args.vertical_mode),
            multi_surface_mode=bool(args.multi_surface_mode),
        ),
        "architectures": [list(value) for value in architectures],
        "hidden_activation_mode": str(args.hidden_activations),
        "hidden_activation_candidates_by_depth": {
            str(depth): [list(value) for value in candidates]
            for depth, candidates in sorted(activation_candidates_by_depth.items())
        },
        "default_activations_by_depth": DEFAULT_ACTIVATIONS_BY_DEPTH,
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(batch_size),
        "validation_fraction": float(args.validation_fraction),
        "boring_keep_probability": float(args.boring_keep_probability),
        "disable_boring_filter": bool(args.disable_boring_filter),
        "train_augment_mirror": not bool(args.disable_train_mirror),
        "val_augment_mirror": bool(args.mirror_validation),
        "mask_feature_names": mask_feature_names,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "random_seed": random_seed,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Output dir: {output_dir}")
    print(f"Device: {device} | amp={amp_enabled} | batch_size={batch_size}")
    print(f"Train attempts={len(train_files)} frames={train_obs.shape[0]}")
    print(f"Val attempts={len(val_files)} frames={val_obs.shape[0]}")
    print(f"obs_dim={obs_dim} act_dim={act_dim}")

    summary_rows: list[dict] = []
    epoch_rows: list[dict] = []
    skipped_rows: list[dict] = []
    candidate_index = 0
    seen_configs: set[tuple[tuple[int, ...], tuple[str, ...]]] = set()
    seen_candidate_ids: set[str] = set()

    for hidden_dims in architectures:
        activation_candidates = activation_candidates_by_depth[len(hidden_dims)]
        for normalized_activations in activation_candidates:
            arch_label = dims_label(hidden_dims)
            requested_activation = activation_label(normalized_activations)
            config_key = (hidden_dims, normalized_activations)
            if config_key in seen_configs:
                skipped_rows.append(
                    {
                        "architecture": arch_label,
                        "requested_activation": requested_activation,
                        "reason": "duplicate normalized architecture/activation",
                    }
                )
                continue
            seen_configs.add(config_key)

            candidate_index += 1
            candidate_id = (
                f"c{candidate_index:03d}_"
                f"arch_{safe_label(arch_label)}_"
                f"act_{safe_label(activation_label(normalized_activations))}"
            )
            if candidate_id in seen_candidate_ids:
                raise RuntimeError(f"Duplicate candidate_id generated: {candidate_id}")
            seen_candidate_ids.add(candidate_id)
            candidate_seed = random_seed + 1000 + candidate_index
            print(
                f"[{candidate_index:03d}] {candidate_id} | "
                f"dims={hidden_dims} activations={normalized_activations}"
            )
            summary, candidate_epoch_rows = train_candidate(
                candidate_id=candidate_id,
                hidden_dims=hidden_dims,
                hidden_activations=normalized_activations,
                train_obs=train_obs,
                train_actions=train_actions,
                val_obs=val_obs,
                val_actions=val_actions,
                device=device,
                epochs=int(args.epochs),
                batch_size=int(batch_size),
                learning_rate=float(args.learning_rate),
                weight_decay=float(args.weight_decay),
                random_seed=candidate_seed,
                amp_enabled=amp_enabled,
            )
            summary["requested_activation"] = requested_activation
            summary["candidate_index"] = candidate_index
            summary_rows.append(summary)
            epoch_rows.extend(candidate_epoch_rows)

    normalized_pairs = [
        (str(row["architecture"]), str(row["activation"]))
        for row in summary_rows
    ]
    if len(set(normalized_pairs)) != len(normalized_pairs):
        raise RuntimeError("Duplicate architecture + activation rows detected in summary.")
    candidate_ids = [str(row["candidate_id"]) for row in summary_rows]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("Duplicate candidate_id rows detected in summary.")

    summary_fieldnames = [
        "candidate_index",
        "candidate_id",
        "architecture",
        "activation",
        "requested_activation",
        "hidden_dims",
        "hidden_activations",
        "parameter_count",
        "epochs",
        "best_epoch",
        "final_train_loss",
        "final_train_eval_loss",
        "best_val_loss",
        "best_val_steer_mae",
        "best_val_gas_accuracy",
        "best_val_brake_accuracy",
        "best_val_gas_bce",
        "best_val_brake_bce",
        "best_val_gas_mae",
        "best_val_brake_mae",
        "train_samples",
        "val_samples",
        "wall_seconds",
        "random_seed",
    ]
    epoch_fieldnames = [
        "candidate_id",
        "epoch",
        "train_loss",
        "val_loss",
        "val_steer_mae",
        "val_gas_accuracy",
        "val_brake_accuracy",
        "val_gas_bce",
        "val_brake_bce",
    ]
    write_csv(output_dir / "summary.csv", summary_rows, fieldnames=summary_fieldnames)
    write_csv(output_dir / "epoch_metrics.csv", epoch_rows, fieldnames=epoch_fieldnames)
    if skipped_rows:
        write_csv(output_dir / "skipped_configs.csv", skipped_rows, fieldnames=["architecture", "requested_activation", "reason"])

    plot_heatmap(summary_rows, output_dir)
    plot_scatter(summary_rows, output_dir, "best_val_loss", "Best validation loss", "validation_loss_vs_parameter_count.png")
    plot_scatter(summary_rows, output_dir, "best_val_steer_mae", "Best validation steer MAE", "steer_mae_vs_parameter_count.png")
    plot_scatter(summary_rows, output_dir, "best_val_gas_accuracy", "Best validation gas accuracy", "gas_accuracy_vs_parameter_count.png")
    plot_scatter(summary_rows, output_dir, "best_val_brake_accuracy", "Best validation brake accuracy", "brake_accuracy_vs_parameter_count.png")
    write_report(output_dir, summary_rows, skipped_rows, config)

    print(f"Trained candidates: {len(summary_rows)}")
    print(f"Skipped incompatible configs: {len(skipped_rows)}")
    print(f"Saved summary to: {output_dir / 'summary.csv'}")
    print(f"Saved report to: {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
