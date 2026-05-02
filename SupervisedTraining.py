import argparse
import glob
import json
import os
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from NeuralPolicy import NeuralPolicy
from ObservationEncoder import ObservationEncoder


DEFAULT_VERTICAL_MODE = False
DEFAULT_MULTI_SURFACE_MODE = False


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def choose_batch_size(num_train_samples: int, device: torch.device) -> int:
    if num_train_samples <= 0:
        return 1
    if device.type == "cuda":
        preferred = 4096
        minimum = 512
    else:
        preferred = 1024
        minimum = 128

    batch_size = min(preferred, num_train_samples)
    if batch_size < minimum:
        batch_size = min(num_train_samples, minimum)
    return max(1, batch_size)


def choose_num_workers(device: torch.device) -> int:
    cpu_count = os.cpu_count() or 1
    if device.type == "cuda":
        return max(0, min(4, cpu_count - 1))
    return 0


def find_attempt_files(base_dir: str = "logs/supervised_data") -> List[str]:
    pattern = os.path.join(base_dir, "**", "attempts", "attempt_*.npz")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No attempt files found under {base_dir}.")
    return sorted(files)


def _neutral_feature_value(name: str) -> float:
    if name.startswith("laser_"):
        return 1.0
    if name.startswith("surface_instruction_"):
        return 1.0
    if name == "support_normal_y":
        return 1.0
    if name == "dt_ratio":
        return 1.0
    return 0.0


def _infer_source_feature_names(obs_dim: int) -> List[str] | None:
    modes = ObservationEncoder.infer_modes_from_dim(obs_dim)
    if modes is None:
        return None
    vertical_mode, multi_surface_mode = modes
    return ObservationEncoder.feature_names(
        vertical_mode=vertical_mode,
        multi_surface_mode=multi_surface_mode,
    )


def convert_observations_to_layout(
    observations: np.ndarray,
    target_vertical_mode: bool,
    target_multi_surface_mode: bool,
) -> np.ndarray:
    """Map observations by feature name instead of slicing by position.

    This is important after the observation layout became configurable. A 53-dim
    v3d+surface dataset stores temporal features after surface+height features,
    while the 34-dim v2d+asphalt model expects those temporal features directly
    after `slip_mean`.
    """

    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim != 2:
        raise ValueError(f"Expected 2D observations, got shape {obs.shape}.")

    source_names = _infer_source_feature_names(obs.shape[1])
    target_names = ObservationEncoder.feature_names(
        vertical_mode=bool(target_vertical_mode),
        multi_surface_mode=bool(target_multi_surface_mode),
    )

    if source_names is None:
        # Historical fallback: keep only canonical feature names that still fit.
        # This intentionally does not preserve any old appended action-leakage
        # columns such as previous gas/brake/steer.
        source_names = ObservationEncoder.feature_names(
            vertical_mode=False,
            multi_surface_mode=False,
        )[: obs.shape[1]]

    source_index = {name: idx for idx, name in enumerate(source_names) if idx < obs.shape[1]}
    converted = np.empty((obs.shape[0], len(target_names)), dtype=np.float32)
    for target_idx, name in enumerate(target_names):
        source_idx = source_index.get(name)
        if source_idx is None:
            converted[:, target_idx] = _neutral_feature_value(name)
        else:
            converted[:, target_idx] = obs[:, source_idx]
    return converted


def mask_action_leakage_features(
    observations: np.ndarray,
    feature_names: Sequence[str],
    mask_feature_names: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """Zero out any direct action/input features if a legacy dataset contains them."""

    if not mask_feature_names:
        return observations, []
    masked = np.asarray(observations, dtype=np.float32).copy()
    name_to_index = {name: idx for idx, name in enumerate(feature_names)}
    applied: List[str] = []
    for name in mask_feature_names:
        idx = name_to_index.get(name)
        if idx is None:
            continue
        masked[:, idx] = 0.0
        applied.append(name)
    return masked, applied


def load_attempt(
    path: str,
    target_vertical_mode: bool = DEFAULT_VERTICAL_MODE,
    target_multi_surface_mode: bool = DEFAULT_MULTI_SURFACE_MODE,
    mask_feature_names: Sequence[str] = (),
) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        observations = convert_observations_to_layout(
            np.asarray(data["observations"], dtype=np.float32),
            target_vertical_mode=target_vertical_mode,
            target_multi_surface_mode=target_multi_surface_mode,
        )
        feature_names = ObservationEncoder.feature_names(
            vertical_mode=bool(target_vertical_mode),
            multi_surface_mode=bool(target_multi_surface_mode),
        )
        observations, masked_features = mask_action_leakage_features(
            observations,
            feature_names=feature_names,
            mask_feature_names=mask_feature_names,
        )
        return dict(
            path=path,
            observations=observations,
            actions=np.asarray(data["actions"], dtype=np.float32),
            masked_features=list(masked_features),
        )


def extract_attempt_group(path: str) -> str:
    attempts_dir = os.path.dirname(path)
    run_dir = os.path.dirname(attempts_dir)
    return os.path.basename(run_dir)


def split_attempt_files(
    attempt_files: Sequence[str],
    validation_fraction: float,
    rng: np.random.Generator,
) -> Tuple[List[str], List[str]]:
    files = list(attempt_files)
    if not files:
        raise ValueError("No attempt files available for splitting.")

    grouped: Dict[str, List[str]] = {}
    for path in files:
        grouped.setdefault(extract_attempt_group(path), []).append(path)

    train_files: List[str] = []
    val_files: List[str] = []
    for _, group_files in sorted(grouped.items()):
        local = list(group_files)
        rng.shuffle(local)
        if len(local) == 1:
            train_files.extend(local)
            continue

        val_count = max(1, int(np.ceil(len(local) * validation_fraction)))
        val_count = min(val_count, len(local) - 1)
        val_files.extend(local[:val_count])
        train_files.extend(local[val_count:])

    if not val_files:
        shuffled = files.copy()
        rng.shuffle(shuffled)
        val_files = [shuffled[0]]
        train_files = shuffled[1:] if len(shuffled) > 1 else shuffled[:]

    return train_files, val_files


def _boring_mask(actions: np.ndarray) -> np.ndarray:
    return (
        (actions[:, 0] > 0.95)
        & (actions[:, 1] < 0.05)
        & (np.abs(actions[:, 2]) < 0.05)
    )


def canonicalize_target_actions(
    observations: np.ndarray,
    actions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observations, dtype=np.float32).copy()
    act = np.asarray(actions, dtype=np.float32).copy()
    act[:, 0] = (act[:, 0] > 0.5).astype(np.float32)
    act[:, 1] = (act[:, 1] > 0.5).astype(np.float32)
    return obs, act


def preprocess_attempt(
    observations: np.ndarray,
    actions: np.ndarray,
    rng: np.random.Generator,
    boring_keep_probability: float,
    max_frames_after_filter: int | None,
    apply_boring_filter: bool,
    apply_frame_cap: bool,
    augment_mirror: bool,
    target_vertical_mode: bool,
    target_multi_surface_mode: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    raw_count = int(observations.shape[0])
    obs, act = canonicalize_target_actions(observations, actions)
    boring_count = int(np.count_nonzero(_boring_mask(act)))

    if apply_boring_filter:
        boring_mask = _boring_mask(act)
        keep_mask = (~boring_mask) | (
            rng.random(act.shape[0]) < float(boring_keep_probability)
        )
        obs = obs[keep_mask]
        act = act[keep_mask]

    after_filter = int(obs.shape[0])

    if apply_frame_cap and max_frames_after_filter is not None and after_filter > max_frames_after_filter:
        chosen = np.sort(
            rng.choice(after_filter, size=int(max_frames_after_filter), replace=False)
        )
        obs = obs[chosen]
        act = act[chosen]

    after_cap = int(obs.shape[0])

    if augment_mirror:
        mirrored_obs = np.stack(
            [
                ObservationEncoder.mirror_observation(
                    sample,
                    vertical_mode=target_vertical_mode,
                    multi_surface_mode=target_multi_surface_mode,
                )
                for sample in obs
            ],
            axis=0,
        ).astype(np.float32)
        mirrored_act = act.copy()
        mirrored_act[:, 2] *= -1.0
        obs = np.concatenate([obs, mirrored_obs], axis=0)
        act = np.concatenate([act, mirrored_act], axis=0)

    return obs, act, dict(
        raw_frames=raw_count,
        boring_frames=boring_count,
        frames_after_filter=after_filter,
        frames_after_cap=after_cap,
        final_frames=int(obs.shape[0]),
    )


def build_dataset_from_attempts(
    attempt_files: Sequence[str],
    rng: np.random.Generator,
    boring_keep_probability: float,
    max_frames_after_filter: int | None,
    apply_boring_filter: bool,
    apply_frame_cap: bool,
    augment_mirror: bool,
    target_vertical_mode: bool,
    target_multi_surface_mode: bool,
    mask_feature_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    per_attempt_stats: List[Dict] = []

    for path in attempt_files:
        attempt = load_attempt(
            path,
            target_vertical_mode=target_vertical_mode,
            target_multi_surface_mode=target_multi_surface_mode,
            mask_feature_names=mask_feature_names,
        )
        obs, act, stats = preprocess_attempt(
            observations=attempt["observations"],
            actions=attempt["actions"],
            rng=rng,
            boring_keep_probability=boring_keep_probability,
            max_frames_after_filter=max_frames_after_filter,
            apply_boring_filter=apply_boring_filter,
            apply_frame_cap=apply_frame_cap,
            augment_mirror=augment_mirror,
            target_vertical_mode=target_vertical_mode,
            target_multi_surface_mode=target_multi_surface_mode,
        )
        observations.append(obs)
        actions.append(act)
        per_attempt_stats.append(
            dict(
                path=path,
                masked_features=list(attempt.get("masked_features", [])),
                **stats,
            )
        )

    if not observations:
        raise ValueError("No observations available after preprocessing.")

    observations_arr = np.concatenate(observations, axis=0)
    actions_arr = np.concatenate(actions, axis=0)

    permutation = rng.permutation(observations_arr.shape[0])
    observations_arr = observations_arr[permutation]
    actions_arr = actions_arr[permutation]

    total_raw = int(sum(item["raw_frames"] for item in per_attempt_stats))
    total_boring = int(sum(item["boring_frames"] for item in per_attempt_stats))
    total_after_filter = int(sum(item["frames_after_filter"] for item in per_attempt_stats))
    total_after_cap = int(sum(item["frames_after_cap"] for item in per_attempt_stats))

    stats = dict(
        attempt_files=len(attempt_files),
        total_frames_before_filter=total_raw,
        boring_frames_before_filter=total_boring,
        frames_after_filter_before_cap=total_after_filter,
        frames_after_cap_before_mirror=total_after_cap,
        frames_final=int(observations_arr.shape[0]),
        apply_boring_filter=bool(apply_boring_filter),
        apply_frame_cap=bool(apply_frame_cap),
        augment_mirror=bool(augment_mirror),
        target_vertical_mode=bool(target_vertical_mode),
        target_multi_surface_mode=bool(target_multi_surface_mode),
        target_observation_dim=int(observations_arr.shape[1]),
        mask_feature_names=list(mask_feature_names),
        per_attempt=per_attempt_stats,
    )
    return observations_arr, actions_arr, stats


def parse_csv_values(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(item) for item in parse_csv_values(value)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a supervised target policy from recorded attempts.")
    parser.add_argument("--data-root", default="logs/supervised_data")
    parser.add_argument("--output-root", default="logs/supervised_runs")
    parser.add_argument("--vertical-mode", type=parse_bool, default=DEFAULT_VERTICAL_MODE)
    parser.add_argument("--multi-surface-mode", type=parse_bool, default=DEFAULT_MULTI_SURFACE_MODE)
    parser.add_argument("--hidden-dim", default="32,16")
    parser.add_argument("--hidden-activation", default="relu,tanh")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--boring-keep-probability", type=float, default=0.9)
    parser.add_argument("--max-frames-after-filter", type=int, default=None)
    parser.add_argument("--disable-boring-filter", action="store_true")
    parser.add_argument("--disable-mirror", action="store_true")
    parser.add_argument(
        "--mask-feature-names",
        default="steer,gas,brake,input_steer,input_gas,input_brake,previous_steer,previous_gas,previous_brake",
        help=(
            "Comma-separated observation feature names to zero before training. "
            "Current canonical observations do not include direct actions, but this "
            "guards against old action-leakage datasets."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=51951)
    args = parser.parse_args()

    data_root = args.data_root
    output_root = args.output_root
    target_vertical_mode = bool(args.vertical_mode)
    target_multi_surface_mode = bool(args.multi_surface_mode)

    hidden_dims = parse_int_list(args.hidden_dim)
    hidden_activations = parse_csv_values(args.hidden_activation)
    batch_size_override = args.batch_size
    epochs = int(args.epochs)
    learning_rate = float(args.learning_rate)
    weight_decay = float(args.weight_decay)
    train_boring_keep_probability = float(args.boring_keep_probability)
    train_max_frames_after_filter = args.max_frames_after_filter
    random_seed = int(args.random_seed)
    mask_feature_names = parse_csv_values(args.mask_feature_names)

    rng_train = np.random.default_rng(random_seed + 1)

    attempt_files = find_attempt_files(data_root)
    train_obs, train_actions, dataset_stats = build_dataset_from_attempts(
        attempt_files=attempt_files,
        rng=rng_train,
        boring_keep_probability=train_boring_keep_probability,
        max_frames_after_filter=train_max_frames_after_filter,
        apply_boring_filter=not bool(args.disable_boring_filter),
        apply_frame_cap=False,
        augment_mirror=not bool(args.disable_mirror),
        target_vertical_mode=target_vertical_mode,
        target_multi_surface_mode=target_multi_surface_mode,
        mask_feature_names=mask_feature_names,
    )

    obs_dim = int(train_obs.shape[1])
    act_dim = int(train_actions.shape[1])

    device = choose_device()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    batch_size = (
        int(batch_size_override)
        if batch_size_override is not None
        else choose_batch_size(num_train_samples=train_obs.shape[0], device=device)
    )
    num_workers = choose_num_workers(device=device)
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0
    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    print(
        f"Training on {device} | batch_size={batch_size} | "
        f"num_workers={num_workers} | amp={amp_enabled}"
    )
    print(f"Dataset attempts: {len(attempt_files)}")
    print(f"Dataset stats: {dataset_stats}")

    model = NeuralPolicy(
        obs_dim=obs_dim,
        hidden_dim=hidden_dims,
        act_dim=act_dim,
        action_mode="target",
        hidden_activation=hidden_activations,
        action_scale=np.ones(act_dim, dtype=np.float32),
        device=device,
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_weights = torch.tensor([1.0, 1.0, 3.0], dtype=torch.float32, device=device)
    criterion = nn.SmoothL1Loss(reduction="none")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_obs), torch.from_numpy(train_actions)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    lidar_tag = (
        f"{'v3d' if target_vertical_mode else 'v2d'}_"
        f"{'surface' if target_multi_surface_mode else 'asphalt'}"
    )
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{lidar_tag}_target_supervised"
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=False)
    best_model_path = os.path.join(run_dir, "best_model.pt")
    curve_path = os.path.join(run_dir, "loss_curve.png")

    train_losses: List[float] = []
    best_train_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_count = 0
        for batch_obs, batch_actions in train_loader:
            batch_obs = batch_obs.to(device, non_blocking=pin_memory)
            batch_actions = batch_actions.to(device, non_blocking=pin_memory)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions = model(batch_obs)
                per_value_loss = criterion(predictions, batch_actions)
                loss = (per_value_loss * loss_weights).mean()

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            batch_size_actual = batch_obs.shape[0]
            running_train_loss += float(loss.detach().cpu()) * batch_size_actual
            train_count += batch_size_actual

        train_loss = running_train_loss / max(train_count, 1)
        train_losses.append(train_loss)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.6f}"
        )

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_epoch = epoch
            model.save(
                best_model_path,
                extra={
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "dataset_stats": dataset_stats,
                    "attempt_files": attempt_files,
                    "vertical_mode": bool(target_vertical_mode),
                    "multi_surface_mode": bool(target_multi_surface_mode),
                    "observation_layout": ObservationEncoder.feature_names(
                        vertical_mode=target_vertical_mode,
                        multi_surface_mode=target_multi_surface_mode,
                    ),
                },
            )

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Supervised Target Policy Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()

    summary = {
        "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_root": data_root,
        "all_attempt_files": attempt_files,
        "dataset_stats": dataset_stats,
        "obs_dim": obs_dim,
        "observation_layout": ObservationEncoder.feature_names(
            vertical_mode=target_vertical_mode,
            multi_surface_mode=target_multi_surface_mode,
        ),
        "vertical_mode": bool(target_vertical_mode),
        "multi_surface_mode": bool(target_multi_surface_mode),
        "act_dim": act_dim,
        "hidden_dims": list(hidden_dims),
        "hidden_activation": hidden_activations[0] if len(hidden_activations) == 1 else list(hidden_activations),
        "hidden_activations": list(hidden_activations),
        "batch_size": batch_size,
        "batch_size_override": batch_size_override,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "amp_enabled": amp_enabled,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "train_boring_keep_probability": train_boring_keep_probability,
        "train_max_frames_after_filter": train_max_frames_after_filter,
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "best_model_path": best_model_path,
        "curve_path": curve_path,
        "device": str(device),
        "random_seed": random_seed,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(f"Best model saved to: {best_model_path}")
    print(f"Training summary saved to: {os.path.join(run_dir, 'summary.json')}")
