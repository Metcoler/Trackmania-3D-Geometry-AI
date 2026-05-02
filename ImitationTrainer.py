import argparse
import csv
import glob
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from Actor import (
    ActionPerturbationConfig,
    AttemptSample,
    AttemptWriter,
    VirtualGamepadActionOutput,
    controller_to_action,
    rising_edge,
)
from Car import Car
from Map import Map
from NeuralPolicy import NeuralPolicy
from ObservationEncoder import ObservationEncoder
from SupervisedTraining import (
    build_dataset_from_attempts,
    choose_batch_size,
    choose_device,
    choose_num_workers,
    parse_bool,
    parse_csv_values,
    parse_int_list,
)
from XboxController import XboxControllerReader, XboxControllerState


def find_latest_supervised_model(pattern: str = "logs/supervised_runs/**/best_model.pt") -> str:
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError("No supervised model found under logs/supervised_runs.")
    return max(files, key=os.path.getmtime)


def find_attempt_files_in_roots(roots: Sequence[str]) -> List[str]:
    files: List[str] = []
    for root in roots:
        root = str(root).strip()
        if not root:
            continue
        files.extend(glob.glob(os.path.join(root, "**", "attempts", "attempt_*.npz"), recursive=True))
        files.extend(glob.glob(os.path.join(root, "attempts", "attempt_*.npz"), recursive=True))
    return sorted(set(files))


def count_attempt_frames(attempt_files: Sequence[str]) -> int:
    total = 0
    for path in attempt_files:
        try:
            with np.load(path) as data:
                total += int(data["observations"].shape[0])
        except Exception:
            # Frame count is only used for user feedback; training will surface
            # any real dataset issue through the normal loading path.
            continue
    return total


def make_grad_scaler(device: torch.device):
    amp_enabled = device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=amp_enabled)
    return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def canonicalize_action(action: np.ndarray, binarize_gas_brake: bool) -> np.ndarray:
    clean = np.asarray(action, dtype=np.float32).reshape(3).copy()
    clean[0] = float(np.clip(clean[0], 0.0, 1.0))
    clean[1] = float(np.clip(clean[1], 0.0, 1.0))
    clean[2] = float(np.clip(clean[2], -1.0, 1.0))
    if binarize_gas_brake:
        clean[0] = 1.0 if clean[0] > 0.5 else 0.0
        clean[1] = 1.0 if clean[1] > 0.5 else 0.0
    return clean.astype(np.float32, copy=False)


@dataclass
class ImitationMixConfig:
    mode: str = "switch"
    num_attempts: int = 20
    initial_agent_probability: float = 0.0
    max_agent_probability: float = 1.00
    agent_hold_seconds: float = 0.10
    random_seed: int = 20260502
    binarize_executed_gas_brake: bool = False

    def probability_for_saved_attempts(self, saved_attempts: int) -> float:
        if self.num_attempts <= 1:
            return float(np.clip(self.initial_agent_probability, 0.0, self.max_agent_probability))

        curriculum_position = float(np.clip(saved_attempts, 0, self.num_attempts - 1))
        curriculum_ratio = curriculum_position / float(self.num_attempts - 1)
        probability = (
            self.initial_agent_probability
            + curriculum_ratio * (self.max_agent_probability - self.initial_agent_probability)
        )
        return float(np.clip(probability, 0.0, self.max_agent_probability))

    def is_complete(self, saved_attempts: int) -> bool:
        return int(saved_attempts) >= int(self.num_attempts)

    def as_dict(self) -> Dict[str, float | int | str | bool]:
        return {
            "mode": self.mode,
            "num_attempts": int(self.num_attempts),
            "initial_agent_probability": float(self.initial_agent_probability),
            "max_agent_probability": float(self.max_agent_probability),
            "agent_hold_seconds": float(self.agent_hold_seconds),
            "random_seed": int(self.random_seed),
            "binarize_executed_gas_brake": bool(self.binarize_executed_gas_brake),
        }


class HumanAgentMixer:
    def __init__(self, config: ImitationMixConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(int(config.random_seed))
        self.agent_until = 0.0

    def mix(
        self,
        human_action: np.ndarray,
        agent_action: np.ndarray,
        agent_probability: float,
    ) -> Tuple[np.ndarray, str]:
        human = canonicalize_action(human_action, binarize_gas_brake=False)
        agent = canonicalize_action(agent_action, binarize_gas_brake=False)
        probability = float(np.clip(agent_probability, 0.0, 1.0))
        mode = str(self.config.mode).strip().lower()

        if mode == "human":
            return canonicalize_action(human, self.config.binarize_executed_gas_brake), "human"

        if mode == "agent":
            return canonicalize_action(agent, self.config.binarize_executed_gas_brake), "agent"

        if mode == "blend":
            if probability <= 1e-6:
                return canonicalize_action(human, self.config.binarize_executed_gas_brake), "human"
            if probability >= 1.0 - 1e-6:
                return canonicalize_action(agent, self.config.binarize_executed_gas_brake), "agent"
            mixed = (1.0 - probability) * human + probability * agent
            return canonicalize_action(mixed, self.config.binarize_executed_gas_brake), "blend"

        if mode != "switch":
            raise ValueError("mix mode must be one of: human, agent, switch, blend")

        now = time.perf_counter()
        if now < self.agent_until:
            return canonicalize_action(agent, self.config.binarize_executed_gas_brake), "agent_hold"

        if self.rng.random() < probability:
            self.agent_until = now + max(0.0, float(self.config.agent_hold_seconds))
            return canonicalize_action(agent, self.config.binarize_executed_gas_brake), "agent"

        return canonicalize_action(human, self.config.binarize_executed_gas_brake), "human"


def build_policy(
    initial_model_path: str,
    obs_dim: int,
    hidden_dims: Sequence[int],
    hidden_activations: Sequence[str],
    device: torch.device,
) -> Tuple[NeuralPolicy, str]:
    normalized = str(initial_model_path).strip()
    if normalized.lower() == "latest":
        normalized = find_latest_supervised_model()

    if normalized and normalized.lower() not in {"none", "random"}:
        policy, _ = NeuralPolicy.load(normalized, map_location=device)
        if int(policy.obs_dim) != int(obs_dim):
            raise ValueError(
                f"Loaded model obs_dim={policy.obs_dim}, but selected observation layout has obs_dim={obs_dim}."
            )
        policy.to(device)
        return policy, normalized

    policy = NeuralPolicy(
        obs_dim=obs_dim,
        hidden_dim=tuple(hidden_dims),
        act_dim=3,
        action_mode="target",
        hidden_activation=tuple(hidden_activations),
        action_scale=np.ones(3, dtype=np.float32),
        device=device,
    )
    return policy, "random"


def fine_tune_policy(
    policy: NeuralPolicy,
    attempt_files: Sequence[str],
    vertical_mode: bool,
    multi_surface_mode: bool,
    mask_feature_names: Sequence[str],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    boring_keep_probability: float,
    augment_mirror: bool,
    random_seed: int,
    batch_size_override: int | None,
) -> Dict[str, float | int]:
    if not attempt_files:
        return {"frames": 0, "epochs": 0, "loss": float("nan")}

    rng = np.random.default_rng(int(random_seed))
    observations, actions, stats = build_dataset_from_attempts(
        attempt_files=attempt_files,
        rng=rng,
        boring_keep_probability=boring_keep_probability,
        max_frames_after_filter=None,
        apply_boring_filter=True,
        apply_frame_cap=False,
        augment_mirror=augment_mirror,
        target_vertical_mode=vertical_mode,
        target_multi_surface_mode=multi_surface_mode,
        mask_feature_names=mask_feature_names,
    )

    device = policy.device
    batch_size = int(batch_size_override) if batch_size_override else choose_batch_size(len(observations), device)
    num_workers = choose_num_workers(device)
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        TensorDataset(torch.from_numpy(observations), torch.from_numpy(actions)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    optimizer = optim.Adam(policy.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    criterion = nn.SmoothL1Loss(reduction="none")
    loss_weights = torch.tensor([1.0, 1.0, 3.0], dtype=torch.float32, device=device)
    amp_enabled = device.type == "cuda"
    scaler = make_grad_scaler(device)
    last_loss = float("nan")

    policy.train()
    for _ in range(max(1, int(epochs))):
        running_loss = 0.0
        count = 0
        for batch_obs, batch_actions in loader:
            batch_obs = batch_obs.to(device, non_blocking=pin_memory)
            batch_actions = batch_actions.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                prediction = policy(batch_obs)
                loss = (criterion(prediction, batch_actions) * loss_weights).mean()
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_loss += float(loss.detach().cpu()) * int(batch_obs.shape[0])
            count += int(batch_obs.shape[0])
        last_loss = running_loss / max(count, 1)

    return {
        "frames": int(observations.shape[0]),
        "attempt_files": int(len(attempt_files)),
        "epochs": int(epochs),
        "loss": float(last_loss),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "raw_frames": int(stats.get("total_frames_before_filter", 0)),
    }


def write_event_row(path: str, row: Dict[str, object]) -> None:
    headers = [
        "attempt_index",
        "saved_attempts",
        "event",
        "agent_probability",
        "mix_mode",
        "frames",
        "finish_time",
        "dense_progress",
        "discrete_progress",
        "distance",
        "train_loss",
        "train_frames",
        "train_attempt_files",
        "train_seconds",
        "model_path",
    ]
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in headers})


def snapshot_to_gamepad_state(snapshot: XboxControllerState) -> XboxControllerState:
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive DAgger-style imitation trainer for Trackmania.")
    parser.add_argument("--map-name", default="AI Training #5")
    parser.add_argument("--vertical-mode", type=parse_bool, default=False)
    parser.add_argument("--multi-surface-mode", type=parse_bool, default=False)
    parser.add_argument("--data-root", default="logs/imitation_data")
    parser.add_argument("--output-root", default="logs/imitation_runs")
    parser.add_argument("--base-training-data-root", default="logs/supervised_data")
    parser.add_argument(
        "--update-data-scope",
        default="attempt",
        choices=["attempt", "session", "all"],
        help=(
            "Data used after A confirmation. 'attempt' fine-tunes only on the "
            "newly accepted attempt; 'session' uses all accepted attempts from "
            "this imitation run; 'all' also includes base supervised data."
        ),
    )
    parser.add_argument("--initial-model-path", default="latest", help="Path, latest, random, or none.")
    parser.add_argument("--hidden-dim", default="32,16")
    parser.add_argument("--hidden-activation", default="relu,tanh")
    parser.add_argument("--mix-mode", default="blend", choices=["human", "agent", "switch", "blend"])
    parser.add_argument(
        "--num-attempts",
        type=int,
        default=20,
        help=(
            "Number of accepted attempts in the imitation curriculum. "
            "Agent control probability is interpolated linearly over these attempts."
        ),
    )
    parser.add_argument("--initial-agent-probability", type=float, default=0.0)
    parser.add_argument("--max-agent-probability", type=float, default=1.00)
    parser.add_argument("--agent-hold-seconds", type=float, default=0.10)
    parser.add_argument("--binarize-executed-gas-brake", action="store_true")
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--boring-keep-probability", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--disable-mirror", action="store_true")
    parser.add_argument(
        "--mask-feature-names",
        default="steer,gas,brake,input_steer,input_gas,input_brake,previous_steer,previous_gas,previous_brake",
    )
    parser.add_argument("--random-seed", type=int, default=20260502)
    args = parser.parse_args()

    vertical_mode = bool(args.vertical_mode)
    multi_surface_mode = bool(args.multi_surface_mode)
    hidden_dims = parse_int_list(args.hidden_dim)
    hidden_activations = parse_csv_values(args.hidden_activation)
    mask_feature_names = parse_csv_values(args.mask_feature_names)

    encoder = ObservationEncoder(
        dt_ref=1.0 / 100.0,
        dt_ratio_clip=3.0,
        vertical_mode=vertical_mode,
        multi_surface_mode=multi_surface_mode,
    )
    mix_config = ImitationMixConfig(
        mode=args.mix_mode,
        num_attempts=int(args.num_attempts),
        initial_agent_probability=float(args.initial_agent_probability),
        max_agent_probability=float(args.max_agent_probability),
        agent_hold_seconds=float(args.agent_hold_seconds),
        random_seed=int(args.random_seed),
        binarize_executed_gas_brake=bool(args.binarize_executed_gas_brake),
    )
    perturbation_config = ActionPerturbationConfig(enabled=False)
    writer = AttemptWriter(
        base_dir=args.data_root,
        map_name=args.map_name,
        encoder=encoder,
        perturbation_config=perturbation_config,
        apply_to_virtual_gamepad=True,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_map_{args.map_name}_{'v3d' if vertical_mode else 'v2d'}_{'surface' if multi_surface_mode else 'asphalt'}_imitation"
    run_dir = os.path.join(args.output_root, run_name)
    os.makedirs(run_dir, exist_ok=False)
    latest_model_path = os.path.join(run_dir, "latest_model.pt")
    events_path = os.path.join(run_dir, "imitation_events.csv")

    device = choose_device()
    policy, loaded_model = build_policy(
        initial_model_path=args.initial_model_path,
        obs_dim=encoder.obs_dim,
        hidden_dims=hidden_dims,
        hidden_activations=hidden_activations,
        device=device,
    )
    policy.save(
        latest_model_path,
        extra={
            "source_model": loaded_model,
            "vertical_mode": vertical_mode,
            "multi_surface_mode": multi_surface_mode,
            "observation_layout": ObservationEncoder.feature_names(vertical_mode, multi_surface_mode),
        },
    )

    config = {
        "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "map_name": args.map_name,
        "writer_run_dir": writer.run_dir,
        "run_dir": run_dir,
        "loaded_model": loaded_model,
        "latest_model_path": latest_model_path,
        "vertical_mode": vertical_mode,
        "multi_surface_mode": multi_surface_mode,
        "observation_dim": encoder.obs_dim,
        "observation_layout": ObservationEncoder.feature_names(vertical_mode, multi_surface_mode),
        "mix_config": mix_config.as_dict(),
        "base_training_data_root": args.base_training_data_root,
        "update_data_scope": args.update_data_scope,
        "update_epochs": int(args.update_epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "boring_keep_probability": float(args.boring_keep_probability),
        "mirror_augmentation": not bool(args.disable_mirror),
        "device": str(device),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=True)

    print(f"Imitation run: {run_dir}")
    print(f"Dataset attempts are saved into: {writer.run_dir}")
    print(f"Loaded model: {loaded_model}")
    print(f"Update data scope: {args.update_data_scope}")
    print("Workflow:")
    print("- Trackmania should listen to the virtual gamepad, not the physical controller.")
    print("- Drive normally; the script sometimes lets the agent act or blends it in.")
    print("- After a finish, press A to save + update the model, or B to discard.")
    print("- Press B during an attempt to discard/restart.")
    print("- Stop with Ctrl+C.")
    print(
        "Curriculum: "
        f"{mix_config.num_attempts} accepted attempts, "
        f"p_agent {mix_config.initial_agent_probability:.3f} -> "
        f"{mix_config.max_agent_probability:.3f}."
    )

    game_map = Map(args.map_name)
    car = Car(game_map, vertical_mode=vertical_mode)
    controller = XboxControllerReader()
    mixer = HumanAgentMixer(mix_config)
    action_output = VirtualGamepadActionOutput()

    attempt_index = 1
    saved_attempts = 0
    state = "waiting_for_start"
    attempt_samples: List[AttemptSample] = []
    last_buttons = {"a": 0, "b": 0}
    finish_info: Dict[str, float] = {}
    current_agent_probability = mix_config.probability_for_saved_attempts(saved_attempts)
    waiting_reset_notice_printed = False

    try:
        while True:
            distances, instructions, info = car.get_data()
            snapshot = controller.snapshot()
            human_action = controller_to_action(snapshot)

            game_time = float(info.get("time", 0.0))
            discrete_progress = float(info.get("discrete_progress", 0.0))
            dense_progress = float(info.get("dense_progress", discrete_progress))
            total_distance = float(info.get("distance", 0.0))
            finished = bool(info.get("done", 0.0) == 1.0)

            if state == "waiting_for_start" and game_time > 0.0:
                encoder.reset()
                attempt_samples = []
                current_agent_probability = mix_config.probability_for_saved_attempts(saved_attempts)
                state = "recording"
                print(
                    f"Attempt {attempt_index:04d} started | "
                    f"agent_probability={current_agent_probability:.3f}"
                )

            observation = encoder.build_observation(
                distances=distances,
                instructions=instructions,
                info=info,
            )
            agent_action = policy.act(observation)
            executed_action, action_source = mixer.mix(
                human_action=human_action,
                agent_action=agent_action,
                agent_probability=current_agent_probability,
            )
            action_output.apply(executed_action, snapshot_to_gamepad_state(snapshot))

            a_pressed = rising_edge(snapshot.button_a, last_buttons["a"])
            b_pressed = rising_edge(snapshot.button_b, last_buttons["b"])
            last_buttons["a"] = snapshot.button_a
            last_buttons["b"] = snapshot.button_b

            if state == "recording":
                crashes = int(info.get("crashes", 0))
                timeout = int(info.get("timeout", 0))
                attempt_samples.append(
                    AttemptSample(
                        observation=observation,
                        action=human_action.copy(),
                        executed_action=executed_action.copy(),
                        game_time=game_time,
                        discrete_progress=discrete_progress,
                        dense_progress=dense_progress,
                        distance=total_distance,
                        speed=float(info.get("speed", 0.0)),
                        side_speed=float(info.get("side_speed", 0.0)),
                        x=float(info.get("x", 0.0)),
                        y=float(info.get("y", 0.0)),
                        z=float(info.get("z", 0.0)),
                        dx=float(info.get("dx", 0.0)),
                        dy=float(info.get("dy", 0.0)),
                        dz=float(info.get("dz", 0.0)),
                        slip_mean=float(info.get("slip_mean", 0.0)),
                        dt_ratio=float(info.get("dt_ratio", 1.0)),
                        finished=int(finished),
                        crashes=crashes,
                        timeout=timeout,
                    )
                )

                print(
                    f"attempt={attempt_index:04d} | saved={saved_attempts:03d} | "
                    f"p_agent={current_agent_probability:.3f} | src={action_source:<10} | "
                    f"prog={dense_progress:6.2f} | t={game_time:6.2f} | "
                    f"speed={float(info.get('speed', 0.0)):6.2f}        ",
                    end="\r",
                )

                if b_pressed:
                    print(f"\nAttempt {attempt_index:04d} discarded by B restart.")
                    writer.log_discard(
                        attempt_index,
                        attempt_samples,
                        dict(
                            time=game_time,
                            discrete_progress=discrete_progress,
                            dense_progress=dense_progress,
                            distance=total_distance,
                            finished=0,
                            crashes=crashes,
                            timeout=timeout,
                        ),
                    )
                    write_event_row(
                        events_path,
                        {
                            "attempt_index": attempt_index,
                            "saved_attempts": saved_attempts,
                            "event": "discard_during_attempt",
                            "agent_probability": current_agent_probability,
                            "mix_mode": mix_config.mode,
                            "frames": len(attempt_samples),
                            "finish_time": game_time,
                            "dense_progress": dense_progress,
                            "discrete_progress": discrete_progress,
                            "distance": total_distance,
                            "model_path": latest_model_path,
                        },
                    )
                    attempt_index += 1
                    attempt_samples = []
                    state = "waiting_for_reset"
                    waiting_reset_notice_printed = False
                elif finished:
                    finish_info = dict(
                        time=game_time,
                        discrete_progress=discrete_progress,
                        dense_progress=dense_progress,
                        distance=total_distance,
                        finished=1,
                        crashes=0,
                        timeout=0,
                    )
                    print(
                        f"\nAttempt {attempt_index:04d} finished in {game_time:.2f}s. "
                        "Press A to save/train or B to discard."
                    )
                    state = "await_finish_confirmation"
                elif game_time <= 0.0:
                    print(f"\nAttempt {attempt_index:04d} reset before finish. Discarded.")
                    writer.log_discard(
                        attempt_index,
                        attempt_samples,
                        dict(
                            time=game_time,
                            discrete_progress=discrete_progress,
                            dense_progress=dense_progress,
                            distance=total_distance,
                            finished=0,
                            crashes=0,
                            timeout=0,
                        ),
                    )
                    attempt_index += 1
                    attempt_samples = []
                    state = "waiting_for_start"

            elif state == "await_finish_confirmation":
                if a_pressed:
                    output_path = writer.save_attempt(attempt_index, attempt_samples, finish_info)
                    print(f"Attempt {attempt_index:04d} saved: {output_path}")
                    saved_attempts += 1
                    if args.update_data_scope == "attempt":
                        attempt_files = [output_path]
                    elif args.update_data_scope == "all":
                        training_roots = [args.base_training_data_root, writer.run_dir]
                        attempt_files = find_attempt_files_in_roots(training_roots)
                    else:
                        training_roots = [writer.run_dir]
                        attempt_files = find_attempt_files_in_roots(training_roots)
                    train_frame_estimate = count_attempt_frames(attempt_files)
                    train_started_at = time.perf_counter()
                    print(
                        "Fitting model... "
                        f"attempts={len(attempt_files)} | "
                        f"frames={train_frame_estimate} | "
                        f"epochs={int(args.update_epochs)} | "
                        f"scope={args.update_data_scope}"
                    )
                    train_stats = fine_tune_policy(
                        policy=policy,
                        attempt_files=attempt_files,
                        vertical_mode=vertical_mode,
                        multi_surface_mode=multi_surface_mode,
                        mask_feature_names=mask_feature_names,
                        epochs=int(args.update_epochs),
                        learning_rate=float(args.learning_rate),
                        weight_decay=float(args.weight_decay),
                        boring_keep_probability=float(args.boring_keep_probability),
                        augment_mirror=not bool(args.disable_mirror),
                        random_seed=int(args.random_seed) + saved_attempts,
                        batch_size_override=args.batch_size,
                    )
                    policy.save(
                        latest_model_path,
                        extra={
                            "saved_attempts": saved_attempts,
                            "last_train_stats": train_stats,
                            "vertical_mode": vertical_mode,
                            "multi_surface_mode": multi_surface_mode,
                            "observation_layout": ObservationEncoder.feature_names(vertical_mode, multi_surface_mode),
                        },
                    )
                    train_seconds = time.perf_counter() - train_started_at
                    print(
                        f"Model updated | saved_attempts={saved_attempts} | "
                        f"loss={float(train_stats['loss']):.6f} | "
                        f"frames={int(train_stats['frames'])} | "
                        f"update_s={train_seconds:.2f} | "
                        f"next_p_agent={mix_config.probability_for_saved_attempts(saved_attempts):.3f}"
                    )
                    print("Press B/reset in Trackmania to start next attempt.")
                    write_event_row(
                        events_path,
                        {
                            "attempt_index": attempt_index,
                            "saved_attempts": saved_attempts,
                            "event": "saved_and_trained",
                            "agent_probability": current_agent_probability,
                            "mix_mode": mix_config.mode,
                            "frames": len(attempt_samples),
                            "finish_time": float(finish_info.get("time", 0.0)),
                            "dense_progress": float(finish_info.get("dense_progress", 0.0)),
                            "discrete_progress": float(finish_info.get("discrete_progress", 0.0)),
                            "distance": float(finish_info.get("distance", 0.0)),
                            "train_loss": float(train_stats["loss"]),
                            "train_frames": int(train_stats["frames"]),
                            "train_attempt_files": int(train_stats["attempt_files"]),
                            "train_seconds": float(train_seconds),
                            "model_path": latest_model_path,
                        },
                    )
                    if mix_config.is_complete(saved_attempts):
                        print(
                            f"Imitation curriculum complete after {saved_attempts} "
                            "accepted attempts."
                        )
                        break
                    attempt_index += 1
                    attempt_samples = []
                    state = "waiting_for_reset"
                    waiting_reset_notice_printed = False
                elif b_pressed:
                    print(f"Attempt {attempt_index:04d} discarded after finish.")
                    writer.log_discard(attempt_index, attempt_samples, finish_info)
                    attempt_index += 1
                    attempt_samples = []
                    state = "waiting_for_reset"
                    waiting_reset_notice_printed = False

            elif state == "waiting_for_reset":
                if not waiting_reset_notice_printed:
                    print("Waiting for Trackmania reset... press B/reset if the next attempt has not started.")
                    waiting_reset_notice_printed = True
                if game_time <= 0.0:
                    state = "waiting_for_start"
                    waiting_reset_notice_printed = False

    except KeyboardInterrupt:
        print("\nStopped imitation training.")
    finally:
        action_output.close()
        controller.close()


if __name__ == "__main__":
    main()
