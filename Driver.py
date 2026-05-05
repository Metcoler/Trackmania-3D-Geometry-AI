import argparse
import glob
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Enviroment import RacingGameEnviroment
from NeuralPolicy import NeuralPolicy
from Individual import Individual


MAP_SPECIALIST_NAMES = (
    "single_surface_flat",
    "multi_surface_flat",
    "single_surface_height",
)
PRACTICAL_INFINITY_TOUCHES = 1_000_000


def find_latest_population(patterns: Optional[List[str]] = None) -> str:
    """Find newest population .npz checkpoint (mini pretrain or TM GA run)."""
    if patterns is None:
        patterns = [
            "Cars Evolution Training Project/logs/mini_pretrain_runs/**/checkpoints/population_gen_*.npz",
            "Cars Evolution Training Project/logs/mini_pretrain_runs/**/final_population.npz",
            "logs/mini_pretrain_runs/**/checkpoints/population_gen_*.npz",
            "logs/mini_pretrain_runs/**/final_population.npz",
            "logs/ga_runs/**/checkpoints/population_gen_*.npz",
            "logs/ga_runs/**/final_population.npz",
            "logs/ga_last_population_*.npz",  # legacy
        ]

    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))

    if not files:
        raise FileNotFoundError(
            "No population .npz found. Check logs/mini_pretrain_runs or logs/ga_runs."
        )

    return max(files, key=os.path.getmtime)


def find_latest_supervised_model(
    pattern: str = "logs/supervised_runs/**/best_model.pt",
) -> str:
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(
            "No supervised model found. Check logs/supervised_runs for best_model.pt."
        )
    return max(files, key=os.path.getmtime)


def find_latest_policy_model() -> str:
    """Find newest deployable NeuralPolicy model, preferring imitation runs."""
    patterns = [
        "logs/imitation_runs/**/latest_model.pt",
        "logs/supervised_runs/**/best_model.pt",
    ]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))

    if not files:
        raise FileNotFoundError(
            "No policy model found. Check logs/imitation_runs for latest_model.pt "
            "or logs/supervised_runs for best_model.pt."
        )

    return max(files, key=os.path.getmtime)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def parse_max_touches(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"inf", "infinite", "infinity", "nekonecno", "nekonečno"}:
        return PRACTICAL_INFINITY_TOUCHES
    touches = int(text)
    if touches < 1:
        raise argparse.ArgumentTypeError("--max-touches must be >= 1 or 'inf'.")
    return touches


def infer_map_name_from_model(model_file: str, extra: Optional[Dict[str, Any]] = None) -> Optional[str]:
    candidates: List[str] = [str(model_file)]
    if extra:
        attempt_files = extra.get("attempt_files") or []
        if isinstance(attempt_files, (list, tuple)):
            candidates.extend(str(path) for path in attempt_files)
    for candidate in candidates:
        normalized = candidate.replace("\\", "/")
        for map_name in MAP_SPECIALIST_NAMES:
            if f"/{map_name}/" in normalized or f"map_{map_name}_" in normalized:
                return map_name
        match = re.search(r"map_(.+?)_v[23]d_", normalized)
        if match:
            return match.group(1)
    return None


def find_latest_map_specialist_model(
    specialist_root: str = "logs/supervised_runs_map_specialists_20260505",
    map_name: Optional[str] = None,
) -> str:
    root = Path(specialist_root)
    if map_name:
        search_root = root / map_name
        if not search_root.exists():
            search_root = root
        patterns = [
            str(search_root / "**" / "best_model.pt"),
            str(root / f"**{map_name}**" / "**" / "best_model.pt"),
        ]
    else:
        patterns = [str(root / "**" / "best_model.pt")]

    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    files = sorted(set(files), key=os.path.getmtime)
    if not files:
        if map_name:
            raise FileNotFoundError(
                f"No map-specialist best_model.pt found for {map_name!r} under {specialist_root!r}."
            )
        raise FileNotFoundError(f"No map-specialist best_model.pt found under {specialist_root!r}.")
    return files[-1]


def infer_hidden_dim(genome_size: int, obs_dim: int, act_dim: int) -> int:
    """
    Infer hidden_dim from flattened MLP genome:

        genome_size = H*(obs_dim + 1) + act_dim*(H + 1)
                    = H*(obs_dim + 1 + act_dim) + act_dim
    """
    denom = obs_dim + 1 + act_dim
    num = genome_size - act_dim
    if denom <= 0:
        raise ValueError("Invalid dimensions for infer_hidden_dim.")

    if num % denom != 0:
        raise ValueError(
            f"Genome size {genome_size} is not compatible with "
            f"obs_dim={obs_dim}, act_dim={act_dim}."
        )

    hidden_dim = num // denom
    if hidden_dim <= 0:
        raise ValueError(
            f"Inferred hidden_dim={hidden_dim}, which is invalid. Check network architecture."
        )
    return hidden_dim


def _read_optional_array(data, key: str) -> Optional[np.ndarray]:
    return np.asarray(data[key]) if key in data.files else None


def _read_optional_scalar_int(data, key: str) -> Optional[int]:
    if key not in data.files:
        return None
    arr = np.asarray(data[key]).reshape(-1)
    if arr.size == 0:
        return None
    return int(arr[0])


def _read_optional_int_tuple(data, key: str) -> Optional[Tuple[int, ...]]:
    if key not in data.files:
        return None
    arr = np.asarray(data[key]).reshape(-1)
    if arr.size == 0:
        return None
    return tuple(int(value) for value in arr)


def _read_optional_str_tuple(data, key: str) -> Optional[Tuple[str, ...]]:
    if key not in data.files:
        return None
    arr = np.asarray(data[key]).reshape(-1)
    if arr.size == 0:
        return None
    return tuple(str(value) for value in arr)


def load_population(
    filename: str,
) -> Tuple[np.ndarray, Dict[str, Optional[np.ndarray]], Dict[str, Any]]:
    """
    Load genomes + optional metrics/meta from population checkpoint.

    Supported formats:
    - mini tkinter pretrain checkpoints (population_gen_XXXX.npz)
    - Trackmania GA checkpoints/final_population.npz
    - legacy ga_last_population_*.npz (if it contains genomes)
    """
    with np.load(filename) as data:
        if "genomes" not in data.files:
            raise ValueError(f"File '{filename}' does not contain 'genomes'.")

        genomes = np.asarray(data["genomes"], dtype=np.float32)
        if genomes.ndim != 2:
            raise ValueError(f"Expected 2D 'genomes' array, got shape {genomes.shape}.")

        metrics: Dict[str, Optional[np.ndarray]] = {
            "fitnesses": _read_optional_array(data, "fitnesses"),
            "progresses": _read_optional_array(data, "progresses"),
            "times": _read_optional_array(data, "times"),
            "finisheds": _read_optional_array(data, "finisheds"),
            "crashes": _read_optional_array(data, "crashes"),
            "distances": _read_optional_array(data, "distances"),
        }
        meta: Dict[str, Any] = {
            "generation": _read_optional_scalar_int(data, "generation"),
            "obs_dim": _read_optional_scalar_int(data, "obs_dim"),
            "hidden_dim": _read_optional_scalar_int(data, "hidden_dim"),
            "act_dim": _read_optional_scalar_int(data, "act_dim"),
            "vertical_mode": _read_optional_scalar_int(data, "vertical_mode"),
            "multi_surface_mode": _read_optional_scalar_int(data, "multi_surface_mode"),
        }
        hidden_dims = _read_optional_int_tuple(data, "hidden_dims")
        if hidden_dims is not None:
            meta["hidden_dims"] = hidden_dims
        hidden_activations = _read_optional_str_tuple(data, "hidden_activations")
        if hidden_activations is None:
            hidden_activations = _read_optional_str_tuple(data, "hidden_activation")
        if hidden_activations is not None:
            meta["hidden_activations"] = hidden_activations
        observation_layout = _read_optional_str_tuple(data, "observation_layout")
        if observation_layout is not None:
            meta["observation_layout"] = observation_layout

    return genomes, metrics, meta


def _build_replay_indices(
    pop_size: int,
    fitnesses: Optional[np.ndarray],
    sort_by_fitness: bool,
    rank_start: int,
    rank_end: Optional[int],
    exact_indices: Optional[List[int]],
) -> np.ndarray:
    if exact_indices is not None and len(exact_indices) > 0:
        selected: List[int] = []
        for idx in exact_indices:
            i = int(idx)
            if i < 0:
                i = pop_size + i
            if i < 0 or i >= pop_size:
                raise IndexError(f"Index {idx} is out of range 0..{pop_size-1}.")
            selected.append(i)
        return np.asarray(selected, dtype=np.int32)

    indices = np.arange(pop_size, dtype=np.int32)
    if sort_by_fitness and fitnesses is not None:
        fitnesses_safe = np.array(
            [(-np.inf if np.isnan(f) else float(f)) for f in fitnesses],
            dtype=np.float32,
        )
        indices = np.argsort(-fitnesses_safe).astype(np.int32)  # descending

    # 1-based rank slice for convenience when replaying "top N"
    start_zero = max(0, int(rank_start) - 1)
    end_zero = None if rank_end is None else int(rank_end)
    return indices[start_zero:end_zero]


def _wait_for_positive_game_time(
    env: RacingGameEnviroment,
    timeout_seconds: float = 5.0,
):
    observation = getattr(env, "previous_observation", None)
    distances, instructions, info = getattr(env, "previous_observation_info", (None, None, {}))
    if float(info.get("time", 0.0)) > 0.0 and observation is not None:
        return observation, info

    deadline = time.perf_counter() + float(timeout_seconds)
    last_info = dict(info)
    while time.perf_counter() < deadline:
        distances, instructions, info = env.observation_info()
        last_info = info
        if float(info.get("time", 0.0)) > 0.0:
            observation = env.build_observation(
                distances=distances,
                instructions=instructions,
                info=info,
            )
            env.previous_observation_info = (distances, instructions, info)
            env.previous_observation = observation
            return observation, info
    return observation, last_info


def _apply_target_steer_deadzone(action: np.ndarray, steer_deadzone: float) -> np.ndarray:
    if steer_deadzone <= 0.0:
        return action
    adjusted = np.asarray(action, dtype=np.float32).copy()
    if adjusted.shape == (3,) and abs(float(adjusted[2])) < float(steer_deadzone):
        adjusted[2] = 0.0
    return adjusted


def _maybe_invert_steer(action: np.ndarray, invert_steer: bool) -> np.ndarray:
    if not invert_steer:
        return action
    adjusted = np.asarray(action, dtype=np.float32).copy()
    if adjusted.ndim == 1 and adjusted.shape[0] >= 3:
        adjusted[2] = -adjusted[2]
    return adjusted


def replay_population(
    map_name: str = "small_map",
    population_file: Optional[str] = None,
    episodes_per_individual: int = 1,
    max_steps: Optional[int] = None,
    env_max_time: float = 60.0,
    max_touches: int = 1,
    never_quit: bool = True,
    action_mode: str = "delta",
    vertical_mode: bool = True,
    multi_surface_mode: bool = True,
    pause_between: bool = True,
    sort_by_fitness: bool = True,
    rank_start: int = 1,
    rank_end: Optional[int] = None,
    exact_indices: Optional[List[int]] = None,
    target_steer_deadzone: float = 0.0,
    invert_steer: bool = False,
) -> None:
    """
    Replay a whole population or selected subset in Trackmania.

    Selection modes:
    - exact_indices=[...] : replay specific indices from saved population
    - otherwise ranks <rank_start, rank_end> after optional fitness sorting
    """
    if population_file is None:
        population_file = find_latest_population()

    print(f"Loading population from: {population_file}")
    genomes, metrics, meta = load_population(population_file)
    fitnesses = metrics.get("fitnesses")
    pop_size, genome_size = genomes.shape

    print(f"Loaded {pop_size} individuals, genome_size={genome_size}")
    if meta.get("generation") is not None:
        print(f"Checkpoint generation: {meta['generation']}")
    if meta.get("vertical_mode") is not None:
        print(f"Checkpoint vertical_mode: {bool(meta['vertical_mode'])}")
    if meta.get("multi_surface_mode") is not None:
        print(f"Checkpoint multi_surface_mode: {bool(meta['multi_surface_mode'])}")

    env = RacingGameEnviroment(
        map_name=map_name,
        never_quit=never_quit,
        action_mode=action_mode,
        vertical_mode=vertical_mode,
        multi_surface_mode=multi_surface_mode,
        max_time=env_max_time,
        max_touches=max_touches,
    )
    obs, info = env.reset()
    obs_dim = obs.shape[0]

    try:
        try:
            act_dim = int(env.action_space.shape[0])
        except Exception:
            act_dim = 3

        file_obs_dim = meta.get("obs_dim")
        file_hidden_dims = meta.get("hidden_dims")
        file_hidden_dim = meta.get("hidden_dim")
        file_hidden_activations = meta.get("hidden_activations")
        file_act_dim = meta.get("act_dim")

        if file_obs_dim is not None and file_obs_dim != obs_dim:
            print(
                f"WARNING: checkpoint obs_dim={file_obs_dim} does not match env obs_dim={obs_dim}"
            )
        file_vertical_mode = meta.get("vertical_mode")
        if file_vertical_mode is not None and bool(file_vertical_mode) != bool(vertical_mode):
            print(
                "WARNING: checkpoint vertical_mode does not match replay vertical_mode. "
                "Set Driver.VERTICAL_MODE accordingly."
            )
        file_multi_surface_mode = meta.get("multi_surface_mode")
        if file_multi_surface_mode is not None and bool(file_multi_surface_mode) != bool(multi_surface_mode):
            print(
                "WARNING: checkpoint multi_surface_mode does not match replay multi_surface_mode. "
                "Set Driver.MULTI_SURFACE_MODE accordingly."
            )
        if file_act_dim is not None and file_act_dim != act_dim:
            print(
                f"WARNING: checkpoint act_dim={file_act_dim} does not match env act_dim={act_dim}"
            )

        if file_hidden_dims is not None:
            hidden_dim = tuple(int(dim) for dim in file_hidden_dims)
        elif file_hidden_dim is not None and file_hidden_dim > 0:
            hidden_dim = int(file_hidden_dim)
        else:
            hidden_dim = infer_hidden_dim(genome_size, obs_dim, act_dim)

        hidden_activation: Any = "tanh"
        if file_hidden_activations is not None:
            file_hidden_activations = tuple(str(value) for value in file_hidden_activations)
            hidden_activation = (
                file_hidden_activations[0]
                if len(file_hidden_activations) == 1
                else list(file_hidden_activations)
            )

        print(
            f"Architecture: obs_dim={obs_dim}, hidden_dim={hidden_dim}, "
            f"hidden_activation={hidden_activation}, act_dim={act_dim}"
        )
        print(f"Replay steer inversion: {bool(invert_steer)}")

        indices = _build_replay_indices(
            pop_size=pop_size,
            fitnesses=fitnesses,
            sort_by_fitness=sort_by_fitness,
            rank_start=rank_start,
            rank_end=rank_end,
            exact_indices=exact_indices,
        )

        if indices.size == 0:
            print("Selection is empty, nothing to replay.")
            return

        if exact_indices:
            print(f"Replaying exact indices: {list(map(int, indices))}")
        else:
            sort_txt = "sorted by saved fitness" if (sort_by_fitness and fitnesses is not None) else "saved order"
            print(
                f"Replaying {indices.size} individuals ({sort_txt}), "
                f"ranks {rank_start}..{rank_end or pop_size}"
            )

        m_progress = metrics.get("progresses")
        m_times = metrics.get("times")
        m_finished = metrics.get("finisheds")
        m_crashes = metrics.get("crashes")
        m_distances = metrics.get("distances")

        for rank_in_selection, idx in enumerate(indices, start=1):
            idx = int(idx)
            genome = genomes[idx]

            print("\n" + "=" * 40)
            print(f"Replay {rank_in_selection}/{indices.size} | population index={idx}")
            if fitnesses is not None:
                fval = float(fitnesses[idx])
                print(f"Saved fitness: {fval:.3f}")
            if m_progress is not None:
                parts = [f"saved progress={float(m_progress[idx]):.2f}%"]
                if m_times is not None:
                    parts.append(f"time={float(m_times[idx]):.2f}")
                if m_distances is not None:
                    parts.append(f"distance={float(m_distances[idx]):.2f}")
                if m_finished is not None:
                    parts.append(f"finished={int(m_finished[idx])}")
                if m_crashes is not None:
                    parts.append(f"crashes={int(m_crashes[idx])}")
                print("Saved metrics: " + ", ".join(parts))
            print("=" * 40)

            individual = Individual(
                obs_dim=obs_dim,
                hidden_dim=hidden_dim,
                act_dim=act_dim,
                genome=genome,
                action_scale=np.ones(act_dim, dtype=np.float32) if str(action_mode).strip().lower() == "target" else None,
                action_mode=action_mode,
                hidden_activation=hidden_activation,
            )

            for ep in range(episodes_per_individual):
                obs, info = env.reset()
                if str(action_mode).strip().lower() == "target":
                    obs, info = _wait_for_positive_game_time(env)
                total_reward = 0.0

                step_count = 0
                while True:
                    if max_steps is not None and step_count >= max_steps:
                        break
                    action = individual.act(obs)
                    if str(action_mode).strip().lower() == "target":
                        action = _apply_target_steer_deadzone(action, target_steer_deadzone)
                    action = _maybe_invert_steer(action, invert_steer)
                    obs, reward, done, truncated, info = env.step(action)
                    total_reward += float(reward)
                    step_count += 1

                    race_term = getattr(env, "race_terminated", 0)
                    info_done = info.get("done", 0.0) == 1.0
                    terminated = done or truncated or info_done or (race_term != 0)
                    if terminated:
                        break

                print(
                    f"  Episode {ep + 1}/{episodes_per_individual} | "
                    f"reward={total_reward:.3f} | "
                    f"finished={int(info.get('finished', getattr(env, 'finished', 0)))} | "
                    f"crashes={int(info.get('crashes', getattr(env, 'crashes', 0)))} | "
                    f"progress={float(info.get('discrete_progress', 0.0)):.2f}% | "
                    f"time={float(info.get('time', 0.0)):.2f}s | "
                    f"distance={float(info.get('distance', 0.0)):.2f}"
                )

            if pause_between and rank_in_selection < indices.size:
                input("Press Enter for next individual...")

    finally:
        env.close()
        print("Environment closed.")


def drive_model(
    map_name: Optional[str],
    model_file: str,
    max_steps: Optional[int] = None,
    env_max_time: float = 60.0,
    max_touches: int = 1,
    never_quit: bool = True,
    action_mode: Optional[str] = None,
    vertical_mode: Optional[bool] = None,
    multi_surface_mode: Optional[bool] = None,
    target_steer_deadzone: float = 0.0,
    invert_steer: Optional[bool] = None,
    wait_for_positive_time: bool = False,
    debug_actions: int = 0,
) -> None:
    policy, extra = NeuralPolicy.load(model_file, map_location="cpu")
    if map_name is None or str(map_name).strip().lower() == "auto":
        map_name = infer_map_name_from_model(model_file, extra)
    if not map_name:
        raise ValueError("Map name could not be inferred. Pass --map-name explicitly.")
    if action_mode is None:
        action_mode = str(getattr(policy, "action_mode", "target"))
    if vertical_mode is None:
        if extra and "vertical_mode" in extra:
            vertical_mode = bool(extra["vertical_mode"])
        else:
            vertical_mode = int(policy.obs_dim) >= 44
    if multi_surface_mode is None:
        if extra and "multi_surface_mode" in extra:
            multi_surface_mode = bool(extra["multi_surface_mode"])
        else:
            multi_surface_mode = int(policy.obs_dim) >= 39
    if invert_steer is None:
        invert_steer = False

    print(f"Loaded model from: {model_file}")
    print(f"Replay map: {map_name}")
    print(
        f"Replay config: action_mode={action_mode}, vertical_mode={vertical_mode}, "
        f"multi_surface_mode={multi_surface_mode}, max_touches={max_touches}, "
        f"invert_steer={bool(invert_steer)}, wait_for_positive_time={bool(wait_for_positive_time)}"
    )
    print(f"Model config: {policy.get_config()}")
    if extra:
        print(f"Model extra: {extra}")
        if "vertical_mode" in extra and bool(extra["vertical_mode"]) != bool(vertical_mode):
            print(
                "WARNING: model vertical_mode does not match replay vertical_mode. "
                "Set Driver.VERTICAL_MODE accordingly."
            )
        if "multi_surface_mode" in extra and bool(extra["multi_surface_mode"]) != bool(multi_surface_mode):
            print(
                "WARNING: model multi_surface_mode does not match replay multi_surface_mode. "
                "Set Driver.MULTI_SURFACE_MODE accordingly."
            )

    env = RacingGameEnviroment(
        map_name=map_name,
        never_quit=never_quit,
        action_mode=action_mode,
        vertical_mode=vertical_mode,
        multi_surface_mode=multi_surface_mode,
        max_time=env_max_time,
        max_touches=max_touches,
    )
    obs, info = env.reset()
    if str(action_mode).strip().lower() == "target" and bool(wait_for_positive_time):
        obs, info = _wait_for_positive_game_time(env)
    if obs.shape[0] != policy.obs_dim:
        raise ValueError(
            f"Model obs_dim={policy.obs_dim} does not match env obs_dim={obs.shape[0]}."
        )
    try:
        total_reward = 0.0
        step_count = 0
        while True:
            if max_steps is not None and step_count >= max_steps:
                break
            raw_action = policy.act(obs)
            action = raw_action
            if str(action_mode).strip().lower() == "target":
                action = _apply_target_steer_deadzone(action, target_steer_deadzone)
            action = _maybe_invert_steer(action, bool(invert_steer))
            if int(debug_actions) > 0 and step_count < int(debug_actions):
                print(
                    "\n"
                    f"debug step={step_count} "
                    f"time={float(info.get('time', 0.0)):.3f} "
                    f"speed={float(info.get('speed', 0.0)):.2f} "
                    f"progress={float(info.get('dense_progress', info.get('progress', 0.0))):.2f} "
                    f"seg_err={float(info.get('segment_heading_error', 0.0)):.3f} "
                    f"next_err={float(info.get('next_segment_heading_error', 0.0)):.3f} "
                    f"raw_action=[{raw_action[0]:.3f},{raw_action[1]:.3f},{raw_action[2]:.3f}] "
                    f"sent_action=[{action[0]:.3f},{action[1]:.3f},{action[2]:.3f}]"
                )
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            step_count += 1
            race_term = getattr(env, "race_terminated", 0)
            info_done = info.get("done", 0.0) == 1.0
            if done or truncated or info_done or (race_term != 0):
                break

        print(
            f"Replay finished | reward={total_reward:.3f} | "
            f"finished={int(info.get('finished', getattr(env, 'finished', 0)))} | "
            f"crashes={int(info.get('crashes', getattr(env, 'crashes', 0)))} | "
            f"progress={float(info.get('progress', info.get('dense_progress', info.get('discrete_progress', 0.0)))):.2f}% | "
            f"block={float(info.get('block_progress', info.get('discrete_progress', 0.0))):.2f}% | "
            f"time={float(info.get('time', 0.0)):.2f}s | "
            f"distance={float(info.get('distance', 0.0)):.2f}"
        )
    finally:
        env.close()
        print("Environment closed.")


def parse_auto_bool_arg(value: str) -> Optional[bool]:
    if str(value).strip().lower() == "auto":
        return None
    return parse_bool(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drive a trained NeuralPolicy in live Trackmania.")
    parser.add_argument("--map-name", default="auto", help="Map to replay. Use 'all' for all map specialists.")
    parser.add_argument("--model-file", default=None, help="Exact .pt model path. If omitted, latest map specialist is used.")
    parser.add_argument(
        "--specialist-root",
        default="logs/supervised_runs_map_specialists_20260505",
        help="Root with map-specific supervised specialists.",
    )
    parser.add_argument("--population-file", default=None, help="Replay a saved population instead of a .pt model.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--env-max-time", type=float, default=60.0)
    parser.add_argument("--max-touches", type=parse_max_touches, default=1)
    parser.add_argument("--never-quit", type=parse_bool, default=True)
    parser.add_argument("--action-mode", choices=("auto", "target", "delta"), default="auto")
    parser.add_argument("--vertical-mode", default="auto", help="auto|true|false")
    parser.add_argument("--multi-surface-mode", default="auto", help="auto|true|false")
    parser.add_argument(
        "--invert-steer",
        default="false",
        help=(
            "auto|true|false. Default false. Use true only for legacy/debug replays that were trained "
            "with an old mirrored observation convention."
        ),
    )
    parser.add_argument("--target-steer-deadzone", type=float, default=0.05)
    parser.add_argument(
        "--wait-for-positive-time",
        type=parse_bool,
        default=False,
        help="Wait for Trackmania game time > 0 before first target action. Usually false because the timer starts after input.",
    )
    parser.add_argument(
        "--debug-actions",
        type=int,
        default=0,
        help="Print the first N live observations/actions before sending them to the gamepad.",
    )
    parser.add_argument("--pause-between-maps", type=parse_bool, default=True)

    parser.add_argument("--episodes-per-individual", type=int, default=1)
    parser.add_argument("--pause-between-individuals", type=parse_bool, default=False)
    parser.add_argument("--sort-by-fitness", type=parse_bool, default=True)
    parser.add_argument("--rank-start", type=int, default=1)
    parser.add_argument("--rank-end", type=int, default=None)
    parser.add_argument("--exact-indices", default="", help="Comma-separated population indices.")
    return parser


def parse_exact_indices(value: str) -> Optional[List[int]]:
    text = str(value).strip()
    if not text:
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    args = build_arg_parser().parse_args()
    vertical_mode = parse_auto_bool_arg(args.vertical_mode)
    multi_surface_mode = parse_auto_bool_arg(args.multi_surface_mode)
    invert_steer = parse_auto_bool_arg(args.invert_steer)
    action_mode = None if args.action_mode == "auto" else args.action_mode

    if args.population_file:
        if args.map_name in {"auto", "all"}:
            raise ValueError("--map-name must be explicit when replaying a population.")
        replay_population(
            map_name=args.map_name,
            population_file=args.population_file,
            episodes_per_individual=args.episodes_per_individual,
            max_steps=args.max_steps,
            env_max_time=args.env_max_time,
            max_touches=args.max_touches,
            never_quit=args.never_quit,
            action_mode=action_mode or "target",
            vertical_mode=True if vertical_mode is None else vertical_mode,
            multi_surface_mode=True if multi_surface_mode is None else multi_surface_mode,
            pause_between=args.pause_between_individuals,
            sort_by_fitness=args.sort_by_fitness,
            rank_start=args.rank_start,
            rank_end=args.rank_end,
            exact_indices=parse_exact_indices(args.exact_indices),
            target_steer_deadzone=args.target_steer_deadzone,
            invert_steer=False if invert_steer is None else bool(invert_steer),
        )
        return

    map_names: List[Optional[str]]
    if str(args.map_name).strip().lower() == "all":
        map_names = list(MAP_SPECIALIST_NAMES)
    else:
        map_names = [None if str(args.map_name).strip().lower() == "auto" else str(args.map_name)]

    for index, map_name in enumerate(map_names):
        model_file = args.model_file
        if model_file is None:
            model_file = find_latest_map_specialist_model(
                specialist_root=args.specialist_root,
                map_name=map_name,
            )
        if index > 0 and args.pause_between_maps:
            input(f"Load map '{map_name}' in Trackmania, then press Enter to continue...")
        drive_model(
            map_name=map_name,
            model_file=model_file,
            max_steps=args.max_steps,
            env_max_time=args.env_max_time,
            max_touches=args.max_touches,
            never_quit=args.never_quit,
            action_mode=action_mode,
            vertical_mode=vertical_mode,
            multi_surface_mode=multi_surface_mode,
            target_steer_deadzone=args.target_steer_deadzone,
            invert_steer=invert_steer,
            wait_for_positive_time=bool(args.wait_for_positive_time),
            debug_actions=int(args.debug_actions),
        )


if __name__ == "__main__":
    main()
