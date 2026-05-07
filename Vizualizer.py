import socket
import sys
import threading
import shutil
from time import sleep, time
import trimesh
import numpy as np
from Map import Map
from Car import Car
from ObservationEncoder import ObservationEncoder
from SurfaceTypes import surface_material_name

CRASH_DEBUG_HOLD_SECONDS = 2.0
crash_debug_latched_until = 0.0
crash_debug_event_count = 0
crash_debug_was_active = False
crash_debug_min_clearance_seen = float("inf")
gameplay_status_last_game_time = None
gameplay_status_fps = 0.0
gameplay_status_physics_hz = 0.0
gameplay_status_physics_ticks = 0

def callback_function(scene: trimesh.Scene):
    # Callback function is called every frame of visualization
    car.visualize_rays(scene)
    car.update_model_view()
    car.update_camera(scene)


def plot_map():
    # Create a scene with the car and the map
    scene = trimesh.Scene()
    scene.add_geometry(car.get_mesh())
    walls_mesh = game_map.get_sensor_walls_mesh() if vertical_mode else game_map.get_walls_mesh()
    scene.add_geometry(walls_mesh)
    scene.add_geometry(game_map.get_road_mesh())
    scene.add_geometry(game_map.get_path_line_mesh())
    scene.add_geometry(game_map.get_path_points_mesh())
    scene.show(callback=callback_function, flags={"cull": False})


def print_fps(frame: int):
    # Print the average FPS every 100 frames
    global start_time_fps, current_fps
    if frame % 100 != 0:
        return
    
    dt = time() - start_time_fps
    if dt == 0:
        dt = 0.01
    current_fps = 100 / dt
    start_time_fps = time()


def format_instruction_window(instructions) -> str:
    values = np.asarray(instructions, dtype=np.float32).reshape(-1)
    return "[" + ", ".join(f"{value:+.0f}" for value in values) + "]"


def format_feature_block(values) -> str:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    return np.array2string(
        vector,
        precision=3,
        suppress_small=False,
        floatmode="fixed",
        max_line_width=240,
    )


def current_surface_summary(data_dictionary) -> tuple[str, float, float, float]:
    wheel_keys = ("fl", "fr", "rl", "rr")
    material_ids = [
        int(round(float(data_dictionary.get(f"ground_material_{wheel}", 80.0))))
        for wheel in wheel_keys
    ]
    material_id = max(set(material_ids), key=material_ids.count)
    material_name = surface_material_name(material_id)
    slips = np.array(
        [float(data_dictionary.get(f"slip_{wheel}", 0.0)) for wheel in wheel_keys],
        dtype=np.float32,
    )
    return material_name, float(np.min(slips)), float(np.mean(slips)), float(np.max(slips))


def current_lidar_crash_summary(data_dictionary) -> tuple[int, int, int, float, float, float, float]:
    global crash_debug_latched_until
    global crash_debug_event_count
    global crash_debug_was_active
    global crash_debug_min_clearance_seen

    clearances = np.asarray(
        data_dictionary.get("laser_clearances", []),
        dtype=np.float32,
    ).reshape(-1)
    min_clearance = float(
        data_dictionary.get(
            "min_laser_clearance",
            float(np.min(clearances)) if clearances.size else float("inf"),
        )
    )
    min_raw = float(data_dictionary.get("min_raw_laser_distance", float("nan")))
    hitbox_offset = float(data_dictionary.get("hitbox_offset_at_min_clearance", float("nan")))
    crash_now = int(np.isfinite(min_clearance) and min_clearance <= 0.0)

    if np.isfinite(min_clearance):
        crash_debug_min_clearance_seen = min(crash_debug_min_clearance_seen, min_clearance)

    if crash_now and not crash_debug_was_active:
        crash_debug_event_count += 1
    if crash_now:
        crash_debug_latched_until = max(
            crash_debug_latched_until,
            time() + CRASH_DEBUG_HOLD_SECONDS,
        )
    crash_debug_was_active = bool(crash_now)
    crash_latched = int(time() <= crash_debug_latched_until)

    return (
        crash_now,
        crash_latched,
        crash_debug_event_count,
        min_clearance,
        crash_debug_min_clearance_seen,
        min_raw,
        hitbox_offset,
    )


def update_gameplay_status_from_time(game_time: float, dt_ref: float) -> tuple[float, float, int]:
    global gameplay_status_last_game_time
    global gameplay_status_fps
    global gameplay_status_physics_hz
    global gameplay_status_physics_ticks

    game_time = float(game_time)
    if not np.isfinite(game_time) or game_time < 0.0:
        gameplay_status_last_game_time = None
        return 0.0, 0.0, 0

    previous_game_time = gameplay_status_last_game_time
    gameplay_status_last_game_time = game_time
    if previous_game_time is None:
        return gameplay_status_fps, gameplay_status_physics_hz, gameplay_status_physics_ticks

    game_dt = game_time - float(previous_game_time)
    if game_dt <= 1e-6:
        return gameplay_status_fps, gameplay_status_physics_hz, gameplay_status_physics_ticks
    if game_dt > 0.25:
        # Reset/finish pauses are not representative gameplay samples.
        return gameplay_status_fps, gameplay_status_physics_hz, gameplay_status_physics_ticks

    dt_ref = max(1e-6, float(dt_ref))
    physics_ticks = max(1, int(round(game_dt / dt_ref)))
    gameplay_status_physics_ticks = physics_ticks
    gameplay_status_physics_hz = 1.0 / max(1e-6, physics_ticks * dt_ref)

    instantaneous_fps = float(np.clip(1.0 / game_dt, 0.0, 240.0))
    if gameplay_status_fps <= 0.0:
        gameplay_status_fps = instantaneous_fps
    else:
        gameplay_status_fps = (0.85 * gameplay_status_fps) + (0.15 * instantaneous_fps)
    return gameplay_status_fps, gameplay_status_physics_hz, gameplay_status_physics_ticks


def build_env_status_line(frame: int, data_dictionary) -> str:
    game_time = float(data_dictionary.get("time", 0.0))
    discrete_progress = float(data_dictionary.get("discrete_progress", 0.0))
    dense_progress = float(data_dictionary.get("dense_progress", discrete_progress))
    speed = float(data_dictionary.get("speed", 0.0))
    try:
        dt_ref = float(encoder.dt_ref)
    except NameError:
        dt_ref = 1.0 / 100.0
    fps, physics_hz, physics_ticks = update_gameplay_status_from_time(game_time, dt_ref)
    physics_hz_norm = 1.0 / float(physics_ticks) if physics_ticks > 0 else 0.0
    physics_delay_norm = float(np.clip(1.0 - physics_hz_norm, 0.0, 1.0))
    return (
        f"env_status: step={frame:<6d} | fps={fps:<6.1f} | "
        f"phys_Hz={physics_hz:<6.1f} | phys_ticks={physics_ticks:<2d} | "
        f"delay={physics_delay_norm:<5.3f} | "
        f"progress={dense_progress:<7.3f} | time={game_time:<7.2f} | speed={speed:<7.2f}"
    )


def build_debug_panel(frame: int, data_dictionary, instructions, observation, mirrored_observation) -> str:
    fps = float(current_fps)
    current_error = float(data_dictionary.get("segment_heading_error", 0.0))
    next_error = float(data_dictionary.get("next_segment_heading_error", 0.0))
    discrete_progress = float(data_dictionary.get("discrete_progress", 0.0))
    dense_progress = float(data_dictionary.get("dense_progress", discrete_progress))
    path_tile_index = int(car.path_tile_index)
    speed = float(data_dictionary.get("speed", 0.0))
    side_speed = float(data_dictionary.get("side_speed", 0.0))
    game_time = float(data_dictionary.get("time", 0.0))
    vertical_speed = float(data_dictionary.get("vertical_speed", 0.0))
    forward_y = float(data_dictionary.get("forward_y", 0.0))
    vertical_lidar_mode = str(data_dictionary.get("vertical_lidar_mode", "-"))
    support_normal_y = float(data_dictionary.get("support_normal_y", 1.0))
    cross_slope = float(data_dictionary.get("cross_slope", 0.0))
    instructions_text = format_instruction_window(instructions)
    surface_text = format_feature_block(
        data_dictionary.get(
            "next_surface_instructions",
            np.ones(Car.SIGHT_TILES, dtype=np.float32),
        )
    )
    height_text = format_feature_block(
        data_dictionary.get(
            "next_height_instructions",
            np.zeros(Car.SIGHT_TILES, dtype=np.float32),
        )
    )
    ray_modes = list(data_dictionary.get("ray_debug_modes", []))
    mode_counts = {}
    for value in ray_modes:
        mode_counts[value] = mode_counts.get(value, 0) + 1
    mode_counts_text = ", ".join(f"{key}:{value}" for key, value in sorted(mode_counts.items()))
    crash_now, crash_latched, crash_events, min_clearance, min_seen_clearance, min_raw_laser, hitbox_offset = current_lidar_crash_summary(data_dictionary)
    summary_line = (
        f"frame={frame:06d}  "
        f"mode={'3D' if encoder.vertical_mode else '2D'}  "
        f"fps={fps:6.2f}  "
        f"time={game_time:6.2f}s  "
        f"disc={discrete_progress:6.2f}%  "
        f"dense={dense_progress:6.2f}%  "
        f"path_idx={path_tile_index:03d}  "
        f"speed={speed:7.2f}  "
        f"side={side_speed:7.2f}  "
        f"crash={crash_now:d}/{crash_latched:d}  "
        f"hits={crash_events:03d}  "
        f"clear={min_clearance:+6.2f}  "
        f"seg_err={current_error:+.3f}  "
        f"next_err={next_error:+.3f}  "
        f"instr={instructions_text}  "
        f"surface={surface_text}  "
        f"height={height_text}"
    )
    secondary_summary = (
        f"vertical_speed={vertical_speed:+7.3f}  "
        f"forward_y={forward_y:+6.3f}  "
        f"support_ny={support_normal_y:+6.3f}  "
        f"cross_slope={cross_slope:+6.3f}  "
        f"min_clear_seen={min_seen_clearance:+6.2f}  "
        f"raw_lidar={min_raw_laser:6.2f}  "
        f"hitbox_off={hitbox_offset:5.2f}  "
        f"ray_modes={mode_counts_text if mode_counts_text else '-'}"
    )
    material_name, slip_min, slip_avg, slip_max = current_surface_summary(data_dictionary)
    slices = ObservationEncoder.section_slices(
        vertical_mode=encoder.vertical_mode,
        multi_surface_mode=encoder.multi_surface_mode,
    )

    lines = [
        summary_line,
        secondary_summary,
        f"current surface: {material_name}  slip min/avg/max={slip_min:.3f}/{slip_avg:.3f}/{slip_max:.3f}",
        "",
        f"obs lasers   : {format_feature_block(observation[slices['lasers']])}",
        f"obs path     : {format_feature_block(observation[slices['path']])}",
        f"obs base     : {format_feature_block(observation[slices['base']])}",
        f"obs slip     : {format_feature_block(observation[slices['slip']])}",
        f"obs temporal : {format_feature_block(observation[slices['temporal']])}",
        "",
        f"mir lasers   : {format_feature_block(mirrored_observation[slices['lasers']])}",
        f"mir path     : {format_feature_block(mirrored_observation[slices['path']])}",
        f"mir base     : {format_feature_block(mirrored_observation[slices['base']])}",
        f"mir slip     : {format_feature_block(mirrored_observation[slices['slip']])}",
        f"mir temporal : {format_feature_block(mirrored_observation[slices['temporal']])}",
    ]
    if "surface" in slices:
        lines.insert(8, f"obs surface  : {format_feature_block(observation[slices['surface']])}")
        lines.insert(16, f"mir surface  : {format_feature_block(mirrored_observation[slices['surface']])}")
    if "height" in slices:
        insert_obs_at = 9 if "surface" in slices else 8
        insert_mir_at = 17 if "surface" in slices else 16
        lines.insert(insert_obs_at, f"obs height   : {format_feature_block(observation[slices['height']])}")
        lines.insert(insert_mir_at, f"mir height   : {format_feature_block(mirrored_observation[slices['height']])}")
    if "vertical" in slices:
        lines.extend(
            [
                f"obs vertical : {format_feature_block(observation[slices['vertical']])}",
                "",
                f"mir vertical : {format_feature_block(mirrored_observation[slices['vertical']])}",
                f"ray elev raw : {format_feature_block(data_dictionary.get('laser_elevation_rates', np.zeros(Car.NUM_LASERS, dtype=np.float32)))}",
            ]
        )
    lines.extend(["", build_env_status_line(frame, data_dictionary)])
    return "\n".join(lines)


def print_live_debug(frame: int, data_dictionary, instructions, observation, mirrored_observation) -> None:
    panel = build_debug_panel(
        frame=frame,
        data_dictionary=data_dictionary,
        instructions=instructions,
        observation=observation,
        mirrored_observation=mirrored_observation,
    )
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.write(panel)
    sys.stdout.write("\n")
    sys.stdout.flush()


def build_dashboard_lines(frame: int, data_dictionary, instructions, observation) -> list[str]:
    fps = float(current_fps)
    game_time = float(data_dictionary.get("time", 0.0))
    discrete_progress = float(data_dictionary.get("discrete_progress", 0.0))
    dense_progress = float(data_dictionary.get("dense_progress", discrete_progress))
    speed = float(data_dictionary.get("speed", 0.0))
    side_speed = float(data_dictionary.get("side_speed", 0.0))
    segment_error = float(data_dictionary.get("segment_heading_error", 0.0))
    next_error = float(data_dictionary.get("next_segment_heading_error", 0.0))
    path_tile_index = int(car.path_tile_index)
    instructions_text = format_instruction_window(instructions)
    surface_text = format_feature_block(
        data_dictionary.get(
            "next_surface_instructions",
            np.ones(Car.SIGHT_TILES, dtype=np.float32),
        )
    )
    height_text = format_feature_block(
        data_dictionary.get(
            "next_height_instructions",
            np.zeros(Car.SIGHT_TILES, dtype=np.float32),
        )
    )
    wetness = float(data_dictionary.get("wetness", 0.0))
    material_name, slip_min, slip_avg, slip_max = current_surface_summary(data_dictionary)
    support_valid = float(data_dictionary.get("support_valid", 0.0))
    support_normal_y = float(data_dictionary.get("support_normal_y", 1.0))
    forward_y = float(data_dictionary.get("forward_y", 0.0))
    vertical_lidar_mode = str(data_dictionary.get("vertical_lidar_mode", "-"))
    ray_modes = list(data_dictionary.get("ray_debug_modes", []))
    mode_counts = {}
    for value in ray_modes:
        mode_counts[value] = mode_counts.get(value, 0) + 1
    mode_counts_text = ", ".join(f"{key}:{value}" for key, value in sorted(mode_counts.items()))
    crash_now, crash_latched, crash_events, min_clearance, min_seen_clearance, min_raw_laser, hitbox_offset = current_lidar_crash_summary(data_dictionary)

    lines = [
        f"frame={frame:06d} fps={fps:5.1f} time={game_time:6.2f}s "
        f"disc={discrete_progress:6.2f}% dense={dense_progress:6.2f}% "
        f"idx={path_tile_index:03d} "
        f"spd={speed:7.2f} side={side_speed:7.2f} "
        f"crash={crash_now:d}/{crash_latched:d} hits={crash_events:03d} "
        f"clear={min_clearance:+6.2f} min={min_seen_clearance:+6.2f}",
        f"heading_err={segment_error:+.3f}/{next_error:+.3f} "
        f"instr={instructions_text} surface={surface_text} height={height_text} wet={wetness:.2f}",
        f"current_surface={material_name} "
        f"slip_min/avg/max={slip_min:.3f}/{slip_avg:.3f}/{slip_max:.3f}",
        f"lidar raw={min_raw_laser:6.2f} hitbox_off={hitbox_offset:5.2f} "
        f"vertical lidar={vertical_lidar_mode} support={support_valid:.0f} support_ny={support_normal_y:+.3f} "
        f"forward_y={forward_y:+.3f} ray_modes={mode_counts_text if mode_counts_text else '-'}",
    ]

    if DEBUG_SURFACE_DASHBOARD_ROWS > 4:
        slices = ObservationEncoder.section_slices(
            vertical_mode=encoder.vertical_mode,
            multi_surface_mode=encoder.multi_surface_mode,
        )
        lines.append(f"obs slip     : {format_feature_block(observation[slices['slip']])}")
        if "surface" in slices:
            lines.append(f"obs surface  : {format_feature_block(observation[slices['surface']])}")
        if "height" in slices:
            lines.append(f"obs height   : {format_feature_block(observation[slices['height']])}")
        lines.append(f"obs temporal : {format_feature_block(observation[slices['temporal']])}")
    lines.append(build_env_status_line(frame, data_dictionary))
    return lines


def print_dashboard_debug(frame: int, data_dictionary, instructions, observation) -> None:
    global previous_dashboard_line_count, last_compact_debug_time

    now = time()
    if now - last_compact_debug_time < DEBUG_PRINT_INTERVAL_SECONDS:
        return
    last_compact_debug_time = now

    lines = build_dashboard_lines(
        frame=frame,
        data_dictionary=data_dictionary,
        instructions=instructions,
        observation=observation,
    )
    terminal_width = max(40, shutil.get_terminal_size((240, 20)).columns)
    clipped_lines = []
    for line in lines:
        if len(line) >= terminal_width:
            line = line[: terminal_width - 4] + "..."
        clipped_lines.append(line.ljust(terminal_width - 1))

    if previous_dashboard_line_count > 0:
        sys.stdout.write(f"\x1b[{previous_dashboard_line_count}F")

    sys.stdout.write("\n".join(clipped_lines))
    sys.stdout.write("\n")
    sys.stdout.flush()
    previous_dashboard_line_count = len(clipped_lines)


if __name__ == "__main__":  
    
    map_name = "surface_test"
    map_name = "height_test"
    
    map_name = "AI Training #5"
    map_name = "single_surface_flat"
    vizualize = True
    vertical_mode = True
    multi_surface_mode = True

    game_map = Map(map_name)
    car = Car(game_map, vertical_mode=vertical_mode)
    encoder = ObservationEncoder(
        vertical_mode=vertical_mode,
        multi_surface_mode=multi_surface_mode,
    )

    start_time_fps = time()
    current_fps = 0.0
    previous_dashboard_line_count = 0
    last_compact_debug_time = 0.0
    crash_debug_latched_until = 0.0
    crash_debug_event_count = 0
    crash_debug_was_active = False
    crash_debug_min_clearance_seen = float("inf")
    gameplay_status_last_game_time = None
    gameplay_status_fps = 0.0
    gameplay_status_physics_hz = 0.0
    gameplay_status_physics_ticks = 0
    DEBUG_PRINT_INTERVAL_SECONDS = 0
    DEBUG_SURFACE_DASHBOARD_ROWS = 20
    
    # Start the visualization thread
    if vizualize:
        window_thread = threading.Thread(target=plot_map, daemon=True)
        window_thread.start()

    # Start the data collection loop
    frame = 0
    while True:
        distances, instructions, data_dictionary = car.get_data()
        observation = encoder.build_observation(distances, instructions, data_dictionary)
        mirrored_observation = ObservationEncoder.mirror_observation(
            observation,
            vertical_mode=encoder.vertical_mode,
            multi_surface_mode=encoder.multi_surface_mode,
        )
        print_fps(frame)
        print_dashboard_debug(frame, data_dictionary, instructions, observation)
        frame += 1
        if vizualize and not window_thread.is_alive():
            break











    
