from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import socket
import struct
import time
from collections import Counter
from pathlib import Path

import numpy as np


PACKET_FLOAT_COUNT = 37
PACKET_SIZE = PACKET_FLOAT_COUNT * 4
TIME_INDEX = 15
DEFAULT_STAGES = [
    (
        "focused_visible",
        "Keep Trackmania focused and visible. Do not cover or minimize it.",
    ),
    (
        "unfocused_visible",
        "Alt-tab away, but keep the Trackmania window visible on screen.",
    ),
    (
        "unfocused_covered",
        "Keep Trackmania unfocused and cover most/all of it with another window.",
    ),
]


def parse_stage(value: str) -> tuple[str, str]:
    if ":" in value:
        name, instruction = value.split(":", 1)
        return name.strip(), instruction.strip()
    name = value.strip()
    return name, f"Prepare state: {name}"


def connect_openplanet(host: str, port: int, timeout_seconds: float) -> socket.socket:
    deadline = time.perf_counter() + max(0.0, timeout_seconds)
    last_error: Exception | None = None
    while time.perf_counter() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=2.0)
            sock.settimeout(1.0)
            return sock
        except OSError as exc:
            last_error = exc
            time.sleep(0.5)
    raise ConnectionError(
        f"Could not connect to Openplanet stream at {host}:{port}. "
        "Close other Python clients using the stream and confirm the plugin is running."
    ) from last_error


def read_packet(sock: socket.socket, recv_buffer: bytearray) -> tuple[dict | None, float]:
    while len(recv_buffer) < PACKET_SIZE:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Openplanet stream closed the connection.")
        recv_buffer.extend(chunk)

    packet = bytes(recv_buffer[:PACKET_SIZE])
    del recv_buffer[:PACKET_SIZE]
    values = struct.unpack(f"<{PACKET_FLOAT_COUNT}f", packet)
    return {"game_time": float(values[TIME_INDEX])}, time.perf_counter()


def wait_for_valid_sample(
    sock: socket.socket,
    recv_buffer: bytearray,
    max_game_dt: float,
    max_wait_seconds: float,
) -> tuple[float, float]:
    deadline = time.perf_counter() + max_wait_seconds
    previous_game_time: float | None = None
    while time.perf_counter() < deadline:
        packet, recv_time = read_packet(sock, recv_buffer)
        if packet is None:
            continue
        game_time = packet["game_time"]
        if not math.isfinite(game_time) or game_time < 0.0:
            previous_game_time = None
            continue
        if previous_game_time is not None:
            game_dt = game_time - previous_game_time
            if 1e-6 < game_dt <= max_game_dt:
                return game_time, recv_time
            if game_dt < 0.0:
                previous_game_time = None
                continue
        previous_game_time = game_time
    raise TimeoutError("Timed out waiting for a valid positive game-time delta.")


def collect_stage(
    sock: socket.socket,
    recv_buffer: bytearray,
    stage_name: str,
    sample_seconds: float,
    dt_ref: float,
    max_game_dt: float,
) -> list[dict]:
    rows: list[dict] = []
    previous_game_time: float | None = None
    previous_recv_time: float | None = None
    stage_start = time.perf_counter()
    sample_index = 0

    while time.perf_counter() - stage_start < sample_seconds:
        packet, recv_time = read_packet(sock, recv_buffer)
        if packet is None:
            continue
        game_time = packet["game_time"]
        if not math.isfinite(game_time) or game_time < 0.0:
            previous_game_time = None
            previous_recv_time = None
            continue

        game_dt = None
        wall_dt = None
        if previous_game_time is not None:
            game_dt = game_time - previous_game_time
        if previous_recv_time is not None:
            wall_dt = recv_time - previous_recv_time

        previous_game_time = game_time
        previous_recv_time = recv_time
        if game_dt is None or game_dt <= 1e-6 or game_dt > max_game_dt:
            continue

        ticks = max(1, int(round(float(game_dt) / max(1e-6, float(dt_ref)))))
        physics_hz = 1.0 / max(1e-6, ticks * float(dt_ref))
        physics_hz_norm = 1.0 / float(ticks)
        physics_delay_norm = 1.0 - physics_hz_norm
        rows.append(
            {
                "stage": stage_name,
                "sample_index": sample_index,
                "recv_time": recv_time,
                "game_time": game_time,
                "game_dt": game_dt,
                "wall_dt": wall_dt if wall_dt is not None else "",
                "game_fps": 1.0 / game_dt,
                "packet_hz_wall": (1.0 / wall_dt) if wall_dt and wall_dt > 1e-6 else "",
                "physics_tick_count": ticks,
                "physics_hz": physics_hz,
                "physics_hz_norm": physics_hz_norm,
                "physics_delay_norm": physics_delay_norm,
            }
        )
        sample_index += 1
    return rows


def summarize_stage(stage_name: str, rows: list[dict]) -> dict:
    valid = len(rows)
    if valid <= 0:
        return {
            "stage": stage_name,
            "valid_deltas": 0,
            "mean_game_fps": "",
            "median_game_fps": "",
            "mean_physics_hz": "",
            "median_physics_hz": "",
            "tick_1_percent": "",
            "tick_2_percent": "",
            "tick_3_percent": "",
            "tick_4_plus_percent": "",
            "below_95hz_percent": "",
        }

    game_fps = np.asarray([float(row["game_fps"]) for row in rows], dtype=np.float64)
    physics_hz = np.asarray([float(row["physics_hz"]) for row in rows], dtype=np.float64)
    ticks = [int(row["physics_tick_count"]) for row in rows]
    counts = Counter(ticks)

    def percent(predicate) -> float:
        return 100.0 * sum(1 for value in ticks if predicate(value)) / float(valid)

    return {
        "stage": stage_name,
        "valid_deltas": valid,
        "mean_game_fps": float(np.mean(game_fps)),
        "median_game_fps": float(np.median(game_fps)),
        "mean_physics_hz": float(np.mean(physics_hz)),
        "median_physics_hz": float(np.median(physics_hz)),
        "tick_1_percent": 100.0 * counts.get(1, 0) / float(valid),
        "tick_2_percent": 100.0 * counts.get(2, 0) / float(valid),
        "tick_3_percent": 100.0 * counts.get(3, 0) / float(valid),
        "tick_4_plus_percent": percent(lambda value: value >= 4),
        "below_95hz_percent": 100.0 * sum(1 for value in physics_hz if value < 95.0) / float(valid),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_plot(path: Path, summaries: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot because matplotlib is unavailable: {exc}")
        return

    labels = [str(row["stage"]) for row in summaries]
    mean_hz = [
        float(row["mean_physics_hz"]) if row["mean_physics_hz"] != "" else 0.0
        for row in summaries
    ]
    below_95 = [
        float(row["below_95hz_percent"]) if row["below_95hz_percent"] != "" else 0.0
        for row in summaries
    ]
    tick_1 = [
        float(row["tick_1_percent"]) if row["tick_1_percent"] != "" else 0.0
        for row in summaries
    ]
    tick_2 = [
        float(row["tick_2_percent"]) if row["tick_2_percent"] != "" else 0.0
        for row in summaries
    ]
    tick_4 = [
        float(row["tick_4_plus_percent"]) if row["tick_4_plus_percent"] != "" else 0.0
        for row in summaries
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].bar(labels, mean_hz, color="#2f6f73", edgecolor="#1f2d2e")
    axes[0].axhline(100.0, color="#202020", linestyle="--", linewidth=1.0, label="100 Hz target")
    axes[0].set_ylabel("Mean physics Hz")
    axes[0].set_ylim(0.0, max(110.0, max(mean_hz, default=0.0) * 1.15))
    axes[0].set_title("Trackmania physics rate by window state")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend()

    bottom = np.zeros(len(labels), dtype=np.float64)
    for values, label, color in [
        (tick_1, "1 tick / 100 Hz", "#2f6f73"),
        (tick_2, "2 ticks / 50 Hz", "#d98c3f"),
        (tick_4, "4+ ticks / <=25 Hz", "#c23b22"),
    ]:
        axes[1].bar(labels, values, bottom=bottom, label=label, color=color, edgecolor="#1f2d2e")
        bottom += np.asarray(values, dtype=np.float64)
    axes[1].plot(labels, below_95, color="#111111", marker="o", linewidth=1.5, label="<95 Hz samples")
    axes[1].set_ylabel("Percent of valid deltas")
    axes[1].set_ylim(0.0, 105.0)
    axes[1].set_title("Physics tick distribution")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(fontsize=8)

    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_report(path: Path, summaries: list[dict], sample_seconds: float) -> None:
    worst = None
    numeric = [row for row in summaries if row["mean_physics_hz"] != ""]
    if numeric:
        worst = min(numeric, key=lambda row: float(row["mean_physics_hz"]))

    table_lines = [
        "| state | valid deltas | mean phys Hz | tick1 % | tick2 % | tick4+ % | <95Hz % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        def fmt(key: str) -> str:
            value = row[key]
            if value == "":
                return ""
            return f"{float(value):.2f}"

        table_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["stage"]),
                    str(row["valid_deltas"]),
                    fmt("mean_physics_hz"),
                    fmt("tick_1_percent"),
                    fmt("tick_2_percent"),
                    fmt("tick_4_plus_percent"),
                    fmt("below_95hz_percent"),
                ]
            )
            + " |"
        )

    verdict = "No valid samples were collected."
    if worst is not None:
        verdict = (
            f"Worst measured state: `{worst['stage']}` with mean physics Hz "
            f"{float(worst['mean_physics_hz']):.2f}."
        )

    text = f"""# Trackmania 100 Hz Alt-Tab Probe

This probe measures the Openplanet game-time delta and quantizes it into the same
physics tick metric used by the live trainer: `ticks = round(game_dt / 0.01)`,
`physics_hz = 100 / ticks`.

Sample length per state: `{sample_seconds:.1f}` seconds.

{verdict}

## Summary

{chr(10).join(table_lines)}

## Interpretation

- `focused_visible` should stay close to `100 Hz`.
- `unfocused_visible` tells us whether merely alt-tabbing is enough to throttle Trackmania.
- `unfocused_covered` separates focus loss from Windows/GPU occlusion throttling.
- `minimized` is expected to be the riskiest state; if it drops, avoid minimizing during live training.

## Practical Checklist

- Keep Trackmania visible in windowed/borderless mode if possible.
- Windows 11: set Trackmania to High Performance GPU and test windowed-game optimizations both on and off.
- Trackmania: use Maximum FPS at least `100/120` and test GPU/CPU synchronization at `1 frame` or `2 frames`.
- NVIDIA: disable Background Application Max Frame Rate for Trackmania and prefer maximum performance.
- AMD: disable Radeon Chill for Trackmania, or set Chill min/max high enough that it cannot fall below `100`.
- Avoid Python foreground loops unless you explicitly want Trackmania to steal focus while you work.

"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure live Trackmania physics Hz under focused/unfocused window states."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--sample-seconds", type=float, default=20.0)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--dt-ref", type=float, default=1.0 / 100.0)
    parser.add_argument("--max-game-dt", type=float, default=0.25)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Default: Experiments/analysis/trackmania_100hz_alt_tab_<timestamp>",
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help="Custom stage as name or name:instruction. Can be repeated.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Collect stages immediately without waiting for Enter between states.",
    )
    args = parser.parse_args()

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("Experiments") / "analysis" / f"trackmania_100hz_alt_tab_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stages = [parse_stage(stage) for stage in args.stage] if args.stage else DEFAULT_STAGES

    print("Connecting to Openplanet stream...")
    print("Close other live Python clients first; the plugin accepts one client at a time.")
    sock = connect_openplanet(str(args.host), int(args.port), float(args.connect_timeout))
    recv_buffer = bytearray()
    print("Connected. Waiting for a valid game-time delta...")
    wait_for_valid_sample(
        sock,
        recv_buffer,
        max_game_dt=float(args.max_game_dt),
        max_wait_seconds=max(5.0, float(args.connect_timeout)),
    )

    all_rows: list[dict] = []
    summaries: list[dict] = []
    try:
        for stage_name, instruction in stages:
            print()
            print(f"Stage: {stage_name}")
            print(instruction)
            if not args.no_prompt:
                input("Press Enter when this state is ready...")
            rows = collect_stage(
                sock,
                recv_buffer,
                stage_name=stage_name,
                sample_seconds=float(args.sample_seconds),
                dt_ref=float(args.dt_ref),
                max_game_dt=float(args.max_game_dt),
            )
            summary = summarize_stage(stage_name, rows)
            summaries.append(summary)
            all_rows.extend(rows)
            mean_hz = summary.get("mean_physics_hz", "")
            below = summary.get("below_95hz_percent", "")
            if mean_hz == "":
                print(f"  collected {len(rows)} valid deltas")
            else:
                print(
                    f"  collected {len(rows)} valid deltas | "
                    f"mean phys_Hz={float(mean_hz):.2f} | <95Hz={float(below):.2f}%"
                )
    finally:
        sock.close()

    sample_fields = [
        "stage",
        "sample_index",
        "recv_time",
        "game_time",
        "game_dt",
        "wall_dt",
        "game_fps",
        "packet_hz_wall",
        "physics_tick_count",
        "physics_hz",
        "physics_hz_norm",
        "physics_delay_norm",
    ]
    summary_fields = [
        "stage",
        "valid_deltas",
        "mean_game_fps",
        "median_game_fps",
        "mean_physics_hz",
        "median_physics_hz",
        "tick_1_percent",
        "tick_2_percent",
        "tick_3_percent",
        "tick_4_plus_percent",
        "below_95hz_percent",
    ]
    write_csv(output_dir / "samples.csv", all_rows, sample_fields)
    write_csv(output_dir / "summary.csv", summaries, summary_fields)
    write_plot(output_dir / "physics_hz_by_window_state.png", summaries)
    write_report(output_dir / "REPORT.md", summaries, sample_seconds=float(args.sample_seconds))
    print()
    print(f"Saved probe output to: {output_dir}")


if __name__ == "__main__":
    main()
