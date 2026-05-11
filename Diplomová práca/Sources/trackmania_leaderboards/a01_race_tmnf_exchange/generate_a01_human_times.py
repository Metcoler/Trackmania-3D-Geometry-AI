from __future__ import annotations

import csv
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


TRACK_ID = 2233
TRACK_NAME = "A01-Race"
TRACK_PAGE_URL = "https://tmnf.exchange/trackshow/2233"
API_DOC_URL = "https://api2.mania.exchange/Method/Index/45"
API_URL = "https://tmnf.exchange/api/replays"
FIELDS = [
    "ReplayId",
    "User.UserId",
    "User.Name",
    "ReplayTime",
    "ReplayRespawns",
    "Validated",
    "Score",
    "Position",
    "IsBest",
    "TrackAt",
    "ReplayAt",
]

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
THESIS_ROOT = REPO_ROOT / "Masters thesis"
IMAGE_DIR = THESIS_ROOT / "Latex" / "images" / "theory"

RAW_JSON = THIS_DIR / "a01_race_replays_raw.json"
FILTERED_CSV = THIS_DIR / "a01_race_leaderboard_top1000.csv"
SUMMARY_JSON = THIS_DIR / "a01_race_leaderboard_summary.json"
FIGURE_PDF = IMAGE_DIR / "theory_trackmania_a01_human_times.pdf"
FIGURE_PNG = IMAGE_DIR / "theory_trackmania_a01_human_times.png"


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "trackmania-ai-thesis/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def build_url(after: int | None = None) -> str:
    query: dict[str, str | int] = {
        "trackId": TRACK_ID,
        "fields": ",".join(FIELDS),
        "count": 1000,
    }
    if after is not None:
        query["after"] = after
    return f"{API_URL}?{urllib.parse.urlencode(query)}"


def fetch_all_pages() -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    after: int | None = None
    seen_after: set[int] = set()

    while True:
        url = build_url(after)
        payload = fetch_json(url)
        results = payload.get("Results", [])
        pages.append({"url": url, "response": payload})

        if not payload.get("More") or not results:
            break

        next_after = int(results[-1]["ReplayId"])
        if next_after in seen_after:
            raise RuntimeError(f"Paging did not advance after ReplayId={next_after}.")
        seen_after.add(next_after)
        after = next_after

    return pages


def flatten_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in pages:
        rows.extend(page["response"].get("Results", []))
    return rows


def is_leaderboard_row(row: dict[str, Any], latest_track_at: str) -> bool:
    return (
        row.get("TrackAt") == latest_track_at
        and row.get("Validated") is True
        and int(row.get("ReplayRespawns") or 0) == 0
        and row.get("Position") is not None
    )


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = int(math.floor((len(sorted_values) - 1) * p))
    return float(sorted_values[idx])


def compute_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    std = float(np.std(values, ddof=1))
    bandwidth = 1.06 * std * (len(values) ** (-1 / 5))
    if not np.isfinite(bandwidth) or bandwidth <= 1e-6:
        bandwidth = 0.03
    z = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z * z).mean(axis=1) / (bandwidth * math.sqrt(2 * math.pi))
    return density


def comma(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def load_cached_rows() -> list[dict[str, Any]]:
    if not FILTERED_CSV.exists():
        return []
    rows: list[dict[str, Any]] = []
    with FILTERED_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({"ReplayTime": float(row["replay_time_ms"])})
    return rows


def write_filtered_csv(rows: list[dict[str, Any]]) -> None:
    with FILTERED_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "position",
                "replay_id",
                "user_id",
                "user_name",
                "replay_time_ms",
                "replay_time_s",
                "score",
                "validated",
                "replay_respawns",
                "track_at",
                "replay_at",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            user = row.get("User") or {}
            writer.writerow(
                {
                    "rank": rank,
                    "position": row.get("Position"),
                    "replay_id": row.get("ReplayId"),
                    "user_id": user.get("UserId"),
                    "user_name": user.get("Name"),
                    "replay_time_ms": row.get("ReplayTime"),
                    "replay_time_s": f"{float(row['ReplayTime']) / 1000.0:.3f}",
                    "score": row.get("Score"),
                    "validated": row.get("Validated"),
                    "replay_respawns": row.get("ReplayRespawns"),
                    "track_at": row.get("TrackAt"),
                    "replay_at": row.get("ReplayAt"),
                }
            )


def make_summary(raw_rows: list[dict[str, Any]], leaderboard_rows: list[dict[str, Any]]) -> dict[str, Any]:
    times = [float(row["ReplayTime"]) / 1000.0 for row in leaderboard_rows]
    sorted_times = sorted(times)
    ranges: dict[str, float] = {}
    for n in (10, 100, 500, 1000):
        if len(sorted_times) >= n:
            ranges[f"top_{n}_range_s"] = sorted_times[n - 1] - sorted_times[0]

    return {
        "track_name": TRACK_NAME,
        "track_id": TRACK_ID,
        "track_page_url": TRACK_PAGE_URL,
        "api_doc_url": API_DOC_URL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_replays": len(raw_rows),
        "filtered_leaderboard_replays": len(leaderboard_rows),
        "filters": {
            "latest_track_at": leaderboard_rows[0]["TrackAt"] if leaderboard_rows else None,
            "validated": True,
            "replay_respawns": 0,
            "position_not_null": True,
            "top_n_used_for_figure": min(1000, len(leaderboard_rows)),
        },
        "best_time_s": sorted_times[0] if sorted_times else None,
        "median_top1000_s": percentile(sorted_times[:1000], 0.5),
        "min_top1000_s": sorted_times[0] if sorted_times else None,
        "max_top1000_s": sorted_times[min(999, len(sorted_times) - 1)] if sorted_times else None,
        **ranges,
    }


def draw_figure(rows: list[dict[str, Any]]) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    top_rows = rows[:1000]
    values = np.array([float(row["ReplayTime"]) / 1000.0 for row in top_rows], dtype=float)
    values.sort()

    best = float(values[0])
    median = float(np.median(values))
    x_min = min(best - 0.05, best - 0.03)
    x_max = max(float(values[-1]) + 0.04, best + 0.53)
    grid = np.linspace(x_min, x_max, 500)
    density = compute_kde(values, grid)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.hist(
        values,
        bins=34,
        density=True,
        color="#BFD7EA",
        edgecolor="#4C78A8",
        linewidth=0.6,
        alpha=0.88,
        label="histogram top 1000",
    )
    ax.plot(grid, density, color="#1F5F99", linewidth=2.4, label="smoothed density estimate")

    ax.axvline(best, color="#C44E52", linewidth=2.0)
    ax.text(best, ax.get_ylim()[1] * 0.92, f"best\n{comma(best)} s", ha="left", va="top", color="#8F1D21", fontsize=9)

    ax.axvline(median, color="#F58518", linewidth=2.0, linestyle="--")
    ax.text(median, ax.get_ylim()[1] * 0.92, f"median\n{comma(median)} s", ha="left", va="top", color="#9A4D00", fontsize=9)

    y_arrow = ax.get_ylim()[1] * 0.78
    ax.annotate(
        "",
        xy=(best + 0.5, y_arrow),
        xytext=(best, y_arrow),
        arrowprops={"arrowstyle": "<->", "color": "#1F2937", "linewidth": 1.6},
    )
    ax.text(best + 0.25, y_arrow * 1.04, "0.5 s", ha="center", va="bottom", color="#1F2937", fontsize=10)

    ranges_text = (
        f"top 10: {comma(values[9] - values[0])} s\n"
        f"top 100: {comma(values[99] - values[0])} s\n"
        f"top 500: {comma(values[499] - values[0])} s\n"
        f"top 1000: {comma(values[999] - values[0])} s"
    )
    ax.text(
        0.98,
        0.95,
        ranges_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1", "linewidth": 0.8},
    )

    ax.set_title("Human leaderboard times on A01-Race")
    ax.set_xlabel("run time [s]")
    ax.set_ylabel("density")
    ax.set_xlim(x_min, x_max)
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    fig.savefig(FIGURE_PNG, bbox_inches="tight", dpi=180)
    plt.close(fig)


def main() -> None:
    cached_rows = load_cached_rows()
    if cached_rows:
        draw_figure(cached_rows[:1000])
        return

    pages = fetch_all_pages()
    raw_rows = flatten_pages(pages)
    latest_track_at = max(row["TrackAt"] for row in raw_rows if row.get("TrackAt"))
    leaderboard_rows = [row for row in raw_rows if is_leaderboard_row(row, latest_track_at)]
    leaderboard_rows.sort(key=lambda row: (int(row["Position"]), int(row["ReplayTime"])))
    top_rows = leaderboard_rows[:1000]

    RAW_JSON.write_text(
        json.dumps(
            {
                "source": {
                    "track_name": TRACK_NAME,
                    "track_id": TRACK_ID,
                    "track_page_url": TRACK_PAGE_URL,
                    "api_doc_url": API_DOC_URL,
                    "api_url": API_URL,
                    "fields": FIELDS,
                },
                "pages": pages,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_filtered_csv(top_rows)
    summary = make_summary(raw_rows, leaderboard_rows)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_figure(top_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {FILTERED_CSV}")
    print(f"Wrote {FIGURE_PDF}")


if __name__ == "__main__":
    main()
