from __future__ import annotations

import math
import tkinter as tk

import numpy as np

from Experiments.tm2d_env import TM2DSimEnv


class TM2DViewer:
    def __init__(self, env: TM2DSimEnv, width: int = 1100, height: int = 800) -> None:
        self.env = env
        self.width = int(width)
        self.height = int(height)
        self.root = tk.Tk()
        self.root.title(f"TM2D Experiment - {env.geometry.map_name}")
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg="#111820")
        self.canvas.pack()
        min_x, min_y, max_x, max_y = env.geometry.bounds
        margin = 40.0
        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)
        self.scale = min((self.width - 2 * margin) / span_x, (self.height - 2 * margin) / span_y)
        self.offset = np.array([margin - min_x * self.scale, margin - min_y * self.scale], dtype=np.float32)
        self.info_text = self.canvas.create_text(
            12,
            12,
            anchor="nw",
            fill="#eaf2ff",
            font=("Consolas", 12),
            text="",
        )
        self.car_item = None
        self.laser_items: list[int] = []
        self._draw_static_map()

    def world_to_canvas(self, point) -> tuple[float, float]:
        point = np.asarray(point, dtype=np.float32)
        canvas = point * self.scale + self.offset
        return float(canvas[0]), float(canvas[1])

    def _draw_static_map(self) -> None:
        for triangle in self.env.geometry.road_triangles:
            coords = []
            for point in triangle:
                coords.extend(self.world_to_canvas(point))
            self.canvas.create_polygon(coords, fill="#56616b", outline="")
        for segment in self.env.geometry.wall_segments:
            x1, y1 = self.world_to_canvas(segment[0])
            x2, y2 = self.world_to_canvas(segment[1])
            self.canvas.create_line(x1, y1, x2, y2, fill="#f2d57e", width=2)
        for idx, point in enumerate(self.env.geometry.path_points):
            x, y = self.world_to_canvas(point)
            color = "#52e37a" if idx == 0 else "#64d8ff"
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline="")

    def update(self, info: dict | None = None) -> None:
        center = self.env.position
        heading = float(self.env.heading)
        length = self.env.physics.car_length
        width = self.env.physics.car_width
        forward = np.array([math.cos(heading), math.sin(heading)], dtype=np.float32)
        right = np.array([-forward[1], forward[0]], dtype=np.float32)
        corners = [
            center + forward * length * 0.5 + right * width * 0.5,
            center + forward * length * 0.5 - right * width * 0.5,
            center - forward * length * 0.5 - right * width * 0.5,
            center - forward * length * 0.5 + right * width * 0.5,
        ]
        coords = []
        for corner in corners:
            coords.extend(self.world_to_canvas(corner))
        if self.car_item is None:
            self.car_item = self.canvas.create_polygon(coords, fill="#4aa3ff", outline="#d6efff")
        else:
            self.canvas.coords(self.car_item, *coords)

        while len(self.laser_items) < self.env.geometry.num_lasers:
            self.laser_items.append(self.canvas.create_line(0, 0, 0, 0, fill="#56f0c5"))
        if info is not None and "laser_endpoints_2d" in info:
            start = self.world_to_canvas(center)
            for item, endpoint in zip(self.laser_items, info["laser_endpoints_2d"]):
                end = self.world_to_canvas(endpoint)
                self.canvas.coords(item, start[0], start[1], end[0], end[1])

        self.canvas.itemconfig(
            self.info_text,
            text=(
                f"t={self.env.time:5.2f}s  speed={self.env.speed:6.2f}  "
                f"progress={self.env.geometry.progress_for_index(self.env.path_index):6.2f}%  "
                f"path={self.env.path_index}/{len(self.env.geometry.path_tiles_xz)-1}"
            ),
        )
        self.root.update_idletasks()
        self.root.update()

