# Trackmania 100 Hz Alt-Tab Probe

Trackmania can drop from the desired 100 Hz physics cadence when the game loses
focus, gets covered by another window, or is minimized. This is likely caused by
background/occlusion throttling in the game, Windows compositor, or GPU driver.
Openplanet cannot fix this by itself because it ticks with the game.

## Measurement

Run the interactive probe while Trackmania and the Openplanet data plugin are
running:

```powershell
python Experiments\trackmania_100hz_probe.py --sample-seconds 20
```

The probe connects to the same `127.0.0.1:9002` stream as live training and
measures these states:

- `focused_visible`: Trackmania focused and visible.
- `unfocused_visible`: another window focused, Trackmania still visible.
- `unfocused_covered`: another window focused and covering Trackmania.
- `minimized`: Trackmania minimized.

Outputs are saved to `Experiments/analysis/trackmania_100hz_alt_tab_<timestamp>/`:

- `samples.csv`: per-frame timing samples.
- `summary.csv`: mean physics Hz and tick distribution per state.
- `physics_hz_by_window_state.png`: quick diagnostic plot.
- `REPORT.md`: short interpretation and checklist.

Only one Python client can use the Openplanet socket at a time. Stop live
training, visualizer, or replay clients before running the probe.

## Interpretation

The relevant metric is not render FPS but physics cadence:

```text
physics_tick_count = round(game_dt / 0.01)
physics_hz = 100 / physics_tick_count
```

Good training conditions should keep almost all samples at `1 tick / 100 Hz`.
If `unfocused_visible` stays near 100 Hz but `covered` or `minimized` drops,
keep Trackmania visible while working. If even `unfocused_visible` drops, focus
loss itself is enough to trigger throttling on this setup.

## Practical Fix Checklist

- Trackmania: use windowed/borderless and keep the window visible if possible.
- Trackmania: set maximum FPS to at least `100` or `120`.
- Trackmania: test GPU/CPU synchronization at `1 frame`, then `2 frames`.
- Windows 11: set Trackmania to High Performance GPU.
- Windows 11: test Optimizations for windowed games both on and off.
- NVIDIA: disable Background Application Max Frame Rate for Trackmania.
- NVIDIA: set Power management mode to Prefer maximum performance.
- NVIDIA laptop: disable Whisper Mode or Battery Boost for this profile.
- AMD: disable Radeon Chill for Trackmania or set Chill min/max above `100`.
- Avoid a Python focus-stealing loop unless you explicitly want Trackmania to
  interrupt normal work.

## References

- Microsoft Windows 11 windowed game optimizations:
  https://support.microsoft.com/en-us/windows/optimizations-for-windowed-games-in-windows-11-3f006843-2c7e-4ed0-9a5e-f9389e535952
- NVIDIA Background Application Max Frame Rate:
  https://www.nvidia.com/content/Control-Panel-Help/vLatest/en-us/mergedProjects/3D%20Settings/Manage_3D_Settings_%28reference%29.htm
- AMD Radeon Chill:
  https://www.amd.com/en/resources/support-articles/faqs/DH-033.html
- Trackmania FPS and GPU/CPU synchronization settings:
  https://www.gamersdecide.com/articles/trackmania-2020-best-settings
