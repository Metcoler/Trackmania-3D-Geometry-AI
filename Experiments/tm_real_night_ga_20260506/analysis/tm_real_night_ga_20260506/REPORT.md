# Live Trackmania GA Night Run 20260506

## Summary
- Source run: `logs\tm_finetune_runs\20260506_004011_tm_seed_map_single_surface_flat_v2d_asphalt_h48x24_p48_src_best_model`
- Map: `single_surface_flat`
- Ranking key: `(finished, progress, -time, -crashes)`
- Population size: `48`
- Logged generations: `82` (final generation `82`)
- Latest replayable population checkpoint: generation `80`
- First finish generation: `49`
- Best finish time: `27.96 s` in generation `51`
- Total finishing individuals across logged generations: `82`
- Max finishers in one generation: `4`

## Final Population
- Finishers: `4` / `48`
- Timeouts: `10` / `48`
- Crashes: `34` / `48`
- Final best time: `27.96 s`
- Final mean progress: `52.88 %`
- Final best progress: `100.00 %`

## Latest Replayable Checkpoint
- Checkpoint generation: `80`
- Finishers: `2` / `48`
- Best time: `27.96 s`
- Mean progress: `65.05 %`

## Interpretation
- This is a positive live-Trackmania sanity result: the GA run did not only improve progress, it produced real finishers on `single_surface_flat`.
- The run is still noisy: most final-population individuals crash, while a smaller elite group finishes or reaches high progress.
- The best time is slower than the clean 2D sandbox results, which is expected because this run evaluates policies through the live game loop.
- The result supports continuing with replay/evaluation of the final population, but it should not yet be presented as a final robust agent.
