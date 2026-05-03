# Vehicle AABB Hitbox Analysis

## Inputs
- Attempt files: `1`
- Frames: `11770`
- Raw laser source: `raw_laser_distances` when available, otherwise reconstructed as `observations[:, :15] * 160.0`
- Mesh estimates:
```json
{
  "visual_main_body": {
    "size": [
      2.1432,
      3.7842,
      0.8822
    ],
    "center": [
      0.0004,
      -0.2737,
      0.4292
    ],
    "note": "MainBody lod1 visual mesh OBJ export."
  },
  "car_primitives": {
    "size": [
      2.1326,
      4.0792,
      1.1792
    ],
    "center": [
      0.0,
      0.1051,
      -0.5795
    ],
    "note": "ManiaPark primitive-style CPlugSolid estimate."
  }
}
```

## Selected Empirical AABB
```json
{
  "half_width": 1.1,
  "front_half_length": 2.15,
  "rear_half_length": 1.95,
  "half_height": 0.6,
  "source": "empirical_mesh_and_supervised_aabb_20260503"
}
```

## Key Checks
- Minimum supervised-minus-AABB margin: `0.216`
- Median supervised-minus-AABB margin: `0.326`
- Minimum q01 supervised-minus-AABB margin: `0.355`
- Median q01 supervised-minus-AABB margin: `0.467`
- Plot written: `True`

## Interpretation
The current AABB is intentionally empirical. It is based on the primitive mesh
footprint and checked against supervised near-contact lidar distances. The
supervised minima should not be treated as exact collision truth because the
latest near-contact run finished successfully; they are mainly a sanity check
that the selected AABB does not exceed observed close-pass distances.
