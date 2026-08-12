# Mixamo Retarget Pipeline

MotionViewer drives a validated Mixamo FBX character from body-only SMPL-X
motion. The supported target rig family is `mixamo`; bare, `mixamorig:`, and
`mixamorig1:` assets differ only by a stored bone-name prefix.

## Pipeline

```
SMPL-X NPZ -> FBX import -> Mixamo preflight/profile
          -> global rest-delta transfer -> contact-aware quality pass -> actor
```

1. Import records and validates the armature object basis without mutating it.
2. Preflight requires the complete 22-bone Mixamo hierarchy, valid parent
   links, non-degenerate lengths, one unambiguous namespace, a rigid uniform
   object basis, and a finite world height in `[0.25, 4] m`. Unsupported rigs
   fail before animation and must remain quarantined in the catalog.
3. `MixamoNameAdapter` maps the one internal Mixamo skeleton contract to the
   imported names. No animation code branches on a prefix.
4. Blender samples source pose matrices once. The pure NumPy solver applies the
   source global rest delta to target rest, reconstructs target local
   quaternions, and evaluates target FK from its own rest-local transforms.
5. `quality` mode applies closed-form arm/leg IK with source pole hints,
   source-relative quaternion velocity limits, and a fixed three-pass contact
   solve over profiled mesh sole anchors. `direct` preserves raw rest-delta
   output for audits. Blender never decides a constraint or modifies a pose.

## Configuration

```yaml
body:
  backend: fbx_skeleton
  fbx_path: assets/fbx/iron.fbx
  bone_map: auto       # auto or mixamo
  retarget_mode: quality # quality or direct
```

The catalog has one `mixamo` profile. `check-fbx --deep` runs the five candidate
binary assets against `gt`, `mdm`, and `omegamotiongpt` in three parallel,
CPU-only Blender processes. `--motion` may be repeated to replace that default
matrix. The report contains proposed statuses, but catalog approval stays
explicit.

## Quality Metrics

`scripts/retarget_quality_audit.py` records mapping coverage, full-skeleton
height-normalized MPJPE, endpoint and segment error, contact drift, profiled sole
height, whole-mesh minimum height, root motion, joint limits, mirror anomalies,
quaternion validity, angular velocity, and representative frames for every
metric peak. The gate requires complete coverage,
bone-length drift below `1e-4`, mean/P95 MPJPE below `4%/8%` of target height,
sole penetration no deeper than `5 mm`, and contact drift below
`max(1 cm, 0.75% of target height)`.
