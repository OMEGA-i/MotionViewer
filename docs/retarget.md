# Retarget Pipelines

Two target rig families are supported and they do not share a solver:
`mixamo` (FBX, below) and `mmd` (PMX, [MMD Retarget Pipeline](#mmd-retarget-pipeline)).

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

# MMD Retarget Pipeline

MMD/PMX characters (Genshin-style rips) are driven from the same body-only
SMPL-X motion, but they need their own solver. The rig adds cancel bones
(`肩P`/`肩C`), axis-limited twist bones (`腕捩`/`手捩`) sitting *on* the FK
chain, `D`-bones that carry all the leg mesh weight through append-copy
constraints, and a four-segment spine.

```
SMPL-X NPZ -> PMX import (mmd_tools) -> rig inspection -> channel plan
          -> per-bone transfer solve -> quaternion channels -> actor
```

## The source frame invariant

Bone frames are built once on the rest skeleton and then carried by each
joint's SMPL-X world rotation, so

```
S(t, b) = R_g[b] @ S_rest(b)
```

holds for every bone. Two things follow. Every transfer collapses to one
constant per-bone calibration, `W(t, b) = S(t, b) @ C_b`. And the rest frame's
roll is free: each calibration is solved against `S_rest`, so a constant roll
cancels exactly. Only its *conditioning* matters, which is why the rest roll
reference is the world axis least parallel to the bone — arm bones lie almost
along world X, and projecting a fixed X reference onto their perpendicular
plane turned rounding noise into roll that flipped `left_collar` by up to 133
degrees between neighbouring frames.

Aiming a posed bone at its posed child looks equivalent, and is for the Y axis
up to shape blend shapes, but it leaves the roll undetermined for exactly those
bones — and roll is what carries twist.

## Per-bone transfer modes

`relative` — `C = S_rest^-1 @ target_rest`, so `W = R_g @ target_rest`. The
character keeps its own rest orientation and receives the source's rotation
about it. Correct wherever the rigs already agree on a bone's rest direction,
and required wherever the target's rest axis is a convention rather than an
anatomical aim: `腰` displays 47 degrees off vertical, `頭` is a vertical bone
while the SMPL-X neck-to-head edge leans 17 degrees forward, `足首` points
further down because of the footwear.

`absolute` — `C = Roll_y(theta)`. A roll about Y cannot move the Y axis, so the
bone direction follows the source exactly *and* the source's rotation about the
bone axis survives. `theta` is solved once at rest, preserving the character's
own roll and therefore its elbow hinge plane and 捩 axes. Required on arms:
MMD binds A-pose against SMPL-X's T-pose rest.

On this rig the split is unambiguous — arms disagree by 23-51 degrees, and
everything else by 1.5-6:

| joint | rest gap | mode |
| --- | --- | --- |
| `手首` / `ひじ` | 51 / 47 deg | absolute |
| `腕` | 24-29 deg | absolute |
| `頭` / `足首` | 17 / 16 deg | relative (bone convention) |
| `上半身*`, `首`, `足`, `ひざ` | 1.5-5.8 deg | relative |

A mode boundary is fine at a real joint — the torso stays upright while the arms
copy the source — but not inside one straight limb: an `absolute` forearm under
a `relative` hand kinks the wrist by the full rest gap.

## Twist redistribution

Putting a whole arm rotation on `腕` candy-wraps the mesh, because the deform
weight is spread across `腕捩` and its `腕捩1/2/3` append copies. The local
basis is split into swing and twist about the bone axis; swing stays on `腕`
and the twist goes to `腕捩` conjugated by its own rest local:

```
B_twist = rest_local^-1 @ twist @ rest_local
```

Without the conjugation the 捩 bone's own rest rotation is applied on top of the
twist, shifting everything below it. With it, the chain below the 捩 bone is
identical to putting the whole rotation on the swing bone, which is the property
the unit tests assert.

## Channels, not assumptions

Local bases are reconstructed against each bone's **actual** parent. Every bone
between the armature root and a driven one is an explicit pass-through channel
(`センター`, `肩P`, `肩C`, `下半身`), so an inserted 捩 bone is a node in the
chain rather than an assumption that a mapped parent telescopes. Blender's rest
matrices are single precision; the quaternion round trip is redone in float64
and the forward model uses the same quaternion Blender receives, because a 1e-7
orthonormality defect is amplified once per level and reached 0.47 degrees at
`頭`, eleven bones down.

IK, rotation limits and negative-influence cancel constraints are muted.
Positive-influence append copies stay live: they are how `足D`/`ひざD`/`足首D`
carry the leg mesh, which holds all of the weight while the FK leg bones hold
none.

Toes are not driven — `左足先EX` carries no weight on this rig family and
`左つま先` is an IK aim. Fingers are not driven either: body-22 has no hand pose.

## Verification

`scripts/_validate_mmd_retarget.py` checks three things that must agree
independently, since a silent disagreement is how a plausible but wrong action
escapes: the NumPy solver against Blender's evaluated pose (covers the channel
write, the real parent chain and every live constraint), each `absolute` bone's
world Y against the source aim, and each `relative` bone against
`R_g @ target_rest`. It also reports arm drop against the source, because a
character with flared kimono sleeves cannot be judged from a silhouette.

Current numbers on Yoimiya over 40 frames: solver vs theory `0.0` deg, Blender
vs solver `0.05` deg (Blender's own float32 pose evaluation), relative transfer
`0.03` deg, absolute aim `0.03` deg, arm drop vs source `0.000` deg.

`scripts/_dump_mmd_rig.py` records the rig's hierarchy, rest alignment,
constraints and per-bone vertex weight — run it first on any new character.
`scripts/_render_mmd_sheet.py` renders a view-by-frame contact sheet, with
`--identity` for the rest-transfer check, `--in-place` to hold the camera still,
`--zoom` for the arms and `--abduction` to sweep the costume clearance.
`scripts/_render_hand_closeup.py` aims a camera at one wrist, which is how the
derived finger curl direction was checked. `scripts/_compose_sheet.py` tiles a
sheet's PNGs into one image.

## Configuration

```yaml
body:
  backend: fbx_skeleton
  fbx_path: assets/fbx/pmx/yoimiya/宵宫.pmx
  fbx_scale: 0.08
  bone_map: mmd
  retarget_mode: direct
```

`retarget_mode` is `direct` for MMD: the contact and joint-limit passes are
Mixamo-profile specific and are not applied.

## The polish pass

The transfer above is exact. It is also not, on its own, what a stylised
character should do, because the two bodies are not the same shape. Measured
against height on this rig:

| segment | character / H | SMPL-X / H | ratio |
| --- | --- | --- | --- |
| shoulder width | 0.118 | 0.223 | **0.53** |
| forearm | 0.118 | 0.166 | 0.71 |
| whole arm | 0.270 | 0.338 | 0.80 |
| whole leg | 0.529 | 0.528 | 1.00 |

Legs match, which is why they never looked wrong. Arms do not: the shoulders are
half as wide and the arm is a fifth shorter, while the costume is not narrower at
all. An arm that hangs at the source's exact angle therefore starts much closer
to the midline and ends up inside the kimono. That is a proportion mismatch, not
a retarget error, and no amount of transfer accuracy fixes it.

``MmdPolishOptions`` groups the departures, all off when ``enabled`` is false:

- ``arm_abduction_degrees`` (12) swings each whole arm chain outward about the
  shoulder, so elbow bend and every angle inside the arm are untouched. Scaled
  by how far the arm is from pointing outward, so a raised or extended arm —
  which never clips — stays exactly on the source.
- ``collar_damping`` (0.45) / ``collar_limit_degrees`` (22) scale the clavicle.
  Fits dump shoulder motion into the collar because it is barely observable: on
  these clips it averages 9-28 deg and peaks at 78, where a real clavicle manages
  about 20. ``肩`` carries real mesh weight, so the raw value reads as a hunched,
  yanked shoulder. Arms are transferred in ``absolute`` mode, so damping the
  collar cannot move them — only the shoulder's deformation changes.
- ``twist_window`` (5) smooths the arm chain's rotation about its own axis only,
  never its aim. Axial rotation is nearly invisible in 2D and fits leave it
  jittery: the bundled walk clip jumps up to 29 deg per frame at 30 fps, which is
  870 deg/s of pronation.
- ``hand_relax`` (1.0) applies a mild resting curl, since body-22 has no hand
  pose and the flat bind pose reads as a mannequin. The flexion axis is derived
  from the rig — fingers flex toward the side the palm faces, identified by the
  thumb — so a model that binds its hands at a different angle still curls
  correctly.

``polish_source_frames`` applies all of it in one place, and the validator calls
that same function, so the reported transfer error is transfer error rather than
the polish measured against un-polished theory. ``polish_deviation_deg`` reports
how far the polish moved the source, so the trade-off stays visible: currently
12-57 deg, almost all of it the collar clamp.

## Reviewing without Blender

`scripts/render_mmd_compare.py` renders the character and the SMPL-X source
skeleton as two passes under one camera; `scripts/build_review_page.py` stacks
them into MP4s and writes an `index.html`. A pose that looks wrong on the
character and also looks wrong on the skeleton came from the motion.
`scripts/_batch_review.py` drives both over a directory of clips.
