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

MMD/PMX characters can be driven from native SOMA-77 or body-only SMPL-X motion
with their own solver. Prefer [native SOMA](soma-direct.md) when displaying the
generated data, including its fingers. The rig adds cancel bones
(`肩P`/`肩C`), axis-limited twist bones (`腕捩`/`手捩`) sitting *on* the FK
chain, `D`-bones that carry all the leg mesh weight through append-copy
constraints, and a four-segment spine.

```
SOMA or SMPL-X NPZ -> PMX import (mmd_tools) -> rig inspection -> channel plan
          -> per-bone transfer solve -> quaternion channels -> actor
```

## The source frame invariant

Bone frames are built once on the rest skeleton and then carried by each
joint's chained source world rotation, so

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

Toes are not driven in the current mapping: the originally checked model used an
unweighted `左足先EX` and `左つま先` as an IK aim. Other models still require a
weight audit. Body-22 has no hand pose; native SOMA drives up to 30 finger bones.

## Verification

`scripts/_validate_mmd_retarget.py` checks three things that must agree
independently, since a silent disagreement is how a plausible but wrong action
escapes: the NumPy solver against Blender's evaluated pose (covers the channel
write, the real parent chain and every live constraint), each `absolute` bone's
world Y against the source aim, and each `relative` bone against
`R_g @ target_rest`. It also reports arm drop against the source, because a
character with flared kimono sleeves cannot be judged from a silhouette.

Historical numbers on Yoimiya over 40 frames: solver vs theory `0.0` deg, Blender
vs solver `0.05` deg (Blender's own float32 pose evaluation), relative transfer
`0.03` deg, absolute aim `0.03` deg, arm drop vs source `0.000` deg.

`scripts/_dump_mmd_rig.py` records the rig's hierarchy, rest alignment,
constraints and per-bone vertex weight — run it first on any new character.
`scripts/_render_mmd_sheet.py` renders a view-by-frame contact sheet, with
`--identity` for the rest-transfer check, `--in-place` to hold the camera still,
`--zoom` for the arms and `--abduction` to sweep the costume clearance.
[`scripts/_render_grip.py`](../scripts/_render_grip.py) aims at the fist centre
and renders two orthogonal views for checking weapon grips. The earlier
[`scripts/archive/_render_hand_closeup.py`](../scripts/archive/_render_hand_closeup.py)
aimed at one wrist and was used to derive the finger curl direction; it is kept
as a historical reference. `scripts/_compose_sheet.py` tiles a sheet's PNGs
into one image.

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

**Opt-in only as of 2026-10-22.** The default is `MmdPolishOptions.enabled=False`,
also exposed as `--faithful`. `--polish` deliberately edits the generated pose.
Root deltas now default to unscaled source metres; `--root-motion legacy-scaled`
reproduces the historical scale. The old scale function's canonical leg names do
not resolve in current source-name maps, so it actually falls back to a height
ratio. Earlier claims that this was a verified leg-length correction were wrong.

Mapped rotations do not guarantee matching contacts on a different body shape.
For generated-motion display, preserve the mapping and treat clothes/props as
separate adaptation problems. The historical artistic rationale used these
measurements on one rig (not a universal anatomical constraint):

| segment | character / H | SMPL-X / H | ratio |
| --- | --- | --- | --- |
| shoulder width | 0.118 | 0.223 | **0.53** |
| forearm | 0.118 | 0.166 | 0.71 |
| whole arm | 0.270 | 0.338 | 0.80 |
| whole leg | 0.529 | 0.528 | 1.00 |

The shorter arms and narrower shoulders can bring a correctly mapped arm into a
wide costume. This is a proportion/costume issue; numerical mapping agreement does
not establish absence of sliding or penetration.

The pose-polish controls in ``MmdPolishOptions`` are off when ``enabled`` is false.
The independently selected ``root_motion`` mode remains in effect:

- ``arm_abduction_degrees`` (12) swings each whole arm chain outward about the
  shoulder, so elbow bend and every angle inside the arm are untouched. Scaled
  by how far the arm is from pointing outward. This deliberately changes the
  silhouette and is not a general collision solve.
- ``collar_damping`` (0.45) / ``collar_limit_degrees`` (22) scale the clavicle.
  This was an artistic response to exaggerated shoulders in earlier fitted clips,
  not a validated anatomical correction. ``肩`` carries real mesh weight. The
  arms' absolute world orientations remain separately prescribed, but their
  positions and the shoulder deformation can change with the collar.
- ``twist_window`` (5) smooths the arm chain's rotation about its own axis only,
  never its aim. Axial rotation is nearly invisible in 2D and fits leave it
  jittery: the bundled walk clip jumps up to 29 deg per frame at 30 fps, which is
  870 deg/s of pronation.
- ``hand_relax`` (1.0) applies a mild resting curl, since body-22 has no hand
  pose and the flat bind pose reads as a mannequin. The flexion axis is derived
  from the rig — fingers flex toward the side the palm faces, identified by the
  thumb. It is synthetic hand animation and requires visual review on each rig.
  Native SOMA finger driving disables it.

``polish_source_frames`` applies the body rotation edits; static hand relaxation
is applied separately by the Blender adapter. The transfer validator uses the
same body-polish function, so solver agreement alone does not prove source
fidelity when polish is enabled. ``polish_deviation_deg`` reports
how far the polish moved the source. Historical probe changes reached tens of
degrees; they must not be presented as faithful generated-motion display.

## Reviewing without Blender

### Optional contact and upper-body passes (2026-10-22)

These switches belong to `scripts/render_mmd_compare.py`, after the direct MMD
retarget. They default off to keep the source mapping as the baseline:

| switch | effect and limit |
| --- | --- |
| `--foot-contact on` | Detect support from source foot markers, measure the animated character's shoe vertices, then bake bounded leg corrections. The root's XY trajectory stays unchanged; a bounded vertical pelvis correction is allowed. Runs before framing and cloth physics. |
| `--floor-z Z` | Fix the world-space floor height in metres, both with and without contact correction. When omitted, the optional contact pass estimates it from the shoes; the legacy off path still estimates it from sampled whole-mesh minima. |
| `--upper-smooth 3` | Apply a symmetric three-frame SO(3) filter to spine, neck, head and collars only. Root pose/translation, lower-body rotations, and shoulder/elbow/wrist local rotations stay unchanged. |
| `--match-camera path/to/info.json` | Reuse the baseline's camera for every rendered frame. The reference must contain the requested views with matching start frame, frame count, resolution and fps. |

Contact is still opt-in: the current character checks include a Furina case with
about 8 mm residual contact error. This is not a guarantee of zero penetration or
perfect planting across all models. Review support transitions, toe support and
airborne frames in motion, using the actual shoe measurements in `foot_contact`
from `info.json` rather than the source-only skate score.

The upper-body filter limits each **local joint correction** to 2 degrees; this
is not a 2-degree bound on the resulting head or hand orientation. Corrections
accumulate down the hierarchy. `upper_smoothing` reports local and global
orientation deviations, signed angular-velocity changes and before/after angular
acceleration. Intentional sharp dance beats can also be softened, so use a short
window and inspect a matched-camera comparison. SOMA-77 currently rejects this
option; it supports SMPL-X body-22 only.

When enabled, smoothing writes a complete derived archive to
`<output>/processed-motion.npz`, leaving the input file untouched. The archive
contains source-path/SHA-256 provenance and the JSON smoothing report. `joints22`
is updated using each frame's original parent-child vectors and the change of
parent orientation, preserving bone lengths and the unchanged lower-body branch.
It does not refit a constant skeleton. Original and derived paths are both
recorded in `info.json`; extra clips supplied through `--frame-motion` receive
the same smoothing before the shared framing hull is computed.

For a clean A/B, render the baseline first, then add `--match-camera` pointing to
that baseline's `info.json`. Pass the **same explicit `--floor-z`** to both runs.
If using the automatic floor estimate, read the baseline's `floor_z` and pass
that value to the second run. Camera matching does not choose the floor for you.
Every view records camera world transforms and optical settings per frame under
`camera_tracks`. Replayed camera keys use linear interpolation.

`--scene carry_box_two_hand` animates one fixed-size box from both hands.
`--scene-grip-mode side|bottom` selects its placement, and
`--scene-fit-start/--scene-fit-end` limit automatic sizing to an inclusive Blender
frame interval. The box has its own world animation and must not be parented to a
wrist again. Without `--scene-contact`, this follows the original hands but does
not make their changing separation fit the fixed box. The original `carry_box`
preset remains available.

The following is the **historical pose-adaptation experiment**, not the faithful
`016783` display. It uses fitted SMPL-X input and changes the character's arms
and fingers to make a side grip. To reproduce that experiment, use its archived
camera/settings and these contact options:

```text
--scene carry_box_two_hand --scene-grip-mode side --scene-contact
--scene-box-size .3192 .22 .15
--bullet --bullet-damping pmx --bullet-cloth-slack 10
--polish --root-motion legacy-scaled
```

Contact fitting calibrates the evaluated hand skin, grips the rear corners, and
keys derived shoulder, elbow, wrist and finger rotations. A bounded, continuous
elbow-plane adjustment avoids the box with the forearms; a smooth box offset
clears the torso. The contact pass leaves the input file and the already-retargeted
root trajectory unchanged; the historical retarget had already scaled that
trajectory. Box dimensions never animate. `--scene-contact` currently rejects
`bottom` placement.
With Bullet enabled, this two-hand box also gets an animated kinematic collider;
only its own collision-layer mask changes. Existing character collision exclusions
are preserved. Use the same Bullet settings and box size in both sides of an A/B.

Reports distinguish three stages: `scene.report` contains the original bone-proxy
fit measurements; `scene.report.hand_contact` is explicitly `pre_physics`; and
`scene.post_physics_clearance` checks the actual evaluated actor triangles against
the box after the full Bullet bake. The historical matched-camera `016783` comparison
checked all 152 frames at 50 fps: before contact fitting, every frame intersected
the box (maximum sampled penetration 54.18 mm); after fitting, no actor/box
triangle intersections were found, including the sleeves and bracelet. This uses
a 0.2 mm inset and discrete animation frames, not continuous collision detection
between frames. Penetration depths are sampled lower bounds. Evidence is saved in
`outputs/props-2026-10-22/box/video_after_bullet_final/info.json` and
`box_contact_bullet_final_validation.json` in the same parent box directory.

This result established sampled collision clearance, **not natural grasping or
motion fidelity**. It is specific to that character and clip. The maximum derived wrist
translation is 82.0 mm, elbow-plane adjustment is 30 degrees, forearm axial twist
can reach 90 degrees, and the right wrist's local rotation differs from the source
by up to 110.2 degrees. Review the close views before reusing the fit elsewhere.
Most contact fingers remain about 2–3.2 mm from the box; the left index and right
little finger remain lifted (about 12.5 and 21.4 mm respectively). The source
motion's raised-box face occlusion and foot motion were retained. Those large
arm/wrist changes make this unsuitable as the default original-motion showcase.

The subsequent native SOMA experiment (`--scene carry_box_supported`) instead
leaves every character channel and root sample unchanged and moves only the box.
On `016783`, the generated hands
rise toward the face and turn away from a common support plane late in the clip.
Keeping a box above the fingers can avoid geometry intersections but does not
establish that the palms support its weight. After the MCP wrist-axis correction,
the current box-only candidate has support gaps over 10 mm in 57 of 152 frames
(maximum 37.45 mm), and frames 109–139 present the backs of the hands toward the
box. Its sampled box/actor triangle audit finds no intersections, including after
the filtered garment bake, but finger-tip proximity is not evidence of a
load-bearing grasp. It is not a solved natural lift.
Retain the unmodified, no-box motion beside it and report these limitations.
An explicitly separate `085015` generated clip was also tried; a shallower box
reduced collisions but still intersected the body in 13 sampled frames, with a
sampled penetration up to 26.1 mm. It is not a successful substitute for the
original clip. Neither candidate justifies silently changing the source motion.

For weapons, `--weapon-fit-mode bounded` opts into the bounded finger solver.
`legacy` remains the default. The bounded diagnostic is a bone-centre proxy,
not a skin-surface collision test, and that mode leaves thumb opposition unsolved.

`--weapon auto --weapon-fit-mode surface` selects the calibrated right-hand grip
for the installed Feixiao character and weapon pair. It checks hashes of the
imported geometry, hand weights and rest skeleton; an unknown asset is rejected
instead of receiving another character's pose. The Feixiao PMX is a three-weapon
pack. This preset selects the independent central axe and retains its full
2.320 m length, excluding the two unheld side weapons from the working scene.
The source PMX and motion files are unchanged. The attachment holds the narrow,
stable blue shaft. With body-only SMPL-X, the grip adds synthetic poses for all
15 finger bones, including thumb opposition. With native SOMA, the source finger
keys are retained unless `--override-source-fingers` explicitly requests the
calibrated grip. The preset records its component selection and evaluated skin contact
samples in `weapon.fit` and `weapon.fingers`. Outline shells are excluded from
the measurements. These are discrete surface samples, not a guarantee against
all intersections; whole-clip weapon/body contact needs its own review.

For this verified pair, **`--weapon-arm-clearance` explicitly opts into** checking left-arm/weapon triangle crossings
and tries a calibrated, smooth shoulder rotation with compensating wrist rotation.
It retains the elbow's local bend and wrist world orientation, and validates the
entire transition before inserting any keys. The historical fitted-SMPL-X `128701` clip used a 17°
peak rotation and 116.6 mm wrist displacement over 0.8 seconds. Limits are 20° and
120 mm; infeasible motion retains its original arm keys and records the failure
in `weapon.left_arm_clearance`. This is an explicit character adaptation, not a
claim that the source arm movement is unchanged. It is disabled by default and
has not established an acceptable deviation for native generated-motion display.

With `--bullet`, the selected axe also gets animated collision proxies built from
small clipped convex pieces. These follow its original geometry and improve hair
and clothing contact without changing the original rigid-body collision masks.
The proxy report is stored in `physics.scene_weapon_colliders`; discrete audit
results still matter because thin hair and hem surfaces can cross between proxies
or simulation steps.

`scripts/render_mmd_compare.py` renders the character and the selected source
skeleton as two passes under one camera; `scripts/build_review_page.py` stacks
them into MP4s and writes an `index.html`. The skeleton panel shows stored joint
positions. On fitted SMPL-X inputs those positions need not equal the FK of the
fitted rotations, so comparing silhouettes alone cannot identify every mapping error.
`scripts/_batch_review.py` drives both over a directory of clips.

# Looking right: cel shading

A correct retarget on a Genshin or Honkai model still looked like clay, and the
reason is not the motion. These models are authored for a non-photoreal shader:
a 16-32 px ramp texture decides the shadow step, a sphere map fakes the
specular, and an outline sells the drawing. `mmd_tools` loads all three, then
wires the ramp into a physically based mix as if it were a colour multiply, and
sets `Sphere Tex Fac` to 0. So every input is present and none of them is doing
its job — smooth diffuse shading over an anime texture, which is the waxy look.

`blender/mmd_toon.py` rebuilds each material the way the model expects:

```
N·L  ->  sharpened step  ->  the model's own shadow tint  ->  x base texture
```

`Shader to RGB` supplies `N·L` including cast shadows, so this is EEVEE-only.
The result is emitted rather than lit, which means **brightness is bounded by the
base texture and cannot blow out** — that alone fixed the washed-out faces, which
three area lights at 900/420/280 W had been overexposing.

Two decisions are read off the model instead of guessed:

- **A material with no toon ramp is left unlit.** Eyes, mouth interiors, teeth
  and brows are authored that way on all three rigs tested; shading them is what
  makes anime eyes look dead.
- **The face gets a much flatter terminator.** Genshin drives face shadow from an
  authored SDF map that a PMX rip does not carry, and a plain `N·L` terminator
  cuts a hard line across the nose and eyes that reads as a blemish. The face
  group is identified by which materials share a base texture with an *unlit*
  material — `face_base_images` — so no material-name matching is needed and it
  works whatever language the rig is in.

The shadow tint is sampled from the bottom row of the model's own ramp, so a warm
skin shadow stays warm.

`add_outline` builds an inverted hull: a copy of the character, inflated by
`Solidify` with flipped normals and a back-face-culled black material, so only
the silhouette survives. The copy keeps the armature modifier and therefore
deforms with the animation. 4.5 mm on a 1.5 m character is about 3 px at 1500 px
wide; the 1.8 mm first attempt was under a pixel and invisible.

`add_toon_lighting` replaces the three-light rig with one sun plus flat ambient.
Cel shading wants a single clean terminator; three lights give three overlapping
ones. The sun's angle is wide (0.42 rad) so both the body terminator and the cast
shadow stay soft. `add_ground` adds a floor that receives that shadow, which is
most of what tells a viewer where the feet are.

Pass `--toon` to `scripts/render_yoimiya.py`, `scripts/render_mmd_compare.py` or
`scripts/_render_mmd_sheet.py`; `--no-outline` and `--no-ground` opt out.

## What is still missing

Verified on Yoimiya, Furina and Silver Wolf. Known gaps, in the order they will
be noticed:

- Hair and cloth need per-model visual review. `--bullet` uses the PMX's rigid
  bodies; `--spring` is the older collision-free approximation. Fast source
  turns can still look unnatural even with secondary motion.
- Source diagnostics do not automatically distinguish bad motion from an
  intentional sharp gesture. `score_motion_quality.py` now uses signed rotation
  vectors; optional upper-body smoothing must be reviewed in motion.
- Face shadow is a flattened ramp, not the game's SDF map.
- Body-22 has no finger pose. Native SOMA retains finger animation; optional weapon
  grip overrides add authored/solved curls. The reviewed
  Feixiao `surface` preset adds thumb opposition and sampled skin contact checks;
  automatic, reliable contact for arbitrary character/weapon pairs remains open.
- **A fixed floor does not establish foot contact.** `_mmd_ground_offset` supplies
  a static rest-mesh offset. The contact-off renderer then estimates a separate
  floor from sampled posed mesh minima, or uses explicit `--floor-z`. It can raise
  the floor to the feet; it does not make the soles follow that plane across the
  clip. Earlier claims that bent legs always hover 4–8 cm, or that the floor is
  always lowered, were not valid general descriptions. Contact tests and props
  must use the reported `floor_z`. The optional shoe-based pass changes the pose
  and remains separate from the default faithful display.

## Secondary motion

The current batch default is Bullet (`--physics bullet`); the single-clip renderer
requires `--bullet`. It bakes the PMX rigid-body setup at the source frame rate
and records the actual result in `info.json`. `--bullet-damping legacy-inverted`
preserves the older showcase's artistic damping adjustment;
`--bullet-damping pmx` preserves the model's authored value. The inversion is not
a validated format/unit conversion. Failed Bullet setup stops the render.

Bullet still requires review on the actual unchanged source motion. Clothing
constraint changes accepted on a polished SMPL-X pose are not automatically
validated on native SOMA. `--bullet-cloth-slack` changes locked garment angular
limits; it is an optional costume adaptation, not a source-motion correction.
`--bullet-cloth-linear-slack` similarly allows a bounded translation on originally
locked garment axes, expressed in metres and defaulting to zero. On native
`100948`, the reviewed 20-degree angular / 0.002 m linear candidate reduces the
visible front-skirt pops while retaining the mapped body and finger animation.
The joint filter excludes connections to other dynamic garment chains so it does
not relax the long panel's root pin. Even with that filter, the 10-degree / 0.002 m
walking candidate keeps the side panel farther back around frames 127–140 than
the zero-linear-slack comparison. These are clip-specific costume settings, not
a validated universal default or a claim of physically accurate cloth.
Neither a low jitter score nor zero sampled intersections establishes natural
cloth behavior. Review consecutive frames and occluded sides as well.

The following describes the historical spring fallback, still available through
`--spring`. Earlier Bullet failures motivated this fallback; they do not show
that PMX rigid bodies inherently fail during motion. Frame rate, initialization,
collision coverage and constraints all affect the current Bullet result.

`blender/mmd_spring.py` keeps the part of that data that is worth keeping and
throws away the solver. The PMX says which bones should swing: a rigid body with
physics mode 1 or 2 is dynamic, and the bone it names is a bone the artist wanted
to move on its own. Those bones get a damped spring, solved here and baked to
quaternion keys.

Each bone's tip is pulled toward where it would be if the bone were rigid, carries
its own velocity, and is then projected back onto its own length. This bounds
each simulated tip's distance from its head; it does not guarantee a stable,
collision-free or plausible skinned surface.

Three things it does *not* do, each for a reason:

- **No added gravity.** This fallback adds lag around the authored pose. Some
  assets are sculpted with hair already lifted, so the assumption that rest means
  a gravity-settled pose is not generally true.
- **No collision.** A spring cannot know about the body, so a skirt can pass
  through a leg. The angle clamp bounds motion but cannot establish a low
  penetration rate.
- **It excludes mapped driven bones.** This preserves their channels, but does
  not make the target's proportions, mesh contacts or all other bones match the source.

For this spring fallback, physics import is metadata-only, and `mmd_tools`'s own rig build is
deliberately skipped: `Model.build()` disconnects the physics bones so the rigid
bodies can own them, and once the bodies are deleted those bones float at the
armature origin and smear across the frame. That was the second failed attempt.

Pass `--spring` to `scripts/render_mmd_compare.py`.

## Expressions

Facial expressions are vertex morphs in the PMX, and the importer was only asked
for `MESH` and `ARMATURE` — so every render before this carried **no morphs at
all**. The face was the model's neutral mesh, which on an anime model reads as
faintly stern. Nothing had been switched on by accident; nothing had been switched
on at all.

`blender/mmd_expression.py` imports them (`morphs=True`) and applies a preset.
Morph names are a de facto standard across the MMD ecosystem, and all three rigs
tested carry the same vocabulary — `にこり` narrows the eyes into a gentle smile,
`口角上げ` lifts the mouth corners, `笑い` closes the eyes into arcs. Each slot
lists aliases, and a slot that resolves to nothing is reported rather than
silently skipped, because a silently neutral face looks like a bug.

The single-clip default is `smile`; the dataset batch default is `soft_smile`.
Both are authored facial overrides, not generated facial animation. Use
`--expression none` when a neutral, unmodified-face comparison is needed.

Scale matters, and it is worth being blunt about it. At showcase framing the head
is about 40 px in a 1000 px frame, and switching the whole preset on changes **72
pixels**. The expression is for close-ups and figure stills; it will not rescue a
full-body video. What *did* make faces look wrong in the earliest batch was
exposure — three area lights at 900/420/280 W blew the face to white with a few
dark slashes left, which reads as a grimace. Cel shading fixed that, and it is a
separate problem from the morphs.

## Retuning the springs

The first shipped spring was `stiffness=0.34, damping=0.76`, and it looked like
someone was pointing a fan at the character: sashes lifted horizontally and hair
splayed. The relevant quantity is not "critically damped" — this discrete spring
is underdamped for any sane stiffness — but how fast the oscillation *decays*. The
complex roots have magnitude `sqrt(damping)`, so that is the per-frame amplitude
ratio: 0.76 decays 13% per frame and leaves about half a second of ringing after
every move, while 0.30 decays 45% per frame and is gone in four.

Defaults are now `stiffness=0.50, damping=0.30`, plus a whole-chain deflection
budget. A per-bone cap alone is length-dependent: at 24 degrees each, a 10-bone
scarf can end up 240 degrees off its rest while a 5-bone ponytail manages 120, so
one setting reads as drape on one chain and as a gale on the other. Each bone now
gets the smaller of the per-bone cap and an equal share of `max_total_degrees`.

One thing tuning cannot fix: some models are *sculpted* mid-motion. Furina's hair
streams horizontally in her rest mesh, and it does so with the simulation switched
off entirely. Making it hang would mean editing the character, not the solver.
