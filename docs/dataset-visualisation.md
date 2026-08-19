# Visualising a T2M dataset on an MMD character

One command turns a directory tree of SMPL-X clips into MP4s and a browsable
page:

```bash
uv run python scripts/visualize_dataset.py --list                       # what is in there
uv run python scripts/visualize_dataset.py --task picks38/gt            # render one task
uv run python scripts/visualize_dataset.py --task picks38/gen --compare picks38/gt
```

`scripts/visualize_dataset.py` discovers the tasks under `--data`, validates every
file against the input contract below, scores the motion, renders the selection,
and writes `index.html` next to the videos. It is resumable, deletes frames once
they are encoded, and collects failures rather than stopping on the first one.

## Input contract

A clip is one `.npz`, loadable with `allow_pickle=False`, holding four arrays:

| key | shape | meaning |
| --- | --- | --- |
| `global_orient` | `(T, 3)` | root orientation, **axis-angle** |
| `body_pose` | `(T, 63)` or `(T, 21, 3)` | the 21 body joints, axis-angle |
| `transl` | `(T, 3)` | root translation, **metres** |
| `joints22` | `(T, 22, 3)` | joint world positions, metres |

`fps` is read if present and assumed to be 30 otherwise. `caption` is picked up
for the page. Everything else in the file is ignored.

Three conventions have to match, and all three are checked:

- **Joint order** is SMPL-X body-22, as listed in `SMPLX_BODY22_NAMES`. The check
  is that every bone in `SMPLX_BODY22_PARENTS` holds a constant length over the
  clip; a permuted order shows up immediately as a limb that changes length.
- **The source space is Y-up**, in metres, with the feet near `Y = 0`.
  `_SOURCE_TO_BLENDER` maps `(x, y, z)` to `(x, -z, y)`.
- **`joints22` is required**, not optional. The rest skeleton is recovered from
  it by `recover_rest_offsets`, and the height ratio that scales root translation
  onto a stylised character comes from that skeleton. No SMPL-X model weights are
  loaded, so joint positions cannot be derived from the pose parameters alone.

### Layout

Any nesting works — every directory containing `.npz` files becomes a task, named
after its path with a `_smplx` suffix stripped:

```
<root>/picks38_smplx/gen/000003.smplx.npz   ->  task "picks38/gen", clip "000003"
<root>/picks38_smplx/gt/000003.smplx.npz    ->  task "picks38/gt",  clip "000003"
```

Clips that share an id across two tasks can be rendered side by side with
`--compare`, which is how a generation is put next to its ground truth.

### How exact does the data have to be?

`global_orient`/`body_pose` and `joints22` are two descriptions of the same
motion, and they never agree perfectly. Reprojecting the joints through forward
kinematics measures the gap: **6.5–8.3 mm mean** on the T2M exports, against
**6.5 mm** on the SOMA clips the pipeline was originally built on. Most of that
residual is the single-constant-offset-per-bone approximation in
`recover_rest_offsets`, not the data. Anything in this range is fine; a gap of
centimetres would mean the two arrays describe different motions.

## Lead-in trimming

Two artefacts sit at the head of these exports. Both are detected per clip by
`detect_trim`, reported on the page, and recorded in `manifest.json`. `--trim
none` disables the whole thing; `--trim N` drops a fixed count.

**A frozen lead-in.** Every `old500` clip holds one pose for exactly 14 frames
(19 of 500 hold 15) before moving, and the onset is smooth — 2.4 mm against a
7.4 mm later median, so there is no pop, just half a second of a statue at the
head of every video. Detected as a leading run of steps under 0.5 mm/frame.

**An anchor pop at frame 0.** `gen` clips start on the conditioning pose and jump
to the first generated frame: a **358 mm** single-frame step in `new500` against
a 9 mm median afterwards, and 277 mm in `picks38/gen`. 496 of 500 `new500` clips
and 32 of 39 `picks38/gen` clips have it. Exactly one transition is affected, so
one frame is dropped.

This one is not cosmetic. `_mmd_root_locations` re-bases the whole root path on
frame 0 (`(roots - roots[0]) * scale`) and the ground offset is measured there
too, so an anchor pose at frame 0 mis-places the entire clip.

Trimming changes what the metrics say, because scoring runs after it:

| task | penalty before | penalty after |
| --- | --- | --- |
| `new500/gen` | 11.79 | **0.73** |
| `picks38/gen` | 12.91 | **2.24** |
| `picks38/gt` | 4.03 | 3.63 |
| `old500/gen` | 1.29 | 1.36 |

The ranking inverts: `new500` looks like the worst set until the anchor frame is
removed, after which it is the cleanest. `old500` gets slightly worse because its
14 frozen frames were diluting every average.

## What is measured but deliberately left alone

These videos are evidence about a motion model, so a flaw in the *source* motion
has to survive into the render. The page badges them instead of fixing them:

- **Floor penetration.** The retarget applies vertical delta faithfully, so a clip
  whose feet drop below their frame-0 level sinks into the floor. The median is
  0.7–1.8 cm, which nothing shows; the worst `new500` clip reaches 25.8 cm. Forcing
  ground contact would rewrite the model's output, so it is measured
  (`ground_penetration_m`) and badged past 5 cm.
- **Run-away root travel**, jitter, and peak angular velocity, likewise.

`score_motion_quality.py` weights each penalty term to reach ~1.0 where the fault
becomes visible: 600 deg/s of turn, 0.5 deg of jitter, 0.35 m/s of foot skate,
and a 10 cm dip with a 3 cm deadband below which nothing is visible.

### Contact metrics need a percentile, not a maximum

Foot skate is the **90th percentile** of horizontal foot speed over contact
frames. Contact is decided by height alone, so the frame where a foot lifts off
sits right at the threshold while already moving fast, and a maximum picks exactly
that frame. Measured on `picks38/gen`, a clip captioned *"walking forward at a
steady pace"* reported **5.42 m/s** at its worst contact frame against a 0.31 m/s
median over the other 28 — a badge claiming 20 km/h of sliding on a clean walk.
The percentile puts the same set at a 0.46 m/s median, against 0.02 m/s on the
SOMA baseline.

`ground_penetration_m` had the subtraction the wrong way round; since `floor` is
the minimum over every frame, `floor - frame0` can never be positive, so the term
was identically zero and its penalty never fired. Both faults are pinned by
`tests/test_motion_scoring.py`.

### What the metrics say about these sets

With the lead-in trimmed and both faults fixed, the turn and skate terms fall to
zero on the 500-clip sets and **jitter is the only term left**:

| set | jitter | penalty |
| --- | --- | --- |
| SOMA baseline | 0.09° | 0.21 |
| `new500/gen` | 0.20° | 0.49 |
| `old500/gen` | 0.20° | 0.59 |
| `picks38/gen` | 0.38° | 1.13 |
| `picks38/gt` | **0.45°** | 1.31 |

The ground truth is the *jitteriest* set, which places the shake before the model:
both tracks pass through the same 77-joint → SMPL-X fit, whose residual is 6.5–8.3
mm per frame. That is the head wobble visible in the renders, and it is a property
of the fit, not of the generator.

Metrics are recomputed on every run and written back over the entries already in
`manifest.json`, so correcting a metric refreshes the page without re-rendering.

## Image quality

Defaults were picked by measuring scale-normalised edge energy on the character,
not by eye:

| setting | relative detail | cost |
| --- | --- | --- |
| 800p, `filter_size` 1.5 | baseline | — |
| 800p, `filter_size` 0.9 | +19% | free |
| 1080p, 0.9 | +38% | 0.92 s/frame |
| **1440p, 0.9** (default) | **+56%** | ~1.2 s/frame |
| 1920p, 0.9 | +72% | 1.58 s/frame |

Two things this settles:

- **`filter_size` was the free win.** Blender defaults the pixel filter to 1.5 px,
  which softens every edge; cel shading is nothing but hard edges and an outline, so
  it loses the most. 0.9 costs nothing and recovers 19%.
- **Samples were never the bottleneck.** 128 render samples measured 5.38 against
  5.41 for 64 — inside the noise, for double the render time. `render_mmd_compare.py`
  had never set `taa_render_samples` at all, so it was on Blender's default of 64;
  raising it would have been pure waste.

H.264 is encoded at CRF 16 rather than 19: 47.0 dB against 45.0 dB PSNR for 62%
more bytes, which matters because large flat colour areas are where H.264 bands
first.

## Facial expression

Expressions come from the model's own vertex morphs, and **the morph's group decides
what its name means**. ``にこり`` sits in the 眉 (brow) block: driving it to 1.0
changes exactly zero pixels in the mouth region. The old `smile` preset used it as
the "smiling eyes" slot, so the whole preset moved 1537 pixels on a face close-up
where ``なごみ`` alone moves 12304. That is why the expression did not read as a
smile — it was softening the brows and nothing else.

The default `smile` drives two slots only: brow ``にこり`` at 0.5 and mouth
``口角上げ`` at 0.55. Both of the things it does *not* do were mistakes worth
recording.

**It does not touch the eyes.** Narrowing them with ``なごみ`` reads as a warmer
smile in a close-up, but these characters are designed around large open eyes:
shrinking them changes who the character looks like, and at full-body scale the
irises vanish into slits and it reads as squinting. The eye morph lives in
``smile_eyes``, opt-in.

**It does not push the mouth hard.** Counting changed pixels said ``口角上げ`` at 1.0
was "too subtle" (456 px), so an earlier version stacked ``にやり`` on top — and on
these models ``にやり`` moves the mouth almost identically to ``口角上げ``, so that
doubled it. Driven that hard the result reads as an *open* mouth, which is worth
understanding:

> These faces draw the mouth as **two strokes with a gap in the middle**, with a
> dark accent at each outer corner. All of that is in the neutral mesh — rendering
> the same frame with ``--expression none`` shows the identical corner marks. Nothing
> is a shadow and the mouth is never open. But stretch and curve those two strokes
> and the pair, either side of the gap, reads as an open mouth with fangs at about
> 40 px. `smile_wide` still exists for a shot where the mouth is large enough to
> carry it.

The lesson: a pixel count is not a look. Judge a morph on a render at the size it
will be seen.

All slots resolve on Yoimiya, Furina and Silver Wolf; the smirk slot used by
`smile_wide` reaches Silver Wolf through the ``にやり２`` alias, since digit width
varies between models.

### The outline was drawing fangs, and it was one colour for everything

The two dark marks at the corners of a closed mouth were **not** the morph, the
texture or a shadow. They were the **inverted-hull outline**, and the root cause was
simpler than the artefact looked: the shell was painted a single near-black for the
whole character.

A PMX carries an `edge_color` **per material**, and on Yoimiya they differ: hair and
nails ask for a warm brown `(0.64, 0.37, 0.15)`, skin and face for
`(0.50, 0.25, 0.00)`, the head ornament for a reddish `(0.50, 0.25, 0.25)`, and only
the clothes for pure black. An inverted hull also shows at every *opening* in a mesh,
not only at the silhouette, so the face shell appears through the eye and mouth holes
— and painted near-black those two specks read as fangs, where in the model's own
colours they are a warm brown that disappears into the lip. Four changes, in the
order they matter:

| change | effect |
| --- | --- |
| use each material's authored `edge_color` | mouth region 63/255 → 89/255, and the hair gets its soft brown edge back |
| convert those colours **sRGB → linear** | a node's `default_value` is linear; sRGB 0.50 fed raw emits as a bright gold line down the jaw, where linear 0.22 is the dark brown intended |
| respect `enabled_toon_edge` per material | 13 of 29 materials say no edge — eyes, lashes, brows, mouth, teeth, skirt — and the game respects it, which is why the game has none of this |
| hold the line at a constant **1.6 px** | thickness is in metres, so a fixed 4.5 mm was 1.4 px at 800p and 3.3 px at 1920p; raising the resolution had silently tripled the outline |

**Two approaches that looked right and were not**, both worth recording because both
sound more principled than the fix:

- *Exclude the face group from the shell.* It removes the marks, and it also removes
  the **hair** outline: the face group is detected by texture sharing, and these rigs
  pack hair and face onto one sheet, so `report["face"]` contains `髮` and `頭飾`.
- *Taper the shell thickness to zero at open boundaries.* Exactly targets where the
  shell shows — except an MMD body is assembled from unwelded pieces, one surface per
  material and hair built from separate cards, so **77%** of Yoimiya's vertices sit on
  an open boundary and **96%** are within two rings. It does not trim the artefact, it
  deletes the outline. Kept as `outline_boundary_rings`, defaulting to 0.

Measured across the three: the final version has both the cleanest mouth (6 dark
pixels against 11) and the *most* outline overall (73538 dark pixels against 67459),
which is the tell that the marks were never extra geometry — only the wrong colour.

`outline_tint` scales the authored colours if the warm skin edge reads too light
against a pale backdrop; 1.0 is faithful, since MMD also draws the edge unlit.

### Whether the expression is visible at all is up to the clip

Two things decide it, and neither is the render:

- **These fits nod the head forward.** Measured against each clip's own rest
  skeleton, the head is pitched down 3.6° on one clip, 19.8° on another and **34.5°**
  on a third, while the torso leans only 1–5°. Head orientation is weakly
  constrained in a SMPL-X fit, and it lands looking at the floor.
- **The character's yaw decides which side faces the camera**, and a fixed
  three-quarter camera sees whatever the motion presents.

Lowering the camera does not fix this: from 0.34 down to 0.0 elevation, iris pixels
went 5810 → 5245 on a head-level clip. And below about 0.10 the camera reaches the
horizon, so the floor stops filling the backdrop and world grey shows behind the
character. `--camera-elevation` exists, but 0.34 stays the default.

What *does* work is aiming at the character instead of at the world:
`--views character_front` (or `character_3q`, offset 32 degrees) puts the camera on
the side the character faces, computed from ``cross(up, left_hip -> right_hip)``.
That heading was validated against travel on three walking clips — dot +1.00 and
+0.98 — and correctly disagrees at -0.98 on the one captioned *"jogging backward"*,
where the person moves backwards while still facing forward. For a comparison the
heading is averaged over both clips of the pair, so the two sides stay the same
view. `--views` accepts it anywhere a preset name goes, including through
`visualize_dataset.py --view`.

## Optional smoothing (`--smooth N`, off by default)

`--smooth 5` box-filters the **torso** rotations — pelvis, the three spine
segments, neck, head and both collars — plus the root path, over 5 frames.

Only the torso, because the torso carries the head and, through the collars, both
arms, so filtering it settles the whole figure. A wrist during a throw moves fast
on purpose and filtering it clips the throw: extending the filter to every joint
bought no extra jitter reduction (63% either way) while tripling how far the pose
moved (p99 3.54° against 1.21°, worst 9.67° against 4.74°).

`joints22` is left byte-identical. The MMD solve reads bone *orientation* from the
SMPL-X rotations and never reads a frame origin — the root path comes from
`transl` — so positions do not drive the character at all, and rebuilding them by
forward kinematics would move a wrist up to 100 mm for no visible effect.

Measured on the 39 `picks38/gt` clips:

| window | jitter | cut | pose moved p50 / p99 / worst |
| --- | --- | --- | --- |
| off | 0.445° | — | — |
| 3 | 0.216° | 52% | 0.00° / 0.66° / 3.67° |
| 5 | 0.163° | 63% | 0.00° / 1.21° / 4.74° |
| 7 | 0.140° | 69% | 0.00° / 1.87° / 8.40° |

5 frames at 30 fps removes content above about 7.5 Hz, which voluntary human
motion does not reach, so what the filter takes out is fit noise rather than
motion.

**What is not established.** The 63% is a reduction in a *source-data* metric —
the second difference of each joint's local rotation. Whether it is visible is a
separate question, and the one attempt to measure it in the rendered image was
inconclusive: tracking the head-region centroid across
`outputs/ab_smooth/smoothing_ab.mp4` gives a 3% drop at the median and 28% at p90,
on an absolute level of 0.28 px, and that measurement is contaminated by the
ponytail's spring motion, which pose smoothing does not touch. The renders do
differ (4.1% of pixels on average), but nothing here demonstrates that the filter
fixes a shake a reviewer complained about. Judge it on the A/B video, not on the
percentage.

Default off, because it edits the motion and these figures are evidence about a
model.

## The camera

**`--camera follow` (the default).** The camera is orthographic, so apparent size
comes from `ortho_scale` rather than distance: framing a box the size of the
character and then translating the camera with the root decouples the figure's size
from how far it walks.

The box is the **root-relative joint hull, unioned over every clip listed for
framing, padded by `--frame-pad`**. Three consequences worth stating precisely:

- **Both sides of a comparison are identical by construction**, because the union
  is over the same pair either way — verified at 1 mm on a 1.77 m box, giving
  character heights of 357–362 px against 359–361 px. Without the union the poses
  alone put gt and gen 1.7% apart at the median and **19.5% apart at worst**, which
  reads as two different shots.
- **Across clips the size still varies**, because the hull depends on the poses in
  the clip: over the 39 `picks38` clips the root-relative hull runs 1.56–1.96 m
  tall and 0.90–1.38 m wide, so apparent size spans about 1.5x. Far tighter than
  trajectory framing, but not uniform. If a figure needs one exact scale across a
  grid of stills, that belongs in `render_paper_figure.py`, not here.
- **The hull is taken from joints, not from the character mesh**, since the mesh is
  only available for the clip currently loaded and a per-side measurement is what
  caused the mismatch above. The 0.12 m pad covers the overhang: measured against
  the mesh box it runs 10–24 cm larger per axis, and the 1.15 camera margin absorbs
  the cases where a truncated clip inverts that.

A static camera cannot do that, because it has to cover the whole trajectory. On
`picks38/gen` 000170 — a 2.6 m circular walk — a static camera left the character
**36% of frame height**, and 000170 is mild; a 12 m clip at 800p renders a 100 px
figure in an 800 px frame. The same clip under a follow camera is **66%**. This is
not a subtlety, it is the difference between a usable showcase video and a speck.

Only the horizontal axes are followed. A camera that tracked Z would cancel the
jump it was meant to show. The path is smoothed over 11 frames
(`--follow-smooth`), because a root sways sideways with every step and a camera
copying it makes the whole shot wobble — the sway belongs on the character.

**The floor needs a pattern under a follow camera.** Tracking the subject cancels
the translation, so on a featureless floor a walk becomes a treadmill and the clip
reads as marching in place. `add_ground(grid_metres=0.5)` adds a world-fixed
checker at 2.0% rendered luminance contrast (242 against 247 of 255) — enough to
give the eye something stationary to measure against, far too little to compete
with a cel-shaded character. It is off for static shots, which already show travel
as the figure crossing the frame.

**`--camera static`** keeps the old behaviour, and `auto` switches on travel past
`--follow-threshold` (1.2 m). Prefer `static` only for a stills figure where the
trajectory itself is the subject.

### Ground contact

The character hovers, and the shadow gives it away. Probed per frame on a walk,
the lowest mesh vertex sits at **+3.8 to +7.7 cm** for the whole clip — never near
zero. Two causes stack:

- `_mmd_ground_offset` measures the **rest** mesh, so the offset grounds an A-pose.
  A posed character has bent legs, which lifts the soles several centimetres.
- `add_ground` used to place the floor at `bounds_min[2]`, the lowest the character
  ever reaches. That guarantees no frame intersects the floor and equally
  guarantees every other frame hovers: a 2.1 cm median gap, about ten pixels at
  800p, which with a soft shadow reads as floating.

Both the video renderer and `render_paper_figure.py` now place the floor at the
**lower quartile** of the per-frame lowest vertex. Planted frames make contact, the
lowest few intersect by a centimetre or two, and the contact shadow hides that.
Only the floor prop moves — no joint angle, root position or vertical delta is
touched, so this does not interact with the penetration metric, which is measured
from the source data.

The first cause is still there; the floor placement compensates for it rather than
removing it. Fixing it properly means re-deriving the offset from the animated
soles inside the retarget, which is a change to `_mmd_ground_offset` that every
render path depends on.

### Shared framing for static shots

`render_mmd_compare.py --frame-motion <other.npz>` frames the camera from the
joint hull of every listed clip, ignoring mesh bounds. Two clips that each list
the other therefore get identical cameras — verified symmetric to within 1 mm on
the worst pair.

This matters for `static`: across the 39 `picks38` pairs the horizontal extent
differs by a median of 1.25x, but the worst pair is **9.1x** — a generation that
ran 31 m against a 3.5 m ground truth. Framed independently, one side would be a
full-height figure and the other an ant, which reads as two different shots rather
than one motion rendered twice. Under `follow` the flag still decides `auto`'s
static-or-follow vote so both sides agree, but the framing box no longer depends on
the trajectory at all.

Panels of unequal length are padded by holding the last frame, never
time-stretched: a generation and its ground truth genuinely differ in duration,
and rescaling one would change its speed.
