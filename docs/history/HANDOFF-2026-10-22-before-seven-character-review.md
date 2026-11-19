# MotionViewer handoff — verified 2026-10-22

This is the current entry point. `AGENTS.md` describes module boundaries, not project status.
The previous handoff and Chinese status are preserved verbatim in `docs/history/` as
historical evidence; their claims are not current requirements or established facts.

Current review entry: [`converted/fidelity_review_20261022/index.html`](converted/fidelity_review_20261022/index.html).
Read the current conclusions below before using any historical measurements later in this file.

Latest user direction: **stop further box work** ("箱子没救了算了"). Preserve the
existing box diagnostics, but do not resume box experiments without a new request.

## Priority correction after user review — generated-motion fidelity

The user challenged the previous cloth and box results and prioritized source fidelity:
visible cloth pops remained, and the box should be above the hands. Preserving an
input file is not the same as preserving the displayed action. The old side-grip box's wrist/arm IK and weapon
avoidance are adapted poses, not acceptable defaults for showing generated motion.

- `t2m_demo400/soma/active` contains native generated SOMA-77 with hand motion;
  `smplx/active` is a fitted body-22 copy with no finger channels. Prefer `--soma`.
- Default pose polish is now off. Root deltas default to source metres, with
  historical height-ratio scaling only under `--root-motion legacy-scaled`.
- Non-thumb finger mapping was wrong: SOMA segment 1 is the metacarpal. Segments
  2/3/4 now drive MMD 1/2/3; thumbs remain 1/2/3 → 0/1/2. Native fingers are not
  overwritten by a weapon grip unless explicitly requested.
- The wrist rest axis is now **Hand → Middle2 (middle-finger MCP)**. The previous
  Middle1 metacarpal-root landmark skewed it by about 18.5 degrees. The corrected
  rest axis is transported by the source Hand rotation; it does not aim at the
  animated finger each frame. Pre-correction videos are diagnostic history.
- The current Yoimiya audit covers **50 mapped bones, including 30 finger bones**.
  Direction/root-delta agreement does not preserve absolute joint or fingertip
  positions across different character proportions, and does not establish contact.
- `--scene carry_box_supported` writes prop channels only. It reports support gaps,
  palm orientation and post-physics mesh collisions separately. This is geometric
  prop placement, not proof of a physically load-bearing grasp. In 016783 the later
  generated pose raises and turns the palms near the head; do not conceal that.
- New work is isolated under `outputs/fidelity-2026-10-22/`. Files marked
  `DIAGNOSTIC_ONLY.json` and early candidate folders are not accepted deliverables.
- Cloth linear slack is an explicit per-character parameter, not a universal fix.
  The short-skirt selector must not open locks on a connected long garment's root;
  all dynamic endpoints must belong to the selected garment family.

Run `_audit_soma_display.py` to compare evaluated mapped rotations/root deltas, but
also verify anatomical landmark choices independently. Agreement with the same
solver cannot prove that a joint's anatomical correspondence was correct.

## Current conclusions — generated motion and separate visual adaptations

| Area | Current status |
| --- | --- |
| Yoimiya dance `100948` | Native SOMA, PMX damping, **20° short-skirt angular slack + ±0.002 m linear slack** passed independent review of all 152 frames. The conspicuous pops improved, but panels still feel stiff. This is the reviewed candidate for this clip, not a universal profile or final user approval. |
| Yoimiya walk `004162` | The formal **20° + ±2 mm, original long-panel damping** diagnostic was reviewed across all 152 frames. Long-panel recoil and slow return remain; **naturalness failed**, with skin/selected-skirt triangle crossings in 99 frames. Earlier foot/shoulder adaptations are not current faithful-display acceptance. |
| Yoimiya box `016783` | `carry_box_supported` animates only a fixed-size box above existing hands. After the corrected MCP wrist mapping, **57/152 frames** have a support gap above 10 mm; maximum **37.45 mm** at frame 124. At least one palm faces away from the box bottom in **109–139**. Fingertip contact and gaps remain, so this is a prop-placement demonstration, not a natural load-bearing carry. |
| Feixiao weapon | The earlier approximately **17° shoulder / 11.7 cm wrist displacement** and synthetic grip are an **adapted** experiment. They are not an acceptable default for displaying native generated motion, nor a current completion claim. |

The fresh box pair is `outputs/fidelity-2026-10-22/box/mcp_{source,supported}_filtered/`;
each contains `info.json`, both full views and `review.blend`. The paired videos and
`mcp_review_summary.json` sit in the parent box directory. They use matching cameras
and PMX / 10° / ±2 mm garment settings on both sides; the dance's 20° setting is separate.
The box's actual post-Bullet actor/box triangle check is zero across 152 rendered
frames with a 0.2 mm inset. This says nothing about continuous-time collisions or
load-bearing support. In a separate before/after-Bullet check, 58 mapped/root-related
bone matrices and root transforms remain exactly unchanged; this verifies physics
did not further alter the mapped pose, not the anatomical mapping itself.

All 152 side-view frames were visually checked across two reviewers. No whole-panel
instantaneous cloth flip was seen; this is not exhaustive proof against garment/body
intersections. `REVIEW_STATUS.json` and `DIAGNOSTIC_ONLY.json` identify superseded
Middle1 wrist, earlier finger mapping and unfiltered garment candidates.

## Objective and current path

Apply native generated SOMA motion to anime characters faithfully and naturally,
producing reviewable videos and research figures. SMPL-X support remains available
for fitted data and other toolkit workflows. Current entry points:

- `scripts/visualize_dataset.py`: SMPL-X dataset → videos + `index.html` + manifest.
- `scripts/render_mmd_compare.py`: SMPL-X or SOMA → character/source frames and `info.json`.
- `scripts/pick_clips.py`: manual selection. Preserve `outputs/picks/` and `data/picks_backup/`.
- `scripts/README.md`: script inventory; `docs/repo-layout.md`: storage and safety rules.

Local Blender is `.local/Blender.app/Contents/MacOS/Blender`; its executable reports
5.2.0 LTS. Background renders and mmd_tools registration work without Blender-MCP.
Keep `.mcp.json` for optional interactive work; no connection is needed for this pipeline.
The generic SMPL-X mesh backend's separate addon is not needed to retarget the supplied
joints to PMX characters. Do not confuse it with the mmd_tools PMX importer.

## Changes in this takeover

- Archived 97 old `outputs/` entries (294 files, 94,103,960 bytes) intact under
  `outputs/archive/2026-10-22/`. `archive-manifest.json` records old/new paths and SHA-256;
  every moved file was checked. No render, asset, motion, selection or measurement deleted.
- Kept `outputs/picks/` and `outputs/motion_scores.json` at their active paths. Original
  absolute paths inside old manifests remain historical provenance, not current paths.
- Restored `scripts/_extract_pmx_zip.py`: `onboard_character.py --archive` still uses it,
  so the previous move into `scripts/archive/` broke that entry point.
- Batch rendering now defaults to Bullet, with explicit `--physics bullet|spring|none`.
  Input content, character assets/textures, implementation, addon and render settings
  participate in reuse. Old videos are retained when a new version is rendered.
- Batch manifests retain actual renderer reports after intermediate frames are removed.
- Fixed SOMA `--bullet` failing to import physics data. Explicit Bullet failures are
  errors; successful reports require an actually baked cache.
- Kept fractional fps via `fps_base`; a cancelled/failed bake restores the scene range.

## Damping correction: the old explanation was wrong

`1 - damping` is **not** a PMX-to-Blender rigid-body parameter conversion. PMX players
and Blender pass rigid-body damping directly to Bullet. The old showcase inversion is
retained explicitly as `--bullet-damping legacy-inverted` (the compatibility default).
`--bullet-damping pmx` preserves authored model values. Do not describe the former as
more physically correct. This is an artistic override that needs per-character visual review.

Evidence: [Blender Bullet interface](https://raw.githubusercontent.com/blender/blender/main/intern/rigidbody/rb_bullet_api.cpp),
[Saba PMX implementation](https://raw.githubusercontent.com/benikabocha/saba/master/src/Saba/Model/MMD/MMDPhysics.cpp),
[Bullet damping implementation](https://raw.githubusercontent.com/bulletphysics/bullet3/master/src/BulletDynamics/Dynamics/btRigidBody.cpp).

## Historical verification checkpoints and limits

These checkpoints describe earlier code and adapted renders. They are retained as
evidence, not the current test count or visual acceptance of the fidelity work above.

- Baseline: 286 tests passed, 3 optional Blender tests skipped; Ruff/build passed.
- Cleanup checkpoint: 315 tests passed, 3 skipped; Ruff/build passed.
- Contact/jitter follow-up: 404 tests passed, 3 skipped; Ruff/build passed.
- Garment/prop follow-up: 482 tests passed, 3 skipped; Ruff/build passed; see the final
  validation record for the additional shared-camera guard check.
- Real Blender: Yoimiya + SMPL-X `active/100948`, 152 frames, both damping modes.
  Each baked 221 rigid bodies / 256 joints at 50 fps. Six matching frames inspected.
- Real Blender: SOMA `active/100948`, first 8 frames, also baked 221 bodies / 256 joints.
  This is a physics smoke test, not full SOMA motion acceptance.
- Retarget validation across 152 frames: worst Blender-vs-solver 0.053913 degrees;
  solver-vs-theory 0.00000342 degrees. Existing polish deliberately deviates from raw
  motion by up to 10.0317 degrees on this clip; numerical agreement is not visual quality.
- See `converted/review_20261022/` for fresh review output and the recorded checks.
  Historical six videos and five checking images remain in `converted/final/`.
- Real batch pipeline: 152-frame H.264 MP4, 518×720, 50 fps, 3.04 s. The batch
  manifest retained the successful Bullet bake report. Six decoded video frames inspected.
- Real reuse checks: a source-code change produced a new video/version and retained
  the old file; the following unchanged run skipped rendering. SHA-256 verified 553
  pre-existing asset/data/output files unchanged (the converted README was intentionally updated).

## Historical contact and jitter adaptation — 2026-10-22

This section preserves an opt-in adaptation experiment. It is not the raw native
SOMA display baseline and must not be used as current fidelity acceptance.

New review: `converted/contact_review_20261022/index.html`. Original NPZ motions,
models and older finals are preserved; the six rendered input hashes were rechecked.
`--foot-contact on` measures evaluated weighted shoe vertices, infers conservative
source support intervals, and applies bounded XY anchoring plus leg IK and small
vertical pelvis compensation before Bullet. Foot world rotations, bone lengths and
root XY are preserved. Heel/toe mode changes inherit the preceding anchor offset;
independently re-centering every mode caused a real slide spike and was rejected.
Extra knee bend is capped near 8 degrees, with a 75 deg/s change-rate guard. Blender
near-straight-leg numerical error can make the realized angle about 0.03 degrees larger.

Real all-frame checks (50 fps), measured only between fully weighted samples in the
same support mode; these are support-marker speeds, not a universal quality score:

| Character / motion | Mean support speed before → after | Contact sole abs error p95 |
| --- | --- | --- |
| Yoimiya / walking 004162 | 10.35 → 1.84 cm/s | 17.2 → 4.1 mm |
| Yoimiya / dance 100948 | 16.03 → 2.14 cm/s | 15.4 → 1.5 mm |
| Furina / walking 004162 | 10.65 → 2.05 cm/s | 22.7 → 8.3 mm |

Yoimiya walking still has at most 2.35 mm penetration. Furina retains 7.83 mm: do not
claim every character is perfectly grounded. Keeping the knee natural takes priority
over forcing a zero-height residual. Contact is opt-in and does not automatically
change all 400 clips. Inferred source joint floors can be ambiguous; provide a known
`--floor-z` where available. Zero contact weight means unclassified, not proof of flight.

Yoimiya / side jump 044005 had no sufficiently slow support interval, so no XY lock
was applied. Penetration relative to the same selected floor fell from 59.8 mm to zero.
The 44 frames with BOTH evaluated shoe soles > 40 mm above the selected floor
retain root height exactly; knee/ankle differences are < 0.003 mm. Source-marker
elevation is a different, heuristic check: all four markers > their individual p5 +
55 mm yields 35 candidate frames, including transition changes up to 1.314 mm root
and 13.002 mm knee. Do NOT claim that all airborne frames are unchanged. An earlier
17-frame figure used pooled left/right p5 baselines and was too narrowly described.
SOMA 100948 also passed a full 152-frame foot audit, with its actual archive joint names.

`--upper-smooth 3` creates a separate complete `processed-motion.npz`: local SO(3)
smoothing only on spines, neck, head and collars, capped at 2 degrees per LOCAL joint.
Root pose/translation and lower-body rotations are unchanged; joints22 are recomputed
consistently. On the dance sample, actual max local/world changes are 0.371/0.870 deg,
and mean treated-joint angular acceleration falls 891 → 404 deg/s². This is deliberately
subtle. `score_motion_quality.py` now measures signed rotation increments; historical
jitter thresholds are not directly comparable to the new metric version.

Video A/B pairs use `--match-camera baseline/info.json`, identical floor/fps/lighting,
and real Bullet bakes. Walk/jump compare foot off/on; dance compares upper smooth 0/3
with foot correction ON on both sides. Do not confuse the dance pair with raw-vs-all.
Encoded-video frames and critical foot-switch frames 48–50 were inspected.

## Historical garment follow-up — superseded after user review

Curated review: `converted/cloth_props_review_20261022/index.html`; final weapon
frames are `outputs/props-2026-10-22/weapon/{before,after}_review/`. Earlier narrow,
wide, and portrait-fit candidates are retained only as intermediate evidence.

The earlier Yoimiya garment candidate used authored PMX damping and
`--bullet-cloth-slack 10`. This opens only originally locked garment angular axes;
existing angular ranges and translation locks remain. It is a Blender adaptation,
not a unit conversion. Legacy defaults are retained, and other characters have not
been visually approved with these settings. The optional `--bullet-hand-colliders`
experiment was OFF in those candidate videos; adding colliders alone did not
reliably eliminate mesh intersections.

`scripts/_audit_mmd_cloth.py` runs the normal renderer and records torso-relative
garment trajectories, original-mesh skin/garment triangle intersections, and separate
nearest-skin diagnostics. It excludes outline copies and shared-vertex triangle
pairs. Neither the nearest-skin sign on an open surface nor a bone trajectory is a
proof of volume penetration or perceptual quality.

Matched 152-frame, 50 fps runs use the same source, foot correction and floor. On
646 selected original skirt vertices, adjacent-frame midpoint deviation p95 falls
12.64 → 2.88 mm for dance 100948 and 12.49 → 3.90 mm for walking 004162. The rendered
skirt was judged improved in that earlier review, but the user subsequently rejected
its remaining visible pops. Temporal movement amplitude remained close to the old
result; it did not freeze the garment. Source images, encoded videos and the
dance peak-deviation frames 118–124 were inspected.

This setting primarily fixes jitter. It does NOT establish less mesh penetration:
dance still has five late fingertip intersection frames, with more intersecting face
pairs, and walking still has wrist/forearm/thumb intersections. See
`outputs/cloth-2026-10-22/accepted-comparison.json` for the exact selection and data.
The earlier adapted walking review additionally used `--arm-clearance-degrees 4`: shoulders
rotate outward by at most 3.959 degrees, with wrist world orientation compensated.
Maximum wrist displacement is 27.03 mm; root and leg matrices are unchanged relative
to the already foot-corrected baseline. Midpoint deviation p95 is 3.51 mm, and
skin/garment intersection frames fall from 147 to 119 (mean intersecting face pairs
26.30 to 18.37). Right forearm crossings disappear, but wrist/thumb residuals remain.
The same adjustment worsened the dance contact check, so dance uses zero clearance.
This option is bounded to 0–4 degrees and is not an all-motion default.

## Historical side-grip box adaptation — rejected for current display

The earlier adapted Yoimiya 016783 review used `--scene carry_box_two_hand --scene-contact
--scene-box-size .3192 .22 .15 --scene-grip-mode side`, PMX damping and cloth slack10.
The fixed-size box has kinematic collision against the character's existing dynamic
physics layers; original body masks/exclusions are unchanged. Hand, wrist and arm
adaptation places hands near the rear corners, keeps elbow transitions continuous,
and shifts the box forward to clear the head. Original root and leg motion remains.
This is a substantial arm adaptation: maximum wrist translation 82.0 mm, forearm
axial twist up to 90 degrees, right wrist local rotation difference up to 110.2 degrees.
Some fingers remain lifted rather than forcing every digit into the box surface.

Both historical 152-frame adapted renders use the same physics, floor and camera. Actual evaluated
whole-actor mesh/box intersections after the Bullet bake drop from all 152 frames
(max sampled penetration 54.18 mm) to zero, including sleeve and bracelet, using a
0.2 mm inset. This checks discrete rendered frames, not continuous inter-frame motion.
`scene.report.hand_contact` explicitly labels pre-physics fitting; use the separate
`scene.post_physics_clearance` to assess the final bake. Source raising/face occlusion
and the clip's own foot motion remain, and need independent visual judgment.
Evidence: `outputs/props-2026-10-22/box/box_contact_bullet_final_validation.json`.


## Historical weapon adaptation — not native-motion acceptance

The geometry and transactional checks below remain useful implementation evidence.
The arm displacement and synthetic grip remain intentional adaptations, which the
user did not accept as the primary display of the generated action.

`--toon --weapon auto --weapon-fit-mode surface` uses a geometry/skin-weight/rest-rig
verified Feixiao right-hand profile. The original weapon PMX contains three separate
weapons; the working scene selects only the central 2.320 m axe, without changing
its dimensions or rewriting the asset. Four fingers wrap the actual 29 x 30 mm blue
shaft and the thumb opposes them; sampled grip penetration is about 0.772 mm.
The former bounded bone-proxy fit is retained for compatibility and uncalibrated
characters still need review. Unknown geometry is rejected by the surface mode.

The non-gripping arm is checked over every input frame. A profile-specific rotation
seed is accepted only if the actual left-arm triangles clear the weapon through
its full cosine transition. For 128701 this turns shoulder and compensating wrist
by 17 degrees over 0.8 seconds, preserving elbow bend and wrist world orientation;
maximum wrist travel is 116.6 mm, within the 120 mm/20 degree limits. It is an
intentional arm adaptation, not negligible denoising. Commit is transactional;
full-frame replay validates root, trunk, legs, gripping arm and left elbow, and
failure restores the original action. Final transaction verification reproduces
the already rendered left-arm matrices exactly (max difference 0.0).

The selected axe gets 70 small kinematic convex collision cells made by clipping
actual triangles into 12 cm Y/Z cells, avoiding a single hull across its large voids.
The cells inherit the dynamic body's existing layer union; original masks are not
changed. Their 1 mm margin and convexification are approximate, so evaluated mesh
inspection remains necessary. A separate real Bullet parent-motion check gives
maximum matrix error 1.19e-7.

On all 180 rendered frames, final non-gripping body/weapon boundary intersections
remain in 10 frames (max 28 triangle pairs); the intermediate selected/surface-grip
version before arm/prop-collision work had 42 frames (max 120). These numbers are
NOT the full-pack-before video's comparison. The former left-glove frames 65–67
and ponytail frame 113 clear. Residuals are right-ear hair in frames 51–59 (frame55
sampled depth 6.94 mm) and a hem tip in frame71 (sampled depth 2.16 mm). Sampled depth
is not an upper bound. Boundary tests exclude the gripping hand and cannot classify
all containment/coplanar cases; grip has a separate surface audit. Close views were
inspected and these residuals are documented rather than hidden.

The historical weapon review uses common square frames and a constant camera-plane fit over
all evaluated body/weapon vertices. This avoids both the old clipped blade and an
overly distant world-AABB shot. Matched camera tracks remain exact; shared
`--frame-motion` framing bypasses this single-motion fit. Both clips' full geometry
is checked against the final image bounds. That camera adjustment did not
additionally change the adapted motion or physics.

## Remaining quality work

1. Resolve the native walking long-panel attachment/collision model; changing only
   damping failed visual review. The formal `walk-native-short20-diagnostic` is not
   an accepted natural result. The dance candidate still has stiff-panel behavior.
   `cloth/tail-structure.json` proves the long panel is a sewn extension of the short
   skirt, with 24 mixed-weight seam vertices. Reparenting its dynamic root directly
   to the waist is not a justified repair and may pull the seam apart. Use a
   continuous garment proxy/cloth model rather than another damping sweep.
2. Keep the original `016783` limitations visible. Only prop placement may change
   in the faithful review; do not restore arm/wrist IK to manufacture contact. Any
   more suitable generated carry sample must have its own visible clip ID.
3. Review native weapon/finger behavior separately from the old adapted surface
   grip and 17-degree arm avoidance. Do not silently overwrite source finger motion.
4. Preserve separate adapted workflows for optional foot/contact work, including
   Furina residuals. `skirt_physics.json:lower_share` selects the wrong region; do not
   use it. Numerical garment metrics only support matched visual comparisons.

## Operating practice

Keep original assets, motions, manual selections and historical final output. New runs
use separate directories. At most three Blender processes; this takeover used no more
than three concurrently for full renders. Inspect frames before claiming visual success.
Do not push/publish anything unless requested. No commit or remote push was made here.
