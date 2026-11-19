# MotionViewer handoff — 2026-10-23 (with 2026-11-17 G1 Teaser addendum)

Current anime-character review: [`converted/character_review_20261023/index.html`](converted/character_review_20261023/index.html).
Current Unitree G1 simulation & paper teaser handoff (2026-11-17): [`docs/g1-teaser-handoff.md`](docs/g1-teaser-handoff.md) (local gallery: `outputs/g1_generated_gallery/index.html`).
The preceding handoff and Chinese status are preserved byte-for-byte in `docs/history/`;
their previous “current” results are historical evidence. `AGENTS.md` defines module boundaries.

## 2026-11-03 addendum 5: original-human body panel

`--panels body` renders the source as a real SMPL-X mesh (`core/smplx_body.py`, NumPy forward
pass: shape blendshapes, pose correctives, LBS; body-22 driven, hands/jaw/eyes at rest), placed with
the skeleton panel's grounding/re-basing and grounded on the lowest vertex. Needs
`--smplx-model <SMPLX_NEUTRAL_2020.npz>` (user's licensed file, in ~/Downloads, never commit); with
`--soma` the matching `smplx/<split>/<clip>.smplx.npz` is found automatically, else `--source-smplx`.
No SMPL-X Blender addon needed. Validation: per-axis RMSE vs the clip's `joints22` equals
`sqrt(fit_mse)` exactly on three clips (27.7/27.8/28.0 mm) — joints22 is the SOMA target, so that
gap is the fit's own loss, not a bug. Bones rigid to 0.0 mm. Review page with this column:
`converted/review_3chars_20261023/index.html`.

## 2026-10-23 addendum 4: hands inside skirt/thigh — proportion problem, fixed with arm abduction

User flagged Nilou's hand buried in the skirt and Furina's pressed into the thigh. Retargeted
arm angles match the source (abduction 4.5/8.7° vs 4.3/7.6°) but hip-joint half-width is 75 mm on
both characters vs 115 mm on the source body, so faithful angles put the hands inside the costume.
The old polish pass's `arm_abduction_degrees` (12) handled this, but every current render uses
`--faithful`. `--arm-clearance-degrees` (cap raised 4 -> 20) applies only the upper-arm opening,
preserving hand world orientation; it fades as the arm is raised. Chosen: **14°** (8° leaves Nilou's
hand on the skirt surface; 18° looks the same as 14° from the side). Max wrist displacement 9.3 cm.
Dance hand/skirt triangle crossings 20 -> 0 frames; walk count rises 12 -> 137 but every remaining
contact is a fingertip on 1-3 vertices of the flared hem's outer edge (verified per frame), not a
buried hand — the count cannot tell these apart, so check side views. My earlier "outward shift
does not work" came from translating the whole hand uniformly, not rotating the shoulder; corrected
in the docs. Videos: `converted/hand_fix_20261023/`. Not yet folded into a profile: profiles forbid
pose keys by design, so pass the flag on the command line.

## 2026-10-23 addendum 3: Furina rear-coat flicker — strips splitting, fixed in a new profile

Neutral settle fixed Nilou; Furina's walk still flickered. Full-size frames show the rear coat
splitting into separate strips and re-joining. `HB1` is a 9-column x 8-row grid named
`HB1_<row>_<col>` (parent of `HB1_3_4` is `HB1_2_4`). Down-column joints are near-rigid (0 mm,
1.4-5.7°); `HB1横` side links allow 35-85 mm and 30°, so columns swing independently.
New `side_link_rules` (profile key; `assets/physics_joint_rules.SideLinkRule`, only narrows,
matches joint names after mmd_tools' `NNN_J.`) at `linear_scale 0.3` cut neighbour-gap rate p95
8.5 -> 3.9 mm/frame. New `bullet.smooth` (`--bullet-smooth`, `core/cloth_despike.smooth_tracks`,
Gaussian in the pelvis frame, sign-aligned quaternion average) at 1.5 frames cut wobble 9.7 -> 7.6
mm and >20 mm single-frame jumps 449 -> 183, swing kept 96%. Both are in
`configs/characters/furina_garment_walk_v2.json`; the reviewed `furina_garment.json` is unchanged.
Dance regression-checked. Ruled out on Furina: damping (zero: 2.0 -> 2.4 mm), leg contact, joint
angle limits. Analysis pitfall hit three times: `arr[:, idx, :3, 3]` with a list `idx` moves that
axis first (shape (N,T,3)); index in two steps.

## 2026-10-23 addendum 2: walk skirt propped open — cause found, fixed (opt-in)

The settle (30 pre-roll frames) holds frame 1's pose. Walk 004162 opens mid-stride (thighs
44.5° apart vs dance 15.6°), so the skirt settles over a forward thigh and the front panels
rest on the leg capsules, propped horizontal until the legs close near frame 90 — the
"fold at 92-94" previously blamed on the solver. `BulletSettings.neutral_settle`
(`--bullet-neutral-settle`, profile `"neutral_settle": true`) holds the driven FK leg bones
at rest for the first half of the settle and smoothsteps them into frame 1's pose in the
second; keys only on pre-roll frames (rendered pelvis/fingertips differ 0.0 mm). Nilou walk:
row 1-2 panel angle while walking 65.5° -> 41.9° (standing 36.7°), garment accel p99/peak
95/415 -> 57/164. Nilou dance unchanged (42.3°). Furina walk barely changes (not propped).
Cost: early-walk hand/skirt crossings 11 frames/1565 pairs -> 12/3224, because the skirt now
hangs where the hand swings. Ruled out: damping (both zero: 65.5° -> 65.2°), substeps, and
negative settle frames (re-baking on positive frames is identical to 0.0 mm; only reading the
negative cache back from a saved .blend is broken). Videos: `converted/walk_skirt_fix_20261023/`.

## 2026-10-23 addendum: cloth snaps fixed (opt-in), hand-through-skirt measured, not fixed

`--bullet-despike` / profile `"bullet": {"despike": true}` runs `blender/mmd_cloth_cleanup.py`
after every bake: each dynamic body's track is taken into the pelvis frame and only frames
whose acceleration is >6x that body's own median (and >20 m/s^2) are refit with a Hermite
bridge; all other frames are bit-for-bit unchanged. It writes keyframes on the bodies and
switches the Bullet world off, so it must stay last. Re-rendered with the exact shipped
commands into `outputs/despike-20261023/` (each has `compare.mp4`, shipped left / despiked right):
Nilou walk peak 415->121 m/s^2, mass-snap frames 14->0, 12.3% of body-frames changed;
Nilou dance 174->137, 1.0%; Furina walk 297->204, 4.4%; Furina dance 240->56, 4.2%.
Swing range kept 91-99% per garment group (85% on the single-body 左裙饰). Default is off;
profiles were not changed.

Rejected inside the solver, all measured on Nilou walk (snap frames / peak m/s^2):
damping 0 (10/464), substeps 30 (15/1306), hand colliders (27/571). Damping is not drag:
skirt lift tracks walking speed at r=0.86 but removing damping changes lift by <1 degree.

Hand-through-skirt, independent triangle-crossing test on Nilou walk (frames / pairs):
shipped 11/1565, despike 10/1383, +vertex pushout 8/1278, hand colliders 8/1791,
colliders+despike 27/4512. Crossing garment vertices sit 13-17 mm *outside* the hand normal
and 18-89 mm away: skirt triangles are wider than the hand, so edges slice between vertices.
Needs finer garment geometry (a mesh-cloth proxy like Yoimiya's), not physics. The vertex
pushout was deleted rather than shipped. Probes: `outputs/review-20261023-mine/`.

Correction to the 2026-10-22 history docs: the MMD/Blender damping "inversion" described there
is not a unit fix — Bullet takes PMX damping directly. The profiles' `damping_mode: pmx` is right.

## User direction and immutable inputs

Display the original generated motion naturally on anime characters. Body pose polish,
foot IK, motion smoothing, arm clearance and source-finger overrides remain off by default.
**Box work was canceled by the user; do not resume it without a new request.**

- Prefer native SOMA77 at `/Users/a26044/Motion data/t2m_demo400/soma/active/`.
  The fitted `smplx/active` body22 copy has no original finger channels.
- Keep source root translation in metres. Non-thumb SOMA 2/3/4 map to MMD1/2/3;
  thumbs 1/2/3 map to 0/1/2. Wrist rest axis is Hand → Middle2 (middle-finger MCP).
- Source audits check mapped orientations and root deltas, not absolute joint/contact
  equivalence across different body proportions. Yoimiya has 50 mapped bones;
  two-segment torso rigs have 49. Both include 30 source finger bones.
- Missing optional Spine2 is now supported; the downstream Chest keeps source world
  orientation contributions. Noncommuting-rotation tests cover this, not just identity poses.
- Source NPZ, PMX, textures and previous media are preserved. Do not reset unrelated
  working-tree changes or commit local assets/renders.

## This review and its limits

Seven local model versions, each with two complete clips: walk004162 plus dance100948
for Yoimiya/Furina/Nilou; walk004162 plus sword-motion128701 for the other four.
These are 152 or 180 frames at50fps. Same-camera before/after pairs are in the review page.

| Character | Current treatment | Residuals / scope |
| --- | --- | --- |
| Yoimiya | Single-surface mesh Cloth on skirt/long tail; remaining hair/accessories retain Bullet | Walk's persistent flaring/recoil improves. Local wrist/skirt and thigh intersections remain; do not claim zero crossings. |
| Furina | Only HB1 internal rear-garment joints get15° half-range and±1mm locked linear allowance | Walk whole-layer snap reduced; layered stiffness remains. Dance regression-checked, not claimed universally better. |
| Nilou | Only70 D/QD internal ribbon joints get±3° on locked angular axes | Dance continuity acceptable. Walk remains stiff before a rapid fold at92–94; improved ending is not a naturalness pass.10° was rejected because it moved/accelerated the fold. |
| Yanqing | Native PMX settings, two-segment torso compatibility, shoe-based ground and fitted camera | Source-support sliding and height differences remain; sword-source hands are mostly open. |
| Liuying Spring Letter | Same workflow; native PMX preserved | Foot support and grasp are not guaranteed. |
| Qingyi | Native PMX preserved; ground/camera corrected | Hair flares backward around15–85 in walk. Root-velocity preroll trials did not establish improvement and were rejected. |
| Coatless Feixiao | Native PMX preserved; ground/camera corrected | Sword-motion91–105 has garment wrapping around raised lower leg. |

Yoimiya's mesh proxy uses an explicit outer surface for the authored double-sided tail,
weighted seam pins, skin/pants collision surfaces and excluded pinned collision faces.
Do not collide it against the overlapping outer clothing: that created persistent jitter.
The physical result is baked into small proxy shape keys before rendering/saving so cold
scene reopening and geometry audits use the same result. The visible garment follows via
Surface Deform; source body bones and original asset files are untouched.

The mesh crossing diagnostics use744 garment vertices/515 faces for both Yoimiya versions.
Walk crossings occur in142/152 after frames versus152 before; peak pair count95 versus61.
Dance has31 after crossing frames versus0 before. Dance full-frame review is conditional: smoother drape, but tighter inner-skirt/thigh
contact around50 and100–104. Keep baseline available as the more conservative contact version;
do not call mesh dance an unconditional replacement. These are surface intersection
diagnostics, not depth/severity or a single quality score. Inspect actual videos and review notes.

## Reproducible entry points

- `scripts/render_mmd_compare.py --soma ... --bullet --physics-profile ...` is the
  native single-clip entry. Profiles in `configs/characters/` validate assetSHA256 and0.08 scale.
- `scripts/_review_native_character.py` adds full-clip geometry framing, a saved scene,
  all-frame source audit, shoe measurements and missing-texture audit.
- `scripts/visualize_dataset.py` forwards profiles and hashes their contents into cache
  provenance, but remains a SMPL-X body22 batch input path; do not call it native SOMA batch.
- `src/motionviewer/assets/physics_profiles.py` handles strict pure config parsing;
  `mmd_joint_rules.py` and `mmd_mesh_cloth.py` apply only garment simulation settings.
- The default display floor now uses full-clip actual shoe minimum, not low hair/cloth
  mesh bounds. Explicit `--floor-z` and foot-contact floor choices remain authoritative.
  Missing shoe skin gives an explicit reported fallback. This never corrects leg motion.
- Profile verification metadata changes its fingerprint. Existing render `info.json`
  retains the exact profile snapshot actually used; later review metadata is separate.

Detailed physics settings: [docs/character-physics-profiles.md](docs/character-physics-profiles.md).
Use separate output directories and matched cameras for comparisons.

## Evidence and environment

`outputs/characters-2026-10-23/` holds full frames, saved scenes, source audits, mesh/sole
measurements, trial folders and final manifests. Start with:

- `yoimiya/review_manifest.json`
- `furina-nilou-review-manifest.json`
- `native_four_review_manifest.json`
- `independent-visual-review.json`
- `protected_files_check.json` and `display_floor_smoke.json`

Rejected trials remain evidence, not defaults. Native four profiles are retrospective
settings equivalents of their renders; `physics_profile_equivalence.json` says this explicitly.
Final check results and protection records are also copied into the review deliverable.

Local Blender5.2LTS and mmd_tools work through background scripts; Blender-MCP is not required.
Use `.venv/bin/python` or `uv run` for Python3.11+, and at most3 concurrent Blender processes.
Before delivery run `uv run ruff check .`, `uv run pytest`, `uv build`.
The working tree includes pre-existing takeover changes and is intentionally uncommitted.

## Next work that is justified by evidence

Preserve the source-body default. If continuing cloth work, prioritize Nilou's release/fold,
Qingyi's initial hair lag and localized wrist/thigh contacts. Evaluate complete clips before
promoting a configuration; increasing all joint limits or damping is not a demonstrated fix.
The source128701 has mostly extended fingers. Adding a prop cannot create a natural grasp
without either a suitable source pose or a separately labeled adapted version.
