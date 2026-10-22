# Handoff: the MMD showcase branch

For an agent picking this up cold. Branch `mmd-smplx-retarget`, head `9244cdb`.
Read `AGENTS.md` first for the module contract, then this file, then
`docs/retarget.md` for the solver derivations.

## What this branch is for

The user is writing a text-to-motion paper. The deliverables are **paper figure
stills** and **web-page videos** showing their T2M model's output driving
Genshin / Honkai: Star Rail / ZZZ characters. Not a product — a figure factory.

Two things follow from that, and both have been stated by the user directly:

- **They never open Blender.** Deliver MP4 plus a double-clickable `index.html`,
  or PNG stills written **inside the repo** (`converted/final/检查图/`). A path
  under `/tmp` is not a deliverable; they said so.
- **Looks beat exactness.** A retarget matching the source to 0.000° can still
  look wrong when the bodies are different shapes. Trade fidelity deliberately,
  name the knob, and report the size of the departure. See the polish pass in
  `docs/retarget.md`.

## The path a clip travels

| stage | file |
| --- | --- |
| pick clips by hand, in 3D | `scripts/pick_clips.py` (Viser) |
| render a whole dataset + page | `scripts/visualize_dataset.py` |
| render one clip, every knob | `scripts/render_mmd_compare.py` |
| solve SMPL-X/SOMA onto MMD bones | `src/motionviewer/blender/retarget/` |
| props, sets, weapons, grips | `src/motionviewer/blender/mmd_props.py` |
| the PMX's own Bullet physics | `src/motionviewer/blender/mmd_bullet.py` |
| cel shading, outline, ground | `src/motionviewer/blender/mmd_toon.py` |
| grip close-ups, two angles | `scripts/_render_grip.py` |

`docs/repo-layout.md` is the fuller map, including what was deleted and how to
rebuild it.

## Data and assets, none of it committed

| what | where |
| --- | --- |
| the 400-clip demo set | `/Users/a26044/Motion data/t2m_demo400` — `smplx/` and `soma/`, each with `active` (300) / `quiet` (100) / `*_gt` / `*_hold`; `meta.json` has per-clip caption, bucket, fit RMSE, padding |
| 20 character packs | `assets/fbx/pmx/<name>/` — git-ignored, own redistribution terms |
| Chinese clip summaries | `data/caption_zh.json`, 400 entries, hand-written |
| the user's shortlist | `data/picks_backup/backup/` — 45 kept active clips + tags, 8 tmp clips |
| every probe number | `data/measurements/` — read its README, including the caveat |

`*_hold` matters: those clips were padded to 3 s and the tail is model
extrapolation. The picker badges them (`补长`) so the user is not shown an
artefact and told it is motion.

## Working, and verified by a render

- **The retarget.** SMPL-X body-22 and SOMA-77 both drive the MMD rigs with no
  per-character configuration, because the mapping is against standard Japanese
  bone names. `scripts/_validate_mmd_retarget.py` is the thing that catches a
  wrong-but-plausible solve; run it after touching the solver.
- **Root translation scaled by leg length, not height.** Yoimiya is 0.4592
  leg/height against the source's 0.5322, so the height rule overshot 15.9% and
  slid the planted foot 15.9 cm per metre travelled. `_mmd_plan.py` measures the
  `hip.L→thigh.L→foot.L` chain and falls back to heights.
- **Cloth, via the PMX's own rigid bodies** (`--bullet`). Leg-through-skirt
  vertices go from 3.1% / 2.9% off to **exactly 0%** on. Three unit errors had
  to be fixed first, all recorded in `f940a05`: damping is inverted between the
  formats, gravity is MMD's `-9.81 * 10 * scale` = −7.848 at 0.08 import scale,
  and Bullet's timestep comes from `scene.render.fps`, which sat at 24 against
  50 fps motion.
- **Weapon grips, three of four.** Qingyi, Yanqing and Liuying hold the hilt
  with natural fingers, checked in two orthogonal camera angles.
- **Character acceptance test.** `_check_hand_clipping.py`, measuring the
  **rise** from bind pose to standing. 3 of 19 flagged.
- **Encoding repair.** Both mojibake directions: GBK read as CP932 (`颜`→`ﾑﾕ`)
  and Shift-JIS read as GBK (`体`→`懱`). 8 characters were rendering magenta.
- **Head clearance** (`--head-clear`, off by default). Rotates the shoulder only;
  every other joint moves 0.000000 mm.

## Open, in the order it will be noticed

1. **`visualize_dataset.py` does not pass `--bullet`.** It still appends
   `--spring`, so the batch path renders with the spring approximation and none
   of the physics fixes. Every video in `converted/final/` was rendered by
   calling `render_mmd_compare.py` directly. Wiring it through is the single
   highest-value remaining change.
2. **Feixiao's grip.** 37 mm shaft against a fist that closes to a ~10 mm ring at
   the anatomical flexion cap; index and little fingers stand 21–24 mm off. Needs
   a wider-spread hand pose, or accept the weapon is too thick. No video shipped.
3. **The thumb is excluded from every grip.** It wraps the opposite way, and the
   same curl that closes the fingers drives it through the grip.
4. **`skirt_physics.json`'s `lower_share` column is wrong.** It selects by
   model-local Z, and `センター` sits at mid-thigh on these rigs, so the band
   landed on shins and feet. Re-measure using the models' own `裙_*` bones —
   Yoimiya has 96, Liuying 112. `mesh_share` is fine.
5. **Ground contact is derived from the rest pose.** `_mmd_ground_offset` puts
   the lowest A-pose vertex at Z=0, so a posed character floats 4–8 cm. Render
   paths hide it by lowering the floor prop. Anything assuming the floor is Z=0 —
   a physics prop, a reflection plane, a contact test — is wrong by that much.
6. **Yanqing is flagged by the clipping test** (+23.0% rise) yet ships a video.
   Its sword guard does clip the index finger; the user looked and accepted it.
   Do not "fix" it without asking.
7. **Face shadow is a flattened ramp**, not the game's SDF map.
8. **The private mirror is stale.** `https://github.com/poisonousID/blender` was
   pushed before the grip and physics work; it is not configured as a remote.
   `origin` (`OMEGA-i/MotionViewer`) is 16 commits behind local on this branch.

## Approaches already tried and rejected — do not re-run these

Each produced **convincing wrong output rather than an error**, which is why they
cost a day each. The recurring failure is a metric that measures the wrong
quantity, four separate times.

| approach | why it failed |
| --- | --- |
| gravity inside the spring solver | the model's rest pose *is* the hanging pose, so gravity is already in it; adding it drags every bone onto the clamp |
| `Model.build()` without ordering it after the retarget | it disconnects the physics bones, which then float at the armature origin |
| a `foot_lock` module | built on the wrong premise. The foot slide was the height-vs-leg translation scale, not slip. Written, then deleted at the user's request |
| cloth-follows-body correlation | blind to the defect. Stretched and correct panels both lag 1 frame at r=0.80 — the shape was wrong, not the timing |
| "widest mesh below the root bone" as skirt flare | measured the character's stride. Produced the impossible claim that *more* damping flared the skirt 32% wider |
| distance to the weapon's shaft axis | an axis is an infinite line, so 13.6 mm from it is satisfied on the palm *and* on the back of the hand. Needs the signed palm-normal component |
| absolute buried-finger share | flagged Yoimiya at 26.8%, then 30.2% — a character that has never clipped. Cuffs and gloves enclose fingers by design |
| ray-casting hand verts at "other mesh objects" | every model is a *single* mesh with 20-odd materials. Returned 0.0% for all 19 |
| "dominant vertex group" to identify a part | `mmd_edge_scale` carries weight 1.0 on **every** vertex. Defeated two separate probes |
| scoring repaired texture names by character plausibility | failed twice. Match against each PMX's own texture list instead |
| fixed Euler angles for weapon alignment | MMD rigs disagree on bone roll; −95° left a racket vertical while the forearm pointed down |
| geometric rules for which end of a weapon is the grip | three different rules each got 3 of 4 right. `WEAPON_BY_CHARACTER` is an explicit table for that reason |
| full-arm FK rebuild for head clearance | least-squares rest offsets left 12–57 mm residual. Rotating the arm rigidly leaves it exact |
| the carry crate (clip 016783) | geometric dead end: hands at 1.28 m put the crate top at 1.59 m against a face at 1.55 m. The user closed it — "木箱不管了" |
| searching finger flexion to 140° | a bound with nothing behind it. The (1.0, 1.25, 0.85) phalanx ratio turns 136° at the knuckle into 170° at the middle joint, folding the finger through itself. **This was the hand deformation.** Cap is 88° |

## Operating rules learned the hard way

- **Never run more than 3 Blender processes.** Six froze the user's machine.
- **Monitor the render and report when it finishes.** Do not go quiet and wait
  for the user to ask; they raised this three times.
- **Look at the output before claiming it works.** Verbatim: "你说你修好了的时候
  能不能自己看看图片或者最终效果". A log line saying 12 bones curled is not
  evidence — one of them was written as `rotation_euler` under quaternion mode
  and did nothing.
- **Grips need two orthogonal views.** From one angle, beside the hand and
  through the hand look identical.
- **Clear `__pycache__`** when a fix appears to have no effect.
- Edit `mmd_props.py` with the Edit tool, not string-replacement shell commands.
  Those corrupted it twice — once placing a function inside a string literal.

## Verification before publishing

```bash
uv run ruff check .          # clean at 9244cdb
uv run pytest                # 286 passed, 3 skipped
uv build
```

Blender integration tests are opt-in: `MOTIONVIEWER_BLENDER_TESTS=1`, and need
the local licensed assets.
