# Reading SOMA-77 directly, instead of the SMPL-X fit

The generated SOMA archives contain `joints77`, `rotmat77` and a full finger
skeleton. The corresponding `smplx_body22_fitted_aa` archives contain fitted body
rotations and no hand-pose fields in the 300 checked `active` files. Going through
that body-only representation loses the finger animation before retargeting starts.
Use `--soma` to display the native generated motion.

The two representations must not be confused: in the four current review clips,
SMPL-X `joints22` is exactly a subset of SOMA `joints77`; it is not the FK output of
the fitted SMPL-X rotations. The historical probe's 26.1–28.3 mm fitting error
describes its fitted model, not an error between those stored joint arrays.

```bash
# same renderer, same shading and camera, different source
blender --background --python scripts/render_mmd_compare.py -- \
    --asset assets/fbx/pmx/yoimiya/宵宫.pmx \
    --soma <clip>.soma.npz --output outputs/x --toon --camera follow \
    --faithful --root-motion source --expression none
```

`--motion` (SMPL-X) remains supported; the two are mutually exclusive. As of the
2026-10-22 fidelity correction, both paths default to no pose polish and root deltas
in source metres. `--polish --root-motion legacy-scaled` explicitly restores the old
body-polish and root-scale settings. This does not reproduce every historical
render: the finger mapping has since been corrected. Optional contact IK,
smoothing, arm corrections and finger overrides are listed separately in
`info.json.motion_fidelity`. An unchanged input file does not mean an unchanged
rendered pose.

## What had to be worked out, and the numbers

The historical probe established the following interpretation:

| claim | measured |
| --- | --- |
| `rotmat77` is a **local** rotation, chained down the tree to get world | bone directions reproduce to **0.031°** chained, against **9.4–20.9°** if the stored matrix is used as a world rotation |
| the rest direction of every bone is recoverable from the clip | reproduces every observed direction to 0.031°, and the recovered directions agree **across all five clips to 0.0007°** — a property of the skeleton, not the clip |
| rigid forward kinematics closes | rebuilding positions from the root gives **≤ 0.0033 mm** against `joints77`; bone lengths are rigid to **0.0003 mm** per frame |

`scripts/_validate_soma_retarget.py` checks those probe files. Its warning when
local and world interpretations agree is a diagnostic heuristic: a nearly static
clip can also make the two readings similar.

On 2026-10-22, independent full FK reconstruction of current `active` clips
`004162`, `100948`, `016783` and `128701` measured a maximum joint-position error
below **0.00083 mm**, using their own `bone_lengths` and chained rotations.
Their `transl` equals `joints77[:, 0]`. The paired SMPL-X translations have a
different fixed origin, but their frame-to-frame trajectories agree to within
0.00017 mm after rebasing. The old character-height root scaling was a separate
retarget modification, now opt-in.

### Where the README is wrong

The probe README says *"不要试图用刚性 FK 从旋转反推位置"*, reporting 14.4 mm of drift,
and attributes it to SOMA's forward pass being an RBF skeleton deformation rather than a
rigid bind. That drift comes from treating the **stored** rotations as world rotations.
Chained, rigid FK closes to numerical precision on the checked files. The 10.7–26.0 mm in the file's own `fk_vs_stored_mm`
is a different quantity: it measures the *official* RBF path, not this one.

The consequence is the useful part: because the skeleton is rigid and the rotations are
world-consistent once chained, `S(t, b) = R_g[b] @ S_rest(b)` — the invariant the whole
MMD solver is built on — holds to numerical precision, so `build_mmd_plan` and `solve_mmd_retarget`
needed no changes at all. They were already written against source *names and frames*
rather than against SMPL-X.

### What the README is right about

Use `bone_lengths` (measured off `joints77`), not `rest_joints77_identity_fk`. The latter
is 6.6% out. This module does exactly that: **directions** are solved from the data,
**lengths** are taken from the file, and positions come from `joints77`.

## The hands

SOMA has 48 finger-chain joints including end markers: each hand has four thumb
joints and five joints for each other digit. Yoimiya's rig has 40 finger bones;
30 receive animation, three segments per digit per hand. Missing target finger
bones are optional, so the actual driven count is reported rather than assumed.

Two mapping decisions:

- **SOMA includes a metacarpal.** `Index1→2` spans the palm; the three phalanges are
  `Index2→3→4→End`. MMD `人指１/２/３` therefore map from SOMA `Index2/3/4`, likewise
  for middle, ring and pinky. The former `1/2/3` mapping was anatomically off by one
  and dropped distal flexion; it was corrected on 2026-10-22. Global transfer retains
  the metacarpal rotation's contribution to the downstream finger directions.
  A target without that metacarpal cannot reproduce its positional deformation.
- **The thumb is offset by one**, since MMD numbers it from the metacarpal: `親指０/１/２`
  take `Thumb1/2/3`.

The wrist's rest longitudinal axis points to `HandMiddle2`, the middle-finger MCP
knuckle. The previous `HandMiddle1` landmark is a laterally offset metacarpal base;
it introduced an approximately 18.5-degree hand-axis error on all four review
clips. Absolute transfer calibrates roll only, so it could not cancel that wrong
aim. The corrected frame is still carried by the source wrist's own rotation;
finger motion does not redefine the wrist orientation. Cached scenes record this
choice in `motionviewer_soma_hand_aim` and must be regenerated when it changes.

This fixes a source-frame mistake, not every anatomical difference. Yoimiya's
display bone axis itself differs from its actual wrist-to-middle-knuckle vector
by about 6 degrees, and its hand proportions differ from SOMA. Matching the chosen
bone axes still does not establish identical palm planes or contact positions.

Fingers use the **`absolute`** transfer, for the same reason the arm chain does: an MMD
hand is bound in a relaxed curl while SOMA's rest hand is flat, so `relative` would add
the source's curl on top of the bind and over-close the fist.

`hand_relax` — the static resting curl that existed *only* because body-22 had no finger
data — is forced off whenever fingers are driven, or it would be posed on top of them.

The current review clips contain time-varying finger rotations, including distal
curl. Their motion amplitude varies by clip; the old probe's finger-speed values
are not a general description of the generated dataset. Weapon attachment keeps
native fingers by default. `--override-source-fingers` explicitly replaces them
with a grip pose; `--weapon-arm-clearance` separately opts into arm adaptation.

## Is it worse anywhere?

This removes the intermediate fit, but it does not establish identical contacts or
physical plausibility on a character with different proportions. Bone directions,
root travel and frame timing can be checked against the generated data; joint positions,
cloth behavior and added props still require a separate visual review.

Two things this does **not** fix:

- The SOMA rest pose is an exact T-pose, same as SMPL-X, so the A-pose bind mismatch on
  the arm chain is unchanged — `absolute` is still needed there.
- Toes stay undriven under the current mapping. The original reviewed rig used
  an unweighted toe extension and an IK aim; this is not a guarantee about every
  PMX character. Jaw, eye rotations and end markers are also unmapped. `Neck2`
  contributes through the mapped head's global rotation.

`load_soma` checks frame/joint dimensions, finite values, parent indices and fps.
When `pose_convention` is present it must be `relative_to_soma_joint_orient`;
older archives without that field remain accepted. These checks do not establish
that an arbitrary new export has the same semantics: verify FK before adopting it.

For a Blender scene, `scripts/_audit_soma_display.py` compares evaluated mapped
bone rotations with the unpolished source solve and checks root deltas across all
frames. It rejects a cached scene with the old finger mapping. This verifies the
implementation of the chosen mapping; it does not prove equal joint positions,
natural grasping, foot contact or collision-free clothing.

## Files

| file | role |
| --- | --- |
| `src/motionviewer/core/soma_fk.py` | chaining, rest recovery, look-at frames |
| `src/motionviewer/blender/retarget/soma_mmd.py` | SOMA-77 → MMD names, fingers included |
| `src/motionviewer/blender/retarget/soma_pipeline.py` | entry point; assembles the same `BootstrapOutput` |
| `scripts/_validate_soma_retarget.py` | re-checks the historical probe's FK interpretation |
| `scripts/_audit_soma_display.py` | checks actual Blender mapped rotations and root deltas |
| `scripts/_probe_soma_vs_smplx.py` | renders the probe set both ways, side by side |
