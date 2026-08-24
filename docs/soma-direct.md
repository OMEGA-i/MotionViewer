# Reading SOMA-77 directly, instead of the SMPL-X fit

The render path was fed `smplx_body22_fitted_aa` — a fit of SMPL-X to the SOMA data.
Measured on `soma_retarget_probe`, that fit costs **26.1–28.3 mm RMSE** and throws away
**55 of the 77 joints**: both hands (52) and the toes (4). Neither is necessary. SOMA can
drive the character directly, and the existing MMD solver takes it unchanged.

```bash
# same renderer, same shading and camera, different source
blender --background --python scripts/render_mmd_compare.py -- \
    --asset assets/fbx/pmx/yoimiya/宵宫.pmx \
    --soma <clip>.soma.npz --output outputs/x --toon --camera follow
```

`--motion` (SMPL-X) still works exactly as before; the two are mutually exclusive and
nothing on the old path changed.

## What had to be worked out, and the numbers

Three properties of the format decide whether this is possible. All three were measured,
and one of them contradicts the probe data's own README.

| claim | measured |
| --- | --- |
| `rotmat77` is a **local** rotation, chained down the tree to get world | bone directions reproduce to **0.031°** chained, against **9.4–20.9°** if the stored matrix is used as a world rotation |
| the rest direction of every bone is recoverable from the clip | reproduces every observed direction to 0.031°, and the recovered directions agree **across all five clips to 0.0007°** — a property of the skeleton, not the clip |
| rigid forward kinematics closes | rebuilding positions from the root gives **≤ 0.0033 mm** against `joints77`; bone lengths are rigid to **0.0003 mm** per frame |

`scripts/_validate_soma_retarget.py` prints that table and exits non-zero if any of it
stops holding, including if the chained and stored readings ever start to agree — which
would mean the export convention changed under us.

### Where the README is wrong

The probe README says *"不要试图用刚性 FK 从旋转反推位置"*, reporting 14.4 mm of drift,
and attributes it to SOMA's forward pass being an RBF skeleton deformation rather than a
rigid bind. That drift comes from treating the **stored** rotations as world rotations.
Chained, rigid FK is exact to 3 µm. The 10.7–26.0 mm in the file's own `fk_vs_stored_mm`
is a different quantity: it measures the *official* RBF path, not this one.

The consequence is the useful part: because the skeleton is rigid and the rotations are
world-consistent once chained, `S(t, b) = R_g[b] @ S_rest(b)` — the invariant the whole
MMD solver is built on — holds **exactly**, so `build_mmd_plan` and `solve_mmd_retarget`
needed no changes at all. They were already written against source *names and frames*
rather than against SMPL-X.

### What the README is right about

Use `bone_lengths` (measured off `joints77`), not `rest_joints77_identity_fk`. The latter
is 6.6% out. This module does exactly that: **directions** are solved from the data,
**lengths** are taken from the file, and positions come from `joints77`.

## The hands

SOMA carries 52 finger joints and Yoimiya's rig has 40 finger bones. 30 of them are now
driven, three segments per finger per hand.

Two mapping decisions:

- **SOMA has one segment more than MMD.** `Index1→2→3→4→End` against
  `人指１→２→３→先`. The first three joints map; `Index4` is left undriven, which costs
  only the distal joint because transfers apply global rotations.
- **The thumb is offset by one**, since MMD numbers it from the metacarpal: `親指０/１/２`
  take `Thumb1/2/3`.

Fingers use the **`absolute`** transfer, for the same reason the arm chain does: an MMD
hand is bound in a relaxed curl while SOMA's rest hand is flat, so `relative` would add
the source's curl on top of the bind and over-close the fist.

`hand_relax` — the static resting curl that existed *only* because body-22 had no finger
data — is forced off whenever fingers are driven, or it would be posed on top of them.

Finger motion in the source is real, not incidental: joints move **1.4–5.8°/frame** at
the median, against 1.3–2.7°/frame for the body.

## Is it worse anywhere?

On the body, no: the transfer modes are inherited unchanged from the SMPL-X path, so the
solve is the same one, fed a source without a 26 mm fit in front of it. Rendered
side by side on the probe set the poses track closely; the visible differences are the
hands, and a ~1% framing difference because the follow box is a joint hull and SOMA's
hull includes fingertips.

Two things this does **not** fix:

- The SOMA rest pose is an exact T-pose, same as SMPL-X, so the A-pose bind mismatch on
  the arm chain is unchanged — `absolute` is still needed there.
- Toes stay undriven. `_SKIP_DRIVE` excludes them because `左足先EX` carries no mesh
  weight on this rig family and `左つま先` is an IK aim, so driving them moves nothing.
  SOMA *has* the data if a rig ever wants it.

## Files

| file | role |
| --- | --- |
| `src/motionviewer/core/soma_fk.py` | chaining, rest recovery, look-at frames |
| `src/motionviewer/blender/retarget/soma_mmd.py` | SOMA-77 → MMD names, fingers included |
| `src/motionviewer/blender/retarget/soma_pipeline.py` | entry point; assembles the same `BootstrapOutput` |
| `scripts/_validate_soma_retarget.py` | re-checks every number above |
| `scripts/_probe_soma_vs_smplx.py` | renders the probe set both ways, side by side |
