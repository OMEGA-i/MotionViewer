# Converted characters and their retarget output

Finished, usable artifacts live here: a `.blend` you can open and play, the MP4s
and review page that show what it does, and the evidence that the retarget is
correct. `outputs/` is scratch space that any run may overwrite; this directory
is where a result is kept once it is worth keeping.

## Nothing here is committed

Every file under `converted/` is ignored except this README, for the same reason
`assets/` is: a converted `.blend` **contains the character**. Textures are
packed into it and the mesh and weights come straight from the source model, so
publishing it redistributes that model. Several bundled PMX files explicitly
prohibit redistribution. Treat a `.blend` here exactly as you would treat the
`.pmx` it came from, and do not weaken the ignore rules.

Motion data is a separate question from the character. `data/examples/` holds
only clips the project may redistribute; anything from a licensed or internal
motion set stays out of git too.

## Layout

```
converted/<character>/
  <character>_<clip>.blend      Retargeted action, textures packed, timeline set
  review/index.html             Open in a browser: MP4s + the decision stills
  review/videos/*.mp4           Source skeleton beside the character, one camera
  retarget_validation.json      Per-bone, per-frame agreement numbers
  rig_dump.json                 The rig as imported: hierarchy, rest, weights
```

## Regenerating

```bash
BLENDER=.local/Blender.app/Contents/MacOS/Blender
ASSET=assets/fbx/pmx/yoimiya/宵宫.pmx
MOTION=data/examples/smplx_body22_fitted_aa/omegamotiongpt.smplx.npz

# 1. The deliverable .blend (add --no-render to skip the still)
$BLENDER --background --python scripts/render_yoimiya.py -- \
  --motion $MOTION --asset $ASSET --output outputs/still --still \
  --save-blend converted/yoimiya/yoimiya_walk.blend

# 2. Correctness numbers. Repeat --motion to check several clips at once.
$BLENDER --background --python scripts/_validate_mmd_retarget.py -- \
  --asset $ASSET --motion $MOTION --frames 0 \
  --output converted/yoimiya/retarget_validation.json

# 3. Source-vs-character comparison, then the review page
$BLENDER --background --python scripts/render_mmd_compare.py -- \
  --asset $ASSET --motion $MOTION --output outputs/compare/walk --views three_quarter,front
uv run python scripts/build_review_page.py \
  --input outputs/compare --output converted/yoimiya/review
```

`scripts/_batch_review.py` drives step 3 over a whole directory of clips.

The retarget design, the per-bone transfer modes and what the polish pass trades
away are documented in [docs/retarget.md](../docs/retarget.md).
