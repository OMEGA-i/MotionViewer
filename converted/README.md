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

## What is here now

Generated 2026-08-15 from the `soma_tmr_test_400m_all_fullpose` test set, `gt`
source, at 800 px portrait with cel shading, outlines, floor shadow and baked
spring bones:

| | clips | videos | size | failures |
| --- | --- | --- | --- | --- |
| `showcase/` (Yoimiya) | 301 of 631 scored | 301 | 159 MB | 0 |
| `showcase_cast/` (Furina, Silver Wolf) | top 60 each | 120 | 69 MB | 0 |
| `figures/yoimiya/` | top 20 | 20 filmstrips, 2 trails | — | 0 |

Verified three ways: every clip rendered, contact sheets eyeballed
(`*/qc/qc_*.png`), and a popping scan over 70 clips found no frame changing more
than 2.3x its clip's median — well inside choreography.

## Layout

```
converted/showcase/              The full run. Start here.
  index.html                     Every clip as a video card, ranked best first
  videos/<rec_id>_<character>.mp4
  manifest.json                  Per-clip scores and captions
  qc/qc_*.png                    One frame per clip, tiled, for a fast scan

converted/figures/<character>/<rec_id>/
  filmstrip.png                  Publication figure: N frames, transparent
  trail.png                      Motion overlay, when the root actually travels
  figures.json                   Which clips got which figure

converted/showcase_cast/         The same top clips on the other characters

converted/<character>/           One-off deliverables for a single clip
  <character>_<clip>.blend       Retargeted action, textures packed, timeline set
  review/index.html              MP4s with the source skeleton beside the
                                 character, plus the decision stills
  retarget_validation.json       Per-bone, per-frame agreement numbers
  rig_dump.json                  The rig as imported: hierarchy, rest, weights
```

Clip order in the showcase is not arbitrary: `scripts/score_motion_quality.py`
ranks the set cleanest-first and then most-expressive, because the retarget is
exact and therefore source quality is the ceiling. Clips above the quality bar are
excluded — on the tested set, 301 of 631 pass. The card for each clip shows its
peak angular velocity and jitter, so a clip that still looks wrong can be checked
against its own numbers.

## Adding a character

```bash
uv run python scripts/onboard_character.py --archive ~/Downloads/model.zip --name <slug>
```

Five gates, each of which has caught a real failure: texture filenames verified
against the PMX's own table, bone mapping, rest-gap sanity against the other rigs,
the numeric retarget check, and an identity T-pose plus a posed frame. It prints
`READY` or `REVIEW`. Then add the slug to `CHARACTERS` in
`scripts/generate_showcase.py`.

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

## Full generation

```bash
CLIPS=.local/soma_all/soma_tmr_test_400m_all_fullpose/clips/t2m

uv run python scripts/score_motion_quality.py --clips $CLIPS --meta .local/soma_meta \
  --output outputs/motion_scores.json

uv run python scripts/generate_showcase.py --clips $CLIPS \
  --scores outputs/motion_scores.json --character yoimiya --limit 0 \
  --resolution 800 --output converted/showcase
```

The generator is resumable — re-running it skips clips whose MP4 already exists,
so an interrupted run costs nothing. It deletes each clip's frames as soon as they
are encoded, because keeping them would be about 18 GB. Roughly 80 s per clip at
800 px, so the full clean set is a few hours.
