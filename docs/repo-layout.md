# What is in here, and what is safe to delete

The repository accumulated 2.5 GB of render debris across the experiments that produced
the current pipeline. This is the map after clearing it, and the rule that decided each
case: **if a Blender run can rebuild it, it goes; if it records a measurement or a
decision, it stays.**

## The render path

Everything on it, in the order a clip travels:

| | |
| --- | --- |
| `scripts/visualize_dataset.py` | batch entry — discovers tasks, validates the contract, scores, renders, writes the page |
| `scripts/render_mmd_compare.py` | single clip; every flag that shapes a render lives here |
| `scripts/pick_clips.py` | Viser picker, with Chinese summaries and the padding marker |
| `src/motionviewer/blender/retarget/` | the solver: mapping tables, plan, per-frame solve |
| `src/motionviewer/blender/mmd_props.py` | hand props, sets, weapons, stages |
| `src/motionviewer/blender/mmd_bullet.py` | the PMX's own rigid bodies, simulated |
| `src/motionviewer/blender/mmd_toon.py` | cel shading, outline, ground |
| `src/motionviewer/core/` | SMPL-X and SOMA kinematics, head clearance |

## Kept because it cannot be regenerated cheaply

| path | why |
| --- | --- |
| `data/measurements/` | every number quoted in a docstring or commit message. Re-running some needs a Blender session per character; see its README, including the caveat on `skirt_physics.json` |
| `data/picks_backup/` | the clip shortlists and tags. Hand-picked, so a lost file is an afternoon |
| `data/caption_zh.json` | 400 hand-written Chinese summaries |
| `converted/soma_probe/` | the only rendered evidence for the SOMA-77 path, which is a reserve rather than the default. Its left panels predate the polish-alias fix |
| `converted/final/` | current deliverables, with `检查图/` holding the verification stills |
| `scripts/archive/` | retired scripts and why — see its README |

## Cleared, and how to rebuild it

`outputs/` held 2.2 GB of PNG frames from abandoned experiments — spring sweeps, wind
tests, outline comparisons — with no MP4 and no manifest between them. `converted/`
held 235 MB of renders of datasets we no longer use.

Anything still wanted comes back from the scripts: `visualize_dataset.py` for a dataset,
`render_mmd_compare.py` for one clip, `_render_grip.py` for grip close-ups,
`render_paper_figure.py` for figures. The motion data and the character assets are the
inputs, and both are outside the repository or git-ignored.

## Not tracked at all

`.local/` (a downloaded Blender, whose main binary is over GitHub's 100 MB per-file
limit), `.venv/`, `outputs/`, `converted/`, and the motion datasets.
