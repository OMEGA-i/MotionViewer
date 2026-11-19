# Render Jobs

Render jobs are YAML configs parsed into `RenderJob`.

Important sections:

- `task`: `continuation`, `text_to_motion`, or `comparison`, plus optional `instruction`.
- `inputs`: path, label, optional explicit format, loader options. Paper path: **one input per job**.
- `timeline`: fps override, frame range, prefix display, `frames_mode`.
- `layout`: `single` (default) or `overlay` (debug overlap). `side_by_side` / `grid` were removed — compose model grids in PPT.
- `body`: mesh backend and neutral hand/face policy (`blender_smplx_addon` or `fbx_skeleton`).
- `style`: palette, temporal ramp, material roughness, ghost snapshots, `freeze_fade_frames`/`freeze_fade_alpha`, `prefix`, `ghost`, `labels`.
- `ground`: trajectory-guided ground mode and contact/carpet thresholds.
- `camera`: base `preset`/`orthographic`/`margin`, optional `views` list for multi-camera output.
- `render`: Blender engine, resolution, samples, frame format.
- `output`: directory, MP4 name, keep frames, manifest name, overwrite behavior.

## Multi-view cameras

Set `camera.views` to emit one PNG sequence + MP4 per preset:

```yaml
camera:
  orthographic: true
  margin: 1.2
  views:
    - preset: three_quarter   # staging: world (default)
    - preset: front           # staging: inplace (default); ghost/ground → none
    - preset: side
```

| Preset | Default staging | Default ghost/ground |
|---|---|---|
| `three_quarter`, `top` | `world` | inherit job style |
| `front`, `side` | `inplace` | `mode: none` |

`inplace` freezes root horizontal motion (source transl X/Z; Blender joints root X/Y) so front/side framing stays centered.

CLI:

- By default, `world` and `inplace` staging groups run as **two parallel Blender processes** (`--parallel-views` / `--no-parallel-views`).
- Multi-view output layout:

```text
outputs/<job>/
  job_bundle.json
  manifest.json          # includes "views": [...]
  blender_status_world.json
  blender_status_inplace.json
  frames/three_quarter/frame_*.png
  frames/front/...
  three_quarter.mp4
  front.mp4
  side.mp4
```

Single-view jobs (no `camera.views`, only `camera.preset`) keep the legacy flat layout: `frames/frame_*.png`, `output.mp4_name`, `blender_status.json`.

Example: `configs/examples/multiview_single_actor.yaml`.

## Qualitative snapshot stills

`motionviewer render qualitative` writes one unlabeled transparent PNG per
source and clip. All sources for a clip use the same sampled times, FBX model,
camera bounds, and camera transform.

PNG filenames contain both the source model and a filesystem-safe caption slug,
for example `omegamotiongpt__a_person_walks_forward.png`. The complete caption
remains in `selection.json` and each clip's `manifest.json`.

- `--snapshot-layout root_aligned` removes each snapshot's horizontal root
  translation, preserves vertical motion, and places snapshots left-to-right at
  `--snapshot-spacing` meter intervals. This is the recommended mode for
  in-place gestures and actions.
- `--snapshot-layout trajectory` retains the generated world trajectory.
- `--material-mode palette` applies one light-to-dark temporal palette to every
  snapshot, including the final pose.
- `--material-mode preserve` retains the FBX material slots and textures on
  every snapshot.
- `--fbx-pool approved` is the default and only samples assets marked
  `status: approved` plus `random_eligible: true` in `assets/fbx/catalog.json`.
  `--fbx-pool all_binary` is an explicit exploratory mode that also samples
  binary catalog assets still pending the full-motion quality gate; visually
  audit those assets before using them in a paper figure.
- `--exclude-fbx iron` (or a comma-separated list) removes known-problematic
  model ids from either pool without changing the FBX catalog itself.

The default `2400x900` landscape canvas is sized for a six-pose strip. The
outputs contain no ground, labels, or background pixels.

## Validation rules

- `layout.mode='single'` → exactly one input.
- `layout.mode='overlay'` → multiple inputs allowed (spatial overlap debugging).
- `camera.views` presets must be unique; views that share a staging must agree on ghost/ground overrides after normalization.

## Package `make-config`

Import a v2 archive with `motionviewer package import`, then build one-actor configs.
Pass `--task` when the bundle has multiple tasks, and exactly one `--sources <name>`.
Render gt separately if needed, then collage in PPT. See `docs/archives.md`.

## Package `preview-package`

`preview-package` creates small sampled previews from a v2 package source. It
materializes a sampled SMPL-X NPZ per selected clip, writes normal RenderJob
YAMLs, and records the selection in `selection.json`.

```bash
uv run motionviewer package preview data/packages/soma_tmr_test.tar.gz \
  --task t2m \
  --source omegamotiongpt \
  --clips 10 \
  --frames 10 \
  --body-pool smplx,fbx-random \
  --fbx-root assets/fbx \
  --seed 123 \
  --output outputs/preview/omegamotiongpt_10clips
```

Pass `--blender /Applications/Blender.app/Contents/MacOS/Blender` to render the
generated configs immediately. `fbx-random` draws only from
`assets/fbx/catalog.json` entries with `status: approved` and
`random_eligible: true`; `assets/fbx/pmx/` is excluded from this pool.

## FBX Body Backend

Per-input FBX body override:

```yaml
inputs:
  - path: motion.smplx.npz
    body:
      backend: fbx_skeleton
      fbx_path: assets/fbx/iron.fbx
      bone_map: auto
```

Validate the non-PMX FBX pool:

```bash
uv run motionviewer fbx check --root assets/fbx
uv run motionviewer fbx check --root assets/fbx --deep --write-report outputs/preview/fbx_validation
```

ASCII FBX files are reported invalid because Blender rejects them in headless
rendering. FBX paths are checked before launching a long render. Deep reports
include catalog profile/status/evidence and proposed statuses; the catalog
itself stays explicit and reviewable.

The Python CLI writes:

- `job_bundle.json`: portable resolved job consumed by Blender.
- `manifest.json`: source hashes, detected formats, bounds, fps, render settings, and optional `views`.
- `blender_status.json` or `blender_status_<staging>.json`: addon detection / runner status.

## Mismatched Input Lengths

Overlay / multi-input jobs may have unequal frame counts. `timeline.frames_mode`:

- `"max"` (default): timeline runs the longest input; shorter inputs freeze and fade.
- `"min"`: trim every input to the shortest.

Use `--dry-run` to validate config and generate bundle/manifest without launching Blender:

```bash
uv run motionviewer render job --config configs/examples/multiview_single_actor.yaml --dry-run
```
