# Visual Style

MotionViewer defaults to comparison-safe presentation with transparent backgrounds for paper-style figures.

## Scale And Layout

- Keep a common scale across actors — **do not** resize independently per camera view.
- Paper comparisons: **one actor per render job**, multiple cameras via `camera.views`, compose model×view grids in PPT.
- `layout.mode: single` (default): one input. `side_by_side` and `grid` were removed.
- `layout.mode: overlay`: intentional spatial overlap for debugging only.
- Default alignment is `start_root`: frame-0 root centered in the layout cell.

## Render Modes

- `task.mode: continuation`: show neutral prefix context and model-colored generated continuation.
- `task.mode: text_to_motion`: no prefix assumptions; legend may show `task.instruction`.

## Prefix Semantics

Shared conditioning prefixes should be visible but not confused with generated continuations.

- `style.prefix.mode: attached` (default): prefix is rendered with neutral color.
- `style.prefix.mode: marker`: adds a small transition marker at `prefix_T`.
- `style.prefix.mode: shared`: draw one neutral prefix in overlay mode.
- `style.ghost.include_prefix: false` by default; continuation ghosts use model colors only.

## Camera And Staging

Static orthographic presets: `three_quarter`, `front`, `side`, `top`.

- World views (`three_quarter`/`top`): keep trajectory; ghost + ground as configured.
- Inplace views (`front`/`side`): freeze root horizontal motion; ghost and ground default to `none` so the silhouette stays clean for PPT.

## Temporal Color

Default `style.temporal_ramp: lavender_to_purple` maps time from light
`rgb(231, 219, 249)` at `t=0` to deep `rgb(130, 81, 219)` at `t=1`.
Ghost snapshots use `style.ghost.alpha`.

`light_to_dark` is still available for older configs; it darkens from
`style.ghost.start_lightness` to `style.ghost.end_lightness`.

- `style.ghost.mode: trail`: snapshots appear only once the sampled frame has been reached, so the trail grows over time.
- `style.ghost.mode: snapshots`: all snapshots are visible for the full render.
- `style.ghost.mode: none`: no static snapshots.

## Labels

- `style.labels.mode: legend` (default): top-right legend with color swatches.
- `style.labels.mode: world`: legacy 3D labels near actors.
- `style.labels.mode: none`: no labels.

## Trajectory-Guided Ground

Modes:

- `none`
- `trajectory_rectangle` (default): one clean rectangle covering the full root trajectory
- `trajectory_carpet`: segmented ribbon mesh under the root trajectory
- `contact_patches`
- `trajectory_ribbon`
- `footprint_trail`
- `coverage_hull`

`trajectory_rectangle` uses `ground.carpet_padding` and `ground.opacity`, and is
computed from the whole clip trajectory once. It avoids progress-dependent
ground clutter. Inplace staging forces `none`.

`trajectory_carpet` remains available for debugging; in continuation mode it is
one connected mesh with prefix/generated material segments.
