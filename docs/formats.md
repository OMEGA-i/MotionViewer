# Data Formats

All inputs are loaded through `MotionFormatRegistry`. Render code should not open motion files directly except inside Blender runner execution using the resolved bundle.

## Loader Contract

Each loader provides:

- `format_id`
- `extensions`
- `capabilities`
- `probe(path)`
- `load(path, options)`
- `validate(sequence)`

Selection is deterministic:

1. Explicit config `format` wins.
2. Otherwise loaders probe suffix and metadata.
3. Equal-confidence matches fail loudly.
4. Unsupported files list supported formats.

## `smplx_body22_fitted_aa`

Current examples in `data/examples/smplx_body22_fitted_aa/` use:

- `joints22`: `(T, 22, 3)`
- `transl`: `(T, 3)`
- `global_orient`: `(T, 3)` axis-angle
- `body_pose`: `(T, 63)` for 21 body joints
- `betas`: `(16,)`
- `fps`: scalar
- `prefix_T`: scalar
- `source`: scalar label
- `format`: `smplx_body22_fitted_aa`
- `fit_mse`: scalar quality metric

This format is mesh-capable through the Blender SMPL-X addon and analysis-capable through `joints22`.

The observed data is y-up, meter-like, body-only, and lacks hands/face/expression. Neutral defaults are used for missing SMPL-X channels.

## Package Protocol Assets

Sample Packing Protocol packages (see `docs/package_protocol.md`) expose per-source files such as:

- `smplx_params.npz` → loaded as `smplx_body22_fitted_aa` (preferred render payload)
- `joints22.npy` → loaded as `joints_npy` (skeleton fallback via `--asset joints22`)

Package ingestion lives in `motionviewer.packages`; once materialized, assets load through the normal format registry above. Optional `joints77.npy` / `native.npy` are retained as inspection variants and are not selected for rendering yet.

## Generic Joints

`joints_npz` reads a configurable `joints`, `joints22`, or `positions` key. `joints_npy` reads a raw `(T, J, 3)` array. These formats can drive skeleton/contact/camera analysis but not real SMPL-X mesh unless a future backend maps them to body parameters.
