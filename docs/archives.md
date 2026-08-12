# Sample Packing Protocol Packages

MotionViewer consumes upstream Sample Packing Protocol **v2.0** packages as the
first-class multi-clip comparison input. A package may be a directory or a
`.tar.gz` / `.tgz` bundle. One bundle holds all tasks (t2m / pred / recon) and
provenances for a `{track}_{split}` eval run.

Protocol layout and semantics live in [`docs/package_protocol.md`](package_protocol.md).
This page documents the MotionViewer consumer workflow.

## Concepts

- **`motionviewer.packages`**: deep package module. External interface is
  `import_package()`, `inspect_package()`, `validate_package()`, and
  `build_render_job_from_package()`.
- **`PackageStore`**: internal tar/directory adapters. Callers never see member
  paths, path-traversal checks, or cache materialization details.
- **Source model**: `gt` plus every protocol source (`model` or
  `reconstruction_level`) are treated uniformly as render inputs.
- **Asset preference**: default render payload is `smplx_params.npz`
  (`smplx_body22_fitted_aa`). Pass `--asset joints22` to force skeleton-only
  `joints22.npy`.
- **Single actor per job**: `make-config` emits one source per RenderJob;
  compose model grids in PPT.

## CLI Workflow

```bash
# Copy a v2 bundle as-is into data/packages/{track}_{split}.tar.gz
uv run motionviewer package import ~/Downloads/soma_tmr_test.tar.gz
uv run motionviewer package import ~/Downloads/soma_tmr_test.tar.gz --force

# Inspect
uv run motionviewer package inspect data/packages/soma_tmr_test.tar.gz
uv run motionviewer package inspect data/packages/soma_tmr_test.tar.gz --json
uv run motionviewer package inspect data/packages/soma_tmr_test.tar.gz --task t2m --clip 'test:abc123'

# Build a RenderJob (one actor). --task required when the bundle has multiple tasks.
uv run motionviewer package config data/packages/soma_tmr_test.tar.gz \
  --task t2m \
  --clip 'test:abc123' \
  --sources omegamotiongpt \
  -o configs/generated/t2m_omegamotiongpt.yaml

# Reconstruction level
uv run motionviewer package config data/packages/soma_tmr_test.tar.gz \
  --task recon \
  --clip 'test:recon1' \
  --sources q04 \
  -o configs/generated/recon_q04.yaml

# Then render like any other config
uv run motionviewer render job --config configs/generated/t2m_omegamotiongpt.yaml --dry-run
```

Defaults:

- `--task` omitted → only legal when the package has exactly one task
- `--clip` omitted → first clip in the selected task (accepts meta `clip_id` or dir name)
- `--sources` omitted → first model source, or first recon level for recon tasks
- Materialized assets land under
  `data/local/packages/<package_stem>/<task>/<clip_id>/<source_id>/`

Imported archives live under `data/packages/`. MotionViewer never unpacks a
bundle wholesale for rendering — only selected assets are materialized.

## Task Mapping

| Protocol task | `RenderJob.task.mode` | Notes |
|---|---|---|
| `t2m` | `text_to_motion` | `meta.caption` → `task.instruction`, prefix hidden |
| `pred` | `continuation` | `prefix_T` preserved in SMPL-X payloads, prefix markers shown |
| `recon` | `comparison` | labels include codebook count and `mse_vs_gt` when present |

## Validation Modes

Used by the package module (CLI `make-config` uses `assets`):

- `none`: parse identity only
- `structural`: manifest/meta shape, task fields, source/recon consistency
- `assets`: referenced files exist and relative paths are safe
- `loadable`: selected assets also load through `MotionFormatRegistry`

## Python API

```python
from pathlib import Path
from motionviewer.packages import (
    PackageRenderRequest,
    PackageSelection,
    build_render_job_from_package,
    import_package,
    inspect_package,
)

imported = import_package("~/Downloads/soma_tmr_test.tar.gz")
inspection = inspect_package(imported)
job = build_render_job_from_package(
    PackageRenderRequest(
        path=imported,
        selection=PackageSelection(task="t2m", clip_id="test:abc123", sources=("omegamotiongpt",)),
        cache_dir=Path("data/local/packages"),
        validation="loadable",
    )
)
```

## Out of Scope

- Producing upstream packages (`package_samples.py` lives upstream)
- Direct rendering of `joints77.npy` / `native.npy` (visible in inspection variants only)
- Protocol v1.0 packages (rejected; re-export with package_samples 2.0)
