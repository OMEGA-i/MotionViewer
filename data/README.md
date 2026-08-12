# Data Layout

Motion data is kept outside the Python package and project source tree.

## `examples/`

Small demo and fixture clips that can live in the repository. The current SMPL-X comparison set is under:

- `examples/smplx_body22_fitted_aa/`

## `raw/`

Local model outputs and baseline exports. This directory is gitignored. Put your own generated motions here, for example:

```text
data/raw/
  ours/
    run_001.smplx.npz
  baselines/
    mdm/
    motiongpt3/
```

Render configs should reference files under `data/` with paths relative to the config file or absolute paths on your machine.

## `packages/`

Imported Sample Packing Protocol **v2.0** archives (see `docs/archives.md` and
`docs/package_protocol.md`). Use `motionviewer package import` to validate and
copy a `.tar.gz` as-is:

```text
data/packages/
  soma_tmr_test.tar.gz
```

This directory is gitignored. Do not unpack bundles wholesale here.

## `local/packages/`

Local cache for assets materialized from Sample Packing Protocol packages.
This directory is gitignored and safe to delete at any time —
`motionviewer package config <package>` re-materializes on demand. Layout:

```text
data/local/packages/
  <package_stem>/
    <task>/
      <clip_id>/
        <source_id>/
          smplx_params.npz
```

Packages are a first-class data-delivery mode: import the archive, inspect, and
selectively materialize through `motionviewer package config` — never unpack
wholesale for rendering.
