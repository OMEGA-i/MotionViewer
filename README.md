# MotionViewer

MotionViewer is a Python toolkit for turning SMPL-X motion sequences into reproducible Blender renders. It validates motion inputs, prepares portable render-job bundles, drives Blender in the background, and records the resolved configuration and input hashes used for every output.

The project is aimed at research figures and model comparisons where scale, trajectory, camera, and visual style must remain consistent.

## Highlights

- SMPL-X mesh rendering through the external SMPL-X Blender addon
- deterministic loaders for SMPL-X NPZ and generic joint arrays
- Sample Packing Protocol package inspection, validation, selection, and preview
- single-actor, multi-view video jobs and transparent qualitative snapshot renders
- optional FBX retargeting, catalog validation, pose grids, and quality audits
- fast CPython test suite; Blender integration tests are opt-in

## Requirements

- macOS or Linux
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- Blender 5.2 LTS for the reference integration environment
- [`smplx_blender_addon`](https://gitlab.tuebingen.mpg.de/jtesch/smplx_blender_addon) plus separately obtained SMPL-X model data
- `ffmpeg` when encoding MP4 output

The addon repository contains code only and uses CC BY-NC 4.0. Its model data has separate terms and is never part of MotionViewer.

## Install

```bash
git clone https://github.com/OMEGA-i/MotionViewer.git
cd MotionViewer
uv sync --dev
uv run motionviewer --help
```

MotionViewer discovers Blender from `--blender`, the `MOTIONVIEWER_BLENDER` environment variable, `PATH`, or common macOS/Linux locations.

```bash
uv run motionviewer doctor
```

## Quick start

Inspect the included redistributable motion sample:

```bash
uv run motionviewer motion inspect \
  data/examples/smplx_body22_fitted_aa/omegamotiongpt.smplx.npz
```

Create and validate a render configuration without launching Blender:

```bash
uv run motionviewer motion config \
  data/examples/smplx_body22_fitted_aa/omegamotiongpt.smplx.npz \
  --output configs/generated/demo.yaml

uv run motionviewer render job \
  --config configs/examples/multiview_single_actor.yaml \
  --dry-run
```

Render with the discovered Blender installation:

```bash
uv run motionviewer render job \
  --config configs/examples/multiview_single_actor.yaml
```

Package and FBX workflows are grouped by domain:

```bash
uv run motionviewer package inspect path/to/package.tar.gz
uv run motionviewer package import path/to/package.tar.gz
uv run motionviewer package preview path/to/package.tar.gz --source model_name
uv run motionviewer fbx check --root assets/fbx
```

## Local data and licensed assets

The repository contains small motion examples only. These paths remain local and are ignored by Git:

- `assets/` — downloaded FBX/PMX characters and local catalogs
- `data/raw/`, `data/local/`, `data/packages/` — research inputs, archives, and caches
- `outputs/` — rendered frames, videos, manifests, and audits
- Blender addon data and SMPL-X weights

Do not remove these ignore rules without checking redistribution terms. See [asset setup](docs/assets.md) and [data layout](data/README.md).

## Development tools

The committed `.mcp.json` starts `blender-mcp` for interactive Blender sessions. Blender-MCP cannot serve commands from a background `blender -b` process; MotionViewer's CLI uses packaged Blender entry scripts for headless work.

Project instructions are hierarchical: start with [AGENTS.md](AGENTS.md), then follow the nearest module-level `AGENTS.md`.

## Documentation

- [Architecture](docs/architecture.md)
- [Blender and addon setup](docs/blender-setup.md)
- [Render jobs](docs/render-jobs.md)
- [Motion formats](docs/formats.md)
- [Sample Packing Protocol](docs/package_protocol.md)
- [Package workflows](docs/archives.md)
- [FBX retargeting](docs/retarget.md)
- [Visual style](docs/visual-style.md)

## License

MotionViewer source code is released under the [MIT License](LICENSE). Third-party addons, body-model data, motion datasets, and character assets retain their own licenses.
