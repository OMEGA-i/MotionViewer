# MotionViewer contributor guide

## Project contract

- MotionViewer turns motion files or Sample Packing Protocol packages into reproducible Blender render jobs.
- Python 3.11+ is the supported host runtime. Core CI targets macOS and Linux.
- Preserve physical scale, trajectory length, coordinate transforms, resolved configuration, and input hashes.
- Render comparisons as one actor per job unless an explicitly debug-only overlay is requested.
- Never commit SMPL-X model weights, Blender addon data, downloaded FBX/PMX characters, local packages, or renders.
- `data/examples/` contains only curated motion samples that the project may redistribute.

## Module seams

- `core/`: pure NumPy domain logic; never imports Blender.
- `loaders/`: converts supported files into `MotionSequence` through `MotionFormatRegistry`.
- `packages/`: validates, indexes, selects, and materializes Sample Packing Protocol inputs.
- `video/`: owns render-job configuration, preparation, manifests, encoding, and batch orchestration.
- `blender/`: Blender adapters and rendering implementation; imports `bpy` lazily.
- `cli.py`: command composition only. Put reusable behavior behind the module interfaces above.

## Verification

Run `uv run ruff check .`, `uv run pytest`, and `uv build` before publishing. Blender integration tests are opt-in with `MOTIONVIEWER_BLENDER_TESTS=1` and require local licensed assets.
