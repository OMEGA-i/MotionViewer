# Video module

- This module translates user configuration into a fully resolved, reproducible job bundle.
- It may import `core`, `loaders`, `packages`, and asset catalog metadata, but never `bpy`.
- Keep Blender process execution at a narrow adapter seam and keep selection/preparation testable in CPython.
- Manifests record resolved inputs, hashes, frame rate, frame range, camera views, and outputs.
- A schema change requires config parsing tests and an update to `docs/render-jobs.md`.
