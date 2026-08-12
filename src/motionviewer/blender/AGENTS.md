# Blender module

- Blender 5.2 LTS is the reference integration environment; avoid hard-coding its macOS installation path.
- Import `bpy` and `mathutils` only inside functions executed by Blender.
- The external SMPL-X addon is probed through operators and properties, not a fixed installation directory.
- Blender adapters consume resolved bundles; they do not reinterpret source formats or package metadata.
- Retarget math remains pure NumPy where possible. Validate with unit tests plus opt-in local Blender tests.
- Blender-MCP requires an interactive Blender process; background rendering uses the packaged entry scripts.
