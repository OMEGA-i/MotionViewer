# Core module

- Keep this module importable in ordinary CPython: no `bpy`, `mathutils`, subprocesses, or filesystem policy.
- Prefer immutable domain values and pure functions that return results instead of mutating callers.
- Coordinate conversion, layout math, palette logic, ground planning, and canonical skeleton semantics live here.
- Test behavior through the public function or type that callers use; avoid tests coupled to intermediate arrays.
