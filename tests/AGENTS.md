# Tests

- Default tests must be hermetic, fast, and runnable without Blender or third-party character assets.
- Generate package fixtures in `tmp_path`; do not depend on `data/local/`, `assets/`, or `outputs/`.
- Mark local Blender tests with the `MOTIONVIEWER_BLENDER_TESTS=1` opt-in gate.
- Assert observable output semantics. Do not lock image tests to a single pixel when lighting or color management affects it.
