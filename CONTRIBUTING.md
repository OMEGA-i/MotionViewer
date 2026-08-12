# Contributing

MotionViewer uses Python 3.11+, `uv`, and Blender 5.2 LTS for integration work.

```bash
uv sync --dev
uv run ruff check .
uv run pytest
uv build
```

Keep pull requests focused and include tests for behavior changes. Do not attach licensed body-model files, downloaded character models, research datasets, or generated renders. Small motion fixtures must have documented redistribution permission.

Blender integration tests are intentionally local because they require separately licensed assets:

```bash
MOTIONVIEWER_BLENDER_TESTS=1 uv run pytest -q tests/test_blender_*.py
```

Follow the nearest `AGENTS.md` when editing a module.
