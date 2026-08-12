"""Blender entry point that validates the external SMPL-X addon and model data."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> int:
    try:
        import addon_utils  # type: ignore
        import bpy  # type: ignore
    except ImportError:
        print("This check must run inside Blender.", file=sys.stderr)
        return 1

    from motionviewer.blender.addon_probe import probe_smplx_addon

    print(f"Blender: {bpy.app.version_string}")
    status = probe_smplx_addon()
    if not status.available:
        print(f"[ISSUE] {status.error}", file=sys.stderr)
        return 1
    print(f"Addon: {status.module} — {'ENABLED' if status.enabled else 'DISABLED'}")
    for operator in status.operators:
        print(f"  {operator}: OK")
    if not status.enabled:
        return 1
    if "scene.smplx_add_gender" not in status.operators:
        print("[ISSUE] scene.smplx_add_gender is unavailable", file=sys.stderr)
        return 1

    module = next((item for item in addon_utils.modules() if item.__name__ == status.module), None)
    module_file = Path(getattr(module, "__file__", "")) if module else None
    model_files: list[Path] = []
    if module_file:
        data_dir = module_file.parent / "data"
        if data_dir.exists():
            model_files = [*data_dir.rglob("*.pkl"), *data_dir.rglob("*.npz")]
    if not model_files:
        print("[ISSUE] No SMPL-X model data files were found", file=sys.stderr)
        return 1
    print(f"Models: {len(model_files)} files found")
    for name, value in status.wm_properties.items():
        print(f"  tool.{name}: {value}")
    print("All checks passed — ready to render!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
