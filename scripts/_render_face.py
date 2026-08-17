"""Face close-up, optionally with an expression applied. Local helper.

Expression work needs the face at a readable size; at showcase framing a head is
about 80 px tall and a mouth is a dozen pixels, which is not enough to tell a
smile from a grimace.

  blender --background --python scripts/_render_face.py -- \
    --asset assets/fbx/pmx/yoimiya/宵宫.pmx --output outputs/face/neutral.png \
    --expression smile
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expression", default="none")
    parser.add_argument("--amount", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=760)
    parser.add_argument("--distance", type=float, default=0.80)
    parser.add_argument("--yaw", type=float, default=18.0, help="Degrees off dead-on")
    parser.add_argument("--no-toon", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

    import bpy  # type: ignore
    import numpy as np
    from mathutils import Vector  # type: ignore

    from motionviewer.blender.mmd_expression import EXPRESSION_PRESETS, apply_expression
    from motionviewer.blender.mmd_import import import_pmx_character
    from motionviewer.blender.mmd_toon import add_toon_lighting, apply_toon_shading
    from motionviewer.blender.render import _normalize_engine
    from motionviewer.blender.scene import clear_scene, setup_world

    toon_dir = ROOT / ".local/blender_mmd_tools/mmd_tools/externals/MikuMikuDance"
    for name in ("toon01.bmp", "toon05.bmp"):
        source = toon_dir / name
        target = ROOT / name
        if source.is_file() and not target.exists():
            target.symlink_to(source)

    clear_scene()
    setup_world(transparent=False)
    armature, meshes = import_pmx_character(bpy, args.asset, label="face", scale=0.08, morphs=True)
    bpy.context.view_layer.update()

    if args.expression != "none":
        report = apply_expression(meshes, args.expression, amount=args.amount)
        print(f"expression: {json.dumps(report, ensure_ascii=False)}")
    else:
        print(f"expression: neutral (available presets: {sorted(EXPRESSION_PRESETS)})")

    head = armature.pose.bones.get("頭")
    if head is None:
        raise SystemExit("no 頭 bone")
    matrix = armature.matrix_world @ head.matrix
    # Aim a little above the head bone's root, which sits at the neck.
    target = matrix.translation + matrix.to_3x3().col[1].normalized() * 0.055

    bounds_min = [float(value) - 0.2 for value in target]
    bounds_max = [float(value) + 0.2 for value in target]
    if not args.no_toon:
        apply_toon_shading(meshes)
        add_toon_lighting(bounds_min, bounds_max)
    else:
        from motionviewer.blender.scene import add_lighting

        add_lighting(bounds_min, bounds_max)

    camera_data = bpy.data.cameras.new("FaceCam")
    camera_data.lens = 62.0
    camera = bpy.data.objects.new("FaceCam", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    angle = np.radians(args.yaw)
    direction = Vector((float(np.sin(angle)), -float(np.cos(angle)), 0.12)).normalized()
    camera.location = target + direction * args.distance
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = (target - camera.location).to_track_quat("-Z", "Y")
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = _normalize_engine("BLENDER_EEVEE")
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 128
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.output)
    bpy.ops.render.render(write_still=True)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
