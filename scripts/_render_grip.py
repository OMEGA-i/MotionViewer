"""Close-ups of the weapon hand, from two angles, aimed at the fist centre.

A held weapon is small in a full-body frame, and judging a grip from a 330 px thumbnail
does not work — three separate wrong grips survived that way here: the blade in the hand
with the hilt pointing outward, fingers curled backwards toward the wrist, and an open
palm with the sword resting against it. Each looked fine at body scale.

Two decisions make this trustworthy where earlier checks were not:

- **Aim at the fist, not the wrist bone.** A grip sits at the knuckles, so a camera
  pointed at ``手首`` frames the forearm and cuts off what matters.
- **Two orthogonal views per frame.** From one angle a weapon passing *beside* the hand
  is indistinguishable from one passing *through* it. A second camera 90 degrees round
  settles it, and getting that wrong is what made a previous report claim success.

  blender --background --python scripts/_render_grip.py -- --characters 彦卿,青衣
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402

from motionviewer.blender.mmd_props import (  # noqa: E402
    WEAPON_BY_CHARACTER,
    close_hand,
    fit_weapon_to_hand,
    load_weapon,
)
from motionviewer.blender.mmd_toon import add_toon_lighting, apply_toon_shading  # noqa: E402
from motionviewer.blender.render import _normalize_engine  # noqa: E402
from motionviewer.blender.retarget.pipeline import create_fbx_actor_from_npz  # noqa: E402
from motionviewer.blender.scene import clear_scene, setup_world  # noqa: E402

ASSETS = ROOT / "assets/fbx/pmx"

#: Character directory -> the PMX inside it. The packs nest inconsistently, so this
#: cannot be derived from the directory name alone.
CHARACTER_MODELS = {
    "星穹铁道—彦卿": "星穹铁道—彦卿/彦卿1.0.pmx",
    "星穹铁道—流萤·春日手信": "星穹铁道—流萤·春日手信/星穹铁道—流萤·春日手信.pmx",
    "星穹铁道—飞霄无外套": "星穹铁道—飞霄（无外套）.pmx",
    "青衣": "青衣.pmx",
}

# Two camera offsets about 90 degrees apart, close enough that a 30 mm shaft fills
# a useful part of the frame at 85 mm.
VIEWS = (("A", Vector((0.22, -0.22, 0.10))), ("B", Vector((-0.22, -0.22, 0.10))))


def fist_centre(armature, hand_bone: str) -> Vector:
    """Midpoint of the curled fingers, which is where a grip actually lies."""
    side = "右" if hand_bone.startswith("右") else "左"
    points = []
    for finger in ("人指", "中指", "薬指", "小指"):
        for segment, use_tail in (("１", False), ("３", True)):
            bone = armature.pose.bones.get(f"{side}{finger}{segment}")
            if bone is None:
                continue
            local = bone.tail if use_tail else bone.head
            points.append(np.array(tuple(armature.matrix_world @ local)))
    if not points:
        return armature.matrix_world @ armature.pose.bones[hand_bone].head
    return Vector(tuple(np.mean(points, axis=0)))


def grip_gap_cm(weapon, fist: Vector) -> float:
    """Distance from the fist centre to the nearest point on the weapon."""
    matrix = weapon.matrix_world
    points = np.array([tuple(matrix @ v.co) for v in weapon.data.vertices])
    return float(np.linalg.norm(points - np.array(tuple(fist)), axis=1).min()) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion",
        type=Path,
        default=Path("/Users/a26044/Motion data/t2m_demo400/smplx/active/128701.smplx.npz"),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "converted/final/检查图/grip")
    parser.add_argument("--frames", default="10,45", help="Comma-separated frames to shoot")
    parser.add_argument("--characters", default="", help="Comma-separated subset")
    parser.add_argument("--resolution", type=int, default=560)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

    wanted = [name for name in args.characters.split(",") if name] or list(CHARACTER_MODELS)
    frames = [int(value) for value in args.frames.split(",") if value]
    args.output.mkdir(parents=True, exist_ok=True)

    for key in wanted:
        match = next((full for full in CHARACTER_MODELS if key in full), None)
        if match is None:
            print(f"GRIP unknown character {key!r}; have {sorted(CHARACTER_MODELS)}", flush=True)
            continue
        entry = WEAPON_BY_CHARACTER.get(match)
        if entry is None:
            print(f"GRIP {match} has no weapon registered", flush=True)
            continue

        clear_scene()
        setup_world()
        model = ASSETS / match / CHARACTER_MODELS[match]
        actor = create_fbx_actor_from_npz(args.motion, label="grip", fbx_path=model, retarget_mode="direct")
        armature = actor.armature
        hand = entry["hand"]

        weapon_path = model.parent / entry["file"]
        if not weapon_path.is_file():
            weapon_path = model.parent.parent / entry["file"]
        weapon = load_weapon(bpy, weapon_path, label="w")
        fit = fit_weapon_to_hand(bpy, weapon, armature, hand, grip=entry["grip"])
        curl = close_hand(
            bpy,
            armature,
            hand,
            frame_start=1,
            frames=max(frames) + 1,
            shaft_m=fit["shaft_mm"] / 1000.0,
            weapon=weapon,
            grip=entry["grip"],
            palm_normal=Vector(fit["palm_normal"]),
        )

        apply_toon_shading(list(actor.mesh_objects) + [weapon])
        add_toon_lighting([-1.0, -1.0, 0.0], [1.0, 1.0, 2.0])

        camera_data = bpy.data.cameras.new("grip")
        camera_data.lens = 85
        camera = bpy.data.objects.new("grip", camera_data)
        bpy.context.collection.objects.link(camera)
        scene = bpy.context.scene
        scene.camera = camera
        scene.render.engine = _normalize_engine("BLENDER_EEVEE")
        scene.render.resolution_x = scene.render.resolution_y = args.resolution
        scene.render.film_transparent = False

        for frame in frames:
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            target = fist_centre(armature, hand)
            gap = grip_gap_cm(weapon, target)
            print(
                f"GRIP {match} f{frame}: shaft {fit['shaft_mm']:.0f} mm, "
                f"{curl['mode']} {curl['solved']}, weapon-to-fist {gap:.2f} cm",
                flush=True,
            )
            for tag, offset in VIEWS:
                camera.location = target + offset
                direction = (target - camera.location).normalized()
                camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
                scene.render.filepath = str(args.output / f"{match[:10]}_{frame:03d}{tag}.png")
                bpy.ops.render.render(write_still=True)
    print(f"GRIP wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
