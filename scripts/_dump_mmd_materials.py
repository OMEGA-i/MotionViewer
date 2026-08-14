"""Dump the material graph mmd_tools builds for a PMX. Local helper.

A Genshin or Honkai model is authored for cel shading: a ramp texture decides the
shadow step, a sphere map fakes the specular, and an outline sells the drawing.
Imported onto a physically based shader those inputs are either ignored or used
as if they were albedo, which is what makes the character read as clay. Before
replacing the shading, record what is actually there.
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
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

    import bpy  # type: ignore

    from motionviewer.blender.mmd_import import import_pmx_character
    from motionviewer.blender.scene import clear_scene

    clear_scene()
    armature, meshes = import_pmx_character(bpy, args.asset, label="mat", scale=0.08)
    bpy.context.view_layer.update()

    report: dict = {"asset": str(args.asset), "meshes": [], "materials": [], "images": []}
    for mesh in meshes:
        report["meshes"].append(
            {
                "name": mesh.name,
                "polygons": len(mesh.data.polygons),
                "materials": [slot.material.name if slot.material else None for slot in mesh.material_slots],
                "modifiers": [(m.name, m.type) for m in mesh.modifiers],
            }
        )

    for material in bpy.data.materials:
        entry: dict = {
            "name": material.name,
            "use_nodes": bool(material.use_nodes),
            "blend_method": str(getattr(material, "blend_method", "")),
            "use_backface_culling": bool(getattr(material, "use_backface_culling", False)),
            "nodes": [],
            "links": [],
            "mmd": {},
        }
        mmd = getattr(material, "mmd_material", None)
        if mmd is not None:
            entry["mmd"] = {
                "ambient_color": list(getattr(mmd, "ambient_color", ())),
                "diffuse_color": list(getattr(mmd, "diffuse_color", ())),
                "specular_color": list(getattr(mmd, "specular_color", ())),
                "shininess": float(getattr(mmd, "shininess", 0.0) or 0.0),
                "is_double_sided": bool(getattr(mmd, "is_double_sided", False)),
                "enabled_toon_edge": bool(getattr(mmd, "enabled_toon_edge", False)),
                "edge_color": list(getattr(mmd, "edge_color", ())),
                "edge_weight": float(getattr(mmd, "edge_weight", 0.0) or 0.0),
                "sphere_texture_type": str(getattr(mmd, "sphere_texture_type", "")),
                "is_shared_toon_texture": bool(getattr(mmd, "is_shared_toon_texture", False)),
                "toon_texture": str(getattr(mmd, "toon_texture", "") or ""),
            }
        if material.use_nodes:
            for node in material.node_tree.nodes:
                node_entry = {"name": node.name, "type": node.bl_idname, "label": node.label}
                if node.bl_idname == "ShaderNodeTexImage" and node.image is not None:
                    node_entry["image"] = node.image.name
                    node_entry["filepath"] = node.image.filepath
                    node_entry["has_data"] = bool(node.image.has_data)
                node_entry["inputs_linked"] = [
                    socket.name for socket in node.inputs if socket.is_linked
                ]
                entry["nodes"].append(node_entry)
            for link in material.node_tree.links:
                entry["links"].append(
                    f"{link.from_node.name}.{link.from_socket.name} -> {link.to_node.name}.{link.to_socket.name}"
                )
        report["materials"].append(entry)

    for image in bpy.data.images:
        report["images"].append(
            {
                "name": image.name,
                "filepath": image.filepath,
                "size": list(image.size),
                "has_data": bool(image.has_data),
                "colorspace": image.colorspace_settings.name,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}: {len(report['materials'])} materials, {len(report['images'])} images")


if __name__ == "__main__":
    main()
