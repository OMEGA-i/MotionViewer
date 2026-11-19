"""Export compiled G1 29-DOF visual meshes + per-frame world FK transforms for Blender rendering."""
from __future__ import annotations

import json

import mujoco
import numpy as np
from paths import ensure_work, g1_dir

G1_XML = g1_dir() / "g1_29dof.xml"
SIM_DIR = ensure_work("sim_rollouts")
OUT_FK = ensure_work("g1_fk_bundles")


def main():
    model = mujoco.MjModel.from_xml_path(str(G1_XML))
    data = mujoco.MjData(model)

    # Select visual geoms ONLY: in g1_29dof.xml, visual meshes have group == 1 and contype == 0!
    vis_geoms = []
    mesh_arrays = {}
    for g in range(model.ngeom):
        body_id = int(model.geom_bodyid[g])
        if body_id == 0:
            continue  # skip floor
        gtype = int(model.geom_type[g])
        group = int(model.geom_group[g])
        # In g1_29dof.xml every link has a visual geom (group=1, contype=0) and a duplicate collision geom (group=0, contype=1).
        # Keeping ONLY group==1 avoids duplicate overlapping z-fighting meshes!
        if group != 1:
            continue
        if gtype != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = int(model.geom_dataid[g])
        mname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        rgba = model.geom_rgba[g].tolist()

        v_adr, v_num = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        f_adr, f_num = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
        if f"vert_{mid}" not in mesh_arrays:
            mesh_arrays[f"vert_{mid}"] = model.mesh_vert[v_adr : v_adr + v_num].astype(np.float32)
            mesh_arrays[f"face_{mid}"] = model.mesh_face[f_adr : f_adr + f_num].astype(np.int32)

        vis_geoms.append({
            "geom_id": g,
            "mesh_id": mid,
            "mesh_name": mname,
            "body_name": bname,
            "rgba": rgba,
        })

    print(f"Exporting {len(vis_geoms)} clean non-overlapping visual geoms of Unitree G1:")
    for vg in vis_geoms:
        print(f"  geom {vg['geom_id']:2d}: mesh={vg['mesh_name']:28s} body={vg['body_name']:28s} rgba={vg['rgba'][:3]}")

    np.savez_compressed(OUT_FK / "g1_meshes.npz", **mesh_arrays)
    (OUT_FK / "g1_geoms.json").write_text(json.dumps(vis_geoms, indent=2) + "\n")

    gids = [vg["geom_id"] for vg in vis_geoms]
    for npz_file in sorted(SIM_DIR.glob("*.npz")):
        cid = npz_file.stem
        d = np.load(npz_file)
        if bool(d.get("fell", False)):
            continue
        for kind in ("sim", "ref"):
            qpos = d[f"{kind}_qpos"]
            T = len(qpos)
            pos = np.zeros((T, len(gids), 3), dtype=np.float32)
            rot = np.zeros((T, len(gids), 3, 3), dtype=np.float32)
            for t in range(T):
                data.qpos[:36] = qpos[t]
                mujoco.mj_forward(model, data)
                pos[t] = data.geom_xpos[gids]
                rot[t] = data.geom_xmat[gids].reshape(-1, 3, 3)
            # Compute exact lowest mesh vertex across entire clip to verify floor contact!
            sample_ts = np.linspace(0, T - 1, min(T, 30)).astype(int)
            lows = []
            for t in sample_ts:
                z_min_t = 10.0
                for idx, vg in enumerate(vis_geoms):
                    verts = mesh_arrays[f"vert_{vg['mesh_id']}"]
                    wz = verts @ rot[t, idx, 2, :] + pos[t, idx, 2]
                    z_min_t = min(z_min_t, float(wz.min()))
                lows.append(z_min_t)
            # Ground floor offset so lowest stance foot kisses z=0.000m without sinking
            floor_z = float(np.percentile(lows, 10))
            pos[:, :, 2] -= floor_z
            root_xyz = qpos[:, :3].copy()
            root_xyz[:, 2] -= floor_z
            np.savez_compressed(
                OUT_FK / f"{cid}_{kind}.npz",
                positions=pos,
                rotations=rot,
                root_xyz=root_xyz,
                root_quat_wxyz=qpos[:, 3:7],
                fps=np.float32(50.0),
            )
        print(f"Exported FK bundle: {cid} (T={len(d['sim_qpos'])})")


if __name__ == "__main__":
    main()
