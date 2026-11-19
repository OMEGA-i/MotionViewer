"""Convert curated SOMA-77 clips to oriented SOMA BVHs and retarget to Unitree G1 via NVIDIA soma-retargeter."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from paths import ensure_work, picks39_csv_dir, soma_active_dir, soma_orient_path
from scipy.spatial.transform import Rotation

SOMA_DIR = soma_active_dir()
ORIENT_PATH = soma_orient_path()
PICKS39_CSV = picks39_csv_dir()

OUT_BVH = ensure_work("bvh_oriented")
OUT_CSV = ensure_work("g1_csv")

CANDIDATES_DEMO400 = [
    "004162",  # walk forward & stand still (user reviewed)
    "018149",  # slow walk swaying arms 3.53m (user reviewed)
    "039936",  # stand -> run forward 3.24m (user reviewed)
    "114905",  # crossover sideways walk 2.84m (user reviewed)
    "044005",  # sideways jump left & stand 3.08m
    "038806",  # walk -> jog transition 3.23m
    "116153",  # sit -> stand up
    "015265",  # crawl on ground -> rise to stand
    "037748",  # kneel on one knee -> rise & palm strike
    "058466",  # stand -> bend forward & touch ground
    "016783",  # carry object forward two steps & stop (user reviewed)
    "013842",  # walk while carrying object in one hand 4.93m
    "046600",  # martial arts stance, shift weight & punch (user reviewed)
    "090589",  # expressive dance, wide arm sweep & spin (user reviewed)
    "100948",  # energetic dance (user reviewed)
    "061377",  # deep squat -> stand & lift overhead
]

CANDIDATES_PICKS39 = [
    "000177",  # straight walk 4.96m
    "000170",  # curved walk turning left 4.82m
    "000182",  # step forward, bend down, pick up & place on left
    "000211",  # bend down to pick up object
    "028509",  # retro dance lateral 1.68m
    "045969",  # steady walk 2.41m
    "011124",  # martial arts squat -> high leg kick -> lunge thrust
    "131585",  # funky hip-hop dance & stop
]


def euler_zyx_deg(R: np.ndarray) -> np.ndarray:
    rot = Rotation.from_matrix(R.reshape(-1, 3, 3))
    return rot.as_euler("ZYX", degrees=True).reshape(R.shape[:-2] + (3,)).astype(np.float64)


def recover_rest_joints(joints: np.ndarray, rotmat77: np.ndarray, parents: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    T, J, _ = joints.shape
    world = np.empty_like(rotmat77, dtype=np.float64)
    for j in range(J):
        p = int(parents[j])
        world[:, j] = rotmat77[:, j] if p < 0 else np.einsum("tij,tjk->tik", world[:, p], rotmat77[:, j])
    rest = np.zeros((J, 3), dtype=np.float64)
    # Match Hips rest height ~1.01m (initial root Y)
    rest[0] = np.array([0.0, float(joints[0, 0, 1]), 0.0], dtype=np.float64)
    for j in range(1, J):
        p = int(parents[j])
        delta = joints[:, j] - joints[:, p]
        norm = np.linalg.norm(delta, axis=1, keepdims=True)
        obs = delta / np.maximum(norm, 1e-12)
        local_dir = np.einsum("tji,tj->ti", world[:, p], obs).mean(axis=0)
        ln = np.linalg.norm(local_dir)
        d_unit = local_dir / ln if ln > 1e-12 else np.array([0.0, 1.0, 0.0])
        rest[j] = rest[p] + d_unit * float(lengths[j])
    return rest


def export_oriented_bvh(soma_npz: Path, out_bvh: Path, O78: np.ndarray) -> dict:
    d = np.load(soma_npz, allow_pickle=True)
    joints = np.asarray(d["joints77"], dtype=np.float64)
    rotmat = np.asarray(d["rotmat77"], dtype=np.float64)
    transl = np.asarray(d["transl"], dtype=np.float64)
    lengths = np.asarray(d["bone_lengths"], dtype=np.float64)
    names77 = [str(x) for x in d["joint_names"]]
    parents77 = np.asarray(d["joint_parents"], dtype=int)
    fps = float(d["fps"])
    T = len(joints)

    rest77 = recover_rest_joints(joints, rotmat, parents77, lengths)

    # Build 78-joint hierarchy (0=Root, 1..77=SOMA77)
    names78 = ["Root"] + names77
    parents78 = [-1] + [p + 1 for p in parents77]  # Hips(-1 -> 0)

    # Plain offsets in cm
    off_plain = np.zeros((78, 3), dtype=np.float64)
    off_plain[1] = rest77[0] * 100.0
    for j in range(1, 77):
        p = int(parents77[j])
        off_plain[j + 1] = (rest77[j] - rest77[p]) * 100.0

    # Oriented offsets: off'(k) = O[parent(k)]^T @ off_plain(k)
    off_oriented = np.zeros_like(off_plain)
    for k in range(1, 78):
        pk = parents78[k]
        off_oriented[k] = O78[pk].T @ off_plain[k]

    # Oriented local rotations: L'(k) = O[parent(k)]^T @ L(k) @ O[k]
    L_oriented = np.zeros((T, 78, 3, 3), dtype=np.float64)
    L_oriented[:, 0] = np.eye(3)
    for k in range(1, 78):
        pk = parents78[k]
        L_plain = rotmat[:, k - 1]
        L_oriented[:, k] = np.einsum("ij,tjk,kl->til", O78[pk].T, L_plain, O78[k])

    eul = euler_zyx_deg(L_oriented)  # (T, 78, 3)
    hips_cm = transl * 100.0  # Root has O[0]=I, so Hips translation in Root frame is unchanged

    children: dict[int, list[int]] = {}
    for k, pk in enumerate(parents78):
        children.setdefault(pk, []).append(k)

    lines = ["HIERARCHY"]
    def emit(k: int, depth: int):
        ind = "  " * depth
        kind = "ROOT" if k == 0 else "JOINT"
        o = off_oriented[k]
        lines.append(f"{ind}{kind} {names78[k]}")
        lines.append(f"{ind}{{")
        lines.append(f"{ind}  OFFSET {o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
        if k in (0, 1):
            lines.append(f"{ind}  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
        else:
            lines.append(f"{ind}  CHANNELS 3 Zrotation Yrotation Xrotation")
        for c in children.get(k, []):
            emit(c, depth + 1)
        lines.append(f"{ind}}}")

    emit(0, 0)
    lines.append("MOTION")
    lines.append(f"Frames: {T}")
    lines.append(f"Frame Time: {1.0 / fps:.6f}")
    for t in range(T):
        vals = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        vals += [hips_cm[t, 0], hips_cm[t, 1], hips_cm[t, 2], *eul[t, 1]]
        for k in range(2, 78):
            vals += list(eul[t, k])
        lines.append(" ".join(f"{v:.6f}" for v in vals))
    out_bvh.write_text("\n".join(lines) + "\n")
    return {"stem": soma_npz.stem.replace(".soma", ""), "T": T, "fps": fps, "caption": str(d["caption"])}


def main():
    orient_data = np.load(ORIENT_PATH)
    O78 = np.asarray(orient_data["orient"], dtype=np.float64)

    exported = []
    for stem in CANDIDATES_DEMO400:
        p = SOMA_DIR / f"{stem}.soma.npz"
        if not p.exists():
            print("Missing:", p)
            continue
        bvh_p = OUT_BVH / f"{stem}.bvh"
        info = export_oriented_bvh(p, bvh_p, O78)
        exported.append(info)
        print(f"Exported oriented BVH: {stem} (T={info['T']}) | {info['caption'][:65]}")

    # Run soma-retargeter NewtonPipeline on GPU
    import soma_retargeter.assets.bvh as bvh_utils
    import soma_retargeter.assets.csv as csv_utils
    import soma_retargeter.pipelines.newton_pipeline as newton_pipeline
    import warp as wp
    from soma_retargeter.utils.space_conversion_utils import (
        SpaceConverter,
        get_facing_direction_type_from_str,
    )

    wp.init()
    with wp.ScopedDevice("cuda:0"):
        bvh_files = sorted(OUT_BVH.glob("*.bvh"))
        importer = bvh_utils.BVHImporter()
        ref_skel, _ = importer.create_skeleton(bvh_files[0])
        # Verify FK of first BVH against source joints77!
        _, test_anim = bvh_utils.load_bvh(bvh_files[0], ref_skel)
        g0 = test_anim.compute_global_transforms(0)
        d0 = np.load(SOMA_DIR / f"{bvh_files[0].stem}.soma.npz")
        fk_err = np.max(np.linalg.norm(g0[1:, :3] - d0["joints77"][0], axis=1))
        print(f"[FK CHECK] {bvh_files[0].stem} frame-0 max joint position error vs joints77: {fk_err*1000:.3f} mm")

        converter = SpaceConverter(get_facing_direction_type_from_str("Mujoco"))
        tx = converter.transform(wp.transform_identity())
        pipeline = newton_pipeline.NewtonPipeline(ref_skel, "soma", "unitree_g1")

        # Retarget one by one so each clip uses its own skeleton bone lengths!
        for bf in bvh_files:
            sk_i, anim_i = bvh_utils.load_bvh(bf)
            pipeline.human_robot_scaler.skeleton = sk_i
            pipeline._landmark_model = None
            pipeline.clear()
            pipeline.add_input_motions([anim_i], [tx], True)
            out_buf = pipeline.execute()[0]
            csv_utils.save_csv(OUT_CSV / f"{bf.stem}.csv", out_buf)
            print(f"[RETARGETED] {bf.stem} -> {OUT_CSV / (bf.stem + '.csv')} (frames={out_buf.num_frames})")

    # Also copy picks39 pre-retargeted CSVs
    for stem in CANDIDATES_PICKS39:
        src = PICKS39_CSV / f"{stem}.csv"
        if src.exists():
            shutil.copy2(src, OUT_CSV / f"p39_{stem}.csv")
            print(f"[COPIED] p39_{stem}.csv")

if __name__ == "__main__":
    main()
