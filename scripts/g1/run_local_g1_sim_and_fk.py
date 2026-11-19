"""Run local 200Hz GEAR-SONIC closed-loop MuJoCo simulation + 36-mesh visual FK export for Unitree G1 videos."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml
from paths import WORK, ensure_work, g1_dir, policy_dir
from scipy.spatial.transform import Rotation

G1_DIR = g1_dir()
POLICY_DIR = policy_dir()
CSV_DIR = ensure_work("g1_csv")
OUT_SIM = ensure_work("sim_rollouts")
OUT_FK = ensure_work("g1_fk_bundles")

ISAACLAB_TO_MUJOCO = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
                      11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
MUJOCO_TO_ISAACLAB = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
                      16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
OMEGA = 10.0 * 2.0 * math.pi
ZETA = 2.0

STIFF_5020 = ARMATURE_5020 * OMEGA * OMEGA
STIFF_7520_14 = ARMATURE_7520_14 * OMEGA * OMEGA
STIFF_7520_22 = ARMATURE_7520_22 * OMEGA * OMEGA
STIFF_4010 = ARMATURE_4010 * OMEGA * OMEGA

DAMP_5020 = 2.0 * ZETA * ARMATURE_5020 * OMEGA
DAMP_7520_14 = 2.0 * ZETA * ARMATURE_7520_14 * OMEGA
DAMP_7520_22 = 2.0 * ZETA * ARMATURE_7520_22 * OMEGA
DAMP_4010 = 2.0 * ZETA * ARMATURE_4010 * OMEGA

EFF_5020 = 25.0
EFF_7520_14 = 88.0
EFF_7520_22 = 139.0
EFF_4010 = 5.0

KPS = np.array([
    STIFF_7520_22, STIFF_7520_22, STIFF_7520_14, STIFF_7520_22, 2 * STIFF_5020, 2 * STIFF_5020,
    STIFF_7520_22, STIFF_7520_22, STIFF_7520_14, STIFF_7520_22, 2 * STIFF_5020, 2 * STIFF_5020,
    STIFF_7520_14, 2 * STIFF_5020, 2 * STIFF_5020,
    STIFF_5020, STIFF_5020, STIFF_5020, STIFF_5020, STIFF_5020, STIFF_4010, STIFF_4010,
    STIFF_5020, STIFF_5020, STIFF_5020, STIFF_5020, STIFF_5020, STIFF_4010, STIFF_4010,
], dtype=np.float64)

KDS = np.array([
    DAMP_7520_22, DAMP_7520_22, DAMP_7520_14, DAMP_7520_22, 2 * DAMP_5020, 2 * DAMP_5020,
    DAMP_7520_22, DAMP_7520_22, DAMP_7520_14, DAMP_7520_22, 2 * DAMP_5020, 2 * DAMP_5020,
    DAMP_7520_14, 2 * DAMP_5020, 2 * DAMP_5020,
    DAMP_5020, DAMP_5020, DAMP_5020, DAMP_5020, DAMP_5020, DAMP_4010, DAMP_4010,
    DAMP_5020, DAMP_5020, DAMP_5020, DAMP_5020, DAMP_5020, DAMP_4010, DAMP_4010,
], dtype=np.float64)

EFFORT_LIMITS = np.array([
    EFF_7520_22, EFF_7520_22, EFF_7520_14, EFF_7520_22, 50.0, 50.0,
    EFF_7520_22, EFF_7520_22, EFF_7520_14, EFF_7520_22, 50.0, 50.0,
    EFF_7520_14, 50.0, 50.0,
    EFF_5020, EFF_5020, EFF_5020, EFF_5020, EFF_5020, EFF_4010, EFF_4010,
    EFF_5020, EFF_5020, EFF_5020, EFF_5020, EFF_5020, EFF_4010, EFF_4010,
], dtype=np.float64)

ACTION_SCALE_MJ = 0.25 * EFFORT_LIMITS / KPS
# Restore exact ankle/waist scales matching policy_parameters.hpp where EFF_5020 (25) != 50 limit
for idx in (4, 5, 10, 11, 13, 14):
    ACTION_SCALE_MJ[idx] = 0.25 * EFF_5020 / STIFF_5020

DEFAULT_ANGLES_MJ = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float64)
DEFAULT_ANGLES_IL = DEFAULT_ANGLES_MJ[MUJOCO_TO_ISAACLAB]

OBS_DIMS = {
    "token_state": 64,
    "his_base_angular_velocity_10frame_step1": 30,
    "his_body_joint_positions_10frame_step1": 290,
    "his_body_joint_velocities_10frame_step1": 290,
    "his_last_actions_10frame_step1": 290,
    "his_gravity_dir_10frame_step1": 30,
    "encoder_mode_4": 4,
    "motion_joint_positions_10frame_step5": 290,
    "motion_joint_velocities_10frame_step5": 290,
    "motion_root_z_position_10frame_step5": 10,
    "motion_root_z_position": 1,
    "motion_anchor_orientation": 6,
    "motion_anchor_orientation_10frame_step5": 60,
    "motion_joint_positions_lowerbody_10frame_step5": 120,
    "motion_joint_velocities_lowerbody_10frame_step5": 120,
    "vr_3point_local_target": 9,
    "vr_3point_local_orn_target": 12,
    "smpl_joints_10frame_step1": 720,
    "smpl_anchor_orientation_10frame_step1": 60,
    "motion_joint_positions_wrists_10frame_step1": 60,
}

# Dynamic non-walking candidates from picks39 (martial arts, hip-hop dance, squat jump, athletic skips, retro dance, pick-place)
TARGET_CIDS = ["011124", "131585", "075931", "123798", "028509", "000182", "034016", "000003"]


def load_csv_motion(csv_path: Path):
    with open(csv_path) as fp:
        r = csv.reader(fp)
        next(r)
        raw = np.array([[float(x) for x in row] for row in r], dtype=np.float64)
    raw = raw[1:]
    while len(raw) > 4 and np.abs(raw[1, 7:36] - raw[0, 7:36]).max() > 35.0:
        raw = raw[1:]
    pos_scale = 0.01 if np.median(np.abs(raw[:, 3])) > 5.0 else 1.0
    root_xyz = raw[:, 1:4] * pos_scale
    root_xyz[:, :2] -= root_xyz[0, :2]

    rot_seq = Rotation.from_euler("xyz", raw[:, 4:7], degrees=True)
    quat_xyzw = rot_seq.as_quat()
    quat_wxyz = np.c_[quat_xyzw[:, 3], quat_xyzw[:, :3]]
    for i in range(1, len(quat_wxyz)):
        if np.dot(quat_wxyz[i], quat_wxyz[i - 1]) < 0:
            quat_wxyz[i] = -quat_wxyz[i]
            quat_xyzw[i] = -quat_xyzw[i]

    j_max = float(np.max(np.abs(raw[:, 7:36])))
    j_mj = np.deg2rad(raw[:, 7:36]) if j_max > 4.5 else raw[:, 7:36].copy()
    j_il = j_mj[:, MUJOCO_TO_ISAACLAB]
    jvel_il = np.gradient(j_il, 1.0 / 50.0, axis=0, edge_order=2)
    qpos_ref = np.c_[root_xyz, quat_wxyz, j_mj]
    return qpos_ref, j_il, jvel_il, quat_xyzw


def main() -> None:
    enc_sess = ort.InferenceSession(str(POLICY_DIR / "model_encoder.onnx"), providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(POLICY_DIR / "model_decoder.onnx"), providers=["CPUExecutionProvider"])
    enc_in = enc_sess.get_inputs()[0]
    dec_in = dec_sess.get_inputs()[0]

    obs_cfg = yaml.safe_load((POLICY_DIR / "observation_config.yaml").read_text())
    enc_offsets, off = {}, 0
    for item in obs_cfg["encoder"]["encoder_observations"]:
        if item["enabled"]:
            enc_offsets[item["name"]] = (off, off + OBS_DIMS[item["name"]])
            off += OBS_DIMS[item["name"]]
    dec_offsets, doff = {}, 0
    for item in obs_cfg["observations"]:
        if item["enabled"]:
            dec_offsets[item["name"]] = (doff, doff + OBS_DIMS[item["name"]])
            doff += OBS_DIMS[item["name"]]

    scene_xml = G1_DIR / "scene_29dof.xml"
    if not scene_xml.exists():
        spec = mujoco.MjSpec.from_file(str(G1_DIR / "g1_29dof.xml"))
        spec.option.timestep = 0.005
        if not any(g.name == "floor" for g in spec.worldbody.geoms):
            spec.worldbody.add_geom(
                name="floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[0, 0, 0.05],
                rgba=[0.85, 0.85, 0.85, 1.0],
                friction=[0.9, 0.005, 0.0001],
            )
        m_tmp = spec.compile()
        if m_tmp.nu == 0:
            for j in range(1, m_tmp.njnt):
                jname = m_tmp.joint(j).name
                act = spec.add_actuator(name=f"act_{jname}", target=jname, trntype=mujoco.mjtTrn.mjTRN_JOINT)
                act.gainprm[0] = 1.0
        scene_xml = WORK / "scene_29dof_gen.xml"
        scene_xml.write_text(spec.to_xml())

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    model.opt.timestep = 0.005
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.dof_armature[6:35] = [
        ARMATURE_7520_22, ARMATURE_7520_22, ARMATURE_7520_14, ARMATURE_7520_22, 2 * ARMATURE_5020, 2 * ARMATURE_5020,
        ARMATURE_7520_22, ARMATURE_7520_22, ARMATURE_7520_14, ARMATURE_7520_22, 2 * ARMATURE_5020, 2 * ARMATURE_5020,
        ARMATURE_7520_14, 2 * ARMATURE_5020, 2 * ARMATURE_5020,
        ARMATURE_5020, ARMATURE_5020, ARMATURE_5020, ARMATURE_5020, ARMATURE_5020, ARMATURE_4010, ARMATURE_4010,
        ARMATURE_5020, ARMATURE_5020, ARMATURE_5020, ARMATURE_5020, ARMATURE_5020, ARMATURE_4010, ARMATURE_4010,
    ]
    data = mujoco.MjData(model)

    # Export visual geoms & meshes once
    vis_geoms = []
    mesh_arrays = {}
    for g in range(model.ngeom):
        body_id = int(model.geom_bodyid[g])
        if body_id == 0 or int(model.geom_group[g]) != 1 or int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = int(model.geom_dataid[g])
        mname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        v_adr, v_num = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        f_adr, f_num = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
        if f"vert_{mid}" not in mesh_arrays:
            mesh_arrays[f"vert_{mid}"] = model.mesh_vert[v_adr : v_adr + v_num].astype(np.float32)
            mesh_arrays[f"face_{mid}"] = model.mesh_face[f_adr : f_adr + f_num].astype(np.int32)
        vis_geoms.append({"geom_id": g, "mesh_id": mid, "mesh_name": mname, "body_name": bname, "rgba": model.geom_rgba[g].tolist()})

    np.savez_compressed(OUT_FK / "g1_meshes.npz", **mesh_arrays)
    (OUT_FK / "g1_geoms.json").write_text(json.dumps(vis_geoms, indent=2) + "\n")
    gids = [vg["geom_id"] for vg in vis_geoms]

    for cid in TARGET_CIDS:
        csv_file = CSV_DIR / f"{cid}.csv"
        if not csv_file.exists():
            continue
        qpos_ref, ref_j_il, ref_jv_il, ref_quat_xyzw = load_csv_motion(csv_file)
        T = len(qpos_ref)

        data.qpos[:] = qpos_ref[0]
        mujoco.mj_forward(model, data)
        l_ank = model.body("left_ankle_roll_link").id
        r_ank = model.body("right_ankle_roll_link").id
        sole_z_f0 = min(data.xpos[l_ank, 2], data.xpos[r_ank, 2]) - 0.035
        if sole_z_f0 < 0.002:
            qpos_ref[:, 2] += 0.004 - sole_z_f0

        mujoco.mj_resetData(model, data)
        data.qpos[:] = qpos_ref[0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        hist_ang_vel = np.zeros((10, 3), dtype=np.float64)
        hist_jpos = np.zeros((10, 29), dtype=np.float64)
        hist_jvel = np.zeros((10, 29), dtype=np.float64)
        hist_act = np.zeros((10, 29), dtype=np.float64)
        hist_grav = np.zeros((10, 3), dtype=np.float64)
        last_action_il = np.zeros(29, dtype=np.float64)

        def push_history(init_fill=False):
            qw, qx, qy, qz = data.qpos[3:7]
            rb = Rotation.from_quat([qx, qy, qz, qw])
            avl = data.qvel[3:6].copy()
            jp = data.qpos[7:36][MUJOCO_TO_ISAACLAB] - DEFAULT_ANGLES_IL
            jv = data.qvel[6:35][MUJOCO_TO_ISAACLAB]
            gl = rb.inv().apply([0.0, 0.0, -1.0])
            if init_fill:
                hist_ang_vel[:] = avl
                hist_jpos[:] = jp
                hist_jvel[:] = jv
                hist_act[:] = last_action_il
                hist_grav[:] = gl
            else:
                hist_ang_vel[:-1] = hist_ang_vel[1:]
                hist_ang_vel[-1] = avl
                hist_jpos[:-1] = hist_jpos[1:]
                hist_jpos[-1] = jp
                hist_jvel[:-1] = hist_jvel[1:]
                hist_jvel[-1] = jv
                hist_act[:-1] = hist_act[1:]
                hist_act[-1] = last_action_il
                hist_grav[:-1] = hist_grav[1:]
                hist_grav[-1] = gl
            return rb

        r_base = push_history(init_fill=True)
        for _ in range(15):
            idx10 = np.zeros(10, dtype=int)
            enc_buf = np.zeros((1, off), dtype=np.float32)
            a, b = enc_offsets["motion_joint_positions_10frame_step5"]
            enc_buf[0, a:b] = ref_j_il[idx10].reshape(-1)
            rel_rot = (r_base.inv() * Rotation.from_quat(ref_quat_xyzw[idx10])).as_matrix()[..., :2].reshape(-1)
            a, b = enc_offsets["motion_anchor_orientation_10frame_step5"]
            enc_buf[0, a:b] = rel_rot
            token = enc_sess.run(None, {enc_in.name: enc_buf})[0]

            dec_buf = np.zeros((1, doff), dtype=np.float32)
            dec_buf[0, dec_offsets["token_state"][0]:dec_offsets["token_state"][1]] = token[0]
            dec_buf[0, dec_offsets["his_base_angular_velocity_10frame_step1"][0]:dec_offsets["his_base_angular_velocity_10frame_step1"][1]] = hist_ang_vel.reshape(-1)
            dec_buf[0, dec_offsets["his_body_joint_positions_10frame_step1"][0]:dec_offsets["his_body_joint_positions_10frame_step1"][1]] = hist_jpos.reshape(-1)
            dec_buf[0, dec_offsets["his_body_joint_velocities_10frame_step1"][0]:dec_offsets["his_body_joint_velocities_10frame_step1"][1]] = hist_jvel.reshape(-1)
            dec_buf[0, dec_offsets["his_last_actions_10frame_step1"][0]:dec_offsets["his_last_actions_10frame_step1"][1]] = hist_act.reshape(-1)
            dec_buf[0, dec_offsets["his_gravity_dir_10frame_step1"][0]:dec_offsets["his_gravity_dir_10frame_step1"][1]] = hist_grav.reshape(-1)

            act_il = dec_sess.run(None, {dec_in.name: dec_buf})[0][0]
            last_action_il = act_il.astype(np.float64)
            q_target_mj = DEFAULT_ANGLES_MJ + act_il[ISAACLAB_TO_MUJOCO] * ACTION_SCALE_MJ
            for _ in range(4):
                tau = KPS * (q_target_mj - data.qpos[7:36]) - KDS * data.qvel[6:35]
                data.ctrl[:] = np.clip(tau, -EFFORT_LIMITS, EFFORT_LIMITS)
                mujoco.mj_step(model, data)
            r_base = push_history(init_fill=False)

        sim_qpos = np.zeros((T, 36), dtype=np.float32)
        fell = False
        for t in range(T):
            idx10 = np.minimum(t + np.arange(10) * 5, T - 1)
            enc_buf = np.zeros((1, off), dtype=np.float32)
            a, b = enc_offsets["motion_joint_positions_10frame_step5"]
            enc_buf[0, a:b] = ref_j_il[idx10].reshape(-1)
            a, b = enc_offsets["motion_joint_velocities_10frame_step5"]
            enc_buf[0, a:b] = ref_jv_il[idx10].reshape(-1)
            rel_rot = (r_base.inv() * Rotation.from_quat(ref_quat_xyzw[idx10])).as_matrix()[..., :2].reshape(-1)
            a, b = enc_offsets["motion_anchor_orientation_10frame_step5"]
            enc_buf[0, a:b] = rel_rot
            token = enc_sess.run(None, {enc_in.name: enc_buf})[0]

            dec_buf = np.zeros((1, doff), dtype=np.float32)
            dec_buf[0, dec_offsets["token_state"][0]:dec_offsets["token_state"][1]] = token[0]
            dec_buf[0, dec_offsets["his_base_angular_velocity_10frame_step1"][0]:dec_offsets["his_base_angular_velocity_10frame_step1"][1]] = hist_ang_vel.reshape(-1)
            dec_buf[0, dec_offsets["his_body_joint_positions_10frame_step1"][0]:dec_offsets["his_body_joint_positions_10frame_step1"][1]] = hist_jpos.reshape(-1)
            dec_buf[0, dec_offsets["his_body_joint_velocities_10frame_step1"][0]:dec_offsets["his_body_joint_velocities_10frame_step1"][1]] = hist_jvel.reshape(-1)
            dec_buf[0, dec_offsets["his_last_actions_10frame_step1"][0]:dec_offsets["his_last_actions_10frame_step1"][1]] = hist_act.reshape(-1)
            dec_buf[0, dec_offsets["his_gravity_dir_10frame_step1"][0]:dec_offsets["his_gravity_dir_10frame_step1"][1]] = hist_grav.reshape(-1)

            act_il = dec_sess.run(None, {dec_in.name: dec_buf})[0][0]
            last_action_il = act_il.astype(np.float64)
            q_target_mj = DEFAULT_ANGLES_MJ + act_il[ISAACLAB_TO_MUJOCO] * ACTION_SCALE_MJ
            for _ in range(4):
                tau = KPS * (q_target_mj - data.qpos[7:36]) - KDS * data.qvel[6:35]
                data.ctrl[:] = np.clip(tau, -EFFORT_LIMITS, EFFORT_LIMITS)
                mujoco.mj_step(model, data)
            sim_qpos[t] = data.qpos[:36]
            r_base = push_history(init_fill=False)
            if data.qpos[2] < 0.28 or r_base.apply([0.0, 0.0, 1.0])[2] < 0.25:
                fell = True

        j_mae = float(np.rad2deg(np.mean(np.abs(sim_qpos[:, 7:] - qpos_ref[:, 7:]))))
        print(f"[{'FELL' if fell else 'OK  '}] {cid}: T={T} ({T/50:.2f}s) MAE={j_mae:.2f} deg")
        if not fell:
            pos = np.zeros((T, len(gids), 3), dtype=np.float32)
            rot = np.zeros((T, len(gids), 3, 3), dtype=np.float32)
            for t in range(T):
                data.qpos[:36] = sim_qpos[t]
                mujoco.mj_forward(model, data)
                pos[t] = data.geom_xpos[gids]
                rot[t] = data.geom_xmat[gids].reshape(-1, 3, 3)
            lows = []
            for t in np.linspace(0, T - 1, min(T, 30)).astype(int):
                z_min_t = min(float((mesh_arrays[f"vert_{vg['mesh_id']}"] @ rot[t, idx, 2, :] + pos[t, idx, 2]).min()) for idx, vg in enumerate(vis_geoms))
                lows.append(z_min_t)
            floor_z = float(np.percentile(lows, 10))
            pos[:, :, 2] -= floor_z
            root_xyz = sim_qpos[:, :3].copy()
            root_xyz[:, 2] -= floor_z
            np.savez_compressed(OUT_FK / f"{cid}_sim.npz", positions=pos, rotations=rot, root_xyz=root_xyz, fps=np.float32(50.0))


if __name__ == "__main__":
    main()
