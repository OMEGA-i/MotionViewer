"""Closed-loop 200Hz MuJoCo physics simulation of Unitree G1 using NVIDIA GEAR-SONIC ONNX policy."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml
from paths import ensure_work, g1_dir, policy_dir
from scipy.spatial.transform import Rotation

G1_DIR = g1_dir()
POLICY_DIR = policy_dir()
CSV_DIR = ensure_work("g1_csv")
OUT_SIM = ensure_work("sim_rollouts")

# Exact constants from gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp
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

# MuJoCo DFS order (29 DOFs)
KPS = np.array([
    STIFF_7520_22, STIFF_7520_22, STIFF_7520_14, STIFF_7520_22, 2*STIFF_5020, 2*STIFF_5020,
    STIFF_7520_22, STIFF_7520_22, STIFF_7520_14, STIFF_7520_22, 2*STIFF_5020, 2*STIFF_5020,
    STIFF_7520_14, 2*STIFF_5020, 2*STIFF_5020,
    STIFF_5020, STIFF_5020, STIFF_5020, STIFF_5020, STIFF_5020, STIFF_4010, STIFF_4010,
    STIFF_5020, STIFF_5020, STIFF_5020, STIFF_5020, STIFF_5020, STIFF_4010, STIFF_4010,
], dtype=np.float64)

KDS = np.array([
    DAMP_7520_22, DAMP_7520_22, DAMP_7520_14, DAMP_7520_22, 2*DAMP_5020, 2*DAMP_5020,
    DAMP_7520_22, DAMP_7520_22, DAMP_7520_14, DAMP_7520_22, 2*DAMP_5020, 2*DAMP_5020,
    DAMP_7520_14, 2*DAMP_5020, 2*DAMP_5020,
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

ACTION_SCALE_MJ = np.array([
    0.25 * EFF_7520_22 / STIFF_7520_22,
    0.25 * EFF_7520_22 / STIFF_7520_22,
    0.25 * EFF_7520_14 / STIFF_7520_14,
    0.25 * EFF_7520_22 / STIFF_7520_22,
    0.25 * EFF_5020 / STIFF_5020,
    0.25 * EFF_5020 / STIFF_5020,
    0.25 * EFF_7520_22 / STIFF_7520_22,
    0.25 * EFF_7520_22 / STIFF_7520_22,
    0.25 * EFF_7520_14 / STIFF_7520_14,
    0.25 * EFF_7520_22 / STIFF_7520_22,
    0.25 * EFF_5020 / STIFF_5020,
    0.25 * EFF_5020 / STIFF_5020,
    0.25 * EFF_7520_14 / STIFF_7520_14,
    0.25 * EFF_5020 / STIFF_5020,
    0.25 * EFF_5020 / STIFF_5020,
    0.25 * EFF_5020 / STIFF_5020,
    0.25 * EFF_5020 / STIFF_5020,
    0.25 * EFF_5020 / STIFF_5020,
    0.25 * EFF_5020 / STIFF_5020,
    0.25 * EFF_5020 / STIFF_5020,
    0.25 * EFF_4010 / STIFF_4010,
    0.25 * EFF_4010 / STIFF_4010,
    0.25 * EFF_5020 / STIFF_5020,
    0.25 * EFF_5020 / STIFF_5020,
    0.25 * EFF_5020 / STIFF_5020,
    0.25 * EFF_5020 / STIFF_5020,
    0.25 * EFF_5020 / STIFF_5020,
    0.25 * EFF_4010 / STIFF_4010,
    0.25 * EFF_4010 / STIFF_4010,
], dtype=np.float64)

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


def build_sim_xml(out_xml: Path) -> Path:
    """Create a clean 200Hz MuJoCo simulation XML using g1_29dof.xml + ground plane + motor actuators."""
    # Check if g1_29dof.xml already has <actuator>; if not, add torque motors
    spec = mujoco.MjSpec.from_file(str(G1_DIR / "g1_29dof.xml"))
    spec.option.timestep = 0.005  # 200 Hz physics
    if not any(g.name == "floor" for g in spec.worldbody.geoms):
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05],
            rgba=[0.85, 0.85, 0.85, 1.0],
            friction=[0.9, 0.005, 0.0001],
        )
    model_tmp = spec.compile()
    if model_tmp.nu == 0:
        for j in range(1, model_tmp.njnt):
            jname = model_tmp.joint(j).name
            act = spec.add_actuator(name=f"act_{jname}", target=jname, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.gainprm[0] = 1.0
    xml_str = spec.to_xml()
    out_xml.write_text(xml_str)
    return out_xml


def load_csv_motion(csv_path: Path):
    with open(csv_path) as fp:
        r = csv.reader(fp)
        next(r)
        raw = np.array([[float(x) for x in row] for row in r], dtype=np.float64)
    # Drop cold-start row(s) just like make_sonic_v1.py
    raw = raw[1:]
    while len(raw) > 4 and np.abs(raw[1, 7:36] - raw[0, 7:36]).max() > 35.0:
        raw = raw[1:]

    # Check root translation units: picks39 CSVs had cm if >10, whereas soma-retargeter new CSVs might be m or cm!
    # Wait: let's inspect raw[0, 3] (root Z)! Standing G1 pelvis Z is ~0.75 m (or ~75 if cm).
    pos_scale = 0.01 if np.median(np.abs(raw[:, 3])) > 5.0 else 1.0
    root_xyz = raw[:, 1:4] * pos_scale
    # Re-zero initial XY so robot starts at (0, 0)
    root_xyz[:, :2] -= root_xyz[0, :2]

    # Root euler XYZ degrees -> quaternion (w,x,y,z) and (x,y,z,w)
    rot_seq = Rotation.from_euler("xyz", raw[:, 4:7], degrees=True)
    quat_xyzw = rot_seq.as_quat()
    quat_wxyz = np.c_[quat_xyzw[:, 3], quat_xyzw[:, :3]]
    for i in range(1, len(quat_wxyz)):
        if np.dot(quat_wxyz[i], quat_wxyz[i - 1]) < 0:
            quat_wxyz[i] = -quat_wxyz[i]
            quat_xyzw[i] = -quat_xyzw[i]

    # Check joint angle units in CSV: degrees vs radians!
    # Wait: if max(abs(raw[:, 7:36])) > 4.0, it's degrees; otherwise radians!
    j_max = float(np.max(np.abs(raw[:, 7:36])))
    j_mj = np.deg2rad(raw[:, 7:36]) if j_max > 4.5 else raw[:, 7:36].copy()
    j_il = j_mj[:, MUJOCO_TO_ISAACLAB]
    jvel_il = np.gradient(j_il, 1.0 / 50.0, axis=0, edge_order=2)
    qpos_ref = np.c_[root_xyz, quat_wxyz, j_mj]
    return qpos_ref, j_il, jvel_il, quat_xyzw, pos_scale, j_max


def main():
    enc_sess = ort.InferenceSession(str(POLICY_DIR / "model_encoder.onnx"), providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(POLICY_DIR / "model_decoder.onnx"), providers=["CPUExecutionProvider"])
    enc_in = enc_sess.get_inputs()[0]
    dec_in = dec_sess.get_inputs()[0]
    print("Encoder ONNX input:", enc_in.name, enc_in.shape, "output:", enc_sess.get_outputs()[0].shape)
    print("Decoder ONNX input:", dec_in.name, dec_in.shape, "output:", dec_sess.get_outputs()[0].shape)

    obs_cfg = yaml.safe_load((POLICY_DIR / "observation_config.yaml").read_text())
    enc_offsets = {}
    off = 0
    for item in obs_cfg["encoder"]["encoder_observations"]:
        if item["enabled"]:
            name = item["name"]
            enc_offsets[name] = (off, off + OBS_DIMS[name])
            off += OBS_DIMS[name]
    print("Computed encoder dim:", off, "motion_anchor_orientation_10frame_step5:", enc_offsets["motion_anchor_orientation_10frame_step5"])
    assert off == enc_in.shape[-1], f"{off} != {enc_in.shape[-1]}"

    dec_offsets = {}
    doff = 0
    for item in obs_cfg["observations"]:
        if item["enabled"]:
            name = item["name"]
            dec_offsets[name] = (doff, doff + OBS_DIMS[name])
            doff += OBS_DIMS[name]
    print("Computed decoder dim:", doff)
    assert doff == dec_in.shape[-1], f"{doff} != {dec_in.shape[-1]}"

    model = mujoco.MjModel.from_xml_path(str(G1_DIR / "scene_29dof.xml"))
    model.opt.timestep = 0.005
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Set exact motor rotor armatures matching policy_parameters.hpp
    armatures = [
        ARMATURE_7520_22, ARMATURE_7520_22, ARMATURE_7520_14, ARMATURE_7520_22, 2*ARMATURE_5020, 2*ARMATURE_5020,
        ARMATURE_7520_22, ARMATURE_7520_22, ARMATURE_7520_14, ARMATURE_7520_22, 2*ARMATURE_5020, 2*ARMATURE_5020,
        ARMATURE_7520_14, 2*ARMATURE_5020, 2*ARMATURE_5020,
        ARMATURE_5020, ARMATURE_5020, ARMATURE_5020, ARMATURE_5020, ARMATURE_5020, ARMATURE_4010, ARMATURE_4010,
        ARMATURE_5020, ARMATURE_5020, ARMATURE_5020, ARMATURE_5020, ARMATURE_5020, ARMATURE_4010, ARMATURE_4010,
    ]
    model.dof_armature[6:35] = armatures
    data = mujoco.MjData(model)
    print(f"MuJoCo model: nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}")

    # Find foot geom IDs to ground reference pose accurately at frame 0
    summary = []
    for csv_file in sorted(CSV_DIR.glob("*.csv")):
        cid = csv_file.stem
        qpos_ref, ref_j_il, ref_jv_il, ref_quat_xyzw, pos_scale, j_max = load_csv_motion(csv_file)
        T = len(qpos_ref)

        # Ground qpos_ref so minimum foot geom across clip touches z=0 cleanly (no penetration)
        data.qpos[:] = qpos_ref[0]
        mujoco.mj_forward(model, data)
        # Check ankle link positions + sole offset
        l_ank = model.body("left_ankle_roll_link").id
        r_ank = model.body("right_ankle_roll_link").id
        sole_z_f0 = min(data.xpos[l_ank, 2], data.xpos[r_ank, 2]) - 0.035
        if sole_z_f0 < 0.002:
            qpos_ref[:, 2] += (0.004 - sole_z_f0)

        # Initialize robot at frame 0
        mujoco.mj_resetData(model, data)
        data.qpos[:] = qpos_ref[0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        # History buffer: 10 frames (oldest to newest, matching StateLogger::GetLatest(10, newest_first=false))
        hist_ang_vel = np.zeros((10, 3), dtype=np.float64)
        hist_jpos = np.zeros((10, 29), dtype=np.float64)
        hist_jvel = np.zeros((10, 29), dtype=np.float64)
        hist_act = np.zeros((10, 29), dtype=np.float64)
        hist_grav = np.zeros((10, 3), dtype=np.float64)
        last_action_il = np.zeros(29, dtype=np.float64)

        def push_history(init_fill=False):
            # Pelvis orientation (w,x,y,z in MuJoCo -> x,y,z,w for scipy)
            qw, qx, qy, qz = data.qpos[3:7]
            r_base = Rotation.from_quat([qx, qy, qz, qw])
            ang_vel_local = data.qvel[3:6].copy()  # freejoint qvel[3:6] is already body-local!
            jpos_il = data.qpos[7:36][MUJOCO_TO_ISAACLAB] - DEFAULT_ANGLES_IL
            jvel_il = data.qvel[6:35][MUJOCO_TO_ISAACLAB]
            grav_local = r_base.inv().apply([0.0, 0.0, -1.0])
            if init_fill:
                hist_ang_vel[:] = ang_vel_local
                hist_jpos[:] = jpos_il
                hist_jvel[:] = jvel_il
                hist_act[:] = last_action_il
                hist_grav[:] = grav_local
            else:
                hist_ang_vel[:-1] = hist_ang_vel[1:]
                hist_ang_vel[-1] = ang_vel_local
                hist_jpos[:-1] = hist_jpos[1:]
                hist_jpos[-1] = jpos_il
                hist_jvel[:-1] = hist_jvel[1:]
                hist_jvel[-1] = jvel_il
                hist_act[:-1] = hist_act[1:]
                hist_act[-1] = last_action_il
                hist_grav[:-1] = hist_grav[1:]
                hist_grav[-1] = grav_local
            return r_base

        # Warmup 15 ticks (0.3s) holding frame 0 so robot settles onto floor contact smoothly
        r_base = push_history(init_fill=True)
        for w_tick in range(15):
            idx10 = np.zeros(10, dtype=int)
            enc_buf = np.zeros((1, off), dtype=np.float32)
            a, b = enc_offsets["encoder_mode_4"]
            enc_buf[0, a] = 0.0
            a, b = enc_offsets["motion_joint_positions_10frame_step5"]
            enc_buf[0, a:b] = ref_j_il[idx10].reshape(-1)
            a, b = enc_offsets["motion_joint_velocities_10frame_step5"]
            enc_buf[0, a:b] = 0.0
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

        # Now track full trajectory T frames at 50 Hz!
        sim_qpos = np.zeros((T, 36), dtype=np.float32)
        fell = False
        fell_frame = -1
        for t in range(T):
            idx10 = np.minimum(t + np.arange(10) * 5, T - 1)
            enc_buf = np.zeros((1, off), dtype=np.float32)
            a, b = enc_offsets["encoder_mode_4"]
            enc_buf[0, a] = 0.0
            a, b = enc_offsets["motion_joint_positions_10frame_step5"]
            enc_buf[0, a:b] = ref_j_il[idx10].reshape(-1)
            a, b = enc_offsets["motion_joint_velocities_10frame_step5"]
            enc_buf[0, a:b] = ref_jv_il[idx10].reshape(-1)
            # Heading-aligned or full base orientation: check C++ default orientation_mode=0 (full base_quat)
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

            up_z = r_base.apply([0.0, 0.0, 1.0])[2]
            if not fell and (data.qpos[2] < 0.28 or up_z < 0.25) and cid != "015265":
                fell = True
                fell_frame = t

        j_mae_deg = float(np.rad2deg(np.mean(np.abs(sim_qpos[:, 7:] - qpos_ref[:, 7:]))))
        sim_span = float(np.max(np.ptp(sim_qpos[:, :2], axis=0)))
        ref_span = float(np.max(np.ptp(qpos_ref[:, :2], axis=0)))
        sim_disp = float(np.linalg.norm(sim_qpos[-1, :2] - sim_qpos[0, :2]))
        z_range = float(np.ptp(sim_qpos[:, 2]))

        np.savez_compressed(
            OUT_SIM / f"{cid}.npz",
            sim_qpos=sim_qpos,
            ref_qpos=qpos_ref.astype(np.float32),
            fps=np.float32(50.0),
            fell=np.bool_(fell),
            j_mae_deg=np.float32(j_mae_deg),
        )
        rec = {
            "cid": cid,
            "T": T,
            "fell": fell,
            "fell_frame": fell_frame,
            "j_mae_deg": round(j_mae_deg, 2),
            "sim_span_m": round(sim_span, 2),
            "ref_span_m": round(ref_span, 2),
            "sim_disp_m": round(sim_disp, 2),
            "z_range_m": round(z_range, 2),
            "pos_scale": pos_scale,
            "j_max_raw": round(j_max, 2),
        }
        summary.append(rec)
        status = "OK  " if not fell else f"FELL@{fell_frame}"
        print(f"[{status}] {cid:12s} T={T:4d} MAE={j_mae_deg:5.2f}deg sim_span={sim_span:4.2f}m (ref={ref_span:4.2f}m) z_rng={z_range:4.2f}m")

    (OUT_SIM / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
