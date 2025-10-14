#!/usr/bin/env python3
"""
20s walking preview using a simple IK-based tripod controller (no external CPG).

Params:
- step rate: 1.0 Hz
- stride: 0.032 m forward, lift: 0.020 m (ramps in after 1 s)
- stance: slide back during stance with 1.2 mm preload ramping in over 0.5 s

Run (macOS):
  ~/.local/mujoco-viewer-venv/bin/mjpython robotics_repos/jethexa/jethexa_mj_lab/scripts/manual_tripod_preview.py
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List
import argparse
import numpy as np

import mujoco as mj
import mujoco.viewer as viewer

HERE = os.path.dirname(os.path.abspath(__file__))
XML_REL = "../mjcf/jethexa_lab.xml"
DEFAULT_XML_PATH = os.path.abspath(os.path.join(HERE, XML_REL))
CALIB_DIR = os.path.join(HERE, "calibration")
CALIB_PATH = os.path.join(CALIB_DIR, "neutral_offsets.json")
SIGN_PATH = os.path.join(CALIB_DIR, "calibrated_signs.json")

LEG_ORDER = ["LF","LM","LR","RR","RM","RF"]


def actuator_index_by_name(model: mj.MjModel) -> Dict[str, int]:
    mjt = mj.mjtObj
    names = {}
    for i in range(model.nu):
        nm = mj.mj_id2name(model, mjt.mjOBJ_ACTUATOR, i)
        names[nm] = i
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml', type=str, default=DEFAULT_XML_PATH, help='Path to MJCF to load (scene or robot)')
    ap.add_argument('--headless', action='store_true', help='Run without GUI and print motion stats')
    ap.add_argument('--seconds', type=float, default=8.0, help='Duration in seconds (GUI/headless); use --no-timer to keep GUI open')
    ap.add_argument('--no-timer', action='store_true', help='Keep GUI open until you close the window')
    args = ap.parse_args()
    xml_path = os.path.abspath(args.xml)
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    act_id = actuator_index_by_name(model)

    # Axis sign per hinge so our angles map to MJ joint coords
    jnt_sign: Dict[str, float] = {}
    for j in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j)
        axis = model.jnt_axis[j]
        s = 1.0
        if abs(axis[2]) > 0.5:
            s = 1.0 if axis[2] >= 0 else -1.0
        jnt_sign[name] = s

    # Load calibrated per-leg signs if available (overrides heuristic)
    calib_signs: Dict[str, Dict[str, float]] = {}
    if os.path.exists(SIGN_PATH):
        try:
            with open(SIGN_PATH, 'r', encoding='utf-8') as fh:
                calib_signs = json.load(fh)
        except Exception:
            calib_signs = {}

    def signed(joint_name: str, angle: float) -> float:
        # Prefer calibrated_signs.json per-leg overrides
        if joint_name.startswith('femur_joint_'):
            leg = joint_name.split('_')[-1]
            s = float(calib_signs.get(leg, {}).get('femur_sign', 1.0)) if calib_signs else jnt_sign.get(joint_name, 1.0)
            return angle * s
        if joint_name.startswith('tibia_joint_'):
            leg = joint_name.split('_')[-1]
            s = float(calib_signs.get(leg, {}).get('tibia_sign', 1.0)) if calib_signs else jnt_sign.get(joint_name, 1.0)
            return angle * s
        return angle * jnt_sign.get(joint_name, 1.0)

    def set_ctrl(joint: str, q: float) -> None:
        aid = act_id.get(f"pos_{joint}")
        if aid is not None:
            data.ctrl[aid] = q

    legs = LEG_ORDER

    # Base height/orientation and quick settle
    data.qpos[2] = 0.10
    data.qpos[3:7] = [1,0,0,0]
    mj.mj_forward(model, data)

    # Parameters
    step_hz = 1.0
    duty = 0.65
    stride_x, stride_y, step_height = 0.030, 0.0, 0.020
    # Uniform stance preload to encourage stable contacts without tilt
    stance_push = {
        leg: 0.0006 for leg in legs
    }

    # Correct tripod pairs: A = LF, RM, LR ; B = RF, LM, RR
    group_a = ["LF", "RM", "LR"]
    group_b = ["RF", "LM", "RR"]

    # Hip mounts and link lengths (meters)
    LEG_BASES = {
        "LF": {"hip": np.array([ +0.09360, +0.050805, 0.0]), "yaw0":  np.pi/4},
        "LM": {"hip": np.array([  0.0   , +0.073535, 0.0]), "yaw0":  np.pi/2},
        "LR": {"hip": np.array([ -0.09360, +0.050805, 0.0]), "yaw0":  3*np.pi/4},
        "RR": {"hip": np.array([ -0.09360, -0.050805, 0.0]), "yaw0": -3*np.pi/4},
        "RM": {"hip": np.array([  0.0   , -0.073535, 0.0]), "yaw0": -np.pi/2},
        "RF": {"hip": np.array([ +0.09360, -0.050805, 0.0]), "yaw0": -np.pi/4},
    }
    L_COXA, L_FEMUR, L_TIBIA = 0.04505032, 0.07703, 0.123

    def rotz(theta: float) -> np.ndarray:
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)

    def ik_leg(p_body: np.ndarray, leg: str) -> np.ndarray:
        base = LEG_BASES[leg]
        hip = np.array(base["hip"], dtype=float)
        yaw0 = float(base["yaw0"])
        v = np.asarray(p_body, dtype=float) - hip
        vloc = rotz(-yaw0) @ v
        x,y,z = float(vloc[0]), float(vloc[1]), float(vloc[2])
        theta0 = math.atan2(y, x)
        c0, s0 = math.cos(theta0), math.sin(theta0)
        xr = c0*x + s0*y
        zr = z
        xr -= L_COXA
        r2 = xr*xr + zr*zr
        D = (r2 - L_FEMUR*L_FEMUR - L_TIBIA*L_TIBIA) / (2*L_FEMUR*L_TIBIA)
        D = max(-1.0, min(1.0, D))
        # Choose knee branch so that after applying joint-axis sign, physical bend direction is consistent across sides.
        tib_sign = jnt_sign.get(f"tibia_joint_{leg}", 1.0)
        branch = -1.0 if tib_sign > 0 else +1.0
        theta2 = math.atan2(branch*math.sqrt(max(0.0, 1 - D*D)), D)
        theta1 = math.atan2(zr, xr) - math.atan2(L_TIBIA*math.sin(theta2), L_FEMUR + L_TIBIA*math.cos(theta2))
        return np.array([theta0, theta1, theta2], dtype=float)

    # Neutral footholds (meters)
    DEFAULT_POSE_M = [
        ( +0.1636, +0.150805, -0.090),
        ( +0.0000, +0.193535, -0.090),
        ( -0.1636, +0.150805, -0.090),
        ( -0.1636, -0.150805, -0.090),
        ( +0.0000, -0.193535, -0.090),
        ( +0.1636, -0.150805, -0.090),
    ]
    neutral = {leg: np.array(DEFAULT_POSE_M[i]) for i, leg in enumerate(legs)}

    # Load cached offsets if present (ensures consistent neutral stance)
    # We ignore stored offsets in this preview for robustness
    stored_offsets: Dict[str, List[float]] = {}

    # Build offsets so IK(neutral) -> joint zeros (kept for reference),
    # but we will command ABSOLUTE joint angles to the position actuators.
    offsets: Dict[str, List[float]] = {leg: [0.0, 0.0, 0.0] for leg in legs}
    for leg in legs:
        qik = ik_leg(neutral[leg], leg)
        qmap = [
            signed(f"coxa_joint_{leg}", qik[0]),
            signed(f"femur_joint_{leg}", qik[1]),
            signed(f"tibia_joint_{leg}", qik[2]),
        ]
        set_ctrl(f"coxa_joint_{leg}", qmap[0])
        set_ctrl(f"femur_joint_{leg}", qmap[1])
        set_ctrl(f"tibia_joint_{leg}", qmap[2])
    for _ in range(200):
        mj.mj_step(model, data)

    # No offsets are written in this preview

    duration = float('inf')
    substeps = 6
    w = 2.0 * math.pi * step_hz
    dt = float(model.opt.timestep)
    sim_t = 0.0

    # Per-leg smoothing and rate limit
    alpha = 0.10
    prev_q = {leg: np.zeros(3, dtype=float) for leg in legs}
    for leg in legs:
        qik0 = ik_leg(neutral[leg], leg)
        prev_q[leg][0] = signed(f"coxa_joint_{leg}", qik0[0])
        prev_q[leg][1] = signed(f"femur_joint_{leg}", qik0[1])
        prev_q[leg][2] = signed(f"tibia_joint_{leg}", qik0[2])

    def foot_traj(phi: float, duty: float,
                  stride_fwd: float, stride_lat: float,
                  step_h: float, push_down: float) -> np.ndarray:
        while phi < 0: phi += 2*math.pi
        while phi >= 2*math.pi: phi -= 2*math.pi
        if phi < 2*math.pi*duty:
            # stance: slide foot backward relative to body to enable forward travel
            s = phi / (2*math.pi*duty)
            x = (0.5 - s) * stride_fwd
            y = (0.5 - s) * stride_lat
            z = -push_down
        else:
            s = (phi - 2*math.pi*duty)/(2*math.pi*(1.0 - duty))
            x = (-0.5 + s)*stride_fwd
            y = (-0.5 + s)*stride_lat
            z = step_h * math.sin(math.pi * s)
        return np.array([x, y, z], dtype=float)

    # Common stepping body
    base_z_target = 0.12
    z_int = 0.0
    def step_once(sim_time: float) -> float:
        nonlocal prev_q, z_int
        stride_phase = max(0.0, sim_time - 1.2)
        stride_gain = min(1.0, stride_phase / 1.5)
        base_z = float(data.qpos[2])
        if base_z > 0.30 or base_z < 0.02:
            stride_gain = 0.0
        # Gentle base-height correction (PI): adjust feet Z uniformly
        z_err = base_z_target - base_z
        z_int = max(-0.01, min(0.01, z_int + z_err*dt))
        z_corr = max(-0.01, min(0.01, 0.5*z_err + 0.1*z_int))
        push_gain = min(1.0, sim_time / 0.9)
        for leg in legs:
            grp = 'A' if leg in group_a else 'B'
            ph0 = 0.0 if grp == 'A' else math.pi
            phi = w * sim_time + ph0
            push = stance_push.get(leg, 0.0005)
            traj = foot_traj(
                phi,
                duty,
                stride_x * stride_gain,
                stride_y * stride_gain,
                step_height * stride_gain,
                push * push_gain,
            )
            p_des = neutral[leg] + traj + np.array([0.0, 0.0, -z_corr])
            qik = ik_leg(p_des, leg)
            # Absolute joint targets for position actuators with calibrated offsets
            q_des = np.array([
                signed(f"coxa_joint_{leg}", qik[0]),
                signed(f"femur_joint_{leg}", qik[1]),
                signed(f"tibia_joint_{leg}", qik[2]),
            ], dtype=float)
            max_rate = np.array([2.0, 2.0, 2.5], dtype=float)
            max_step = max_rate * dt
            dq = np.clip(q_des - prev_q[leg], -max_step, max_step)
            prev_q[leg] = prev_q[leg] + dq
            prev_q[leg] = (1.0 - alpha) * prev_q[leg] + alpha * q_des
            set_ctrl(f"coxa_joint_{leg}", float(prev_q[leg][0]))
            set_ctrl(f"femur_joint_{leg}", float(prev_q[leg][1]))
            set_ctrl(f"tibia_joint_{leg}", float(prev_q[leg][2]))
        mj.mj_step(model, data)
        return sim_time + dt

    if args.headless:
        end_t = args.seconds
        ctrl_hist = []
        qpos_hist = []
        while sim_t < end_t:
            for _ in range(substeps):
                sim_t = step_once(sim_t)
            ctrl_hist.append(np.copy(data.ctrl))
            qpos_hist.append(np.copy(data.qpos))
        arr = np.stack(ctrl_hist, axis=0)
        qarr = np.stack(qpos_hist, axis=0)
        # Report RMS motion per actuator and hinge qpos to confirm movement
        rms_ctrl = np.sqrt(np.mean((arr - arr.mean(axis=0))**2, axis=0)).mean()
        # Skip freejoint (first 7 qpos entries)
        hinge_q = qarr[:, 7:7+18]
        rms_qpos = np.sqrt(np.mean((hinge_q - hinge_q.mean(axis=0))**2, axis=0)).mean()
        base_z = qarr[:,2]
        print(f"Headless preview ran {args.seconds:.2f}s, ctrl RMS(mean)={float(rms_ctrl):.4f}, hinge qpos RMS(mean)={float(rms_qpos):.4f}, base_z mean={float(base_z.mean()):.4f}, min={float(base_z.min()):.4f}")
        return
    else:
        if args.no-timer if False else False:
            pass
        if args.no_timer:
            # On-screen preview without timer; print periodic diagnostics to stdout
            ground_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'ground')
            foot_ids = {leg: mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f'foot_{leg}') for leg in legs}
            last_print = time.time()
            with viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as h:
                while h.is_running():
                    for _ in range(substeps):
                        sim_t = step_once(sim_t)
                    now = time.time()
                    if now - last_print >= 0.5:
                        # Count contacts with ground per leg and report base height
                        present = {leg: False for leg in legs}
                        for k in range(data.ncon):
                            c = data.contact[k]
                            if c.geom1 == ground_id:
                                g_other = c.geom2
                            elif c.geom2 == ground_id:
                                g_other = c.geom1
                            else:
                                continue
                            for leg, gid in foot_ids.items():
                                if g_other == gid:
                                    present[leg] = True
                                    break
                        active = ''.join([leg.lower() if present[leg] else '.' for leg in legs])
                        print(f"t={sim_t:5.2f}s base_z={data.qpos[2]:.3f} contacts={active}")
                        last_print = now
                    h.sync()
            return
        else:
            end_time = time.time() + float(args.seconds)
            with viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as h:
                while h.is_running() and time.time() < end_time:
                    for _ in range(substeps):
                        sim_t = step_once(sim_t)
                    h.sync()
            # Ensure we exit after the requested duration
            return


if __name__ == "__main__":
    main()
