#!/usr/bin/env python3
from __future__ import annotations

"""Render baseline policy (circle CPG + IK) with simple overlays.

Keys: [space]=pause  [Q]=quit
Overlays: per-leg circle center, controller radius, support polygon.
"""

import argparse
import sys
import json
import math
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

# Ensure repo root import when run as a script (mjpython tools/..)
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from control.cpg_circle import (
    CirclePolicy,
    TripodPhaser,
    foot_target_hi1,
    motion_center,
    stance_update,
    swing_update,
    stride_limits_about_cw,
    GaitEngine,
)
from tools.p0p1_verify import (
    LEG_ORDER,
    hip_frame_axes,
    paper_circle_params,
    link_lengths_from_xml,
    support_polygon_from_contacts,
    point_to_polygon_margin,
)
from control.ik_analytic import LegGeom, ik3_hi1_closed


def load_circle_policy(cfg_path: Path, m, d):
    cfg = json.loads(Path(cfg_path).read_text())
    a1, a2, a3 = link_lengths_from_xml(m)
    # Use current hip height for d_cw per leg
    zH = float(np.mean([float(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"coxa_{s}{p}")][2]) for s, p in [("L","F"),("L","M"),("L","R"),("R","F"),("R","M"),("R","R")]]))
    r_paper, d_cw = paper_circle_params(a1, a2, a3, zH)
    pol = CirclePolicy(
        r_paper=float(cfg.get("r_paper", cfg.get("r_paper_m", r_paper))),
        r_inscribed_min=float(cfg.get("r_inscribed_min", cfg.get("r_inscribed_min_m", r_paper))),
        r_ctrl=float(cfg.get("r_ctrl", cfg.get("r_ctrl_recommended_m", r_paper))),
        s=float(cfg.get("s", cfg.get("paper_scale_s", 1.0))),
        alpha=float(cfg.get("alpha", cfg.get("alpha_inscribed", 0.8))),
        d_cw=d_cw,
        height=-zH,  # place stance plane at ground (hip_z above ground)
    )
    return pol


def joint_signs_for_leg(m, leg: str):
    # Determine sign from joint axis z component (expects hinge about z)
    signs = {}
    for jname in (f"coxa_joint_{leg}", f"femur_joint_{leg}", f"tibia_joint_{leg}"):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            axis = m.jnt_axis[jid]
            s = 1.0
            if abs(axis[2]) > 0.5:
                s = 1.0 if axis[2] > 0 else -1.0
            signs[jname] = s
        else:
            signs[jname] = 1.0
    return signs


def engine_numeric_ik_leg(m, d, leg: str, target_world: np.ndarray, iters: int = 30, lam: float = 1e-3, alpha: float = 0.5, allow_coxa: bool = True):
    """Solve 3-DoF leg IK directly in engine joint coordinates using body Jacobians.

    Updates d.qpos in-place for the three joints of the leg toward target_world.
    Returns (q_coxa, q_femur, q_tibia).
    """
    # Joint and body ids
    j_coxa = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"coxa_joint_{leg}")
    j_fem  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"femur_joint_{leg}")
    j_tib  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"tibia_joint_{leg}")
    dof_all = [int(m.jnt_dofadr[j_coxa]), int(m.jnt_dofadr[j_fem]), int(m.jnt_dofadr[j_tib])]
    qaddrs = [int(m.jnt_qposadr[j_coxa]), int(m.jnt_qposadr[j_fem]), int(m.jnt_qposadr[j_tib])]
    if allow_coxa:
        col_idx = [0,1,2]
    else:
        col_idx = [1,2]  # femur, tibia only
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}")
    b_foot = int(m.geom_bodyid[gid]) if gid >= 0 else None

    for _ in range(max(1, iters)):
        mujoco.mj_forward(m, d)
        # Current foot center
        Rb = d.xmat[b_foot].reshape(3,3)
        p_curr = d.xpos[b_foot] + Rb @ m.geom_pos[gid]
        err = np.asarray(target_world, dtype=float) - p_curr
        if float(np.linalg.norm(err)) < 1e-4:
            break
        # Jacobian for body center (approximate for geom center)
        Jp = np.zeros((3, m.nv))
        Jr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, Jp, Jr, b_foot)
        J = Jp[:, [dof_all[i] for i in col_idx]]  # 3xk (k=2 or 3)
        JT = J.T
        H = JT @ J + (lam**2) * np.eye(J.shape[1])
        dq = np.linalg.solve(H, JT @ err)
        dq = np.clip(dq, -0.25, 0.25)
        # Update qpos
        if allow_coxa:
            d.qpos[qaddrs[0]] = float(d.qpos[qaddrs[0]] + alpha * dq[0])
            d.qpos[qaddrs[1]] = float(d.qpos[qaddrs[1]] + alpha * dq[1])
            d.qpos[qaddrs[2]] = float(d.qpos[qaddrs[2]] + alpha * dq[2])
        else:
            d.qpos[qaddrs[1]] = float(d.qpos[qaddrs[1]] + alpha * dq[0])
            d.qpos[qaddrs[2]] = float(d.qpos[qaddrs[2]] + alpha * dq[1])
        # Respect limits
        lims = [m.jnt_range[j_coxa], m.jnt_range[j_fem], m.jnt_range[j_tib]]
        for k in range(3):
            lo, hi = lims[k]
            d.qpos[qaddrs[k]] = float(np.clip(d.qpos[qaddrs[k]], lo, hi))
    return np.array([d.qpos[qaddrs[0]], d.qpos[qaddrs[1]], d.qpos[qaddrs[2]]], dtype=float)


def parse_env_or_xml(env: str | None, xml: str | None) -> str:
    if env:
        cfg = json.loads(Path(env).read_text())
        return cfg["xml_path"]
    if xml:
        return xml
    raise SystemExit("Provide --env or --xml")


def total_com(m, d) -> np.ndarray:
    mass = np.array(m.body_mass).reshape(-1)
    xpos = np.array(d.xpos)
    M = float(np.sum(mass))
    return (mass[:, None] * xpos).sum(axis=0) / max(1e-9, M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None, help="Path to env JSON (with xml_path)")
    ap.add_argument("--xml", default=None, help="Direct XML path (fallback)")
    ap.add_argument("--controller", default="baseline:cpg_circle")
    ap.add_argument("--circle-policy", default="configs/policy/workspace_circle.json")
    ap.add_argument("--overlay", default="workspace,com,contacts,phases")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--v-cmd", type=float, default=0.0)
    ap.add_argument("--yaw-cmd", type=float, default=0.5)
    ap.add_argument("--omega", type=float, default=0.25, help="Cycle frequency (Hz) in preview mode")
    ap.add_argument("--amp-scale", type=float, default=1.0, help="Scale radius r_ctrl in preview (0..1]")
    ap.add_argument("--lift", type=float, default=0.0, help="Swing lift height (m)")
    ap.add_argument("--duty", type=float, default=1.0, help="Stance fraction (0..1]")
    ap.add_argument("--mode", choices=["preview","stand"], default="preview",
                    help="preview: CPG+IK targets; stand: hold current joint pose (no motion)")
    ap.add_argument("--lock-coxa", action="store_true", help="Keep coxa angles fixed (femur/tibia-only IK)")
    ap.add_argument("--p2final_v21", action="store_true", help="Use anchored stance + event-driven swing timing (v2.1)")
    args = ap.parse_args()

    xml_path = parse_env_or_xml(args.env, args.xml)
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    pol = load_circle_policy(Path(args.circle_policy), m, d)
    # Preview rate; keep omega and scale XY by amp-scale around cw when generating targets
    pol.omega = 2.0 * math.pi * float(args.omega)
    pol.lift = float(args.lift)
    pol.duty = float(args.duty)
    phaser = TripodPhaser(("LF","RM","LR"), ("RF","LM","RR"))
    controller_kind = args.controller.strip().split(':', 1)[0]
    # Policy hook: try loading a simple linear tanh policy if chosen
    policy_fn = None
    if controller_kind == 'policy':
        try:
            from rl.policy_loader import load_policy
            obs_size = int(m.nq + m.nv)
            act_size = int(m.nu)
            policy_fn = load_policy(args.checkpoint or '', obs_size, act_size)
        except Exception:
            policy_fn = None
    enabled = {k.strip(): True for k in args.overlay.split(',') if k.strip()}
    paused = False

    # Keyboard toggles
    def on_key(keycode):
        nonlocal paused
        if keycode in (ord(' '),):
            paused = not paused
        elif keycode in (ord('w'), ord('W')):
            enabled['workspace'] = not enabled.get('workspace', True)
        elif keycode in (ord('c'), ord('C')):
            enabled['com'] = not enabled.get('com', True)
        elif keycode in (ord('f'), ord('F')):
            enabled['contacts'] = not enabled.get('contacts', True)
        elif keycode in (ord('p'), ord('P')):
            enabled['phases'] = not enabled.get('phases', True)

    # helper drawing functions for user scene
    def draw_sphere(viewer, pos, radius, rgba=(1,0,0,0.8)):
        scn = viewer.user_scn
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g,
                             mujoco.mjtGeom.mjGEOM_SPHERE,
                             np.array([radius, radius, radius], dtype=np.float64),
                             np.array(pos, dtype=np.float64),
                             np.eye(3, dtype=np.float64).reshape(-1),
                             np.array(rgba, dtype=np.float32))
        scn.ngeom += 1

    def draw_line(viewer, p1, p2, width=1.0, rgba=(1,1,1,0.8)):
        scn = viewer.user_scn
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_connector(g,
                             mujoco.mjtGeom.mjGEOM_LINE,
                             float(width),
                             np.array(p1, dtype=np.float64),
                             np.array(p2, dtype=np.float64))
        g.rgba[:] = np.array(rgba, dtype=np.float32)
        scn.ngeom += 1

    with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as viewer:
        GRID_TOP_LEFT = getattr(mujoco.mjtGridPos, 'mjGRID_TOP_LEFT', None) or getattr(mujoco.mjtGridPos, 'mjGRID_TOPLEFT', None)
        GRID_BOTTOM_RIGHT = getattr(mujoco.mjtGridPos, 'mjGRID_BOTTOM_RIGHT', None) or getattr(mujoco.mjtGridPos, 'mjGRID_BOTTOMRIGHT', None)
        def overlay_text(title: str, text: str):
            if hasattr(viewer, 'add_overlay') and GRID_TOP_LEFT is not None:
                viewer.add_overlay(GRID_TOP_LEFT, title, text)
        def overlay_text_grid(grid, title: str, text: str):
            if hasattr(viewer, 'add_overlay') and grid is not None:
                viewer.add_overlay(grid, title, text)
        t = 0.0
        base_phi = 0.0
        # Capture initial joint positions for a gentle ramp of ctrl targets
        # Map joint name -> qpos index
        jidx = {}
        for leg in LEG_ORDER:
            for jname in (f"coxa_joint_{leg}", f"femur_joint_{leg}", f"tibia_joint_{leg}"):
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid >= 0:
                    jidx[jname] = int(m.jnt_qposadr[jid])
        q_init = d.qpos.copy()
        # Precompute actuator->joint mapping for stand mode
        act_map = []
        for leg in LEG_ORDER:
            for jname in (f"coxa_joint_{leg}", f"femur_joint_{leg}", f"tibia_joint_{leg}"):
                aname = f"pos_{jname}"
                aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if aid >= 0 and jid >= 0:
                    act_map.append((aid, int(m.jnt_qposadr[jid])))

        # per-actuator target rate limiter (servo max speed ~5.82 rad/s @ 11.1V)
        omega_max = 5.82
        dctrl_max = omega_max * float(m.opt.timestep)
        prev_ctrl = np.array(d.ctrl, copy=True)

        # v2.1 unified engine (anchored stance)
        engine = GaitEngine(LEG_ORDER, pol, swing_apex=max(0.007, float(args.lift) if args.lift>0 else 0.012)) if args.p2final_v21 else None
        # v2.1 sagittal-only preference: freeze yaw for straight/curve (allow for in-place turns)
        prefer_sagittal_global = bool(args.p2final_v21 and (abs(args.v_cmd) > 1e-6))
        # Per-leg frozen coxa target when sagittal-only is active
        freeze_coxa: dict[str, float] = {}

        while viewer.is_running():
            if not paused:
                # Drive either baseline or policy
                if controller_kind == 'baseline':
                    if args.mode == 'stand':
                        # Hold current pose: copy qpos to position-actuator ctrl each step
                        for aid, qadr in act_map:
                            lo, hi = m.actuator_ctrlrange[aid]
                            d.ctrl[aid] = float(np.clip(q_init[qadr], lo, hi))
                        mujoco.mj_step(m, d)
                        t += float(m.opt.timestep)
                    else:
                        # advance phases and set position-actuator targets via IK
                        base_phi = (base_phi + pol.omega * float(m.opt.timestep)) % (2*math.pi)
                        a1, a2, a3 = link_lengths_from_xml(m)
                        for leg in LEG_ORDER:
                            hip_pos, hx, hy, hz = hip_frame_axes(m, d, leg)
                            if args.p2final_v21 and engine is not None:
                                torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot') if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot') >= 0 else 0
                                torso_pos = d.xpos[torso]; Rb = d.xmat[torso].reshape(3,3)
                                hip_frames = {L: hip_frame_axes(m, d, L) for L in LEG_ORDER}
                                footW = {}
                                for L in LEG_ORDER:
                                    gidL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{L}")
                                    bL = int(m.geom_bodyid[gidL])
                                    footW[L] = d.xpos[bL] + d.xmat[bL].reshape(3,3) @ m.geom_pos[gidL]
                                contact_now = {}
                                for L in LEG_ORDER:
                                    gidL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{L}")
                                    flag = 0
                                    for ci in range(d.ncon):
                                        c = d.contact[ci]
                                        if c.geom1 == gidL or c.geom2 == gidL:
                                            flag = 1; break
                                    contact_now[L] = bool(flag)
                                targets = engine.step(float(m.opt.timestep), torso_pos, Rb, hip_frames, footW, contact_now, args.yaw_cmd, float(args.amp_scale))
                                p_hi1 = targets[leg].copy()
                            else:
                                phi_leg = phaser.phase_for_leg(leg, base_phi)
                                p_hi1 = foot_target_hi1(phi_leg, pol).copy()
                                if args.amp_scale != 1.0:
                                    cx = pol.d_cw
                                    p_hi1[0] = cx + float(args.amp_scale) * (p_hi1[0] - cx)
                                    p_hi1[1] = float(args.amp_scale) * p_hi1[1]
                            # If coxa is unlocked, enforce stride arc limits to avoid yaw accumulation
                            if not args.lock_coxa:
                                # compute cw and stride limits relative to the real motion center in hip frame
                                cw = hip_pos + pol.d_cw*hx
                                torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot') if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot') >= 0 else 0
                                torso_pos = d.xpos[torso]
                                Rb = d.xmat[torso].reshape(3,3)
                                cm_body, dpsi_dt = motion_center(args.v_cmd, args.yaw_cmd, 0.0)
                                cm_world = torso_pos + cm_body[0]*Rb[:,0] + cm_body[1]*Rb[:,1]
                                cm_xy = np.array([np.dot(cm_world-hip_pos, hx), np.dot(cm_world-hip_pos, hy)])
                                cw_xy = np.array([np.dot((cw-hip_pos), hx), np.dot((cw-hip_pos), hy)])
                                r_i = float(np.linalg.norm(cw_xy - cm_xy))
                                lim = stride_limits_about_cw(cw_xy, pol.r_ctrl*float(args.amp_scale), cm_xy, r_i)
                                if lim is not None:
                                    th0, th1 = lim
                                    th = math.atan2(p_hi1[1], p_hi1[0]-pol.d_cw)
                                    if th < th0:
                                        r = pol.r_ctrl*float(args.amp_scale)
                                        p_hi1[0] = pol.d_cw + r*math.cos(th0)
                                        p_hi1[1] = r*math.sin(th0)
                                    elif th > th1:
                                        r = pol.r_ctrl*float(args.amp_scale)
                                        p_hi1[0] = pol.d_cw + r*math.cos(th1)
                                        p_hi1[1] = r*math.sin(th1)
                            # Adjust per-leg so foot center rides at ground + foot radius
                            foot_r = float(m.geom_size[gid][0]) if gid >= 0 else 0.012
                            dz = ((-hip_pos[2] + foot_r) - pol.height)
                            p_hi1[2] += dz
                            # Solve IK directly in engine coordinates using Jacobians
                            p_world = hip_pos + p_hi1[0]*hx + p_hi1[1]*hy + p_hi1[2]*hz
                            # Allow yaw only for explicit turn-in-place (v_cmd≈0 and yaw_cmd≠0), or when lock_coxa is set
                            allow_yaw = (not prefer_sagittal_global) or (abs(args.v_cmd) < 1e-6 and abs(args.yaw_cmd) > 1e-6)
                            allow_yaw = allow_yaw and (not args.lock_coxa)
                            q = engine_numeric_ik_leg(m, d, leg, p_world, iters=20, lam=1e-3, alpha=0.7, allow_coxa=allow_yaw)
                            # Smoothly ramp targets from initial pose to IK targets over ~0.5s
                            ramp = min(1.0, t / 2.0)
                            for jname, qref in (
                                (f"coxa_joint_{leg}", q[0]),
                                (f"femur_joint_{leg}", q[1]),
                                (f"tibia_joint_{leg}", q[2]),
                            ):
                                aname = f"pos_{jname}"
                                aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
                                if aid >= 0:
                                    lo, hi = m.actuator_ctrlrange[aid]
                                    # Blend from the current joint angle towards IK qref
                                    qi = q_init[jidx[jname]] if jname in jidx else 0.0
                                    q_blend = qi + ramp * (qref - qi)
                                    # Sagittal-only: freeze coxa at first observed value when active
                                    raw = float(np.clip(q_blend, lo, hi))
                                    if prefer_sagittal_global and jname.startswith("coxa_joint_"):
                                        if leg not in freeze_coxa:
                                            jidc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
                                            freeze_coxa[leg] = float(d.qpos[int(m.jnt_qposadr[jidc])])
                                        raw = float(np.clip(freeze_coxa[leg], lo, hi))
                                    # Slew-limit target to emulate servo max speed
                                    prev = float(prev_ctrl[aid])
                                    lim = np.clip(raw, prev - dctrl_max, prev + dctrl_max)
                                    d.ctrl[aid] = lim
                                    prev_ctrl[aid] = lim
                        mujoco.mj_step(m, d)
                        t += float(m.opt.timestep)
                elif controller_kind == 'policy' and policy_fn is not None:
                    obs = np.concatenate([d.qpos, d.qvel]).astype(np.float32)
                    act = policy_fn(obs)
                    if act.shape[0] == m.nu:
                        lo = m.actuator_ctrlrange[:,0] if m.actuator_ctrlrange.size else -np.inf
                        hi = m.actuator_ctrlrange[:,1] if m.actuator_ctrlrange.size else +np.inf
                        d.ctrl[:] = np.clip(act, lo, hi)
                mujoco.mj_step(m, d)
                t += float(m.opt.timestep)

            # Clear any previous user geoms; connectors append to this scene
            viewer.user_scn.ngeom = 0

            a1, a2, a3 = link_lengths_from_xml(m)
            # Compute motion center in body frame and map to world
            torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot') if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot') >= 0 else 0
            torso_pos = d.xpos[torso]
            R = d.xmat[torso].reshape(3,3)
            cm_body, dpsi_dt = motion_center(args.v_cmd, args.yaw_cmd, 0.0)
            cm_world = torso_pos + cm_body[0]*R[:,0] + cm_body[1]*R[:,1]

            # Support polygon and COM overlay
            if enabled.get('com', True):
                hull = support_polygon_from_contacts(m, d, [f"foot_{leg}" for leg in LEG_ORDER])
                if hull.shape[0] >= 3:
                    # draw hull as polyline
                    for i in range(hull.shape[0]):
                        p1 = np.array([hull[i][0], hull[i][1], 0.0])
                        p2 = np.array([hull[(i+1)%hull.shape[0]][0], hull[(i+1)%hull.shape[0]][1], 0.0])
                        draw_line(viewer, p1, p2, width=2.0, rgba=(0,1,0,1))
                com = total_com(m, d)
                overlay_text('COM margin', '')
                if hull.shape[0] >= 3:
                    margin = point_to_polygon_margin(com[:2], hull)
                    overlay_text('', f"margin={margin:+.3f} m")
                overlay_text('', f"COM=({com[0]:+.3f},{com[1]:+.3f})")

            # Workspace circles, cm, AEP/PEP arcs
            if enabled.get('workspace', True):
                # draw cm as a small sphere
                draw_sphere(viewer, cm_world, 0.01, rgba=(1,1,0,0.9))

                for leg in LEG_ORDER:
                    hip_pos, hx, hy, hz = hip_frame_axes(m, d, leg)
                    rpap, d_cw = paper_circle_params(a1, a2, a3, hip_pos[2])
                    cw = hip_pos + d_cw * hx
                    # circle ring as polyline at (ground + foot radius) relative to hip
                    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}")
                    foot_r = float(m.geom_size[gid][0]) if gid >= 0 else 0.012
                    height_z = -hip_pos[2] + foot_r
                    th = np.linspace(0, 2*math.pi, 64, endpoint=False)
                    ring = np.stack([cw + pol.r_ctrl * (np.cos(thi)*hx + np.sin(thi)*hy) + height_z*hz for thi in th])
                    for i in range(ring.shape[0]):
                        draw_line(viewer, ring[i], ring[(i+1) % ring.shape[0]], width=1.0, rgba=(1,0,0,0.6))

                    # stride limits
                    # represent cm in hip frame XY
                    cm_xy_hi = np.array([np.dot(cm_world-hip_pos, hx), np.dot(cm_world-hip_pos, hy)])
                    cw_xy = np.array([np.dot(cw-hip_pos, hx), np.dot(cw-hip_pos, hy)])
                    r_i = float(np.linalg.norm(cw_xy - cm_xy_hi))
                    lim = stride_limits_about_cw(cw_xy, pol.r_ctrl, cm_xy_hi, r_i)
                    if lim is not None:
                        th0, th1 = lim
                        arc_th = np.linspace(th0, th1, 32)
                        arc = np.stack([cw + pol.r_ctrl * (np.cos(thi)*hx + np.sin(thi)*hy) + height_z*hz for thi in arc_th])
                        for i in range(arc.shape[0]-1):
                            draw_line(viewer, arc[i], arc[i+1], width=2.5, rgba=(0,0,1,0.8))

            # Contacts and slip vectors
            if enabled.get('contacts', True):
                for ci in range(d.ncon):
                    c = d.contact[ci]
                    pos = np.array(c.pos)
                    fr = np.zeros(6, dtype=np.float64)
                    # API expects contact index, not the struct
                    mujoco.mj_contactForce(m, d, ci, fr)
                    mag = float(fr[2])
                    alpha = min(0.9, 0.3 + 0.6*min(1.0, abs(mag)/200.0))
                    draw_sphere(viewer, pos, 0.004, rgba=(1,0,1,alpha))

            # Overlays: COM margin, support polygon, and phases (simple clocks)
            if enabled.get('phases', True):
                base_phi = (t * pol.omega) % (2*math.pi)
                for leg in LEG_ORDER:
                    phi = phaser.phase_for_leg(leg, base_phi)
                    overlay_text_grid(GRID_BOTTOM_RIGHT, leg, f"phi={phi:.2f}")

            if enabled.get('com', True):
                # weight vs sum normal forces display (quick readout)
                mg = float(np.sum(np.array(m.body_mass))) * 9.81
                # sum normals
                totalN = 0.0
                for ci in range(d.ncon):
                    fr = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(m, d, ci, fr)
                    totalN += float(max(0.0, fr[2]))
                overlay_text("Weight vs ΣN", f"{mg:6.2f} N vs {totalN:6.2f} N")

            viewer.sync()


if __name__ == "__main__":
    main()
