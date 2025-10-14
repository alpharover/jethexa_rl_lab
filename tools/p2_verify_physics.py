#!/usr/bin/env python3
from __future__ import annotations

"""
Headless physics verification (P2): gravity/mass, actuator limits, static weight
balance, free-fall gravity fit, and momentum under finite impulse.

Usage examples:
  mjpython tools/p2_verify_physics.py --xml mjcf/jethexa_lab.xml --test gravity
  mjpython tools/p2_verify_physics.py --xml mjcf/jethexa_lab.xml --test limits
  mjpython tools/p2_verify_physics.py --xml mjcf/jethexa_lab.xml --test static
  mjpython tools/p2_verify_physics.py --xml mjcf/jethexa_lab.xml --test freefall
  mjpython tools/p2_verify_physics.py --xml mjcf/jethexa_lab.xml --test push
  mjpython tools/p2_verify_physics.py --xml mjcf/jethexa_lab.xml --test all
"""

import argparse, json, math
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import mujoco


def _model_and_data(xml: str):
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_gravity_mass(xml: str) -> Dict[str, Any]:
    m, d = _model_and_data(xml)
    g = np.array(m.opt.gravity, dtype=float)
    gnorm = float(np.linalg.norm(g))
    total_mass = float(np.sum(np.array(m.body_mass)))
    ok = abs(gnorm - 9.81) <= 0.2 and (0.5 <= total_mass <= 10.0)
    return {"test": "gravity_mass", "ok": bool(ok), "gravity": g.tolist(), "g_norm": gnorm, "total_mass_kg": total_mass}


def test_actuator_limits(xml: str) -> Dict[str, Any]:
    m, _ = _model_and_data(xml)
    if m.nu == 0:
        return {"test": "actuator_limits", "ok": True, "unclamped": [], "forcerange": []}
    fr = np.array(m.actuator_forcerange) if m.actuator_forcerange.size else np.zeros((m.nu, 2))
    def bad(lo, hi):
        return (np.isclose(lo, 0) & np.isclose(hi, 0)) | (np.isinf(lo) | np.isinf(hi)) | (hi <= 0)
    badmask = [bool(b) for b in bad(fr[:, 0], fr[:, 1])]
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    unclamped = [names[i] for i, b in enumerate(badmask) if b]
    ok = len(unclamped) == 0
    return {"test": "actuator_limits", "ok": bool(ok), "unclamped": unclamped, "forcerange": fr.tolist()}


def _sum_normal_forces(m, d, foot_prefix: str = "foot_") -> float:
    """Sum upward (world-z) normal force from contacts that involve foot geoms.

    Uses mj_contactForce and rotates the contact-frame normal into world frame
    via c.frame (first 3 numbers are the world normal vector).
    """
    totalN = 0.0
    for ci in range(d.ncon):
        c = d.contact[ci]
        g1 = c.geom1; g2 = c.geom2
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1) if g1 >= 0 else ""
        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2) if g2 >= 0 else ""
        if (n1 and n1.startswith(foot_prefix)) or (n2 and n2.startswith(foot_prefix)):
            fr = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(m, d, ci, fr)
            # world normal vector is c.frame[0:3]
            nx, ny, nz = float(c.frame[0]), float(c.frame[1]), float(c.frame[2])
            Fn = float(max(0.0, fr[0]))  # normal magnitude (nonnegative)
            # upward world-z component
            totalN += max(0.0, Fn * nz)
    return float(totalN)


def test_static_weight(xml: str, seconds: float = 2.0) -> Dict[str, Any]:
    m, d = _model_and_data(xml)
    # Lift robot a bit so it settles under gravity into contact during the run
    if m.njnt > 0 and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        d.qpos[2] = max(0.2, float(d.qpos[2]))
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
    # Hold current pose by setting position actuator targets to current qpos each step
    act_map = []
    for i in range(m.nu):
        aname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if not aname:
            continue
        # assume position actuators named "pos_<joint>"
        if aname.startswith("pos_"):
            jname = aname[4:]
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                act_map.append((i, int(m.jnt_qposadr[jid])))
    dt = float(m.opt.timestep)
    steps = max(1, int(seconds / dt))
    normals: List[float] = []
    for _ in range(steps):
        for aid, qadr in act_map:
            lo, hi = m.actuator_ctrlrange[aid]
            d.ctrl[aid] = float(np.clip(d.qpos[qadr], lo, hi))
        mujoco.mj_step(m, d)
        normals.append(_sum_normal_forces(m, d))
    normals = np.array(normals, dtype=float)
    mg = float(np.sum(np.array(m.body_mass))) * 9.81
    meanN = float(np.mean(normals))
    rel_err = float(abs(meanN - mg) / max(1e-6, mg))
    ok = rel_err <= 0.05
    return {"test": "static_weight", "ok": bool(ok), "mean_sum_normal": meanN, "mg": mg, "rel_err": rel_err, "stdev": float(np.std(normals))}


def test_freefall(xml: str, seconds: float = 0.25) -> Dict[str, Any]:
    m, d = _model_and_data(xml)
    # Lift base (robot) above ground to avoid contact
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot")
    if bid < 0:
        bid = 0
    # If the model has a free joint, its qpos starts at index 0; still, record base z from body xpos
    if m.njnt > 0 and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        d.qpos[2] = 1.0
    # Zero velocities
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    dt = float(m.opt.timestep)
    steps = max(2, int(seconds / dt))
    z: List[float] = []
    t: List[float] = []
    for k in range(steps):
        mujoco.mj_step(m, d)
        z.append(float(d.xpos[bid][2]))  # world z of base body
        t.append((k + 1) * dt)
    z = np.array(z); t = np.array(t)
    # Fit z(t) = z0 + v0 t - 0.5 g t^2  => quadratic fit
    A = np.stack([np.ones_like(t), t, -0.5 * t * t], axis=1)
    coeff, *_ = np.linalg.lstsq(A, z, rcond=None)
    ghat = float(coeff[2])
    ok = abs(ghat - 9.81) <= 0.05
    return {"test": "freefall", "ok": bool(ok), "g_hat": ghat, "seconds": seconds, "dt": dt}


def _com_xy(m, d) -> np.ndarray:
    # mass-weighted COM of all bodies
    mass = np.array(m.body_mass).reshape(-1)
    xpos = np.array(d.xpos)
    M = float(np.sum(mass))
    return (mass[:, None] * xpos).sum(axis=0) / max(1e-6, M)


def test_push_momentum(xml: str, deltav: float = 0.2, steps_push: int = 10) -> Dict[str, Any]:
    m, d = _model_and_data(xml)
    # Pick the robot root body (name 'robot' if present else 0)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot")
    if bid < 0:
        bid = 0
    dt = float(m.opt.timestep)
    mass = float(m.body_mass[bid])
    J = mass * abs(deltav)
    F = J / (steps_push * dt)
    # Measure base (freejoint) x-velocity
    jfree = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root_free")
    dofadr = int(m.jnt_dofadr[jfree]) if jfree >= 0 else 0
    v_before = float(d.qvel[dofadr + 0]) if m.jnt_type[jfree] == mujoco.mjtJoint.mjJNT_FREE else 0.0
    # Apply finite impulse spread across K steps; zero after each step
    for _ in range(steps_push):
        d.xfrc_applied[bid, :3] = [F, 0, 0]
        mujoco.mj_step(m, d)
        d.xfrc_applied[bid, :] = 0.0
    mujoco.mj_step(m, d)  # guard step
    # After force window + guard step, read base linear velocity x
    v_after = float(d.qvel[dofadr + 0]) if jfree >= 0 else 0.0
    dv_measured = float(v_after - v_before)
    ok = abs(dv_measured - abs(deltav)) <= 0.12 and bool((d.xfrc_applied == 0).all())
    return {"test": "push_momentum", "ok": bool(ok), "dv_measured": dv_measured, "dv_target": abs(deltav), "dt": dt, "steps": steps_push, "xfrc_zero": bool((d.xfrc_applied == 0).all())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--test", choices=["gravity", "limits", "static", "freefall", "push", "all"], default="all")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--deltav", type=float, default=0.2)
    ap.add_argument("--push-steps", type=int, default=10)
    args = ap.parse_args()

    out: Dict[str, Any] = {}
    if args.test in ("gravity", "all"):
        out["gravity_mass"] = test_gravity_mass(args.xml)
    if args.test in ("limits", "all"):
        out["actuator_limits"] = test_actuator_limits(args.xml)
    if args.test in ("static", "all"):
        out["static_weight"] = test_static_weight(args.xml, seconds=args.seconds)
    if args.test in ("freefall", "all"):
        out["freefall"] = test_freefall(args.xml, seconds=min(0.25, args.seconds))
    if args.test in ("push", "all"):
        out["push_momentum"] = test_push_momentum(args.xml, deltav=args.deltav, steps_push=args.push_steps)
    # Overall ok if all present subtests are ok
    oks = [v.get("ok", False) for v in out.values()]
    overall_ok = all(oks) if oks else True
    out["ok"] = bool(overall_ok)
    print(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
