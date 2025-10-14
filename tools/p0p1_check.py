#!/usr/bin/env python3
"""
P0–P1 check kit for JetHexa MuJoCo model.

Features (CLI entry points):
- Determinism smoke: run twice and compare state hash
- PD step response: position-actuator step on a named joint
- Workspace audit: sample hip->foot XY reach and compare to analytic circle
- Safe push: apply a finite impulse (Δv) to a body over K steps, clearing xfrc each step

Examples:
  mjpython tools/p0p1_check.py --xml mjcf/jethexa_lab.xml --determinism
  mjpython tools/p0p1_check.py --xml mjcf/jethexa_lab.xml --pd-step femur_joint_LF --step 0.2 --settle 1.0 --plot
  mjpython tools/p0p1_check.py --xml mjcf/jethexa_lab.xml --workspace LF --plot
  mjpython tools/p0p1_check.py --xml mjcf/jethexa_lab.xml --push robot --dir x --deltav 0.2 --steps 10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import mujoco as mj
except Exception as e:  # pragma: no cover
    print(f"ERROR: mujoco import failed: {e}")
    sys.exit(2)


ARTIFACTS = os.path.join(os.getcwd(), "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)


# -------------------------- Utilities --------------------------

def load(xml_path: str) -> Tuple[mj.MjModel, mj.MjData]:
    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)
    return m, d


def state_bytes(m: mj.MjModel, d: mj.MjData) -> bytes:
    parts: List[np.ndarray] = [
        np.array([d.time], dtype=np.float64),
        np.array(d.qpos, dtype=np.float64),
        np.array(d.qvel, dtype=np.float64),
        np.array(d.ctrl, dtype=np.float64),
    ]
    return b"|".join(p.tobytes() for p in parts)


def state_hash(m: mj.MjModel, d: mj.MjData) -> str:
    return hashlib.sha256(state_bytes(m, d)).hexdigest()


def actuator_index_by_name(m: mj.MjModel) -> Dict[str, int]:
    names: Dict[str, int] = {}
    for i in range(m.nu):
        nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_ACTUATOR, i)
        names[nm] = i
    return names


def joint_info(m: mj.MjModel, joint_name: str) -> Tuple[int, int]:
    j_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, joint_name)
    qadr = int(m.jnt_qposadr[j_id])
    return j_id, qadr


def body_info(m: mj.MjModel, body_name: str) -> int:
    return mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, body_name)


def geom_info(m: mj.MjModel, geom_name: str) -> int:
    return mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, geom_name)


# ----------------------- Determinism check ----------------------

def run_determinism(xml: str, steps: int = 1000) -> Dict[str, object]:
    m, d = load(xml)
    # First run
    mj.mj_forward(m, d)
    for _ in range(steps):
        mj.mj_step(m, d)
    h0 = state_hash(m, d)
    # Re-init and run again
    m2, d2 = load(xml)
    mj.mj_forward(m2, d2)
    for _ in range(steps):
        mj.mj_step(m2, d2)
    h1 = state_hash(m2, d2)
    return {"equal": h0 == h1, "hash0": h0, "hash1": h1, "steps": steps}


# ----------------------- PD step response -----------------------

@dataclass
class PDResult:
    joint: str
    step: float
    kp: Optional[float]
    kv: Optional[float]
    t90: Optional[float]
    overshoot: Optional[float]
    image: Optional[str]


def _actuator_params(m: mj.MjModel, aid: int) -> Tuple[Optional[float], Optional[float]]:
    # MuJoCo position actuators use gain/bias params; XML can set kp/kv attributes.
    kp = None
    kv = None
    try:
        # Newer MuJoCo stores kp in actuator_gainprm[aid,0] for position servos
        kp = float(m.actuator_gainprm[aid, 0])
    except Exception:
        pass
    try:
        kv = float(m.actuator_biasprm[aid, 1])
    except Exception:
        pass
    return kp, kv


def run_pd_step(xml: str, joint_name: str, step: float, settle: float, plot: bool) -> PDResult:
    import numpy as np
    m, d = load(xml)
    act_name = f"pos_{joint_name}"
    aid = actuator_index_by_name(m).get(act_name)
    if aid is None:
        raise RuntimeError(f"Actuator '{act_name}' not found")
    j_id, qadr = joint_info(m, joint_name)

    mj.mj_forward(m, d)
    # Warmup
    for _ in range(20):
        mj.mj_step(m, d)

    dt = float(m.opt.timestep)
    N = max(1, int(round(settle / dt)))
    t = np.linspace(0.0, settle, num=N, endpoint=False)
    q = np.zeros(N, dtype=float)
    ctrl = np.zeros(N, dtype=float)

    # Step to target and hold
    for i in range(N):
        d.ctrl[aid] = step
        ctrl[i] = float(d.ctrl[aid])
        mj.mj_step(m, d)
        q[i] = float(d.qpos[qadr])

    kp, kv = _actuator_params(m, aid)

    # Metrics: t90 and overshoot
    t90 = None
    overshoot = None
    if step != 0.0:
        target = step
        # first time reaching 90% of target with correct sign
        sign = 1.0 if target >= 0 else -1.0
        thr = 0.9 * abs(target)
        for i in range(N):
            if sign * q[i] >= thr:
                t90 = i * dt
                break
        overshoot = (np.max(sign * q) - abs(target)) / max(1e-9, abs(target))

    # Plot if requested
    img = None
    if plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 3))
            plt.plot(t, q, label="q")
            plt.axhline(step, color="k", ls="--", label="target")
            plt.xlabel("t [s]")
            plt.ylabel(f"{joint_name} [rad]")
            plt.legend()
            img = os.path.join(ARTIFACTS, f"pd_step_{joint_name}.png")
            plt.tight_layout()
            plt.savefig(img)
            plt.close()
        except Exception as e:  # pragma: no cover
            img = None
            print(f"[warn] plotting failed: {e}")

    return PDResult(joint=joint_name, step=step, kp=kp, kv=kv, t90=t90, overshoot=overshoot, image=img)


# ----------------------- Workspace audit ------------------------

def _leg_names(leg: str) -> Tuple[str, str, str, str]:
    leg = leg.upper()
    return (
        f"coxa_joint_{leg}",
        f"femur_joint_{leg}",
        f"tibia_joint_{leg}",
        f"foot_{leg}",
    )


def _link_lengths(m: mj.MjModel, leg: str) -> Tuple[float, float, float]:
    # a1: |coxa->femur| parent->child body_pos in parent's frame
    # a2: |femur->tibia|
    # a3: |tibia->foot geom| (geom_pos in tibia body)
    femur_body = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"femur_{leg}")
    coxa_body = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"coxa_{leg}")
    tibia_body = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"tibia_{leg}")
    a1 = float(np.linalg.norm(m.body_pos[femur_body]))  # femur pos in coxa frame
    a2 = float(np.linalg.norm(m.body_pos[tibia_body]))  # tibia pos in femur frame
    foot_geom = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, f"foot_{leg}")
    a3 = float(np.linalg.norm(m.geom_pos[foot_geom]))  # foot geom in tibia frame
    return a1, a2, a3


def _hip_height(m: mj.MjModel, d: mj.MjData, leg: str) -> float:
    # world Z of the coxa body frame
    bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"coxa_{leg}")
    return float(d.xpos[bid][2])


def _foot_xy_radius_samples(m: mj.MjModel, d: mj.MjData, leg: str, samples: int = 800) -> np.ndarray:
    rng = np.random.default_rng(0)
    coxa, femur, tibia, foot = _leg_names(leg)
    jids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, jn) for jn in (coxa, femur, tibia)]
    addrs = [int(m.jnt_qposadr[j]) for j in jids]
    ranges = [m.jnt_range[j] for j in jids]
    foot_gid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, foot)
    coxa_bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"coxa_{leg}")

    # Save/restore state around sampling
    qsave = np.copy(d.qpos)
    try:
        radii = np.zeros(samples, dtype=float)
        for i in range(samples):
            for k, a in enumerate(addrs):
                lo, hi = ranges[k]
                d.qpos[a] = rng.uniform(lo, hi)
            mj.mj_forward(m, d)
            foot_p = np.array(d.geom_xpos[foot_gid])
            hip_p = np.array(d.xpos[coxa_bid])
            dx, dy = foot_p[0] - hip_p[0], foot_p[1] - hip_p[1]
            radii[i] = math.hypot(dx, dy)
        return radii
    finally:
        d.qpos[:] = qsave
        mj.mj_forward(m, d)


def run_workspace(xml: str, leg: str, plot: bool) -> Dict[str, object]:
    m, d = load(xml)
    mj.mj_forward(m, d)
    leg = leg.upper()
    a1, a2, a3 = _link_lengths(m, leg)
    z_H = _hip_height(m, d, leg)
    L = a2 + a3
    # Analytic circle (per instructions): beta from hip height, diameter from projected length
    beta = math.asin(max(-1.0, min(1.0, z_H / max(1e-9, L))))
    d_analytic = (L * math.cos(beta)) / 2.0
    r_analytic = d_analytic / 2.0

    radii = _foot_xy_radius_samples(m, d, leg)
    r90 = float(np.quantile(radii, 0.90))
    r95 = float(np.quantile(radii, 0.95))
    rmax = float(np.max(radii))

    img = None
    if plot:
        try:
            import matplotlib.pyplot as plt
            theta = np.linspace(0, 2*math.pi, 200)
            cx = a1 + d_analytic/2.0
            circ_x = cx + r_analytic * np.cos(theta)
            circ_y = 0 + r_analytic * np.sin(theta)
            # Sample a subset for scatter
            idx = np.linspace(0, len(radii)-1, num=min(1000, len(radii)), dtype=int)
            # Build XY from radii assuming circle center on +X
            # This is illustrative; main metric is radius comparison
            plt.figure(figsize=(4,4))
            plt.scatter(radii[idx], np.zeros_like(idx), s=6, label="r samples", alpha=0.3)
            plt.plot(circ_x, circ_y, 'r--', label="analytic circle")
            plt.axis('equal')
            plt.title(f"Workspace {leg}: r95={r95:.3f} (m) vs anal r={r_analytic:.3f}")
            plt.legend()
            img = os.path.join(ARTIFACTS, f"workspace_{leg}.png")
            plt.tight_layout()
            plt.savefig(img)
            plt.close()
        except Exception as e:  # pragma: no cover
            img = None
            print(f"[warn] plotting failed: {e}")

    return {
        "leg": leg,
        "a1": a1, "a2": a2, "a3": a3,
        "hip_z": z_H,
        "r90": r90, "r95": r95, "rmax": rmax,
        "analytic": {"beta": beta, "diameter": d_analytic, "radius": r_analytic},
        "image": img,
    }


# ------------------------- Safe push ---------------------------

def run_push(xml: str, body: str, direction: str, deltav: float, steps: int) -> Dict[str, object]:
    m, d = load(xml)
    mj.mj_forward(m, d)
    bid = body_info(m, body)
    dt = float(m.opt.timestep)
    mass = float(m.body_mass[bid])
    # Direction vector
    dir_map = {
        'x': np.array([1,0,0], dtype=float),
        '+x': np.array([1,0,0], dtype=float),
        '-x': np.array([-1,0,0], dtype=float),
        'y': np.array([0,1,0], dtype=float),
        '+y': np.array([0,1,0], dtype=float),
        '-y': np.array([0,-1,0], dtype=float),
    }
    vdir = dir_map.get(direction.lower())
    if vdir is None:
        raise ValueError("direction must be one of: x, -x, y, -y")

    J = mass * float(deltav)
    K = max(1, int(steps))
    Fmag = J / (dt * K)
    F = Fmag * vdir

    # Clear xfrc_applied each step, apply only for K steps
    for i in range(K):
        d.xfrc_applied[:] = 0.0
        d.xfrc_applied[bid, 0:3] = F
        mj.mj_step(m, d)
    # Guard step to ensure cleared
    d.xfrc_applied[:] = 0.0
    mj.mj_step(m, d)

    return {
        "body": body, "mass": mass, "dt": dt,
        "deltav": float(deltav), "steps": int(K),
        "impulse": float(J), "force_peak": float(Fmag),
        "xfrc_applied_zero_after": bool(np.allclose(d.xfrc_applied, 0.0)),
    }


# --------------------------- CLI ------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Path to MJCF XML")
    ap.add_argument("--determinism", action="store_true", help="Run determinism smoke")
    ap.add_argument("--pd-step", dest="pd_joint", default=None, help="Run PD step on joint name (e.g., femur_joint_LF)")
    ap.add_argument("--step", type=float, default=0.2, help="Step magnitude (rad) for PD test")
    ap.add_argument("--settle", type=float, default=1.0, help="Duration (s) to hold the step")
    ap.add_argument("--workspace", default=None, help="Leg tag for workspace audit (e.g., LF)")
    ap.add_argument("--plot", action="store_true", help="Save plots under ./artifacts")
    ap.add_argument("--push", default=None, help="Body name for safe push (e.g., robot)")
    ap.add_argument("--dir", dest="push_dir", default="x", help="Push direction: x,-x,y,-y")
    ap.add_argument("--deltav", type=float, default=0.2, help="Target Δv (m/s) for push")
    ap.add_argument("--steps", type=int, default=10, help="Number of steps to spread impulse over")
    args = ap.parse_args()

    out: Dict[str, object] = {}
    if args.determinism:
        out["determinism"] = run_determinism(args.xml)
    if args.pd_joint:
        r = run_pd_step(args.xml, args.pd_joint, args.step, args.settle, args.plot)
        out["pd_step"] = {
            "joint": r.joint,
            "step": r.step,
            "kp": r.kp,
            "kv": r.kv,
            "t90": r.t90,
            "overshoot": r.overshoot,
            "image": r.image,
        }
    if args.workspace:
        out["workspace"] = run_workspace(args.xml, args.workspace, args.plot)
    if args.push:
        out["push"] = run_push(args.xml, args.push, args.push_dir, args.deltav, args.steps)

    if not out:
        ap.print_help()
        return
    print(json.dumps(out, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()

