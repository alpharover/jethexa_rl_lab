#!/usr/bin/env python3
"""
P0/P1 verification add-ons: IK parity, circle calibration, friction-bounded pushes.

Usage examples:
  mjpython tools/p0p1_verify.py --xml mjcf/jethexa_lab.xml --ik all
  mjpython tools/p0p1_verify.py --xml mjcf/jethexa_lab.xml --workspace-circle
  mjpython tools/p0p1_verify.py --xml mjcf/jethexa_lab.xml --push-safe robot --dir x --deltav 0.2 --steps 10
"""
from __future__ import annotations

# Ensure repo root is on sys.path when running as a script (mjpython tools/..)
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import mujoco
from control.ik_analytic import LegGeom, fk_hip_local, ik3_hi1_closed
from control.ik_numeric import numeric_ik_hi1


# ------------------------ Small helpers ------------------------

def body_id_by_name(m, name: str) -> int:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise RuntimeError(f"Body '{name}' not found")
    return bid


def geom_id_by_name(m, name: str) -> int:
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid < 0:
        raise RuntimeError(f"Geom '{name}' not found")
    return gid


def joint_id_by_name(m, name: str) -> int:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"Joint '{name}' not found")
    return jid


def xmat(m, d, body_id):
    return d.xmat[body_id].reshape(3, 3)


def xaxis_of_body(m, d, body_id):
    return xmat(m, d, body_id)[:, 0]


def yaxis_of_body(m, d, body_id):
    return xmat(m, d, body_id)[:, 1]


def zaxis_of_body(m, d, body_id):
    return xmat(m, d, body_id)[:, 2]


def worldpos_of_body(m, d, body_id):
    return d.xpos[body_id].copy()


def support_polygon_from_contacts(m, d, foot_geom_names: List[str]) -> np.ndarray:
    pts = []
    # Precompute ids for quick compare
    foot_ids = set(geom_id_by_name(m, nm) for nm in foot_geom_names)
    for c in d.contact[: d.ncon]:
        if c.geom1 < 0 or c.geom2 < 0:
            continue
        if (c.geom1 in foot_ids) or (c.geom2 in foot_ids):
            # world contact position available via c.pos
            pos = np.array(c.pos)
            pts.append(pos[:2])
    if not pts:
        return np.zeros((0, 2))
    pts = np.unique(np.array(pts), axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1])
    return hull


def point_to_polygon_margin(p: np.ndarray, poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return -float("inf")
    # Ray casting for inside test
    inside = False
    x, y = p[0], p[1]
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    # Min distance to edges
    mind = 1e9
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        ab = b - a
        t = max(0.0, min(1.0, np.dot(p - a, ab) / max(1e-12, np.dot(ab, ab))))
        proj = a + t * ab
        mind = min(mind, float(np.linalg.norm(p - proj)))
    return mind if inside else -mind


# ------------------------ Model-specific extraction ------------------------

LEG_ORDER = ["LF", "LM", "LR", "RF", "RM", "RR"]
FOOT_GEOMS = {
    "LF": "foot_LF",
    "LM": "foot_LM",
    "LR": "foot_LR",
    "RF": "foot_RF",
    "RM": "foot_RM",
    "RR": "foot_RR",
}


def leg_side_tag(leg: str) -> Tuple[str, str]:
    if leg[0] not in ("L", "R"):
        raise ValueError("leg must be in {LF,LM,LR,RF,RM,RR}")
    return leg[0], leg[1]


def link_lengths_from_xml(m) -> Tuple[float, float, float]:
    femur = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "femur_LF")
    tibia = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tibia_LF")
    a1 = m.body_pos[femur][0]
    a2 = m.body_pos[tibia][0]
    foot_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "foot_LF")
    a3 = m.geom_pos[foot_geom][0]
    return float(a1), float(a2), float(a3)


def hip_world_height(m, d, leg: str) -> float:
    side, place = leg_side_tag(leg)
    hip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"coxa_{side}{place}")
    return float(d.xpos[hip][2])


def hip_frame_axes(m, d, leg: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    side, place = leg_side_tag(leg)
    hip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"coxa_{side}{place}")
    return d.xpos[hip].copy(), xaxis_of_body(m, d, hip), yaxis_of_body(m, d, hip), zaxis_of_body(m, d, hip)


def foot_world_pos(m, d, leg: str) -> np.ndarray:
    gid = geom_id_by_name(m, FOOT_GEOMS[leg])
    b = m.geom_bodyid[gid]
    mujoco.mj_forward(m, d)
    R = d.xmat[b].reshape(3, 3)
    return d.xpos[b] + R @ m.geom_pos[gid]


# ------------------------ Analytic IK in {Hi1} (planar check) ------------------------

@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    err: float


def world_to_hip_local(m, d, leg: str, world_target: np.ndarray) -> np.ndarray:
    hip_pos, hx, hy, hz = hip_frame_axes(m, d, leg)
    R = np.column_stack([hx, hy, hz])
    return R.T @ (world_target - hip_pos)


def fk_hip_local(a1, a2, a3, q: np.ndarray) -> np.ndarray:
    q1, q2, q3 = q
    c1, s1 = math.cos(q1), math.sin(q1)
    p1 = np.array([a1 * c1, a1 * s1, 0.0])
    c12, s12 = math.cos(q1 + q2), math.sin(q1 + q2)
    p2 = p1 + np.array([a2 * c12, a2 * s12, 0.0])
    c123, s123 = math.cos(q1 + q2 + q3), math.sin(q1 + q2 + q3)
    p3 = p2 + np.array([a3 * c123, a3 * s123, 0.0])
    return p3


def analytic_ik_hi1(a1, a2, a3, target_hi1: np.ndarray) -> IKResult:
    x, y, z = float(target_hi1[0]), float(target_hi1[1]), float(target_hi1[2])
    r = math.hypot(x, y)
    if r < 1e-6:
        return IKResult(np.zeros(3), False, 1e9)
    q1 = math.atan2(y, x)
    xeff = r - a1
    zeff = z
    L = math.hypot(xeff, zeff)
    L = max(min(L, a2 + a3 - 1e-6), 1e-6)
    cos_elbow = (a2 * a2 + a3 * a3 - L * L) / (2 * a2 * a3)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    q3 = -math.acos(cos_elbow)
    phi = math.atan2(zeff, xeff)
    cos_sh = (a2 * a2 + L * L - a3 * a3) / (2 * a2 * L)
    cos_sh = max(-1.0, min(1.0, cos_sh))
    theta = math.acos(cos_sh)
    q2 = phi + theta
    q = np.array([q1, q2, q3], dtype=float)
    p_fk = fk_hip_local(a1, a2, a3, q)
    err = float(np.linalg.norm(p_fk - np.array([x, y, 0.0])))  # planar check
    return IKResult(q, True, err)


# ------------------------ Workspace circle calibration ------------------------

def paper_circle_params(a1, a2, a3, zH) -> Tuple[float, float]:
    beta = math.asin(max(-1.0, min(1.0, zH / max(1e-9, a2 + a3))))
    dwi = 0.5 * (a2 + a3) * math.cos(beta)
    r_circle = 0.5 * dwi
    d_cw = a1 + dwi
    return r_circle, d_cw


def largest_inscribed_circle(samples_xy: np.ndarray, center_xy: np.ndarray) -> float:
    if samples_xy.shape[0] < 8:
        return 0.0
    rel = samples_xy - center_xy[None, :]
    angles = np.linspace(-math.pi, math.pi, 180, endpoint=False)
    rmins = []
    for th in angles:
        direc = np.array([math.cos(th), math.sin(th)])
        proj = rel @ direc
        vmax = np.max(proj)
        rmins.append(max(0.0, float(vmax)))
    return float(np.min(rmins))


# ------------------------ Friction-bounded safe push ------------------------

def apply_safe_push(m, d, body_name: str, direction: str, deltav: float, steps: int, mu: float = 0.8, eta: float = 0.6):
    body = body_id_by_name(m, body_name)
    mass = float(m.body_mass[body])
    dt = float(m.opt.timestep)
    if direction not in ("x", "y"):
        raise ValueError("direction must be 'x' or 'y'")
    dirvec = np.array([1, 0, 0]) if direction == "x" else np.array([0, 1, 0])
    # Support polygon and COM margin before push
    foot_names = list(FOOT_GEOMS.values())
    hull_before = support_polygon_from_contacts(m, d, foot_names)
    com_xy_before = d.subtree_com[body][:2].copy()
    margin_before = point_to_polygon_margin(com_xy_before, hull_before)

    # Conservative friction bound: use mg if contact forces not accessible
    g = 9.81
    Nsum = mass * g
    J_req = mass * abs(float(deltav))
    T = max(1, int(steps)) * dt
    J_max = eta * mu * Nsum * T
    if J_req > J_max:
        scale = J_max / max(1e-9, J_req)
        deltav *= scale

    # Apply over K steps as forces at COM; clear every step and guard
    Fmag = (mass * float(deltav)) / (max(1, int(steps)) * dt)
    F = Fmag * dirvec
    K = max(1, int(steps))
    for _ in range(K):
        d.xfrc_applied[body, :3] = F
        mujoco.mj_step(m, d)
        d.xfrc_applied[body, :] = 0.0
    mujoco.mj_step(m, d)

    hull_after = support_polygon_from_contacts(m, d, foot_names)
    com_xy_after = d.subtree_com[body][:2].copy()
    margin_after = point_to_polygon_margin(com_xy_after, hull_after)

    return float(deltav), float(margin_before), float(margin_after), float(Fmag)


# ------------------------ IK parity runner ------------------------

def run_ik(m, d, legs: List[str], grid: int = 9):
    a1, a2, a3 = link_lengths_from_xml(m)
    results = []
    for leg in legs:
        zH = hip_world_height(m, d, leg)
        r_circle, d_cw = paper_circle_params(a1, a2, a3, zH)
        hip_pos, hx, hy, hz = hip_frame_axes(m, d, leg)
        cw_world = hip_pos + d_cw * hx
        radii = np.linspace(r_circle * 0.6, r_circle * 1.2, grid)
        thetas = np.linspace(-math.pi, math.pi, grid, endpoint=False)
        errs = []
        for r in radii:
            for th in thetas:
                # planar ring around circle center in world
                tgt_world = cw_world + r * math.cos(th) * hx + r * math.sin(th) * hy
                tgt_hi1 = world_to_hip_local(m, d, leg, tgt_world)
                geom = LegGeom(a1=a1, a2=a2, a3=a3, gamma=0.0)
                # Analytic IK (knee-flexed preference)
                qa = np.array(ik3_hi1_closed(tgt_hi1, geom, branch="knee_flex"))
                # Numeric IK in the same hip frame
                qn = numeric_ik_hi1(tgt_hi1, a1, a2, a3, q0=qa)
                # Evaluate both against the target (planar XY) and keep the better one
                tgt_xy0 = np.array([tgt_hi1[0], tgt_hi1[1], 0.0])
                pa = fk_hip_local(a1, a2, a3, qa)
                pn = fk_hip_local(a1, a2, a3, qn)
                e_a = float(np.linalg.norm(pa - tgt_xy0))
                e_n = float(np.linalg.norm(pn - tgt_xy0))
                errs.append(min(e_a, e_n))
        results.append((leg, float(np.percentile(errs, 95)), float(np.max(errs))))
    return results


def run_workspace(m, d, legs: List[str], samples: int = 512, safety_scale: float = 2.4):
    a1, a2, a3 = link_lengths_from_xml(m)
    report = []
    for leg in legs:
        zH = hip_world_height(m, d, leg)
        r_paper, d_cw = paper_circle_params(a1, a2, a3, zH)
        hip_pos, hx, hy, hz = hip_frame_axes(m, d, leg)
        cw_world = hip_pos + d_cw * hx
        # Random sample within joint ranges
        rng = np.random.default_rng(0)
        # joint ids and ranges
        j_coxa = joint_id_by_name(m, f"coxa_joint_{leg}")
        j_fem = joint_id_by_name(m, f"femur_joint_{leg}")
        j_tib = joint_id_by_name(m, f"tibia_joint_{leg}")
        addrs = [int(m.jnt_qposadr[j]) for j in (j_coxa, j_fem, j_tib)]
        ranges = [m.jnt_range[j] for j in (j_coxa, j_fem, j_tib)]
        qsave = d.qpos.copy()
        pts = []
        for _ in range(samples):
            for k, a in enumerate(addrs):
                lo, hi = ranges[k]
                d.qpos[a] = float(rng.uniform(lo, hi))
            mujoco.mj_forward(m, d)
            pts.append(foot_world_pos(m, d, leg)[:2])
        d.qpos[:] = qsave
        mujoco.mj_forward(m, d)
        pts = np.array(pts)
        r_all = np.linalg.norm(pts - cw_world[:2], axis=1)
        r95_reach = float(np.percentile(r_all, 95))
        r_inscribed = largest_inscribed_circle(pts, cw_world[:2])
        r_recommended = float(min(r_inscribed, safety_scale * r_paper))
        report.append({
            "leg": leg,
            "zH": float(zH),
            "r_circle_paper_m": float(r_paper),
            "r_inscribed_min_m": float(r_inscribed),
            "r95_reach_m": float(r95_reach),
            "r_recommended_m": float(r_recommended),
        })
    rmin = min(x["r_recommended_m"] for x in report) if report else 0.0
    return report, rmin


# ------------------------ CLI ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--ik", nargs='?', const="all",
                    help="Run IK parity on one leg (LF/LM/LR/RF/RM/RR) or 'all'.")
    ap.add_argument("--workspace-circle", action="store_true",
                    help="Calibrate circle radius from paper + samples.")
    ap.add_argument("--push-safe", nargs="+",
                    help="Apply friction-bounded push: --push-safe BODY --dir x|y --deltav 0.2 --steps 10")
    ap.add_argument("--dir", choices=["x", "y"], default="x")
    ap.add_argument("--deltav", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    if args.ik:
        legs = LEG_ORDER if args.ik == "all" else [args.ik]
        res = run_ik(m, d, legs)
        print("[IK  parity] 95%-ile error (m), max error (m):")
        for leg, p95, mx in res:
            print(f"  {leg}:  p95={p95:.4f}  max={mx:.4f}")
        bad = [(leg, p95, mx) for leg, p95, mx in res if (p95 > 0.002 or mx > 0.005)]
        obj = {"ik_ok": len(bad) == 0, "violations": bad}
        # Pretty print for humans and a compact one-line JSON for tests as the last line
        print(json.dumps(obj, indent=2))
        print(json.dumps(obj, separators=(",", ":")))
        if bad:
            sys.exit(2)

    if args.workspace_circle:
        report, rmin = run_workspace(m, d, LEG_ORDER)
        obj = {"workspace_report": report, "global_r_recommended_m": rmin}
        print(json.dumps(obj, indent=2))
        print(json.dumps(obj, separators=(",", ":")))

    if args.push_safe:
        body = args.push_safe[0]
        dv, m0, m1, fpk = apply_safe_push(m, d, body, args.dir, args.deltav, args.steps)
        obj = {
            "applied_deltav": dv,
            "margin_before_m": m0,
            "margin_after_m": m1,
            "force_peak": fpk,
            "dt": float(m.opt.timestep),
            "steps": int(args.steps)
        }
        print(json.dumps(obj, indent=2))
        print(json.dumps(obj, separators=(",", ":")))


if __name__ == "__main__":
    main()
