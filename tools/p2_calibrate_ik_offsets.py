#!/usr/bin/env python3
from __future__ import annotations

"""Calibrate per-leg analytic-IK → engine joint angle zero offsets.

For each leg, sample a small ring inside the controller circle, solve
analytic IK in hip-local {Hi1} and numeric IK in engine coordinates,
then fit a robust per-joint offset delta so that:

   q_engine ≈ S · (q_hi1 + delta)

Writes configs/calib/ik_offsets.json with per-leg {sign, delta} and XML hash.
"""

import argparse, json, math, os, sys, hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import mujoco

# repo-root imports
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.p0p1_verify import LEG_ORDER, hip_frame_axes, link_lengths_from_xml, paper_circle_params
from control.ik_analytic import LegGeom, choose_branch


def angle_wrap(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def engine_numeric_ik_leg(m, d, leg: str, target_world: np.ndarray,
                          iters: int = 40, lam: float = 3e-4, alpha: float = 0.9) -> np.ndarray:
    j_coxa = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"coxa_joint_{leg}")
    j_fem  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"femur_joint_{leg}")
    j_tib  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"tibia_joint_{leg}")
    dof_all = [int(m.jnt_dofadr[j_coxa]), int(m.jnt_dofadr[j_fem]), int(m.jnt_dofadr[j_tib])]
    qaddrs = [int(m.jnt_qposadr[j_coxa]), int(m.jnt_qposadr[j_fem]), int(m.jnt_qposadr[j_tib])]
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}")
    b_foot = int(m.geom_bodyid[gid])
    for _ in range(max(1, iters)):
        mujoco.mj_forward(m, d)
        Rb = d.xmat[b_foot].reshape(3,3)
        p_curr = d.xpos[b_foot] + Rb @ m.geom_pos[gid]
        err = np.asarray(target_world, float) - p_curr
        if float(np.linalg.norm(err)) < 2e-4:
            break
        Jp = np.zeros((3, m.nv)); Jr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, Jp, Jr, b_foot)
        cols = [dof_all[0], dof_all[1], dof_all[2]]
        J = Jp[:, cols]
        H = J.T @ J + (lam**2) * np.eye(3)
        dq = np.linalg.solve(H, J.T @ err)
        dq = np.clip(dq, -0.2, 0.2)
        for k in range(3):
            adr = qaddrs[k]
            lo, hi = m.jnt_range[[j_coxa, j_fem, j_tib][k]]
            d.qpos[adr] = float(np.clip(d.qpos[adr] + alpha * dq[k], lo, hi))
    return np.array([d.qpos[qaddrs[0]], d.qpos[qaddrs[1]], d.qpos[qaddrs[2]]], float)


def compute_xml_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--out", default="configs/calib/ik_offsets.json")
    ap.add_argument("--angles", type=int, default=16)
    ap.add_argument("--radii", type=int, default=2)
    args = ap.parse_args()

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    a1,a2,a3 = link_lengths_from_xml(m)
    zH = float(np.mean([hip_frame_axes(m,d,leg)[0][2] for leg in LEG_ORDER]))
    r_paper, d_cw = paper_circle_params(a1,a2,a3,zH)

    # Sample angles and radii (inside r_paper to stay away from limits)
    ths = np.linspace(-math.pi, math.pi, int(args.angles), endpoint=False)
    rads = np.linspace(0.6*r_paper, 0.9*r_paper, int(args.radii))

    out: Dict[str, Dict[str, List[float]]] = {}
    for leg in LEG_ORDER:
        hip_pos, hx, hy, hz = hip_frame_axes(m, d, leg)
        # joint sign from axis z
        s = []
        for jn in (f"coxa_joint_{leg}", f"femur_joint_{leg}", f"tibia_joint_{leg}"):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            s.append(1.0 if m.jnt_axis[jid][2] >= 0 else -1.0)
        s = np.array(s, float)

        deltas: List[np.ndarray] = []
        for r in rads:
            for th in ths:
                # hip-local target on ring at ground
                p_hi1 = np.array([d_cw + r*math.cos(th), r*math.sin(th), -zH], float)
                # analytic IK (prefer knee-flex), then numeric engine to world target
                q_hi1 = choose_branch(p_hi1.copy(), LegGeom(a1,a2,a3,0.0), (
                    tuple(m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"coxa_joint_{leg}")]),
                    tuple(m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"femur_joint_{leg}")]),
                    tuple(m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"tibia_joint_{leg}")]),
                ))
                p_w = hip_pos + p_hi1[0]*hx + p_hi1[1]*hy + p_hi1[2]*hz
                q_eng = engine_numeric_ik_leg(m, d, leg, p_w, iters=40)
                dq = s * q_eng - q_hi1
                deltas.append(angle_wrap(dq))
        deltas = np.stack(deltas)
        # robust location: median
        delta = np.median(deltas, axis=0)
        out[leg] = {"sign": s.tolist(), "delta": delta.tolist()}

    meta = {
        "xml": args.xml,
        "xml_sha256": compute_xml_sha256(args.xml),
        "angles": int(args.angles),
        "radii": int(args.radii),
    }
    result = {"meta": meta, "legs": out}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"ok": True, "saved": str(args.out), **meta}, separators=(",", ":")))


if __name__ == "__main__":
    main()

