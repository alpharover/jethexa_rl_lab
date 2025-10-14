#!/usr/bin/env python3
from __future__ import annotations

"""Probe CPG phases and basic slip/scuff counters over a short run."""

import argparse
import sys
import json
import math
from collections import defaultdict

import numpy as np
import mujoco

# Ensure repo root import when run as a script
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from control.cpg_circle import CirclePolicy, TripodPhaser, foot_target_hi1
from tools.p0p1_verify import LEG_ORDER, hip_frame_axes, paper_circle_params, link_lengths_from_xml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    a1, a2, a3 = link_lengths_from_xml(m)
    zH = float(np.mean([d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"coxa_{s}{p}")][2] for s,p in [("L","F"),("L","M"),("L","R"),("R","F"),("R","M"),("R","R")]]))
    r_paper, d_cw = paper_circle_params(a1, a2, a3, zH)
    pol = CirclePolicy(r_ctrl=r_paper*1.4, d_cw=d_cw)
    phaser = TripodPhaser(("LF","RM","LR"), ("RF","LM","RR"))

    t = 0.0
    dt = float(m.opt.timestep)
    phi = 0.0
    phi_rate = pol.omega * dt

    scuff = defaultdict(int)

    steps = int(args.seconds / dt)
    for _ in range(steps):
        # advance phases and produce targets (not applied to actuators here; probe only)
        phi = (phi + phi_rate) % (2 * math.pi)
        for leg in LEG_ORDER:
            hip_pos, hx, hy, hz = hip_frame_axes(m, d, leg)
            phi_leg = phaser.phase_for_leg(leg, phi)
            p_hi1 = foot_target_hi1(phi_leg, pol)
            # convert to world for diagnostics
            p_world = hip_pos + p_hi1[0] * hx + p_hi1[1] * hy + p_hi1[2] * hz
            # simple scuff heuristic: if z target below hip plane, count
            if p_hi1[2] <= pol.height + 1e-4:
                scuff[leg] += 1
        mujoco.mj_step(m, d)
        t += dt

    print(json.dumps({"scuff_counts": scuff, "seconds": args.seconds}, indent=2))


if __name__ == "__main__":
    main()
