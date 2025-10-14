#!/usr/bin/env python3
from __future__ import annotations

"""Headless baseline runner for the circle-CPG + IK controller.

This is a minimal harness that steps the model for a short duration and logs a
few summary stats. Intended for local validation during P2.
"""

import argparse
import sys
import json
import math
from pathlib import Path

import numpy as np
import mujoco

# Ensure repo root when invoked as a script
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from control.cpg_circle import CirclePolicy, TripodPhaser, foot_target_hi1
from control.ik_numeric import numeric_ik_hi1
from tools.p0p1_verify import LEG_ORDER, link_lengths_from_xml, hip_frame_axes, paper_circle_params


def set_ctrl_targets_from_q(m, d, joint_name_to_q, dctrl_max: float | None = None):
    # Position actuators named "pos_<joint>"
    for jname, qref in joint_name_to_q.items():
        aname = f"pos_{jname}"
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        if aid >= 0:
            lo, hi = m.actuator_ctrlrange[aid]
            raw = float(np.clip(qref, lo, hi))
            if dctrl_max is not None:
                prev = float(d.ctrl[aid])
                raw = float(np.clip(raw, prev - dctrl_max, prev + dctrl_max))
            d.ctrl[aid] = raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="configs/env/hex_v0.json")
    ap.add_argument("--circle-policy", default="configs/policy/workspace_circle.json")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--logdir", default="runs/cpg_v1")
    args = ap.parse_args()

    cfg = json.loads(Path(args.env).read_text())
    xml = cfg["xml_path"]
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    pol_cfg = json.loads(Path(args.circle_policy).read_text())
    a1, a2, a3 = link_lengths_from_xml(m)
    zH = float(np.mean([d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"coxa_{s}{p}")][2] for s,p in [("L","F"),("L","M"),("L","R"),("R","F"),("R","M"),("R","R")]]))
    r_paper, d_cw = paper_circle_params(a1, a2, a3, zH)
    pol = CirclePolicy(
        r_paper=float(pol_cfg.get("r_paper", r_paper)),
        r_inscribed_min=float(pol_cfg.get("r_inscribed_min", r_paper)),
        r_ctrl=float(pol_cfg.get("r_ctrl", r_paper)),
        s=float(pol_cfg.get("s", 1.0)),
        alpha=float(pol_cfg.get("alpha", 0.8)),
        d_cw=d_cw,
        height=0.0,
    )
    phaser = TripodPhaser(("LF","RM","LR"), ("RF","LM","RR"))

    dt = float(m.opt.timestep)
    steps = int(args.seconds / dt)
    base_phi = 0.0
    phi_rate = pol.omega * dt

    omega_max = 5.82  # rad/s at 11.1V -> HX-35H spec
    dctrl_max = omega_max * dt

    for k in range(steps):
        base_phi = (base_phi + phi_rate) % (2.0 * math.pi)
        # per-leg foot target in {Hi1} then IK to joint refs
        for leg in LEG_ORDER:
            hip_pos, hx, hy, hz = hip_frame_axes(m, d, leg)
            phi_leg = phaser.phase_for_leg(leg, base_phi)
            p_hi1 = foot_target_hi1(phi_leg, pol)
            # numeric IK in the hip-local frame
            q = numeric_ik_hi1(p_hi1, a1, a2, a3)
            # set actuator targets
            set_ctrl_targets_from_q(m, d, {
                f"coxa_joint_{leg}": q[0],
                f"femur_joint_{leg}": q[1],
                f"tibia_joint_{leg}": q[2],
            }, dctrl_max=dctrl_max)
        mujoco.mj_step(m, d)

    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    (Path(args.logdir)/"summary.json").write_text(json.dumps({"seconds": args.seconds}, indent=2)+"\n")


if __name__ == "__main__":
    main()
