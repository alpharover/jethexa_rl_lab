#!/usr/bin/env python3
from __future__ import annotations

"""Tune controller circle radius from paper and sampled inscribed metrics."""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

import mujoco
import numpy as np

# Ensure repo root when invoked as a script (import namespace package 'tools')
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.p0p1_verify import LEG_ORDER, link_lengths_from_xml, hip_world_height, paper_circle_params, run_workspace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--out", default="configs/policy/workspace_circle.json")
    ap.add_argument("--scale", type=float, default=1.4)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    a1, a2, a3 = link_lengths_from_xml(m)
    zH = float(np.mean([hip_world_height(m, d, leg) for leg in LEG_ORDER]))
    r_paper, d_cw = paper_circle_params(a1, a2, a3, zH)

    report, _ = run_workspace(m, d, LEG_ORDER, samples=512, safety_scale=args.scale)
    r_inscribed_min = float(min(r["r_inscribed_min_m"] for r in report))
    alpha = 0.80
    r_ctrl = float(min(args.scale * r_paper, alpha * r_inscribed_min))

    out = {
        "r_paper": float(r_paper),
        "r_inscribed_min": float(r_inscribed_min),
        "s": float(args.scale),
        "alpha": float(alpha),
        "r_ctrl": float(r_ctrl),
        "hip_z_nom": float(zH),
        "a": [float(a1), float(a2), float(a3)],
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"saved": str(args.out), **out}, indent=2))


if __name__ == "__main__":
    main()
