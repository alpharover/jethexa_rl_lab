#!/usr/bin/env python3
from __future__ import annotations

"""Tune controller circle radius from paper/inscribed metrics, with sweeps.

Adds optional sweeps over alpha (inscribed scaling) and scale s (paper
scaling). When sweeping, runs a short headless preview for each candidate to
collect gates and emits artifacts for audit.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any
from datetime import datetime

import mujoco
import numpy as np

# Ensure repo root when invoked as a script (import namespace package 'tools')
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.p0p1_verify import LEG_ORDER, link_lengths_from_xml, hip_world_height, paper_circle_params, run_workspace
from tools.p2_verify_motion import run_preview


def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _run_short_preview(xml: str, seconds: float = 6.0, amp_scale: float = 0.06) -> Dict[str, Any]:
    # Use small amplitude and default omega; check both locked and unlocked cases
    res = run_preview(xml, seconds=seconds, amp_scale=amp_scale, omega=0.05,
                      lock_coxa=True, lift=0.0, duty=1.0,
                      enforce_stride=False, substeps=24, ik_iters=20,
                      v_cmd=0.0, yaw_cmd=0.0)
    res2 = run_preview(xml, seconds=seconds, amp_scale=amp_scale, omega=0.05,
                       lock_coxa=False, lift=0.0, duty=1.0,
                       enforce_stride=True, substeps=2, ik_iters=10,
                       v_cmd=0.0, yaw_cmd=0.0)
    return {"locked": res, "unlocked": res2, "ok": bool(res.get("ok", False) and res2.get("limits_ok", True))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--out", default="configs/policy/workspace_circle.json")
    ap.add_argument("--scale", type=float, default=2.4, help="Paper scale s when not sweeping")
    ap.add_argument("--alpha", type=float, default=1.0, help="Inscribed scale alpha when not sweeping")
    ap.add_argument("--alpha-sweep", nargs=2, type=float, default=None, metavar=("ALPHA_MIN","ALPHA_MAX"),
                    help="Sweep alpha in [min,max] (inclusive). Implies safety audit.")
    ap.add_argument("--alpha-steps", type=int, default=13, help="Number of alpha samples in sweep")
    ap.add_argument("--scale-sweep", nargs=2, type=float, default=None, metavar=("S_MIN","S_MAX"),
                    help="Sweep paper scale s in [min,max] (inclusive). Optional.")
    ap.add_argument("--scale-steps", type=int, default=9, help="Number of s samples in sweep")
    ap.add_argument("--band", nargs=2, type=float, default=(0.12, 0.16), metavar=("R_MIN","R_MAX"),
                    help="Target r_ctrl band to pin (meters)")
    ap.add_argument("--artifacts", default="artifacts/circle_sweep.jsonl", help="Where to write sweep results")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    a1, a2, a3 = link_lengths_from_xml(m)
    zH = float(np.mean([hip_world_height(m, d, leg) for leg in LEG_ORDER]))
    r_paper, d_cw = paper_circle_params(a1, a2, a3, zH)

    # Sample inscribed radii once (joint limits dependent, but good proxy)
    report, _ = run_workspace(m, d, LEG_ORDER, samples=384, safety_scale=max(1.0, args.scale))
    r_inscribed_min = float(min(r["r_inscribed_min_m"] for r in report))

    # Helper to compute r_ctrl and metrics
    def r_from(s: float, alpha: float) -> float:
        return float(min(s * r_paper, alpha * r_inscribed_min))

    band_lo, band_hi = float(args.band[0]), float(args.band[1])

    # Sweep if requested
    chosen: Dict[str, Any] | None = None
    if args.alpha_sweep or args.scale_sweep:
        alphas = [args.alpha]
        if args.alpha_sweep:
            a0, a1s = float(args.alpha_sweep[0]), float(args.alpha_sweep[1])
            if args.alpha_steps <= 1:
                alphas = [a0]
            else:
                alphas = list(np.linspace(a0, a1s, args.alpha_steps))
        scales = [args.scale]
        if args.scale_sweep:
            s0, s1s = float(args.scale_sweep[0]), float(args.scale_sweep[1])
            if args.scale_steps <= 1:
                scales = [s0]
            else:
                scales = list(np.linspace(s0, s1s, args.scale_steps))

        # Prepare artifact sink
        art_path = Path(args.artifacts)
        _ensure_dir(art_path)
        # Evaluate grid
        best_key = None
        for s in scales:
            for alpha in alphas:
                r_ctrl = r_from(s, alpha)
                # Quick reject if outside a generous envelope
                if r_ctrl < 0.06 or r_ctrl > 0.22:
                    rec = {
                        "s": float(s), "alpha": float(alpha), "r_ctrl": float(r_ctrl),
                        "skip": True, "reason": "r_ctrl_out_of_envelope"
                    }
                    with art_path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    continue
                # Temporarily write a policy with this candidate so run_preview picks it up
                tmp = {
                    "r_paper": float(r_paper),
                    "r_inscribed_min": float(r_inscribed_min),
                    "s": float(s), "alpha": float(alpha), "r_ctrl": float(r_ctrl),
                    "hip_z_nom": float(zH), "a": [float(a1), float(a2), float(a3)],
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "_sweep": True
                }
                # Save to the same out path so p2_verify_motion reads it
                Path(args.out).write_text(json.dumps(tmp) + "\n")
                metrics = _run_short_preview(args.xml, seconds=6.0, amp_scale=0.06)
                locked = metrics.get("locked", {})
                unlocked = metrics.get("unlocked", {})
                ok_band = (band_lo <= r_ctrl <= band_hi)
                ok_locked = bool(locked.get("ring_ok", False) and locked.get("ground_ok", False) and locked.get("tracking_ok", False))
                ok_unlocked = bool(unlocked.get("limits_ok", False) and unlocked.get("contact_ok", False))
                ok = bool(ok_locked and ok_unlocked)
                rec = {
                    "s": float(s), "alpha": float(alpha), "r_ctrl": float(r_ctrl),
                    "band_ok": ok_band, "ok_locked": ok_locked, "ok_unlocked": ok_unlocked, "ok": ok,
                    "locked": locked, "unlocked": unlocked
                }
                with art_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                # Selection heuristic: prefer in‑band and ok; among those, smallest r_ctrl; tie‑break by lowest locked.ring.mean
                def key_tuple(r):
                    inband = 0 if r["band_ok"] else 1
                    notok = 0 if r["ok"] else 1
                    ring_mean = max([v.get("mean_abs", 1.0) for v in r.get("locked", {}).get("ring", {}).values()] or [1.0])
                    return (inband, notok, r["r_ctrl"], ring_mean)
                if best_key is None or key_tuple(rec) < best_key:
                    best_key = key_tuple(rec)
                    chosen = rec

        if chosen is None:
            # Fall back to current settings if no candidate evaluated
            chosen = {"s": float(args.scale), "alpha": float(args.alpha), "r_ctrl": r_from(args.scale, args.alpha), "ok": False}

        # Persist best candidate to config
        out = {
            "r_paper": float(r_paper),
            "r_inscribed_min": float(r_inscribed_min),
            "s": float(chosen["s"]),
            "alpha": float(chosen.get("alpha", args.alpha)),
            "r_ctrl": float(chosen["r_ctrl"]),
            "hip_z_nom": float(zH),
            "a": [float(a1), float(a2), float(a3)],
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "audit": {
                "band": [band_lo, band_hi],
                "ok": bool(chosen.get("ok", False)),
                "artifacts": str(art_path)
            }
        }
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps({"saved": str(args.out), **out}, indent=2))

    else:
        alpha = float(args.alpha)
        r_ctrl = r_from(args.scale, alpha)
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
