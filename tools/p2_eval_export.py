#!/usr/bin/env python3
from __future__ import annotations

"""Export headless evaluation summaries for canonical scenarios.

Writes artifacts/episodes.json (one record per scenario) and
artifacts/summary.json (lightweight aggregate) using the same headless
preview engine as tools/p2_verify_motion.py.
"""

import json, math, sys
from pathlib import Path
import argparse

import mujoco

from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.p2_verify_motion import run_preview


def run_scenario(xml: str, seconds: float, amp_scale: float, omega: float, v_cmd: float, yaw_cmd: float, gates: dict):
    out = run_preview(xml, seconds=seconds, amp_scale=amp_scale, omega=omega,
                      lock_coxa=True, lift=0.0, duty=1.0,
                      enforce_stride=False, substeps=48, ik_iters=30,
                      v_cmd=v_cmd, yaw_cmd=yaw_cmd, gates=gates)
    out2 = run_preview(xml, seconds=seconds, amp_scale=amp_scale, omega=omega,
                       lock_coxa=False, lift=0.0, duty=1.0,
                       enforce_stride=True, substeps=2, ik_iters=10,
                       v_cmd=v_cmd, yaw_cmd=yaw_cmd, gates=gates)
    return {"locked": out, "unlocked": out2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--amp-scale", type=float, default=0.028)
    ap.add_argument("--omega", type=float, default=0.05)
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--seed", type=int, default=638109)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    episodes = []
    # Gate defaults aligned with P2-FINAL v2.0
    gates = {
        "slip_dist_mean": 1.0e-3,
        "slip_dist_max":  2.0e-3,
        "slip_spd_mean":  5.0e-4,
        "slip_spd_p95":   1.5e-3,
        "yaw_share_straight_mean": 0.20,
        "yaw_share_straight_peak": 0.35,
        "yaw_share_curve_mean": 0.35,
        "yaw_share_curve_peak": 0.50,
        "yaw_share_turn_mean": 0.60,
        "yaw_share_turn_peak": 0.75,
        "exc_fem_deg": 6.0,
        "exc_tib_deg": 10.0,
        "vz_mean": 5.0e-4,
        "vz_p95":  1.5e-3,
    }
    for name, (v, w) in {
        "straight": (0.12, 0.0),
        "turn": (0.0, 0.2),
        "curve": (0.12, 0.3),
    }.items():
        res = run_scenario(args.xml, args.seconds, args.amp_scale, args.omega, v, w, gates)
        episodes.append({"scenario": name, "seconds": args.seconds, "amp_scale": args.amp_scale, "omega": args.omega, **res})

    # Save episodes and lightweight summary
    ep_path = Path(args.outdir) / "episodes.json"
    sm_path = Path(args.outdir) / "summary.json"
    with ep_path.open("w") as f:
        json.dump(episodes, f, indent=2)
    # Also write JSONL for quick scanning
    epjl_path = Path(args.outdir) / "episodes.jsonl"
    with epjl_path.open("w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    ok = all(ep["unlocked"].get("ok_v2", False) for ep in episodes)
    # Determinism stamp
    import subprocess, hashlib
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        git_sha = "unknown"
    try:
        cfg = Path("configs/policy/workspace_circle.json").read_bytes()
        cfg_sha = hashlib.sha256(cfg).hexdigest()
    except Exception:
        cfg_sha = "unknown"
    try:
        mjv = mujoco.__version__ if hasattr(mujoco, "__version__") else "unknown"
    except Exception:
        mjv = "unknown"
    stamp = {
        "git_sha": git_sha,
        "config_sha256": cfg_sha,
        "mujoco_version": mjv,
    }
    with sm_path.open("w") as f:
        json.dump({"ok": bool(ok), "count": len(episodes), **stamp}, f, indent=2)
    print(json.dumps({"episodes": str(ep_path), "episodes_jsonl": str(epjl_path), "summary": str(sm_path), "ok": bool(ok), **stamp}, indent=2))


if __name__ == "__main__":
    main()
