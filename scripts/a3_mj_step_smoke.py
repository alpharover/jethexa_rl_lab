#!/usr/bin/env python3
import os, time, json, argparse
from pathlib import Path
import mujoco
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml', default='mjcf/jethexa_lab.xml')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--log', default='runs/update_metrics.jsonl')
    ap.add_argument('--proof', default='.proofs/a3_mj_sim.json')
    args = ap.parse_args()

    Path('runs').mkdir(exist_ok=True)
    Path('.proofs').mkdir(exist_ok=True)

    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)

    start = time.time()
    with open(args.log, 'a') as flog:
        for i in range(args.steps):
            mujoco.mj_step(m, d)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                fps = (i + 1) / max(elapsed, 1e-9)
                rec = {"i": i + 1, "sim_time": float(d.time), "fps": float(fps)}
                flog.write(json.dumps(rec) + "\n")

    duration = time.time() - start
    fps = args.steps / max(duration, 1e-9)
    proof = {
        "steps": args.steps,
        "duration_s": round(duration, 3),
        "fps": round(fps, 1),
        "egl": os.getenv("MUJOCO_GL", "unset"),
        "ok_fps_ge_50": bool(fps >= 50.0)
    }
    with open(args.proof, 'w') as f:
        json.dump(proof, f, indent=2)
    print(f"[a3_mj_step_smoke] {args.steps} steps @ {fps:.1f} fps ; EGL={proof['egl']}")
if __name__ == "__main__":
    main()
