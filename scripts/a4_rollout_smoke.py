#!/usr/bin/env python3
import os, time, json, hashlib
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl")
import mujoco
import numpy as np
xml = "mjcf/jethexa_lab.xml"
m = mujoco.MjModel.from_xml_path(xml)
d = mujoco.MjData(m)
T = 2.0; dt = m.opt.timestep; steps = int(T/dt)
qpos0, qvel0 = d.qpos.copy(), d.qvel.copy()
t0 = time.time()
for _ in range(steps):
  mujoco.mj_step(m, d)  # zero control
dur = time.time() - t0
fps = steps/dur if dur>0 else float("inf")
h = hashlib.sha256(d.qpos.tobytes()+d.qvel.tobytes()).hexdigest()[:16]
out = {"steps":steps, "duration_s":round(dur,3), "fps":fps, "hash":h, "egl":os.getenv("MUJOCO_GL")}
Path(".proofs").mkdir(exist_ok=True)
Path(".proofs/a4_rollout.json").write_text(json.dumps(out, indent=2))
print(out)
