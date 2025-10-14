#!/usr/bin/env python3
import os, json, hashlib
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl")
import mujoco, numpy as np, jax, jax.numpy as jnp

def run_once():
  m = mujoco.MjModel.from_xml_path("mjcf/jethexa_lab.xml")
  d = mujoco.MjData(m)
  steps = int(1.0/m.opt.timestep)
  for _ in range(steps):
    mujoco.mj_step(m, d)
  return hashlib.sha256(d.qpos.tobytes()+d.qvel.tobytes()).hexdigest()[:16]

# Seed both frameworks (JAX split shown for future parity)
np.random.seed(0); _ = jax.random.split(jax.random.PRNGKey(0), 2)
h1 = run_once()
np.random.seed(0); _ = jax.random.split(jax.random.PRNGKey(0), 2)
h2 = run_once()
ok = (h1==h2)
out = {"hash1":h1, "hash2":h2, "ok_deterministic": ok}
Path(".proofs/a4_seed.json").write_text(json.dumps(out, indent=2))
print(out)
exit(0 if ok else 2)
