import os, json
from pathlib import Path

# GPU/JAX probe
import jax, jax.numpy as jnp

Path('.proofs').mkdir(exist_ok=True)

devs = [str(d) for d in jax.devices()]
x = jnp.ones((512, 512)); y = jnp.ones((512, 512))
val = jax.jit(lambda a,b: (a @ b).sum())(x, y).block_until_ready()
with open('.proofs/a2_jax_cuda.json', 'w') as f:
    json.dump({"devices": devs, "sum": float(val)}, f, indent=2)

# MuJoCo EGL probe
os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
xml = "<mujoco><worldbody><body name='b'><geom type='sphere' size='0.01'/></body></worldbody></mujoco>"
_ = mujoco.MjModel.from_xml_string(xml)
with open('.proofs/a2_mujoco_egl.json', 'w') as f:
    json.dump({"ok": True, "gl": os.getenv("MUJOCO_GL")}, f, indent=2)
