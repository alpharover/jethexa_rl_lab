#!/usr/bin/env python3
import os, json, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl")
import mujoco
xml = Path("mjcf/jethexa_lab.xml")
try:
  m = mujoco.MjModel.from_xml_path(str(xml))
  info = {
    "xml_used": str(xml),
    "njnt": int(m.njnt),
    "nq": int(m.nq),
    "ngeom": int(m.ngeom),
    "nmesh": int(m.nmesh),
    "egl": os.getenv("MUJOCO_GL"),
    "ok_compile": True
  }
except Exception as e:
  info = {"xml_used": str(xml), "ok_compile": False, "error": f"{type(e).__name__}: {e}"}
Path(".proofs").mkdir(exist_ok=True)
Path(".proofs/a4_mj_compile.json").write_text(json.dumps(info, indent=2))
print(info)
sys.exit(0 if info.get("ok_compile") else 2)
