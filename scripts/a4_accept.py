#!/usr/bin/env python3
import json, sys
from pathlib import Path

def j(p): return json.loads(Path(p).read_text())
comp = j(".proofs/a4_mj_compile.json")
roll = j(".proofs/a4_rollout.json")
seed = j(".proofs/a4_seed.json")
ok_compile = comp.get("ok_compile", False)
ok_no_fallback = str(comp.get("xml_used","" )).endswith("mjcf/jethexa_lab.xml")
ok_fps = roll.get("fps", 0) >= 50.0
ok_seed = seed.get("ok_deterministic", False)
ok_no_ckpts = not (Path("runs/checkpoints").exists() and any(Path("runs/checkpoints").iterdir()))
report = {
  "ok_mj_compile": ok_compile,
  "ok_no_fallback": ok_no_fallback,
  "ok_rollout_fps_ge_50": ok_fps,
  "ok_seed_deterministic": ok_seed,
  "ok_no_ckpts": ok_no_ckpts,
  "pass_all": all([ok_compile, ok_no_fallback, ok_fps, ok_seed, ok_no_ckpts])
}
Path(".proofs/A4_ACCEPT.json").write_text(json.dumps(report, indent=2))
print(report)
sys.exit(0 if report["pass_all"] else 2)
