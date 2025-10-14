#!/usr/bin/env python3
import json, sys, glob
from pathlib import Path

ROOT = Path(".")
proofs = ROOT/".proofs"
proofs.mkdir(exist_ok=True)

def contains(path, needle):
    try:
        return needle in Path(path).read_text()
    except Exception:
        return False

def trend_ok(mfile):
    vals = []
    with open(mfile) as f:
        for line in f:
            try:
                vals.append(json.loads(line)["loss"])
            except Exception:
                pass
    if len(vals) < 20:
        return False, {"reason":"too_few_points","n":len(vals)}
    n = len(vals)
    head = sum(vals[: max(10, n//10) ])/max(1, min(10, n//10))
    tail = sum(vals[-max(10, n//10): ])/max(1, min(10, n//10))
    ratio = (head / max(1e-9, tail))
    return (ratio >= 3.0), {"head":head, "tail":tail, "ratio":ratio}

ok_guard_pass  = contains(".proofs/a6_guard_pass.log", "fresh-start accepted")
ok_guard_block = contains(".proofs/a6_guard_block.log", "RESUME BLOCKED")
ok_no_ckpts    = (len(glob.glob("runs/*.npz")) + len(glob.glob("runs/checkpoints/*.npz")))==0

# Schema check
eval_path = Path("runs/eval_a6_baseline.json")
try:
    e = json.loads(eval_path.read_text())
    req = ["edge_rho_p5","duty_asym_p90","pass_no_termination"]
    ok_eval_schema = all(k in e for k in req)
except Exception:
    ok_eval_schema = False

ok_trend, trend_meta = trend_ok("runs/update_metrics_a6.jsonl")

accept = {
  "ok_guard_pass": bool(ok_guard_pass),
  "ok_guard_block": bool(ok_guard_block),
  "ok_loss_improves": bool(ok_trend),
  "ok_eval_schema": bool(ok_eval_schema),
  "ok_no_ckpts": bool(ok_no_ckpts),
  "trend": trend_meta,
}
accept["pass_all"] = all(accept[k] for k in list(accept.keys()) if k.startswith("ok_"))

(proofs/"A6_ACCEPT.json").write_text(json.dumps(accept, indent=2))
print(json.dumps(accept, indent=2))
if not accept["pass_all"]:
    sys.exit(2)
