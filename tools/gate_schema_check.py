#!/usr/bin/env python3
import json, os
from pathlib import Path
from typing import Dict, List, Any

OLD_ROOT = Path("/Users/alpha_dev/robotics_repos/jethexa/jethexa_mj_lab")
OUT_DIR = Path("/Users/alpha_dev/robotics_repos/jethexa/jethexa_rl/.proofs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

required = [
    "edge_rho_p5",
    "duty_asym_p90",
    "com_to_edge_p10",
    "pass_no_termination",
]

cand_dirs = [OLD_ROOT / "rl2" / "runs", OLD_ROOT / "remote_artifacts", OLD_ROOT / "archive"]
json_files: List[Path] = []
for base in cand_dirs:
    if base.is_dir():
        for p in base.rglob("eval*.json"):
            if p.is_file():
                json_files.append(p)

records: List[Dict[str, Any]] = []
for p in json_files:
    rec: Dict[str, Any] = {"path": str(p)}
    try:
        with p.open("r") as f:
            data = json.load(f)
        missing = [k for k in required if k not in data]
        rec["missing"] = missing
        rec["ok"] = len(missing) == 0
    except Exception as e:
        rec["error"] = f"{e.__class__.__name__}: {e}"
        rec["ok"] = False
        rec["missing"] = required
    records.append(rec)

missing_tally: Dict[str, int] = {k: 0 for k in required}
for r in records:
    for k in r.get("missing", []):
        missing_tally[k] += 1

ok_count = sum(1 for r in records if r.get("ok"))
total = len(records)
ok_ratio = (ok_count / total) if total else 0.0
report: Dict[str, Any] = {
    "total": total,
    "ok_count": ok_count,
    "ok_ratio": round(ok_ratio, 3),
    "ok_ge_0p75": ok_ratio >= 0.75,
    "required_fields": required,
    "missing_tally": missing_tally,
    "samples": records[:25],
}
if total == 0:
    report["reason"] = "No eval JSON files found under expected directories"
elif ok_ratio < 0.75:
    report["reason"] = f"Insufficient schema coverage: ok={ok_count}/{total}"

out_path = OUT_DIR / "gate_schema.json"
with out_path.open("w") as f:
    json.dump(report, f, indent=2)
print("Wrote {} (ok_ratio={})".format(out_path, report["ok_ratio"]))
