#!/usr/bin/env python3
import json
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    schema_path = repo_root / 'eval/eval_schema_v0.json'
    runs_dir = repo_root / 'runs'
    out_path = repo_root / '.proofs/gate_schema_v0.json'

    schema = json.loads(schema_path.read_text()) if schema_path.exists() else {"required": []}
    required = list(schema.get('required', []))
    files = sorted(runs_dir.glob('eval*.json')) if runs_dir.exists() else []

    ok = 0
    records = []
    for p in files:
        try:
            d = json.loads(p.read_text())
            miss = [k for k in required if k not in d]
            is_ok = len(miss) == 0
            ok += int(is_ok)
            records.append({"path": str(p), "ok": is_ok, "missing": miss})
        except Exception as e:
            records.append({
                "path": str(p),
                "ok": False,
                "error": f"{e.__class__.__name__}: {e}",
                "missing": required,
            })

    ratio = (ok / len(files)) if files else 0.0
    report = {"total": len(files), "ok_count": ok, "ok_ratio": round(ratio, 3), "records": records}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    raise SystemExit(main())
