#!/usr/bin/env python3
import json
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    schema_path = repo_root / "eval/eval_schema_v0.json"
    runs_dir = repo_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / "eval_a3_smoke.json"

    schema = json.loads(schema_path.read_text()) if schema_path.exists() else {"required": []}
    required = list(schema.get("required", []))
    optional = list(schema.get("optional", []))

    # Prepare a minimal record satisfying required keys; values are placeholders
    record = {}
    for k in required:
        if k == "pass_no_termination":
            record[k] = True
        else:
            record[k] = 0.0
    # Include optionals with benign defaults
    for k in optional:
        if k not in record:
            record[k] = 0.0

    out_path.write_text(json.dumps(record, indent=2))
    print(json.dumps({"path": str(out_path), "keys": sorted(list(record.keys()))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
