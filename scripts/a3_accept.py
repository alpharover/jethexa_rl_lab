#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def file_ok(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def any_exists(paths):
    return any(file_ok(p) for p in paths)


def scan_no_ckpts(repo_root: Path) -> bool:
    runs = repo_root / 'runs'
    if not runs.exists():
        return True
    patterns = [
        '/*.ckpt', '/**/*.ckpt',
        '/*.pt', '/**/*.pt',
        '/*.pth', '/**/*.pth',
        '/*checkpoint*', '/**/*checkpoint*',
        '/*ckpt*', '/**/*ckpt*',
    ]
    for pat in patterns:
        if list(runs.glob(pat)):
            return False
    return True


def main():
    repo_root = Path(__file__).resolve().parent.parent
    proofs = repo_root / '.proofs'

    # ok_guard_* inferred from available logs (A2 or A3)
    ok_guard_pass = any_exists([
        proofs / 'a3_guard_pass.log',
        proofs / 'a2_guard_pass.log',
        proofs / 'guard_pass.log',
    ])
    ok_guard_block = any_exists([
        proofs / 'a3_guard_block.log',
        proofs / 'a2_guard_block.log',
        proofs / 'guard_block.log',
    ])

    # ok_mj_sim from a3_mj_sim.json
    ok_mj_sim = False
    mj_path = proofs / 'a3_mj_sim.json'
    if mj_path.exists():
        try:
            mj = json.loads(mj_path.read_text())
            ok_mj_sim = bool(mj.get('ok', False))
        except Exception:
            ok_mj_sim = False

    # ok_eval_schema from gate_schema_v0.json
    ok_eval_schema = False
    schema_gate = proofs / 'gate_schema_v0.json'
    if schema_gate.exists():
        try:
            g = json.loads(schema_gate.read_text())
            total = int(g.get('total', 0))
            ok_count = int(g.get('ok_count', 0))
            ok_eval_schema = (total > 0 and ok_count > 0)
        except Exception:
            ok_eval_schema = False

    ok_no_ckpts = scan_no_ckpts(repo_root)

    pass_all = all([ok_guard_pass, ok_guard_block, ok_mj_sim, ok_eval_schema, ok_no_ckpts])
    out = {
        'ok_guard_pass': ok_guard_pass,
        'ok_guard_block': ok_guard_block,
        'ok_mj_sim': ok_mj_sim,
        'ok_eval_schema': ok_eval_schema,
        'ok_no_ckpts': ok_no_ckpts,
        'pass_all': pass_all,
    }

    out_path = proofs / 'A3_ACCEPT.json'
    proofs.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out))
    return 0 if pass_all else 2


if __name__ == '__main__':
    sys.exit(main())
