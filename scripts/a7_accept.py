#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import glob


ROOT = Path(".")


def parse_guard_flags(path: Path):
    flags = {}
    try:
        for line in path.read_text().splitlines():
            if "=" in line:
                k, v = line.strip().split("=", 1)
                flags[k.strip()] = v.strip()
    except Exception:
        pass
    return flags


def load_metrics(path: Path):
    rows = []
    try:
        with path.open() as f:
            for ln in f:
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    pass
    except Exception:
        pass
    return rows


def rl_signal_present(metrics_rows):
    if not metrics_rows:
        return False
    # Condition (a): ep_return_mean improved by >= 100 absolute across updates
    rets = [r.get("ep_return_mean", 0.0) for r in metrics_rows]
    if len(rets) >= 2 and (max(rets) - min(rets) >= 100.0):
        return True
    # Condition (b): explained_variance increases by >= 0.1 absolute across updates
    evs = [r.get("explained_variance", 0.0) for r in metrics_rows]
    if len(evs) >= 2 and (max(evs) - min(evs) >= 0.1):
        return True
    # Condition (c): approx_kl in [0.0, 0.02] AND clip_fraction in [0.05, 0.3] for at least half updates
    kls = [r.get("approx_kl", 1.0) for r in metrics_rows]
    clips = [r.get("clip_fraction", 0.0) for r in metrics_rows]
    good = 0
    for kl, cf in zip(kls, clips):
        if (0.0 <= float(kl) <= 0.02) and (0.05 <= float(cf) <= 0.3):
            good += 1
    if good >= max(1, len(metrics_rows) // 2):
        return True
    return False


def schema_ok_eval(eval_path: Path):
    req_keys = {"env_id", "seed", "episodes", "return_mean", "return_std", "len_mean", "len_std"}
    try:
        data = json.loads(eval_path.read_text())
        return set(data.keys()) == req_keys
    except Exception:
        return False


def main():
    proofs = ROOT / ".proofs"
    proofs.mkdir(parents=True, exist_ok=True)

    metrics_file = ROOT / "runs" / "update_metrics_a7.jsonl"
    eval_file = ROOT / "runs" / "eval_a7_baseline.json"
    guard_file = ROOT / "runs" / "guard_a7.log"

    rows = load_metrics(metrics_file)
    flags = parse_guard_flags(guard_file)

    # guards_clean: device present and finite flags true
    device_present = bool(flags.get("device"))
    grads_finite = flags.get("grads_finite", "false").lower() == "true"
    params_finite = flags.get("params_finite", "false").lower() == "true"
    nan_in_loss = flags.get("nan_in_loss", "true").lower() == "false"
    nan_in_grad = flags.get("nan_in_grad", "true").lower() == "false"
    guards_clean = device_present and grads_finite and params_finite and nan_in_loss and nan_in_grad

    # rl_signal_present according to criteria
    rl_signal = rl_signal_present(rows)

    # schema_ok
    schema_ok = schema_ok_eval(eval_file)

    # no checkpoints anywhere under runs/
    no_ckpts = (len(glob.glob("runs/ckpt*")) + len(glob.glob("runs/*.npz"))) == 0

    # fresh_start
    fresh_start = flags.get("fresh_start", "false").lower() == "true"

    env_id = None
    if rows:
        env_id = rows[0].get("env_id", None)
    if env_id is None:
        try:
            env_id = json.loads(eval_file.read_text()).get("env_id")
        except Exception:
            env_id = "unknown"

    accept = {
        "name": "A7_PPO_MuJoCo_Smoke",
        "env_id": env_id,
        "gates": {
            "guards_clean": bool(guards_clean),
            "rl_signal_present": bool(rl_signal),
            "schema_ok": bool(schema_ok),
            "no_checkpoints": bool(no_ckpts),
            "fresh_start": bool(fresh_start),
        },
        "evidence": {
            "metrics_file": str(metrics_file),
            "eval_file": str(eval_file),
            "guard_logs": str(guard_file),
        },
    }
    accept["pass_all"] = all(accept["gates"].values())

    proof_path = proofs / "A7_ACCEPT.json"
    proof_path.write_text(json.dumps(accept, indent=2))
    print(json.dumps(accept, indent=2))
    if not accept["pass_all"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
