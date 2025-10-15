import json, os, subprocess

XML = os.path.join("mjcf", "jethexa_lab.xml")


def run(cmd):
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _last_json(stdout: str):
    lines = [l for l in stdout.splitlines() if l.strip().startswith("{")]
    return json.loads(lines[-1]) if lines else {}


def mjpython():
    if os.path.exists("ci-mjpython"):
        return "./ci-mjpython"
    return "mjpython"


def _amp_scale_from_policy(target_radius=0.003):
    cfg = os.path.join("configs", "policy", "workspace_circle.json")
    try:
        data = json.loads(open(cfg).read())
        r_ctrl = float(data.get("r_ctrl", 0.06))
        if r_ctrl > 1e-6:
            amp = target_radius / r_ctrl
            return float(max(0.02, min(0.10, amp)))
    except Exception:
        pass
    return 0.06


def _base_cmd(v_cmd: float, yaw_cmd: float):
    amp = f"{_amp_scale_from_policy():.5f}"
    return [
        mjpython(),
        "tools/p2_verify_motion.py",
        "--xml",
        XML,
        "--seconds",
        "8.0",
        "--omega",
        "0.05",
        "--amp-scale",
        amp,
        "--v-cmd",
        f"{v_cmd}",
        "--yaw-cmd",
        f"{yaw_cmd}",
        "--gate-slip-rel-pct",
        "3.0",
        "--gate-slip-abs-mm",
        "4.0",
        "--gate-slip-speed-mean-mmps",
        "1.5",
        "--gate-yaw-share-straight",
        "0.25",
        "--gate-yaw-share-curve",
        "0.25",
        "--gate-yaw-share-turn",
        "0.45",
        "--gate-exc-femur-deg",
        "8",
        "--gate-exc-tibia-deg",
        "12",
        "--gate-vz-p95-mmps",
        "2.0",
        "--geo-xyrms-mm",
        "3",
        "--geo-ring-mean-mm",
        "2",
        "--geo-ground-mean-mm",
        "1",
        "--p2final_v21",
    ]


def test_turn_in_place_final():
    if not os.path.exists(XML):
        return
    p = run(_base_cmd(v_cmd=0.0, yaw_cmd=0.2))
    assert p.returncode == 0, p.stderr
    out = _last_json(p.stdout)
    assert out["ok"] is True
    assert out["unlocked"]["ok_v2"] is True


def test_straight_line_final():
    if not os.path.exists(XML):
        return
    p = run(_base_cmd(v_cmd=0.12, yaw_cmd=0.0))
    assert p.returncode == 0, p.stderr
    out = _last_json(p.stdout)
    assert out["ok"] is True
    assert out["unlocked"]["ok_v2"] is True


def test_curved_path_final():
    if not os.path.exists(XML):
        return
    p = run(_base_cmd(v_cmd=0.12, yaw_cmd=0.3))
    assert p.returncode == 0, p.stderr
    out = _last_json(p.stdout)
    assert out["ok"] is True
    assert out["unlocked"]["ok_v2"] is True


def test_parity_harness():
    if not os.path.exists(XML):
        return
    cmd = _base_cmd(v_cmd=0.12, yaw_cmd=0.0)
    cmd += ["--parity", "--parity-tol", "0.10"]
    p = run(cmd)
    assert p.returncode == 0, p.stderr
    out = _last_json(p.stdout)
    assert out.get("parity", {}).get("ok", False) is True
