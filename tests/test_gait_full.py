import json, subprocess, os
import math

XML = os.path.join("mjcf", "jethexa_lab.xml")

def run(cmd):
    return subprocess.run(cmd, check=False, capture_output=True, text=True)

def _last_json(stdout: str):
    lines = [l for l in stdout.splitlines() if l.strip().startswith("{")]
    return json.loads(lines[-1]) if lines else {}

def mjpython():
    # Prefer hermetic runner if present (AGENTS.md runner), else fall back to system mjpython
    if os.path.exists("ci-mjpython"):
        return "./ci-mjpython"
    return "mjpython"

def _amp_scale_from_policy(target_radius=0.003):
    """Choose amp-scale so r_ctrl * amp-scale ~= target_radius (meters)."""
    cfg = os.path.join("configs","policy","workspace_circle.json")
    try:
        data = json.loads(open(cfg).read())
        r_ctrl = float(data.get("r_ctrl", 0.06))
        if r_ctrl > 1e-6:
            amp = target_radius / r_ctrl
            return float(max(0.02, min(0.10, amp)))
    except Exception:
        pass
    return 0.06


def test_turn_in_place():
    if not os.path.exists(XML):
        return
    amp = f"{_amp_scale_from_policy():.5f}"
    p = run([mjpython(), "tools/p2_verify_motion.py", "--xml", XML,
             "--seconds", "8.0", "--omega", "0.05", "--amp-scale", amp,
             "--v-cmd", "0.0", "--yaw-cmd", "0.2"])
    assert p.returncode == 0, p.stderr
    out = _last_json(p.stdout)
    # Headline gates for turn-in-place: ring compliance + contact quality + limit-time <= 3%
    assert out["locked"]["ok"] is True
    assert out["unlocked"]["limits_ok"] is True
    assert out["unlocked"]["contact_ok"] is True


def test_straight_line():
    if not os.path.exists(XML):
        return
    amp = f"{_amp_scale_from_policy():.5f}"
    p = run([mjpython(), "tools/p2_verify_motion.py", "--xml", XML,
             "--seconds", "8.0", "--omega", "0.05", "--amp-scale", amp,
             "--v-cmd", "0.15", "--yaw-cmd", "0.0"])
    assert p.returncode == 0, p.stderr
    out = _last_json(p.stdout)
    # For the preview gait, use existing gates as proxies: tracking + ground compliance should be OK
    assert out["locked"]["ok"] is True
    assert out["locked"]["ground_ok"] is True


def test_curved_path():
    if not os.path.exists(XML):
        return
    amp = f"{_amp_scale_from_policy():.5f}"
    p = run([mjpython(), "tools/p2_verify_motion.py", "--xml", XML,
             "--seconds", "8.0", "--omega", "0.05", "--amp-scale", amp,
             "--v-cmd", "0.12", "--yaw-cmd", "0.3"])
    assert p.returncode == 0, p.stderr
    out = _last_json(p.stdout)
    # Ring metric should be within preview thresholds; contact quality OK
    assert out["locked"]["ring_ok"] is True
    assert out["locked"]["contact_ok"] is True


def test_early_touchdown_adaptation():
    # Pure function test (no MuJoCo): early touchdown triggers swing->stance and phi snap to AEP
    from control.cpg_circle import PhaseState, update_phase_on_touchdown
    st = PhaseState(phi=1.2, mode="swing", phi_AEP=2.4, phi_PEP=5.5)
    st2 = update_phase_on_touchdown(st, contact_now=True)
    assert st2.mode == "stance"
    # phi should be set to AEP (mod 2π)
    assert abs((st2.phi - 2.4 + math.pi*2) % (math.pi*2)) < 1e-9
