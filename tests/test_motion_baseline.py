import json, os, subprocess, sys

XML = os.path.join("mjcf","jethexa_lab.xml")

def mjpython():
    return os.environ.get("MJPY", sys.executable)

def run(args):
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def last_json(s):
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return json.loads(lines[-1]) if lines else {}


def test_motion_headless_small_arc():
    if not os.path.exists(XML):
        return
    # Choose amp-scale so r_ctrl * amp ~ 3 mm to keep absolute amplitude stable
    amp = 0.04
    try:
        data = json.loads(open(os.path.join("configs","policy","workspace_circle.json")).read())
        r_ctrl = float(data.get("r_ctrl", 0.06))
        if r_ctrl > 1e-6:
            amp = max(0.02, min(0.10, 0.003 / r_ctrl))
    except Exception:
        pass
    p = run([mjpython(),"tools/p2_verify_motion.py","--xml",XML,"--seconds","8.0","--omega","0.05","--amp-scale",f"{amp:.5f}"])
    assert p.returncode == 0, p.stderr
    out = last_json(p.stdout)
    assert out.get("locked",{}).get("tracking_ok", False), out
    assert out.get("locked",{}).get("ring_ok", False), out
    assert out.get("locked",{}).get("ground_ok", False), out
    assert out.get("locked",{}).get("limits_ok", False), out
    assert out.get("locked",{}).get("contact_ok", False), out
    # unlocked run is not fully gated yet; require at least limits_ok
    assert out.get("unlocked",{}).get("limits_ok", False), out
