import os, subprocess, json

XML = os.path.join("mjcf", "jethexa_lab.xml")

def mjpython():
    return os.environ.get("MJPYTHON", "mjpython")

def run(cmd):
    return subprocess.run(cmd, check=False, capture_output=True, text=True)

def last_json(stdout: str):
    lines = [l for l in stdout.splitlines() if l.strip().startswith("{")]
    return json.loads(lines[-1]) if lines else {}

def test_support_polygon_margin_nonnegative():
    if not os.path.exists(XML):
        return
    # Use push-safe helper to step the simulation; it prints margins before/after
    p = run([mjpython(),"tools/p0p1_verify.py","--xml",XML,
             "--push-safe","robot","--dir","x","--deltav","0.0","--steps","1"])  # zero push, just step
    assert p.returncode == 0, p.stderr
    out = last_json(p.stdout)
    # After one step at neutral pose, margin should be >= 0 if feet are on ground
    assert out.get("margin_after_m", -1.0) >= -1e-3, out

