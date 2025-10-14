import json, subprocess, os

XML = os.path.join("mjcf", "jethexa_lab.xml")

def run(cmd):
    return subprocess.run(cmd, check=False, capture_output=True, text=True)

def _last_json(stdout: str):
    lines = [l for l in stdout.splitlines() if l.strip().startswith("{")]
    return json.loads(lines[-1]) if lines else {}

def test_push_safe_static_margin():
    if not os.path.exists(XML):
        return
    p = run(["mjpython","tools/p0p1_verify.py","--xml",XML,
             "--push-safe","robot","--dir","x","--deltav","0.2","--steps","10"])
    assert p.returncode==0, p.stderr
    out = _last_json(p.stdout)
    assert out.get("margin_after_m", -1.0) >= 0.0, out

