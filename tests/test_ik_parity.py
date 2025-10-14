import os, json, subprocess, math

XML = os.path.join("mjcf", "jethexa_lab.xml")

def mjpython():
    return os.environ.get("MJPYTHON", "mjpython")

def run(cmd):
    return subprocess.run(cmd, check=False, capture_output=True, text=True)

def last_json(stdout: str):
    lines = [l for l in stdout.splitlines() if l.strip().startswith("{")]
    return json.loads(lines[-1]) if lines else {}

def test_ik_parity():
    if not os.path.exists(XML):
        return
    # Use p0p1_verify IK runner which reports 95%/max errors
    p = run([mjpython(),"tools/p0p1_verify.py","--xml",XML,"--ik","all"])
    out = last_json(p.stdout)
    assert "ik_ok" in out, p.stdout
    assert out["ik_ok"], f"IK parity violations: {out.get('violations')}\nSTDERR={p.stderr}"
