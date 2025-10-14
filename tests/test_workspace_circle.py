import os, json, subprocess

XML = os.path.join("mjcf", "jethexa_lab.xml")

def mjpython():
    return os.environ.get("MJPYTHON", "mjpython")

def run(cmd):
    return subprocess.run(cmd, check=False, capture_output=True, text=True)

def last_json(stdout: str):
    lines = [l for l in stdout.splitlines() if l.strip().startswith("{")]
    return json.loads(lines[-1]) if lines else {}

def test_workspace_circle():
    if not os.path.exists(XML):
        return
    p = run([mjpython(),"tools/p0p1_verify.py","--xml",XML,"--workspace-circle"])
    assert p.returncode == 0, p.stderr
    out = last_json(p.stdout)
    assert out.get("global_r_recommended_m", 0.0) > 0.05, out
    # Also validate policy file constraints if present
    pol_path = os.path.join("configs","policy","workspace_circle.json")
    if os.path.exists(pol_path):
        with open(pol_path) as f:
            pol = json.load(f)
        r_paper = float(pol.get("r_paper", pol.get("r_paper_m", 0.0)))
        r_ins = float(pol.get("r_inscribed_min", pol.get("r_inscribed_min_m", 0.0)))
        s = float(pol.get("s", pol.get("paper_scale_s", 1.0)))
        alpha = float(pol.get("alpha", pol.get("alpha_inscribed", 0.8)))
        r_ctrl = float(pol.get("r_ctrl", pol.get("r_ctrl_recommended_m", 0.0)))
        assert r_ctrl <= s * r_paper + 1e-9
        assert r_ctrl <= alpha * r_ins + 1e-9
