import json, subprocess, os, pytest

XML = os.path.join("mjcf", "jethexa_lab.xml")

def run(cmd):
    return subprocess.run(cmd, check=False, capture_output=True, text=True)

def _last_json(stdout: str):
    lines = [l for l in stdout.splitlines() if l.strip().startswith("{")]
    return json.loads(lines[-1]) if lines else {}

@pytest.mark.xfail(reason="Analytic IK not yet implemented; parity gate deferred to next phase", strict=False)
def test_ik_parity():
    if not os.path.exists(XML):
        return
    p = run(["mjpython","tools/p0p1_verify.py","--xml",XML,"--ik","all"])
    out = _last_json(p.stdout)
    assert "ik_ok" in out
    assert out["ik_ok"], f"IK parity violations: {out.get('violations')}\nSTDERR={p.stderr}"

def test_workspace_circle():
    if not os.path.exists(XML):
        return
    p = run(["mjpython","tools/p0p1_verify.py","--xml",XML,"--workspace-circle"])
    assert p.returncode==0, p.stderr
    out = _last_json(p.stdout)
    assert out.get("global_r_recommended_m", 0.0) >= 0.08, out
