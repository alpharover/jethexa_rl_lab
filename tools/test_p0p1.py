
# pytest -q tests/test_p0p1.py --xml mjcf/jethexa_lab.xml
import json, subprocess, sys, os, shutil, tempfile, pathlib

def mjpython():
    # Try to find mjpython on PATH; otherwise fall back to sys.executable
    return shutil.which("mjpython") or sys.executable

def run(cmd):
    p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.stdout

def test_determinism(request):
    xml = request.config.getoption("--xml")
    out = run([mjpython(), "tools/p0p1_check.py", "--xml", xml, "--determinism"])
    j = json.loads(out)
    assert j["determinism"]["equal"], f"State hash changed under fixed rollout: {j['determinism']}"

def test_workspace_radius_reasonable(request):
    xml = request.config.getoption("--xml")
    out = run([mjpython(), "tools/p0p1_check.py", "--xml", xml, "--workspace", "LF"])
    j = json.loads(out)
    r95 = j["workspace"]["sample_radii_m"]["r_p95"]
    # Sanity band for this hexapod scale
    assert 0.04 < r95 < 0.15, f"LF r95 out of expected band: {r95}"

def pytest_addoption(parser):
    parser.addoption("--xml", action="store", default="mjcf/jethexa_lab.xml", help="Path to MJCF XML")
