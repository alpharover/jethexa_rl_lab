import json, os, subprocess, sys

XML = os.path.join("mjcf","jethexa_lab.xml")

def mjpython():
    # Allow overriding the interpreter via env var so CI or local runs without
    # mujoco's mjpython shim can still execute headless tools.
    return os.environ.get("MJPY", sys.executable)

def run(args):
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def last_json(s):
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return json.loads(lines[-1]) if lines else {}


def test_gravity_and_limits():
    if not os.path.exists(XML):
        return
    p = run([mjpython(),"tools/p2_verify_physics.py","--xml",XML,"--test","gravity"])
    assert p.returncode == 0, p.stderr
    out = last_json(p.stdout)
    assert out.get("gravity_mass",{}).get("ok", False), out
    p2 = run([mjpython(),"tools/p2_verify_physics.py","--xml",XML,"--test","limits"])
    out2 = last_json(p2.stdout)
    assert out2.get("actuator_limits",{}).get("ok", False), out2


def test_static_weight_and_push_momentum():
    if not os.path.exists(XML):
        return
    p = run([mjpython(),"tools/p2_verify_physics.py","--xml",XML,"--test","static","--seconds","2.0"])
    out = last_json(p.stdout)
    assert out.get("static_weight",{}).get("ok", False), out
    p2 = run([mjpython(),"tools/p2_verify_physics.py","--xml",XML,"--test","push","--deltav","0.2","--push-steps","10"])
    out2 = last_json(p2.stdout)
    assert out2.get("push_momentum",{}).get("ok", False), out2
