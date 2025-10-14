import os
import json
import math

import numpy as np
import pytest


XML_DEFAULT = os.path.join(os.getcwd(), "mjcf", "jethexa_lab.xml")


@pytest.mark.skipif(not os.path.exists(XML_DEFAULT), reason="model XML not found")
def test_determinism_smoke():
    from tools import p0p1_check as kit
    res = kit.run_determinism(XML_DEFAULT, steps=300)
    assert isinstance(res, dict)
    assert res["equal"] is True
    assert isinstance(res["hash0"], str) and isinstance(res["hash1"], str)


@pytest.mark.skipif(not os.path.exists(XML_DEFAULT), reason="model XML not found")
def test_workspace_reasonable():
    from tools import p0p1_check as kit
    res = kit.run_workspace(XML_DEFAULT, leg="LF", plot=False)
    # Basic sanity: positive radii and plausible magnitudes for this morphology
    r95 = float(res["r95"]) ; rmax = float(res["rmax"])
    assert 0.05 <= r95 <= 0.25
    assert rmax >= r95
    # Analytic estimate is model/assumption-dependent; just ensure it's finite.
    rc = float(res["analytic"]["radius"])
    assert math.isfinite(rc)
