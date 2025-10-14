#!/usr/bin/env python3
import json, os, sys, time, traceback
from pathlib import Path
from typing import Any, Dict, List
import math
import xml.etree.ElementTree as ET

# Use local repo root (jethexa_rl) and its vendored mjcf/
OLD_ROOT = Path(__file__).resolve().parents[1]
MJCF_DIR = OLD_ROOT / "mjcf"
OUT_DIR = Path("/Users/alpha_dev/robotics_repos/jethexa/jethexa_rl/.proofs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

result: Dict[str, Any] = {
    "model_path": None,
    "link_lengths": None,
    "joint_axes": None,
    "gravity_sign": None,
    "contact_params": None,
    "notes": [],
}

workspace: Dict[str, Any] = {
    "AEP_deg": None,
    "PEP_deg": None,
    "radius_estimate_m": None,
    "notes": [],
}

# Discover an MJCF file to use
model_xml = None
if MJCF_DIR.is_dir():
    cands = [p for p in MJCF_DIR.glob("*.xml") if p.is_file()]
    if cands:
        # Prefer jethexa_lab.xml if present
        for p in cands:
            if p.name == "jethexa_lab.xml":
                model_xml = p
                break
        if model_xml is None:
            model_xml = cands[0]
else:
    result["notes"].append("No mjcf directory found")

if model_xml:
    result["model_path"] = str(model_xml)
else:
    result["notes"].append("No model XML discovered; emitting placeholders")

# Try a light XML parse for axes and gravity without mujoco runtime
axes: List[Any] = []
gravity_sign = None
try:
    if model_xml:
        root = ET.parse(model_xml).getroot()
        # gravity from option tag
        opt = root.find("option")
        if opt is not None and opt.get("gravity"):
            try:
                gz = float(opt.get("gravity").split()[2]) if len(opt.get("gravity").split())>=3 else float(opt.get("gravity"))
                gravity_sign = 1 if gz > 0 else -1
            except Exception:
                gravity_sign = -1
        # joint axes (grab up to 12 revolute joints)
        for j in root.findall(".//joint"):
            a = j.get("axis")
            if a:
                try:
                    vec = [float(x) for x in a.split()]
                    axes.append(vec)
                except Exception:
                    axes.append(a)
            if len(axes) >= 12:
                break
        if not axes:
            result["notes"].append("No joint axes found in XML; will use placeholders")
except Exception as e:
    result["notes"].append(f"XML parse failure: {e}")

# Attempt optional mujoco import for link estimates (best‑effort)
link_lengths = None
contact = None
try:
    import mujoco  # type: ignore
    if model_xml:
        m = mujoco.MjModel.from_xml_path(str(model_xml))
        # Heuristic: compute avg geom size as proxy for link lengths
        if m.ngeom > 0:
            sizes = []
            for gi in range(m.ngeom):
                sz = m.geom_size[gi*3:(gi+1)*3]
                # take max dimension per geom
                sizes.append(max(sz))
            if sizes:
                avg = sum(sizes)/len(sizes)
                link_lengths = {
                    "avg_geom_extent_m": float(avg),
                    "n_geoms": int(m.ngeom),
                }
        # Contact parameters from default (mu, solref/solimp)
        contact = {
            "friction": [float(x) for x in m.geom_friction[:3]],
        }
        if gravity_sign is None:
            gz = m.opt.gravity[2]
            gravity_sign = 1 if gz > 0 else -1
except Exception as e:
    result["notes"].append(f"mujoco import/compile skipped: {e.__class__.__name__}: {e}")

# Fallbacks/placeholders
if gravity_sign is None:
    gravity_sign = -1
if not axes:
    axes = [[0.0, 0.0, 1.0]] * 12
if link_lengths is None:
    link_lengths = {
        "avg_geom_extent_m": None,
        "n_geoms": None,
        "reason": "No mujoco runtime; using placeholders",
    }
if contact is None:
    contact = {
        "friction": [0.8, 0.005, 0.0001],
        "reason": "No mujoco runtime; using placeholders",
    }

result["joint_axes"] = axes
result["gravity_sign"] = gravity_sign
result["link_lengths"] = link_lengths
result["contact_params"] = contact

# Workspace heuristics (AEP/PEP, radius) — placeholders unless better info present
try:
    # crude estimate: limb reach ~ 0.18 m, AEP/PEP ~ ±60 deg for hexapod
    radius = 0.18
    AEP = 60.0
    PEP = -60.0
    # If XML had joint range hints, could refine; skipping for A0
    workspace.update({
        "AEP_deg": AEP,
        "PEP_deg": PEP,
        "radius_estimate_m": radius,
    })
except Exception:
    workspace["notes"].append("Workspace defaults used")

# Emit JSON artifacts
geom_path = OUT_DIR / "geom_scan.json"
work_path = OUT_DIR / "workspace_probe.json"
with geom_path.open("w") as f:
    json.dump(result, f, indent=2)
with work_path.open("w") as f:
    json.dump(workspace, f, indent=2)
print(f"Wrote {geom_path} and {work_path}")
