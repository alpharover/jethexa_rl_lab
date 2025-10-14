#!/usr/bin/env python3
import os, shutil, sys, xml.etree.ElementTree as ET
from pathlib import Path

ROOT_NEW = Path(__file__).resolve().parents[1]
ROOT_OLD = Path("/Users/alpha_dev/robotics_repos/jethexa/jethexa_mj_lab")
xml_path = ROOT_NEW/"mjcf/jethexa_lab.xml"
dst_root = ROOT_NEW/"meshes/simplify"
dst_root.mkdir(parents=True, exist_ok=True)

mesh_files = set()
root = ET.parse(xml_path).getroot()
for m in root.findall(".//mesh"):
  f = (m.get("file") or "").strip()
  if f and not f.startswith("/"):
    mesh_files.add(f)

copied, missing = [], []
for rel in sorted(mesh_files):
  src = ROOT_OLD/rel
  dst = ROOT_NEW/rel
  dst.parent.mkdir(parents=True, exist_ok=True)
  if src.is_file():
    if not dst.exists():
      shutil.copy2(src, dst)
    copied.append(str(rel))
  else:
    missing.append(str(rel))

out = ROOT_NEW/".proofs/a4_vendor_meshes.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(__import__("json").dumps({"copied":copied,"missing":missing}, indent=2))
print(f"[a4_vendor] copied={len(copied)} missing={len(missing)} → {out}")
if missing: sys.exit(3)
