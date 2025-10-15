#!/usr/bin/env python3
from __future__ import annotations

"""Headless 2D (top-down) renderer for the baseline gait.

Generates MP4s without any OpenGL by drawing world-XY overlays using Pillow
and writing frames via imageio. Shows:
  - Foot positions (contact=green, flight=magenta)
  - Body COM (yellow)
  - Per-leg workspace ring (controller r_ctrl about circle center cw)
  - Support polygon hull (white)

Usage:
  mjpython tools/render_gait_topdown.py --scenario straight --out docs/final_gait_straight.mp4
  mjpython tools/render_gait_topdown.py --scenario turn --out docs/final_gait_turn.mp4
  mjpython tools/render_gait_topdown.py --scenario curve --out docs/final_gait_curve.mp4
"""

import argparse, json, math, sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import mujoco
from PIL import Image, ImageDraw
import imageio

from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.p0p1_verify import (
    LEG_ORDER,
    hip_frame_axes,
    link_lengths_from_xml,
    paper_circle_params,
    support_polygon_from_contacts,
)
from tools.p2_verify_motion import engine_numeric_ik_leg, foot_world
from control.cpg_circle import (
    CirclePolicy,
    TripodPhaser,
    foot_target_hi1,
    stride_limits_about_cw,
    motion_center,
)
from control.ik_analytic import LegGeom, choose_branch


def draw_frame(img: Image.Image, xy_data: Dict[str, Any], scale=600, center=(400, 400)):
    draw = ImageDraw.Draw(img)
    cx, cy = center

    def to_px(p):
        # world XY meters -> image pixels (y up)
        return (int(cx + scale * float(p[0])), int(cy - scale * float(p[1])))

    # Background grid
    for r in np.linspace(0.05, 0.35, 7):
        draw.ellipse([cx - r*scale, cy - r*scale, cx + r*scale, cy + r*scale], outline=(30,30,30))
    draw.line([cx-360, cy, cx+360, cy], fill=(60,60,60))
    draw.line([cx, cy-360, cx, cy+360], fill=(60,60,60))

    # Workspace rings per leg
    for leg, ring in xy_data.get('rings', {}).items():
        pts = [to_px(p) for p in ring]
        for i in range(len(pts)):
            draw.line([pts[i], pts[(i+1)%len(pts)]], fill=(80,80,200))

    # Support polygon
    hull = xy_data.get('hull')
    if isinstance(hull, np.ndarray) and hull.shape[0] >= 3:
        pts = [to_px(p) for p in hull]
        draw.polygon(pts, outline=(255,255,255))

    # Feet: contact vs flight
    for leg, p in xy_data.get('feet', {}).items():
        pos = to_px(p['xy'])
        col = (0,255,0) if p['contact'] else (255,0,255)
        r = 6
        draw.ellipse([pos[0]-r,pos[1]-r,pos[0]+r,pos[1]+r], fill=col)

    # Body COM
    com = xy_data.get('com')
    if com is not None:
        px = to_px(com)
        draw.ellipse([px[0]-5,px[1]-5,px[0]+5,px[1]+5], fill=(255,255,0))


def simulate_and_render(xml: str, seconds: float, amp_scale: float, omega: float, v_cmd: float, yaw_cmd: float, out_path: Path):
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    a1, a2, a3 = link_lengths_from_xml(m)
    zH = float(np.mean([float(hip_frame_axes(m, d, leg)[0][2]) for leg in LEG_ORDER]))
    r_paper, d_cw = paper_circle_params(a1, a2, a3, zH)
    # Load pinned r_ctrl
    pol_conf = {}
    try:
        pol_conf = json.loads(Path('configs/policy/workspace_circle.json').read_text())
    except Exception:
        pol_conf = {}
    r_ctrl = float(pol_conf.get('r_ctrl', r_paper))
    pol = CirclePolicy(r_paper=r_paper, r_inscribed_min=r_paper, r_ctrl=r_ctrl, s=float(pol_conf.get('s',1.0)), alpha=float(pol_conf.get('alpha',1.0)), d_cw=d_cw, height=-zH, lift=0.0, duty=1.0, omega=2*math.pi*omega)
    phaser = TripodPhaser(("LF","RM","LR"),("RF","LM","RR"))

    dt = float(m.opt.timestep)
    steps = max(2, int(seconds / dt))
    # Map actuators for setting control
    act_map = {}
    for leg in LEG_ORDER:
        for jname in (f"coxa_joint_{leg}", f"femur_joint_{leg}", f"tibia_joint_{leg}"):
            aname = f"pos_{jname}"
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid >= 0:
                act_map[(leg, jname)] = aid

    # IK offsets if present
    ik_off = {}
    try:
        off = json.loads(Path("configs/calib/ik_offsets.json").read_text())
        ik_off = off.get("legs", {})
    except Exception:
        pass

    # Foot geom sizes (radius)
    foot_r = {}
    for leg in LEG_ORDER:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}")
        foot_r[leg] = float(m.geom_size[gid][0]) if gid >= 0 else 0.012

    # Writer
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=int(1.0/dt))

    base_phi = 0.0
    for k in range(steps):
        base_phi = (base_phi + pol.omega * dt) % (2*math.pi)
        # command
        for leg in LEG_ORDER:
            hip_pos, hx, hy, hz = hip_frame_axes(m, d, leg)
            phi_leg = phaser.phase_for_leg(leg, base_phi)
            p_hi1 = foot_target_hi1(phi_leg, pol).copy()
            cx = pol.d_cw
            p_hi1[0] = cx + float(amp_scale) * (p_hi1[0] - cx)
            p_hi1[1] = float(amp_scale) * p_hi1[1]
            dz = ((-hip_pos[2] + foot_r[leg]) - pol.height)
            p_hi1[2] += dz
            # Map analytic IK to engine and set targets
            g = LegGeom(a1=a1, a2=a2, a3=a3, gamma=0.0)
            q_hi1 = choose_branch(p_hi1, g, ((-math.pi, math.pi),)*3)
            delta = np.array(ik_off.get(leg, {}).get("delta", [0.0,0.0,0.0]), dtype=float)
            q_leg = np.array([q_hi1[0]+delta[0], q_hi1[1]+delta[1], q_hi1[2]+delta[2]], dtype=float)
            for idx, jname in enumerate((f"coxa_joint_{leg}", f"femur_joint_{leg}", f"tibia_joint_{leg}")):
                aid = act_map.get((leg, jname), -1)
                if aid >= 0:
                    lo, hi = m.actuator_ctrlrange[aid]
                    d.ctrl[aid] = float(np.clip(q_leg[idx], lo, hi))
        mujoco.mj_step(m, d)

        # Collect top-down features
        torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot')
        com = np.array(d.subtree_com[torso][:2])
        hull = support_polygon_from_contacts(m, d, [f"foot_{leg}" for leg in LEG_ORDER])

        feat = {"feet": {}, "rings": {}, "hull": hull, "com": com}
        for leg in LEG_ORDER:
            pw = foot_world(m, d, leg)
            # contact flag
            in_contact = 0
            for ci in range(d.ncon):
                c = d.contact[ci]
                if c.geom1 >= 0 and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) == f"foot_{leg}"):
                    in_contact = 1; break
                if c.geom2 >= 0 and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) == f"foot_{leg}"):
                    in_contact = 1; break
            feat['feet'][leg] = {"xy": pw[:2], "contact": bool(in_contact)}
            # ring points
            hip_pos, hx, hy, hz = hip_frame_axes(m, d, leg)
            cw = hip_pos + pol.d_cw * hx
            th = np.linspace(0, 2*math.pi, 48, endpoint=False)
            ring = np.stack([cw[:2] + pol.r_ctrl * (np.cos(t)*hx[:2] + np.sin(t)*hy[:2]) for t in th])
            feat['rings'][leg] = ring

        # Draw
        img = Image.new('RGB', (800, 800), color=(0,0,0))
        draw_frame(img, feat, scale=650, center=(400, 500))
        writer.append_data(np.asarray(img))

    writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["straight","turn","curve"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", default="configs/env/hex_v0.json")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--omega", type=float, default=0.05)
    ap.add_argument("--amp-scale", type=float, default=0.027)
    args = ap.parse_args()

    cfg = {}
    try:
        cfg = json.loads(Path(args.env).read_text())
    except Exception:
        pass
    xml = cfg.get("xml_path", "mjcf/jethexa_lab.xml")

    if args.scenario == 'straight':
        v, w = 0.12, 0.0
    elif args.scenario == 'turn':
        v, w = 0.0, 0.2
    else:
        v, w = 0.12, 0.3

    simulate_and_render(xml, args.seconds, args.amp_scale, args.omega, v, w, Path(args.out))
    print(json.dumps({"saved": args.out, "seconds": args.seconds, "omega": args.omega}, indent=2))


if __name__ == '__main__':
    main()

