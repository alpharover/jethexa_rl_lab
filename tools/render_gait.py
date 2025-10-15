#!/usr/bin/env python3
from __future__ import annotations

"""Off-screen renderer for the baseline gait preview (three scenarios).

Produces small MP4 clips in docs/ for straight, turn-in-place, and curved.
This is a tiny wrapper around the same target generator used by
tools/p2_verify_motion.py, but it captures frames to a video file using
MuJoCo's off-screen renderer. If imageio is unavailable, it falls back to
writing PNG frames to docs/ and prints a hint to stitch them with ffmpeg.

Usage:
  mjpython tools/render_gait.py --scenario straight --out docs/final_gait_straight.mp4
  mjpython tools/render_gait.py --scenario turn --out docs/final_gait_turn.mp4
  mjpython tools/render_gait.py --scenario curve --out docs/final_gait_curve.mp4
"""

import argparse
import os
import sys
from pathlib import Path
import math

import numpy as np
import mujoco

# Repo root on sys.path
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.p2_verify_motion import run_preview  # reuse the gait target loop


def render_frames(m: mujoco.MjModel, d: mujoco.MjData, T: int, width: int = 960, height: int = 540):
    """Yield RGB frames using MuJoCo's renderer (works headless)."""
    # Try the simple Renderer API (MuJoCo >= 3.0)
    try:
        renderer = mujoco.Renderer(m, width, height)
        for _ in range(T):
            renderer.update_scene(d)
            yield renderer.render()
    except Exception:
        # Low-level fallback via mjv/mjr
        scn = mujoco.MjvScene(m, maxgeom=2048)
        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_100)
        rect = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_defaultCamera(cam)
        for _ in range(T):
            mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
            mujoco.mjr_render(rect, scn, con)
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(rgb, None, rect, con)
            yield rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="configs/env/hex_v0.json")
    ap.add_argument("--scenario", choices=["straight", "turn", "curve"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--omega", type=float, default=0.05)
    ap.add_argument("--amp-scale", type=float, default=0.08)
    args = ap.parse_args()

    cfg = {}
    try:
        cfg = __import__('json').loads(Path(args.env).read_text())
    except Exception:
        pass
    xml_path = cfg.get("xml_path", "mjcf/jethexa_lab.xml")

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    # Scenario commands
    if args.scenario == "straight":
        v_cmd, yaw_cmd = 0.15, 0.0
    elif args.scenario == "turn":
        v_cmd, yaw_cmd = 0.0, 0.6
    else:  # curve
        v_cmd, yaw_cmd = 0.12, 0.3

    # We'll advance the simulation using the same gait loop used by the motion verifier,
    # but we need to capture frames ourselves. So we run a shadow loop: step one tick,
    # then render a frame.
    # To leverage run_preview's target generator without code duplication we reuse its
    # internal stepping by calling it for 1 dt repeatedly via a tiny shim that patches
    # its seconds argument and returns after one step. For simplicity here, we replicate
    # the stepping logic at a coarse level by calling mj_step and rendering every dt.
    # We use a small wrapper around run_preview to ensure the controller state is warmed up.

    # Warm-up: one short call to establish offsets/IK cache (no rendering)
    _ = run_preview(xml_path, seconds=min(0.1, args.seconds), amp_scale=args.amp_scale, omega=args.omega,
                    lock_coxa=False, lift=0.0, duty=1.0, enforce_stride=True, substeps=2, ik_iters=5,
                    v_cmd=v_cmd, yaw_cmd=yaw_cmd)

    # Main capture run: re-create model/data to reset time and record deterministically
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    # Drive the preview loop by calling the same run_preview but intercept stepping
    # by duplicating the key stepping section inline here (to avoid refactoring tools/).
    from tools.p2_verify_motion import (CirclePolicy, TripodPhaser, foot_target_hi1, stride_limits_about_cw,
                                        LegGeom, choose_branch, LEG_ORDER, hip_frame_axes, link_lengths_from_xml, paper_circle_params)
    a1, a2, a3 = link_lengths_from_xml(m)
    zH = float(np.mean([float(hip_frame_axes(m, d, leg)[0][2]) for leg in LEG_ORDER]))
    r_paper, d_cw = paper_circle_params(a1, a2, a3, zH)
    pol = CirclePolicy(r_paper=r_paper, r_inscribed_min=r_paper, r_ctrl=r_paper, s=1.4, alpha=0.8, d_cw=d_cw, height=-zH, lift=0.0, duty=1.0, omega=2*math.pi*args.omega)
    # Load pinned r_ctrl if present
    try:
        cfg_pol = __import__('json').loads(Path("configs/policy/workspace_circle.json").read_text())
        pol.r_ctrl = float(cfg_pol.get("r_ctrl", pol.r_ctrl))
    except Exception:
        pass
    phaser = TripodPhaser(("LF","RM","LR"),("RF","LM","RR"))

    # per-actuator target rate limiter matching the verifier (5.82 rad/s)
    omega_max = 5.82
    dctrl_max = omega_max * float(m.opt.timestep)
    prev_ctrl = np.array(d.ctrl, copy=True)

    # Map actuators for setting ctrl
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
        ik_off = __import__('json').loads(Path("configs/calib/ik_offsets.json").read_text())
    except Exception:
        ik_off = {}

    # Frame writer
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out.with_suffix("")
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Determine number of simulation steps
    dt = float(m.opt.timestep)
    steps = max(2, int(args.seconds / dt))
    # Video sink (imageio if available)
    writer = None
    use_imageio = False
    try:
        import imageio
        writer = imageio.get_writer(str(out), fps=int(1.0/dt))
        use_imageio = True
    except Exception:
        writer = None

    frame_gen = render_frames(m, d, steps)

    base_phi = 0.0
    for step_idx in range(steps):
        # advance phase
        base_phi = (base_phi + pol.omega * dt) % (2*math.pi)
        for leg in LEG_ORDER:
            hip_pos, hx, hy, hz = hip_frame_axes(m, d, leg)
            phi_leg = phaser.phase_for_leg(leg, base_phi)
            p_hi1 = foot_target_hi1(phi_leg, pol).copy()
            cx = pol.d_cw
            p_hi1[0] = cx + float(args.amp_scale) * (p_hi1[0] - cx)
            p_hi1[1] = float(args.amp_scale) * p_hi1[1]
            dz = ((-hip_pos[2] + float(m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}")][0])) - pol.height)
            p_hi1[2] += dz
            g = LegGeom(a1=a1, a2=a2, a3=a3, gamma=0.0)
            q_hi1 = choose_branch(p_hi1, g, ((-math.pi, math.pi),) * 3)
            # Map to engine joint signs + offsets if available
            delta = np.array(ik_off.get(leg, {}).get("delta", [0.0,0.0,0.0]), dtype=float)
            q_leg = np.array([q_hi1[0]+delta[0], q_hi1[1]+delta[1], q_hi1[2]+delta[2]], dtype=float)
            for k, jnm in enumerate((f"coxa_joint_{leg}", f"femur_joint_{leg}", f"tibia_joint_{leg}")):
                aid = act_map.get((leg, jnm), -1)
                if aid < 0:
                    continue
                lo, hi = m.actuator_ctrlrange[aid]
                raw = float(np.clip(q_leg[k], lo, hi))
                prev = float(prev_ctrl[aid])
                raw = float(np.clip(raw, prev - dctrl_max, prev + dctrl_max))
                d.ctrl[aid] = raw
                prev_ctrl[aid] = raw
        mujoco.mj_step(m, d)
        # Render and write
        frame = next(frame_gen)
        if use_imageio:
            writer.append_data(frame)
        else:
            # Fallback: PNG frames
            import PIL.Image as Image
            Image.fromarray(frame).save(frames_dir / f"{step_idx:06d}.png")

    if writer is not None and use_imageio:
        writer.close()
    elif not use_imageio:
        print(f"[render_gait] imageio not available; wrote PNG frames to {frames_dir}.\n"
              f"Combine with: ffmpeg -r {int(1.0/dt)} -i {frames_dir}/%06d.png -c:v libx264 -pix_fmt yuv420p {out}")


if __name__ == "__main__":
    main()

