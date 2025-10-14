#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive MuJoCo teleop for JetHexa — clean, single-file.
Features:
  • Stable stance-hold (targets=0) on load
  • Tripod in-place stepping preview (diagonal tripods)
  • Yaw-in-place preview (coxa sweep)
  • External push impulses (apply world-space forces)
  • Minimal key conflicts (passive viewer; UI panes hidden)
  • Works with mjpython or python (pip install mujoco)

Keymap (printed at start):
  [1] Hold (stance)        [2] Tripod step         [3] Yaw left      [4] Yaw right
  [U] Forward push         [J] Left push           [K] Backward push [L] Right push
  [I] Up push              [M] Down push           [R] Random push
  [,] Decrease speed       [.] Increase speed      [/] Reset speed
  [Q] Quit

Tripod groups (diagonals): A = LF+RM+LR,  B = RF+LM+RR

Notes:
  • We hide the viewer UI panes to reduce hotkey collisions. The viewer still
    has a few baked-in keys (space pauses, etc.). Avoid using those during teleop.
  • Actuation assumes MuJoCo <position> actuators: ctrl[i] = desired position (rad).
  • 0 control is your "good stance" per your XML; we only add small deltas.
"""
import argparse
import itertools
import math
import os
import random
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer


# --------------------------- Small utilities ---------------------------

def leg_tags():
    return ["LF", "LM", "LR", "RF", "RM", "RR"]

def build_indices(model):
    """Collect actuator indices by (joint_type, leg_tag)."""
    tags = leg_tags()
    # Expect names like: pos_coxa_joint_LF, etc.
    def aid(name):
        try:
            return model.actuator(name).id
        except KeyError as e:
            raise RuntimeError(f"Actuator '{name}' not found in model.") from e

    a = {}
    for t in ["coxa", "femur", "tibia"]:
        for leg in tags:
            a[(t, leg)] = aid(f"pos_{t}_joint_{leg}")
    return a

def side_sign(leg):
    """+1 for left legs, -1 for right legs."""
    return +1.0 if leg.startswith("L") else -1.0

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def print_keymap():
    print(
        "\n[teleop] keymap:\n"
        "  1 Hold (stance)    2 Tripod step     3 Yaw left       4 Yaw right\n"
        "  U +X push          J +Y push         K -X push         L -Y push\n"
        "  I +Z push          M -Z push         R random push\n"
        "  , slower           . faster          / reset speed     Q quit\n"
    )


# --------------------------- Control modes -----------------------------

class Modes:
    HOLD = "hold"
    TRIPOD = "tripod"
    YAW_L = "yaw_left"
    YAW_R = "yaw_right"


# --------------------------- Controller --------------------------------

class HexaTeleop:
    """
    Simple target generator for position actuators.

    We keep a "base" target at zero, and add small, sinusoidal deltas
    per joint depending on mode. All deltas are side-signed so L/R
    move consistently despite flipped joint axes.

    We also support timed world-space push impulses via mj_applyFT.
    """
    def __init__(self, model, data, args):
        self.m = model
        self.d = data
        self.aid = build_indices(model)

        self.mode = Modes.HOLD
        self.speed = args.speed  # global gait frequency scale (Hz)
        self.A_coxa = args.coxa   # rad; 0 by default to avoid lateral drift
        self.A_femur = args.femur # rad
        self.A_tibia = args.tibia # rad (negative bends "down" if sign not flipped)

        # Diagonal tripod grouping
        self.tripod_A = {"LF", "RM", "LR"}
        self.tripod_B = {"RF", "LM", "RR"}

        # Force impulse bookkeeping
        self.impulse_until = 0.0
        self.impulse_force = np.zeros(3)
        self.impulse_point = np.zeros(3)  # apply at body COM (0,0,0 in body)
        self.push_body = self.m.body("robot").id  # root body name in your XML

        # Cache actuator ranges for safe clamping
        self.act_ctrlrange = self._actuator_ctrlranges()

        # Pre-zero the controls to ensure stance hold
        self._zero_controls()

        # Calibrate per-leg lift signs so +delta q lifts the foot in +Z.
        # This avoids left/right axis mirroring issues discovered in teleop.
        self.lift_sign = {leg: {"femur": 1.0, "tibia": 1.0} for leg in leg_tags()}
        try:
            self._calibrate_lift_signs()
        except Exception as e:
            print(f"[teleop] lift sign calibration skipped: {e}")

    def _zero_controls(self):
        self.d.ctrl[:] = 0.0

    def hard_hold(self):
        """Immediate stop: switch to HOLD and zero control targets."""
        self.mode = Modes.HOLD
        self._zero_controls()

    def _actuator_ctrlranges(self):
        """Return per-actuator control range (lo,hi) in a dict keyed by actuator id."""
        lo = self.m.actuator_ctrlrange[:, 0].copy()
        hi = self.m.actuator_ctrlrange[:, 1].copy()
        return {i: (lo[i], hi[i]) for i in range(self.m.nu)}

    # ---------------- Pushes / impulses ----------------

    def schedule_push(self, world_dir, magnitude, duration=0.035):
        """Schedule a brief world-space push on the 'robot' body."""
        v = np.array(world_dir, dtype=float)
        n = np.linalg.norm(v)
        if n == 0.0:
            return
        v /= n
        self.impulse_force = magnitude * v
        self.impulse_until = self.d.time + max(1e-3, duration)

    def _apply_push_if_needed(self):
        """Apply force during the scheduled window using mj_applyFT."""
        if self.d.time >= self.impulse_until:
            return
        # torque = 0, apply at body's COM
        zero3 = np.zeros(3, dtype=np.float64)
        mujoco.mj_applyFT(self.m, self.d,
                          self.impulse_force, zero3,
                          self.impulse_point,
                          self.push_body, self.d.qfrc_applied)

    # ---------------- Sign calibration ----------------

    def _geom_world_pos(self, geom_name: str):
        try:
            gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        except Exception:
            gid = -1
        if gid < 0:
            return None
        return np.array(self.d.geom_xpos[gid], dtype=float)

    def _calibrate_lift_signs(self, eps=0.01):
        """Pick femur/tibia signs so +delta increases the foot's world Z height."""
        mujoco.mj_forward(self.m, self.d)
        for leg in leg_tags():
            gname = f"foot_{leg}"
            p0 = self._geom_world_pos(gname)
            if p0 is None:
                continue
            for jt in ("femur", "tibia"):
                aid = self.aid[(jt, leg)]
                j_id = int(self.m.actuator_trnid[aid, 0])
                qj = int(self.m.jnt_qposadr[j_id])
                qsave = float(self.d.qpos[qj])
                try:
                    self.d.qpos[qj] = qsave + eps
                    mujoco.mj_forward(self.m, self.d)
                    p1 = self._geom_world_pos(gname)
                    dz = (p1[2] - p0[2]) if p1 is not None else 0.0
                    self.lift_sign[leg][jt] = 1.0 if dz >= 0.0 else -1.0
                finally:
                    self.d.qpos[qj] = qsave
                    mujoco.mj_forward(self.m, self.d)

    # -------------- Diagnostics --------------
    def probe_kinematics(self, eps=0.01):
        """Print per-leg femur/tibia foot-Z response and current signs."""
        mujoco.mj_forward(self.m, self.d)
        print("[diag] foot Z response to +eps q (mm):")
        header = "leg   femur_dz  tibia_dz   signs(F,T)"
        print("[diag] " + header)
        for leg in leg_tags():
            gname = f"foot_{leg}"
            p0 = self._geom_world_pos(gname)
            if p0 is None:
                print(f"[diag] {leg:>2}   (no foot geom)")
                continue
            row = {"femur": float('nan'), "tibia": float('nan')}
            for jt in ("femur", "tibia"):
                aid = self.aid[(jt, leg)]
                j_id = int(self.m.actuator_trnid[aid, 0])
                qj = int(self.m.jnt_qposadr[j_id])
                qsave = float(self.d.qpos[qj])
                try:
                    self.d.qpos[qj] = qsave + eps
                    mujoco.mj_forward(self.m, self.d)
                    p1 = self._geom_world_pos(gname)
                    dz = (p1[2] - p0[2]) * 1e3 if p1 is not None else float('nan')
                    row[jt] = dz
                finally:
                    self.d.qpos[qj] = qsave
                    mujoco.mj_forward(self.m, self.d)
            sF = int(self.lift_sign[leg]["femur"]) if leg in self.lift_sign else 0
            sT = int(self.lift_sign[leg]["tibia"]) if leg in self.lift_sign else 0
            print(f"[diag] {leg:>2}   {row['femur']:8.2f}  {row['tibia']:8.2f}    ({sF:+d},{sT:+d})")

    def print_mapping(self):
        """Print actuator->joint mapping and joint axes for each leg/joint."""
        print("[map] actuator -> joint (id) axis | foot geom? | match")
        for leg in leg_tags():
            has_foot = (self._geom_world_pos(f"foot_{leg}") is not None)
            for jt in ("coxa","femur","tibia"):
                try:
                    aid = self.aid[(jt, leg)]
                    j_id = int(self.m.actuator_trnid[aid, 0])
                    # Joint name (robust to API differences)
                    try:
                        jname = self.m.joint(j_id).name
                    except Exception:
                        jname = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j_id) or "<unknown>"
                    # Axis (supports either flat or (njnt,3) layout)
                    ax = self.m.jnt_axis
                    try:
                        if getattr(ax, 'ndim', 1) == 2 and ax.shape[0] == self.m.njnt:
                            axis = ax[j_id]
                        else:
                            axis = ax[j_id*3:(j_id+1)*3]
                    except Exception:
                        axis = (0.0, 0.0, 0.0)
                    # Match check
                    actual = 'coxa' if 'coxa' in jname else ('femur' if 'femur' in jname else ('tibia' if 'tibia' in jname else '?'))
                    match = 'OK' if actual == jt else f'MISMATCH:{actual}'
                    print(f"[map] {leg}:{jt:6s} -> {jname:18s} (#{j_id:02d})  axis=({axis[0]:+.0f},{axis[1]:+.0f},{axis[2]:+.0f})  foot={has_foot}  {match}")
                except Exception as e:
                    print(f"[map] {leg}:{jt:6s} -> <error: {e}>")

    # ---------------- Gaits ----------------

    def step(self, dt):
        """
        Compute new actuator targets for this time step based on the selected mode.
        We never overwrite the viewer’s changes; only set ctrl each step.
        """
        self._apply_push_if_needed()

        if self.mode == Modes.HOLD:
            # Immediate hold at zero targets
            self._zero_controls()
            return

        t = self.d.time
        w = 2.0 * math.pi * max(0.1, self.speed)  # angular frequency

        # Start from zero targets
        targets = np.zeros_like(self.d.ctrl)

        # In-place stepping uses diagonal tripods with π phase offset
        for leg in leg_tags():
            s = side_sign(leg)  # +1 for L*, -1 for R*

            if self.mode in (Modes.TRIPOD, Modes.YAW_L, Modes.YAW_R):
                # Base phase: tripod A = 0, tripod B = π
                phase = 0.0 if leg in self.tripod_A else math.pi

                # Define swing/stance by the sign of the base sine.
                # Positive half-cycle = swing; negative = stance.
                base = math.sin(w * t + phase)
                swing = max(0.0, base)
                stance = max(0.0, -base)

                # Femur: primarily contributes to lift; phase-shift by -pi/2 so
                # it peaks near mid-swing and is near 0 at touch-down.
                femur_wave = math.sin(w * t + phase - math.pi / 2.0)
                s_f = self.lift_sign[leg]["femur"]
                # Gate femur by swing, to minimize sliding during stance
                targets[self.aid[("femur", leg)]] = s_f * self.A_femur * femur_wave * swing

                # Tibia: lift only during swing to avoid sliding while in contact.
                s_t = self.lift_sign[leg]["tibia"]
                targets[self.aid[("tibia", leg)]] = s_t * self.A_tibia * swing

                # Coxa: for yaw, apply side-signed offset during stance only.
                if self.mode in (Modes.YAW_L, Modes.YAW_R) and self.A_coxa > 0.0:
                    yaw_dir = +1.0 if self.mode == Modes.YAW_L else -1.0
                    # Use smooth stance gate so only grounded tripod generates yaw torque.
                    targets[self.aid[("coxa", leg)]] = yaw_dir * s * self.A_coxa * stance

        # Clamp to actuator ctrlranges, then write
        for i in range(self.m.nu):
            lo, hi = self.act_ctrlrange[i]
            self.d.ctrl[i] = clamp(targets[i], lo, hi)


# --------------------------- Main / viewer -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True, help="Path to MJCF XML (e.g., mjcf/jethexa_lab.xml)")
    parser.add_argument("--speed", type=float, default=1.2, help="Base gait frequency (Hz)")
    parser.add_argument("--coxa", type=float, default=0.00, help="Coxa swing amplitude (rad) [default: 0]")
    parser.add_argument("--femur", type=float, default=0.22, help="Femur lift amplitude (rad)")
    parser.add_argument("--tibia", type=float, default=0.40, help="Tibia wave amplitude (rad)")
    parser.add_argument("--realtime", action="store_true", help="Keep wall-clock pace")
    parser.add_argument("--dt-mult", type=float, default=1.0, help="Step multiplier (speed sim up/down)")
    parser.add_argument("--no-ui", action="store_true", help="Also hide the left/right UI panes")
    args = parser.parse_args()

    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)

    # Teleop state
    tele = HexaTeleop(m, d, args)
    # Print calibration summary so we can sanity-check signs
    try:
        lift_map = " ".join(
            f"{leg}:(F{int(tele.lift_sign[leg]['femur']):+d},T{int(tele.lift_sign[leg]['tibia']):+d})"
            for leg in leg_tags())
        print(f"[teleop] lift signs femur/tibia: {lift_map}")
        print(f"[teleop] coxa amplitude = {tele.A_coxa:.3f} rad (set --coxa to enable yaw)")
    except Exception:
        pass

    paused = False
    quit_requested = False

    # Key handler (ASCII-ish). The viewer still processes some of its own keys; we choose non-conflicting ones.
    def on_key(keycode):
        nonlocal paused, quit_requested
        try:
            key = chr(keycode)
        except Exception:
            return
        k = key.upper()

        if k == "Q":
            quit_requested = True
        elif key == " ":
            # Let space continue to pause the sim for convenience (viewer also toggles)
            paused = not paused
        elif k == "1":
            tele.hard_hold()
            print("[teleop] mode = HOLD (hard stop)")
        elif k == "2":
            tele.mode = Modes.TRIPOD
            print("[teleop] mode = TRIPOD (in-place)")
        elif k == "3":
            if tele.A_coxa <= 1e-6:
                tele.A_coxa = 0.08
                print(f"[teleop] enabling coxa sway for yaw: A_coxa={tele.A_coxa:.2f} rad")
            tele.mode = Modes.YAW_L
            print("[teleop] mode = YAW LEFT (in-place)")
        elif k == "4":
            if tele.A_coxa <= 1e-6:
                tele.A_coxa = 0.08
                print(f"[teleop] enabling coxa sway for yaw: A_coxa={tele.A_coxa:.2f} rad")
            tele.mode = Modes.YAW_R
            print("[teleop] mode = YAW RIGHT (in-place)")

        # Pushes (world axes): U/J/K/L/I/M, plus R random
        elif k == "U":  # +X
            tele.schedule_push([+1, 0, 0], magnitude=80.0)
        elif k == "K":  # -X
            tele.schedule_push([-1, 0, 0], magnitude=80.0)
        elif k == "L":  # -Y (right)
            tele.schedule_push([0, -1, 0], magnitude=80.0)
        elif k == "J":  # +Y (left)
            tele.schedule_push([0, +1, 0], magnitude=80.0)
        elif k == "I":  # +Z (up)
            tele.schedule_push([0, 0, +1], magnitude=120.0)
        elif k == "M":  # -Z (down)
            tele.schedule_push([0, 0, -1], magnitude=120.0)
        elif k == "R":  # random push
            v = np.random.randn(3); v[2] = max(-0.2, v[2])  # prefer lateral nudges
            tele.schedule_push(v, magnitude=80.0)
        elif k == "G":
            tele.print_mapping()

        # Speed controls
        elif key == ",":
            tele.speed = max(0.2, tele.speed * 0.8)
            print(f"[teleop] speed = {tele.speed:.2f} Hz")
        elif key == ".":
            tele.speed = min(5.0, tele.speed * 1.25)
            print(f"[teleop] speed = {tele.speed:.2f} Hz")
        elif key == "/":
            tele.speed = args.speed
            print(f"[teleop] speed reset = {tele.speed:.2f} Hz")
        elif k == "D":
            tele.probe_kinematics()

    print_keymap()

    # Use the passive viewer so we fully own stepping.
    with mujoco.viewer.launch_passive(
        m, d,
        key_callback=on_key,
        show_left_ui=(not args.no_ui),
        show_right_ui=(not args.no_ui)
    ) as viewer:

        # Hide both panes by default if requested.
        if args.no_ui:
            with viewer.lock():
                viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD

        last = time.time()
        while viewer.is_running() and not quit_requested:
            now = time.time()
            frame_dt = max(0.0, now - last)
            last = now

            if not paused:
                # Substep to approximate physics timestep rate
                n_sub = max(1, min(200, int(round(frame_dt / m.opt.timestep)) or 1))
                n_sub = int(n_sub * max(1.0, args.dt_mult))
                for _ in range(n_sub):
                    tele.step(m.opt.timestep)
                    mujoco.mj_step(m, d)

            # Sync GUI with physics (state-only is sufficient here)
            viewer.sync(state_only=True)

            if args.realtime:
                # Sleep a little to avoid CPU spin if ahead of realtime
                time.sleep(0.001)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[teleop] ERROR: {e}")
        sys.exit(1)
